"""Download and store raw Figma Help Center articles."""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HELP_CENTER_ROOT = "https://help.figma.com"
HELP_CENTER_SITEMAP_URL = f"{HELP_CENTER_ROOT}/hc/sitemap.xml"
HELP_CENTER_SOURCE_TYPE = "help_center"
EXCLUDED_SECTION_PHRASES = (
    "dev mode",
    "work together in files",
)
DEFAULT_USER_AGENT = "figma-rag-ingestion/0.1 (+https://help.figma.com)"
XML_NAMESPACES = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@dataclass(frozen=True)
class ArticleMetadata:
    document_id: str
    title: str
    source_url: str
    source_type: str
    product_area: str
    breadcrumbs: tuple[str, ...]


@dataclass(frozen=True)
class DownloadSummary:
    discovered_urls: int
    downloaded_pages: int
    skipped_existing: int
    skipped_out_of_scope: int


def build_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_sitemap_urls(
    session: requests.Session,
    sitemap_url: str = HELP_CENTER_SITEMAP_URL,
    timeout_seconds: float = 30.0,
) -> list[str]:
    seen_sitemaps: set[str] = set()
    pending_sitemaps = [sitemap_url]
    article_urls: set[str] = set()

    while pending_sitemaps:
        current_sitemap = normalize_url(pending_sitemaps.pop())
        if current_sitemap in seen_sitemaps:
            continue

        response = session.get(current_sitemap, timeout=timeout_seconds)
        response.raise_for_status()
        sitemap_urls, page_urls = parse_sitemap_xml(response.content)
        article_urls.update(url for url in page_urls if is_help_center_article_url(url))
        pending_sitemaps.extend(sitemap_urls)
        seen_sitemaps.add(current_sitemap)

    return sorted(article_urls)


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    normalized = parsed._replace(query="", fragment="", path=path)
    return normalized.geturl()


def is_help_center_article_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != "help.figma.com":
        return False
    if not parsed.path.startswith("/hc/en-us/articles/"):
        return False
    return True


def fetch_article_html(
    session: requests.Session,
    url: str,
    timeout_seconds: float = 30.0,
) -> str:
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def extract_article_metadata(url: str, html: str) -> ArticleMetadata:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    breadcrumbs = extract_breadcrumbs(soup)
    product_area = infer_product_area(breadcrumbs, title)
    document_id = build_document_id(url, title)
    return ArticleMetadata(
        document_id=document_id,
        title=title,
        source_url=normalize_url(url),
        source_type=HELP_CENTER_SOURCE_TYPE,
        product_area=product_area,
        breadcrumbs=breadcrumbs,
    )


def extract_title(soup: BeautifulSoup) -> str:
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    heading = soup.find("h1")
    if heading and heading.get_text(strip=True):
        return heading.get_text(strip=True)

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return "untitled"


def extract_breadcrumbs(soup: BeautifulSoup) -> tuple[str, ...]:
    breadcrumb_script = soup.find("script", attrs={"type": "application/ld+json"})
    if breadcrumb_script and breadcrumb_script.string:
        for candidate in parse_jsonld_candidates(breadcrumb_script.string):
            if candidate.get("@type") != "BreadcrumbList":
                continue
            elements = candidate.get("itemListElement", [])
            names = [
                item.get("name", "").strip()
                for item in elements
                if isinstance(item, dict) and item.get("name")
            ]
            if names:
                return tuple(names)

    nav = soup.find(attrs={"aria-label": re.compile("breadcrumb", re.IGNORECASE)})
    if nav:
        names = [node.get_text(" ", strip=True) for node in nav.find_all(["a", "span", "li"])]
        names = [name for name in names if name]
        if names:
            return tuple(dict.fromkeys(names))

    return ()


def parse_jsonld_candidates(raw_json: str) -> list[dict]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        graph = payload.get("@graph")
        if isinstance(graph, list):
            return [item for item in graph if isinstance(item, dict)]
        return [payload]
    return []


def parse_sitemap_xml(xml_content: bytes) -> tuple[list[str], list[str]]:
    root = ET.fromstring(xml_content)
    root_tag = strip_xml_namespace(root.tag)
    loc_nodes = [node for node in root.findall(".//sitemap:loc", XML_NAMESPACES) if node.text]

    if root_tag == "sitemapindex":
        sitemap_urls = [normalize_url(node.text) for node in loc_nodes]
        return sitemap_urls, []

    if root_tag == "urlset":
        page_urls = [normalize_url(node.text) for node in loc_nodes]
        return [], page_urls

    return [], []


def strip_xml_namespace(tag: str) -> str:
    if "}" not in tag:
        return tag
    return tag.rsplit("}", 1)[-1]


def infer_product_area(breadcrumbs: Iterable[str], title: str) -> str:
    candidates = [text for text in breadcrumbs if text]
    if candidates:
        return slugify(candidates[0])
    return slugify(title) or "unknown"


def is_article_in_scope(metadata: ArticleMetadata) -> tuple[bool, str | None]:
    searchable_parts = [metadata.source_url, metadata.title, *metadata.breadcrumbs]
    normalized_text = " ".join(part.casefold() for part in searchable_parts if part)
    for phrase in EXCLUDED_SECTION_PHRASES:
        normalized_phrase = phrase.casefold()
        slug_phrase = normalized_phrase.replace(" ", "-")
        if normalized_phrase in normalized_text or slug_phrase in normalized_text:
            return False, phrase
    return True, None


def build_document_id(url: str, title: str) -> str:
    parsed = urlparse(url)
    article_segment = parsed.path.rstrip("/").split("/")[-1]
    if article_segment:
        return f"help-center-{slugify(article_segment)}"
    return f"help-center-{slugify(title) or 'untitled'}"


def slugify(value: str) -> str:
    ascii_text = value.casefold()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    return ascii_text.strip("-")


def load_existing_manifest_urls(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    urls: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source_url = record.get("source_url")
        if source_url:
            urls.add(normalize_url(source_url))
    return urls


def write_article_html(output_dir: Path, document_id: str, html: str, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document_id}.html"
    if output_path.exists() and not overwrite:
        return output_path

    output_path.write_text(html, encoding="utf-8")
    return output_path


def append_manifest_record(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_manifest_record(metadata: ArticleMetadata, raw_file_path: Path) -> dict:
    return {
        "document_id": metadata.document_id,
        "title": metadata.title,
        "source_url": metadata.source_url,
        "source_type": metadata.source_type,
        "product_area": metadata.product_area,
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": raw_file_path.as_posix(),
    }


def download_help_center_articles(
    output_dir: Path,
    manifest_path: Path,
    sitemap_url: str = HELP_CENTER_SITEMAP_URL,
    delay_seconds: float = 0.5,
    timeout_seconds: float = 30.0,
    overwrite: bool = False,
    limit: int | None = None,
) -> DownloadSummary:
    session = build_session()
    discovered_urls = fetch_sitemap_urls(
        session=session,
        sitemap_url=sitemap_url,
        timeout_seconds=timeout_seconds,
    )
    if limit is not None:
        discovered_urls = discovered_urls[:limit]

    existing_urls = load_existing_manifest_urls(manifest_path)
    downloaded_pages = 0
    skipped_existing = 0
    skipped_out_of_scope = 0

    for index, url in enumerate(discovered_urls):
        if url in existing_urls and not overwrite:
            skipped_existing += 1
            continue

        html = fetch_article_html(session=session, url=url, timeout_seconds=timeout_seconds)
        metadata = extract_article_metadata(url=url, html=html)
        in_scope, _ = is_article_in_scope(metadata)
        if not in_scope:
            skipped_out_of_scope += 1
            continue

        raw_file_path = write_article_html(
            output_dir=output_dir,
            document_id=metadata.document_id,
            html=html,
            overwrite=overwrite,
        )
        append_manifest_record(
            manifest_path=manifest_path,
            record=build_manifest_record(metadata, raw_file_path),
        )
        downloaded_pages += 1
        print(f"Downloaded: {metadata.source_url} -> {raw_file_path.as_posix()}")

        is_last_item = index == len(discovered_urls) - 1
        if delay_seconds > 0 and not is_last_item:
            time.sleep(delay_seconds)

    return DownloadSummary(
        discovered_urls=len(discovered_urls),
        downloaded_pages=downloaded_pages,
        skipped_existing=skipped_existing,
        skipped_out_of_scope=skipped_out_of_scope,
    )
