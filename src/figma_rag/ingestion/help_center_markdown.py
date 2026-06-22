"""Convert raw Figma Help Center HTML into retrieval-friendly Markdown."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

MAIN_CONTENT_SELECTORS = (
    ".article-body",
    "[data-swiftype-name='body']",
    ".article-content",
    ".article-info",
    "main article .article-body",
    "main article",
    "main",
)
GLOBAL_REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
)
ARTICLE_REMOVE_SELECTORS = (
    "nav",
    "form",
    "button",
    "select",
    "textarea",
    "input",
    ".breadcrumbs",
    ".sub-nav",
    ".article-footer",
    ".article-ticket-form",
    ".article-votes-question",
    "#feedback-success",
    "fl-course-navigator",
    "fl-metabar",
    "fl-mobile-metabar",
    ".hide",
    "[hidden]",
    "[aria-hidden='true']",
)
BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "blockquote",
    "pre",
    "table",
    "img",
}
REQUIRED_MANIFEST_FIELDS = (
    "document_id",
    "title",
    "source_url",
    "source_type",
    "product_area",
    "raw_file_path",
)
LOW_VALUE_EXACT = {
    "yes",
    "no",
    "submit",
    "search",
    "sign up",
    "contact sales",
    "was this article helpful?",
    "tell us more",
    "submit article feedback",
    "-- please choose an option --",
}


@dataclass(frozen=True)
class ManifestRecord:
    """Metadata required to convert one raw Help Center document."""

    document_id: str
    title: str
    source_url: str
    source_type: str
    product_area: str
    raw_file_path: Path
    fetch_timestamp: str | None = None


@dataclass(frozen=True)
class ContentBlock:
    """One semantic article block and its position in the heading hierarchy."""

    block_type: str
    text: str
    markdown: str
    heading_path: tuple[str, ...]
    level: int | None = None


@dataclass(frozen=True)
class ConversionFailure:
    """A document that could not be converted."""

    document_id: str
    raw_file_path: Path
    error: str


@dataclass(frozen=True)
class ConversionSummary:
    """Result of converting all records in a raw manifest."""

    total_documents: int
    converted_documents: int
    failed_documents: int
    output_dir: Path
    processed_manifest_path: Path
    failures: tuple[ConversionFailure, ...]


def normalize_whitespace(text: str) -> str:
    """Collapse HTML whitespace without adding spaces before punctuation."""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def is_low_value_text(text: str) -> bool:
    """Return whether a text block is too small or generic to retain."""

    normalized = normalize_whitespace(text).casefold()
    return not normalized or len(normalized) < 3 or normalized in LOW_VALUE_EXACT


def load_manifest_records(manifest_path: Path) -> list[ManifestRecord]:
    """Load and validate raw Help Center JSONL manifest records."""

    records: list[ManifestRecord] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected an object in {manifest_path} at line {line_number}"
                )

            missing = [field for field in REQUIRED_MANIFEST_FIELDS if not payload.get(field)]
            if missing:
                fields = ", ".join(missing)
                raise ValueError(
                    f"Missing required fields in {manifest_path} at line "
                    f"{line_number}: {fields}"
                )

            records.append(
                ManifestRecord(
                    document_id=str(payload["document_id"]),
                    title=str(payload["title"]),
                    source_url=str(payload["source_url"]),
                    source_type=str(payload["source_type"]),
                    product_area=str(payload["product_area"]),
                    raw_file_path=Path(payload["raw_file_path"]),
                    fetch_timestamp=(
                        str(payload["fetch_timestamp"])
                        if payload.get("fetch_timestamp")
                        else None
                    ),
                )
            )

    return records


def clean_help_center_html(html: str) -> BeautifulSoup:
    """Parse HTML and remove global non-article elements."""

    soup = BeautifulSoup(html, "html.parser")
    for selector in GLOBAL_REMOVE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    return soup


def extract_article_nodes(soup: BeautifulSoup) -> tuple[Tag | None, Tag]:
    """Find the optional article header and required main article container."""

    main_node = None
    for selector in MAIN_CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate and normalize_whitespace(candidate.get_text(" ", strip=True)):
            main_node = candidate
            break

    if main_node is None:
        raise ValueError("Could not find a main content node")

    header_node = soup.select_one(".article-header")
    if header_node and not normalize_whitespace(header_node.get_text(" ", strip=True)):
        header_node = None

    for root in (header_node, main_node):
        if root is None:
            continue
        for selector in ARTICLE_REMOVE_SELECTORS:
            for node in root.select(selector):
                node.decompose()

    return header_node, main_node


def html_to_blocks(html: str, source_url: str) -> list[ContentBlock]:
    """Convert article HTML into ordered semantic content blocks."""

    soup = clean_help_center_html(html)
    header_node, main_node = extract_article_nodes(soup)
    roots: list[Tag] = []
    header_is_disjoint = (
        header_node is not None
        and header_node not in main_node.descendants
        and main_node not in header_node.descendants
    )
    if header_is_disjoint and header_node is not None:
        roots.append(header_node)
    roots.append(main_node)

    heading_stack: list[tuple[int, str]] = []
    blocks: list[ContentBlock] = []
    for root in roots:
        for node in _iter_semantic_blocks(root):
            block = _node_to_block(node, source_url, heading_stack)
            if block is not None:
                blocks.append(block)

    if not blocks:
        raise ValueError("Article content did not contain any useful blocks")
    return blocks


def _iter_semantic_blocks(container: Tag) -> Iterable[Tag]:
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        if child.name in BLOCK_TAGS:
            yield child
        else:
            yield from _iter_semantic_blocks(child)


def _node_to_block(
    node: Tag,
    source_url: str,
    heading_stack: list[tuple[int, str]],
) -> ContentBlock | None:
    name = node.name.lower()
    level = int(name[1]) if re.fullmatch(r"h[1-6]", name) else None

    if level is not None:
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if is_low_value_text(text):
            return None
        heading_stack[:] = [heading for heading in heading_stack if heading[0] < level]
        heading_stack.append((level, text))
        return ContentBlock(
            block_type="heading",
            text=text,
            markdown=f"{'#' * level} {_render_inline_children(node, source_url)}",
            heading_path=tuple(heading[1] for heading in heading_stack),
            level=level,
        )

    markdown = _render_block(node, source_url)
    text = normalize_whitespace(node.get_text(" ", strip=True))
    if not markdown or is_low_value_text(text):
        return None
    return ContentBlock(
        block_type=_block_type(name),
        text=text,
        markdown=markdown,
        heading_path=tuple(heading[1] for heading in heading_stack),
    )


def _block_type(tag_name: str) -> str:
    if tag_name in {"ul", "ol"}:
        return "list"
    if tag_name == "img":
        return "image"
    return tag_name


def _render_block(node: Tag, source_url: str) -> str:
    if node.name == "p":
        return _render_inline_children(node, source_url)
    if node.name in {"ul", "ol"}:
        return _render_list(node, source_url)
    if node.name == "blockquote":
        content = _render_container(node, source_url)
        return "\n".join(f"> {line}" if line else ">" for line in content.splitlines())
    if node.name == "pre":
        code_node = node.find("code")
        code = (code_node or node).get_text().strip("\n")
        language = _code_language(code_node)
        return f"```{language}\n{code}\n```"
    if node.name == "table":
        return _render_table(node, source_url)
    if node.name == "img":
        return _render_image(node, source_url)
    return _render_container(node, source_url)


def _render_container(container: Tag, source_url: str) -> str:
    parts: list[str] = []
    for node in _iter_semantic_blocks(container):
        if re.fullmatch(r"h[1-6]", node.name):
            level = int(node.name[1])
            rendered = f"{'#' * level} {_render_inline_children(node, source_url)}"
        else:
            rendered = _render_block(node, source_url)
        if rendered:
            parts.append(rendered)
    if parts:
        return "\n\n".join(parts)
    return _render_inline_children(container, source_url)


def _render_inline_children(node: Tag, source_url: str) -> str:
    rendered = "".join(_render_inline(child, source_url) for child in node.children)
    rendered = re.sub(r"[ \t\r\f\v]+", " ", rendered)
    rendered = re.sub(r" *\n *", "\n", rendered)
    rendered = re.sub(r"\s+([,.!?;:])", r"\1", rendered)
    return rendered.strip()


def _render_inline(node: NavigableString | Tag, source_url: str) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name == "br":
        return "\n"
    if name == "img":
        return _render_image(node, source_url)

    content = _render_inline_children(node, source_url)
    if name == "a":
        href = str(node.get("href", "")).strip()
        if not content or not href or href.startswith("#"):
            return content
        return f"[{content}]({urljoin(source_url, href)})"
    if name in {"strong", "b"} and content:
        return f"**{content}**"
    if name in {"em", "i"} and content:
        return f"*{content}*"
    if name == "code" and content:
        delimiter = "``" if "`" in content else "`"
        return f"{delimiter}{content}{delimiter}"
    if name in {"s", "del"} and content:
        return f"~~{content}~~"
    return content


def _render_image(node: Tag, source_url: str) -> str:
    alt = normalize_whitespace(str(node.get("alt", "")))
    if not alt or is_low_value_text(alt):
        return ""
    src = str(
        node.get("src") or node.get("data-src") or node.get("data-original") or ""
    ).strip()
    if src:
        return f"![{alt}]({urljoin(source_url, src)})"
    return f"[Image: {alt}]"


def _render_list(list_node: Tag, source_url: str, depth: int = 0) -> str:
    lines: list[str] = []
    items = list_node.find_all("li", recursive=False)
    for index, item in enumerate(items, start=1):
        marker = f"{index}." if list_node.name == "ol" else "-"
        inline_parts: list[str] = []
        nested_lists: list[Tag] = []
        for child in item.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                nested_lists.append(child)
            elif isinstance(child, Tag) and child.name == "p":
                inline_parts.append(_render_inline_children(child, source_url))
            else:
                inline_parts.append(_render_inline(child, source_url))

        text = normalize_whitespace(" ".join(part for part in inline_parts if part))
        if text:
            indent = "  " * depth
            lines.append(f"{indent}{marker} {text}")
        for nested in nested_lists:
            nested_markdown = _render_list(nested, source_url, depth + 1)
            if nested_markdown:
                lines.append(nested_markdown)
    return "\n".join(lines)


def _code_language(code_node: Tag | None) -> str:
    if code_node is None:
        return ""
    classes = code_node.get("class", [])
    for class_name in classes:
        match = re.fullmatch(r"(?:language|lang)-([A-Za-z0-9_+-]+)", str(class_name))
        if match:
            return match.group(1)
    return ""


def _render_table(table: Tag, source_url: str) -> str:
    rows: list[list[str]] = []
    header_flags: list[bool] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        rows.append([_table_cell(cell, source_url) for cell in cells])
        header_flags.append(any(cell.name == "th" for cell in cells))

    if not rows:
        return normalize_whitespace(table.get_text(" ", strip=True))

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if not header_flags[0]:
        rows.insert(0, [f"Column {index}" for index in range(1, width + 1)])

    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _table_cell(cell: Tag, source_url: str) -> str:
    value = _render_inline_children(cell, source_url).replace("\n", "<br>")
    return value.replace("|", r"\|")


def blocks_to_markdown(blocks: Iterable[ContentBlock]) -> str:
    """Render semantic blocks as a clean Markdown body."""

    return "\n\n".join(block.markdown for block in blocks if block.markdown).strip() + "\n"


def render_markdown_document(record: ManifestRecord, blocks: Iterable[ContentBlock]) -> str:
    """Add source metadata as YAML-compatible front matter to a Markdown body."""

    metadata = {
        "document_id": record.document_id,
        "title": record.title,
        "source_url": record.source_url,
        "source_type": record.source_type,
        "product_area": record.product_area,
    }
    front_matter = ["---"]
    front_matter.extend(
        f"{key}: {json.dumps(value, ensure_ascii=True)}" for key, value in metadata.items()
    )
    front_matter.append("---")
    return "\n".join(front_matter) + "\n\n" + blocks_to_markdown(blocks)


def resolve_raw_file_path(record: ManifestRecord, repository_root: Path) -> Path:
    """Resolve manifest paths while accepting absolute and repository-relative values."""

    if record.raw_file_path.is_absolute():
        return record.raw_file_path
    return repository_root / record.raw_file_path


def convert_help_center_document(
    record: ManifestRecord,
    output_dir: Path,
    repository_root: Path,
) -> Path:
    """Convert and write one raw Help Center document."""

    raw_file_path = resolve_raw_file_path(record, repository_root)
    html = raw_file_path.read_text(encoding="utf-8")
    blocks = html_to_blocks(html, record.source_url)
    markdown = render_markdown_document(record, blocks)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record.document_id}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_processed_manifest_record(record: ManifestRecord, output_path: Path) -> dict:
    """Build the processed JSONL metadata for one successful conversion."""

    payload = {
        "document_id": record.document_id,
        "title": record.title,
        "source_url": record.source_url,
        "source_type": record.source_type,
        "product_area": record.product_area,
        "raw_file_path": record.raw_file_path.as_posix(),
        "processed_file_path": output_path.as_posix(),
    }
    if record.fetch_timestamp:
        payload["fetch_timestamp"] = record.fetch_timestamp
    return payload


def write_processed_manifest(manifest_path: Path, records: Iterable[dict]) -> None:
    """Replace the processed JSONL manifest with successful document records."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=True) + "\n" for record in records
    )
    manifest_path.write_text(content, encoding="utf-8")


def convert_help_center_corpus(
    manifest_path: Path,
    output_dir: Path,
    processed_manifest_path: Path,
    repository_root: Path,
) -> ConversionSummary:
    """Convert all manifest documents, retaining failures for the CLI to report."""

    records = load_manifest_records(manifest_path)
    processed_records: list[dict] = []
    failures: list[ConversionFailure] = []

    for record in records:
        try:
            output_path = convert_help_center_document(
                record=record,
                output_dir=output_dir,
                repository_root=repository_root,
            )
            processed_records.append(build_processed_manifest_record(record, output_path))
        except Exception as exc:
            failures.append(
                ConversionFailure(
                    document_id=record.document_id,
                    raw_file_path=record.raw_file_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    write_processed_manifest(processed_manifest_path, processed_records)
    return ConversionSummary(
        total_documents=len(records),
        converted_documents=len(processed_records),
        failed_documents=len(failures),
        output_dir=output_dir,
        processed_manifest_path=processed_manifest_path,
        failures=tuple(failures),
    )
