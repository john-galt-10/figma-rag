"""Entrypoint for downloading raw Figma Help Center articles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.ingestion.help_center import download_help_center_articles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download raw Figma Help Center HTML and append manifest records."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "figma_docs" / "help_center",
        help="Directory where raw HTML files will be stored.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "figma_docs" / "manifest.jsonl",
        help="Path to the JSONL manifest file.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Sleep interval between article requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Request timeout for sitemap and article fetches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of candidate URLs processed from the sitemap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing HTML files and re-append manifest entries.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = download_help_center_articles(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    print(f"Discovered URLs: {summary.discovered_urls}")
    print(f"Downloaded pages: {summary.downloaded_pages}")
    print(f"Skipped existing: {summary.skipped_existing}")
    print(f"Skipped out of scope: {summary.skipped_out_of_scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
