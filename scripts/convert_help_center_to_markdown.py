"""Entrypoint for converting raw Figma Help Center HTML to Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.ingestion.help_center_markdown import convert_help_center_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw Figma Help Center HTML into structured Markdown."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "figma_docs" / "manifest.jsonl",
        help="Path to the authoritative raw JSONL manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "figma_docs" / "help_center",
        help="Directory where generated Markdown documents will be written.",
    )
    parser.add_argument(
        "--processed-manifest-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "figma_docs" / "manifest.jsonl",
        help="Path to the regenerated processed JSONL manifest.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = convert_help_center_corpus(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        processed_manifest_path=args.processed_manifest_path,
        repository_root=REPO_ROOT,
    )

    for failure in summary.failures:
        print(
            f"Failed: {failure.document_id} ({failure.raw_file_path.as_posix()}): "
            f"{failure.error}",
            file=sys.stderr,
        )

    print(f"Manifest documents: {summary.total_documents}")
    print(f"Converted documents: {summary.converted_documents}")
    print(f"Failed documents: {summary.failed_documents}")
    print(f"Markdown directory: {summary.output_dir.as_posix()}")
    print(f"Processed manifest: {summary.processed_manifest_path.as_posix()}")
    return 1 if summary.failed_documents else 0


if __name__ == "__main__":
    raise SystemExit(main())
