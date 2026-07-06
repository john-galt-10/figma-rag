"""Find exact character offsets for a Markdown span."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.evaluation.rag_dataset import find_span_offsets, read_md  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for span lookup."""
    parser = argparse.ArgumentParser(
        description="Compute Python character offsets for an exact Markdown substring."
    )
    parser.add_argument(
        "--document-path",
        type=Path,
        required=True,
        help="Markdown document path, absolute or relative to the repository root.",
    )
    parser.add_argument(
        "--span-text",
        default=None,
        help="Exact substring to locate. Use --span-file for long multi-line spans.",
    )
    parser.add_argument(
        "--span-file",
        type=Path,
        default=None,
        help="UTF-8 text file containing the exact substring to locate.",
    )
    parser.add_argument(
        "--occurrence",
        type=int,
        default=None,
        help="Zero-based occurrence index to use when the same span appears repeatedly.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional exact text that must immediately precede the selected span.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional exact text that must immediately follow the selected span.",
    )
    return parser


def resolve_repo_path(path: Path) -> Path:
    """Resolve an absolute path or a repository-relative path."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_span_argument(span_text: str | None, span_file: Path | None) -> str:
    """Read the exact span from either a command-line argument or a UTF-8 file."""
    if bool(span_text) == bool(span_file):
        raise ValueError("provide exactly one of --span-text or --span-file")
    if span_file is not None:
        return resolve_repo_path(span_file).read_text(encoding="utf-8")
    return span_text or ""


def main() -> int:
    """Run the span lookup CLI."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        document_path = resolve_repo_path(args.document_path)
        exact_span = read_span_argument(args.span_text, args.span_file)
        document_text = read_md(document_path)
        start, end = find_span_offsets(
            document_text=document_text,
            exact_span=exact_span,
            occurrence=args.occurrence,
            prefix=args.prefix,
            suffix=args.suffix,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = {
        "document_path": args.document_path.as_posix(),
        "char_span": [start, end],
        "extracted_text": document_text[start:end],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
