"""Validate a JSONL RAG evaluation dataset with document character spans."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.evaluation.rag_dataset import (  # noqa: E402
    load_jsonl,
    normalized_query_key,
    read_md,
    span_preview,
    validate_entry,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate RAG evaluation JSONL entries and print span previews."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=REPO_ROOT / "data" / "eval" / "codex_set.jsonl",
        help="JSONL dataset to validate.",
    )
    return parser


def resolve_repo_path(path: Path) -> Path:
    """Resolve an absolute path or a repository-relative path."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> int:
    """Run dataset validation and print a manual semantic review report."""
    args = build_parser().parse_args()
    dataset_path = resolve_repo_path(args.dataset_path)

    try:
        entries = load_jsonl(dataset_path)
        if not entries:
            raise ValueError(f"No entries found in {dataset_path}")

        query_keys = [normalized_query_key(entry.get("query", "")) for entry in entries]
        duplicate_query_keys = [
            query_key for query_key, count in Counter(query_keys).items() if count > 1
        ]
        if duplicate_query_keys:
            raise ValueError(f"Duplicate normalized queries found: {duplicate_query_keys}")

        for row_index, entry in enumerate(entries, start=1):
            try:
                validate_entry(entry, REPO_ROOT)
            except ValueError as exc:
                raise ValueError(f"Entry {row_index}: {exc}") from exc

    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validated entries: {len(entries)}")
    print("")
    print("Manual semantic review report:")
    for row_index, entry in enumerate(entries, start=1):
        print(f"{row_index}. {entry['query']}")
        for target_index, target in enumerate(entry["targets"], start=1):
            document_path = target["document_path"]
            start, end = target["char_span"]
            document_text = read_md(REPO_ROOT / document_path)
            print(f"   target {target_index}: {document_path} [{start}, {end}]")
            print(f"   span: {span_preview(document_text[start:end])}")
        for answer_point in entry["expected_answer_points"]:
            print(f"   answer point: {answer_point}")
        print("")

    print("Validation passed.")
    print(
        "Semantic support still requires human review: confirm each answer point is "
        "directly supported by the printed span preview."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
