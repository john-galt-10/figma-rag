"""Map annotated question targets to overlapping documentation chunks."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_SET_PATH = REPO_ROOT / "data" / "eval" / "golden_set.json"
DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260626-1212.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "retrieval_test"
CHUNKS_FILENAME_PATTERN = re.compile(
    r"^chunks_(?P<strategy>[^_]+)_(?P<model>.+?)_t\d+_o\d+_(?P<timestamp>\d{8}-\d{4})$"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the relevant chunk mapping script."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a JSONL mapping from annotated question IDs to overlapping "
            "documentation chunk IDs and answer coverage scores."
        )
    )
    parser.add_argument(
        "--test-set-path",
        type=Path,
        default=DEFAULT_TEST_SET_PATH,
        help="Annotated Q&A JSONL file containing targets with document_id and char_span.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Chunk JSONL file containing chunk_id, document_id, and char_span.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output JSONL file to write without modifying the Q&A dataset. "
            "By default, writes a named artifact under data/eval/retrieval_test/."
        ),
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of objects."""
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")

            rows.append(row)

    return rows


def validate_char_span(value: Any, label: str) -> tuple[int, int]:
    """Validate and return a half-open character span."""
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(position, int) for position in value)
    ):
        raise ValueError(f"{label} must be a two-integer list like [start, end]")

    start, end = value
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{label} must satisfy 0 <= start <= end")

    return start, end


def overlap_length(first_span: tuple[int, int], second_span: tuple[int, int]) -> int:
    """Return the number of overlapping characters between two half-open spans."""
    overlap_start = max(first_span[0], second_span[0])
    overlap_end = min(first_span[1], second_span[1])
    return max(0, overlap_end - overlap_start)


def build_default_output_path(test_set_path: Path, chunks_path: Path) -> Path:
    """Build a traceable default output path from the input and chunk artifact names."""
    test_set_name = test_set_path.stem
    chunks_name = chunks_path.stem
    match = CHUNKS_FILENAME_PATTERN.match(chunks_name)

    if match:
        chunk_label = (
            f"{match.group('strategy')}_{match.group('model')}_{match.group('timestamp')}"
        )
    else:
        chunk_label = chunks_name.removeprefix("chunks_")

    output_filename = f"{test_set_name}_relevant_chunks_{chunk_label}.jsonl"
    return DEFAULT_OUTPUT_DIR / output_filename


def index_chunks_by_document(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group chunks by document ID while preserving chunk-file order."""
    chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id")
        document_id = chunk.get("document_id")
        char_span = validate_char_span(
            chunk.get("char_span"),
            f"chunk row {row_index} char_span",
        )

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"chunk row {row_index} must include a non-empty chunk_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"chunk row {row_index} must include a non-empty document_id")

        chunks_by_document[document_id].append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "char_span": char_span,
            }
        )

    return dict(chunks_by_document)


def map_question_to_chunks(
    question: dict[str, Any],
    chunks_by_document: dict[str, list[dict[str, Any]]],
    row_index: int,
) -> dict[str, Any]:
    """Create one output mapping row for a question."""
    query = question.get("query")
    query_id = question.get("query_id")
    query_type = question.get("query_type")
    answer_type = question.get("answer_type")
    expected_answer_points = question.get("expected_answer_points")
    targets = question.get("targets")

    if not isinstance(query, str) or not query:
        raise ValueError(f"question row {row_index} must include a non-empty query")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"question row {row_index} must include a non-empty query_id")
    if not isinstance(query_type, str) or not query_type:
        raise ValueError(f"question {query_id} must include a non-empty query_type")
    if not isinstance(answer_type, str) or not answer_type:
        raise ValueError(f"question {query_id} must include a non-empty answer_type")
    if not isinstance(expected_answer_points, list):
        raise ValueError(f"question {query_id} must include an expected_answer_points list")
    if not isinstance(targets, list):
        raise ValueError(f"question row {row_index} must include a targets list")

    answer_chars = 0
    overlap_chars_by_chunk: dict[str, int] = defaultdict(int)
    chunk_order: list[str] = []
    target_metadata = []

    for target_index, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            raise ValueError(f"question {query_id} target {target_index} must be an object")

        document_id = target.get("document_id")
        relevant_url = target.get("relevant_url")
        target_span = validate_char_span(
            target.get("char_span"),
            f"question {query_id} target {target_index} char_span",
        )

        if not isinstance(document_id, str) or not document_id:
            raise ValueError(
                f"question {query_id} target {target_index} must include a non-empty document_id"
            )

        answer_chars += target_span[1] - target_span[0]
        target_metadata.append(
            {
                "document_id": document_id,
                "char_span": list(target_span),
                "relevant_url": relevant_url,
            }
        )

        for chunk in chunks_by_document.get(document_id, []):
            overlap_chars = overlap_length(target_span, chunk["char_span"])
            if overlap_chars == 0:
                continue

            chunk_id = chunk["chunk_id"]
            if chunk_id not in overlap_chars_by_chunk:
                chunk_order.append(chunk_id)
            overlap_chars_by_chunk[chunk_id] += overlap_chars

    if answer_chars == 0 and overlap_chars_by_chunk:
        raise ValueError(f"question {query_id} has overlapping chunks but zero answer_chars")

    chunks = [
        {
            "chunk_id": chunk_id,
            "overlap_score": round(overlap_chars_by_chunk[chunk_id] / answer_chars, 6),
            "overlap_chars": overlap_chars_by_chunk[chunk_id],
        }
        for chunk_id in chunk_order
    ]

    return {
        "query_id": query_id,
        "query": query,
        "query_type": query_type,
        "answer_type": answer_type,
        "expected_answer_points": expected_answer_points,
        "answer_chars": answer_chars,
        "targets": target_metadata,
        "chunks": chunks,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL file, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    """Run the relevant chunk mapping workflow."""
    args = build_parser().parse_args()
    output_path = args.output_path or build_default_output_path(
        test_set_path=args.test_set_path,
        chunks_path=args.chunks_path,
    )

    questions = load_jsonl(args.test_set_path)
    chunks = load_jsonl(args.chunks_path)
    chunks_by_document = index_chunks_by_document(chunks)

    output_rows = [
        map_question_to_chunks(
            question=question,
            chunks_by_document=chunks_by_document,
            row_index=row_index,
        )
        for row_index, question in enumerate(questions, start=1)
    ]

    write_jsonl(output_path, output_rows)
    print(f"Wrote {len(output_rows)} question mappings to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
