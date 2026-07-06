"""Utilities for building and validating document-span RAG datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_md(path: Path) -> str:
    """Read a Markdown file as UTF-8 text using Python character indexing."""
    return path.read_text(encoding="utf-8")


def find_span_offsets(
    document_text: str,
    exact_span: str,
    occurrence: int | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
) -> tuple[int, int]:
    """Return the half-open character offsets for an exact span in a document.

    Args:
        document_text: Full Markdown document text.
        exact_span: Exact substring to locate.
        occurrence: Optional zero-based occurrence index when the same span repeats.
        prefix: Optional text that must immediately precede the selected span.
        suffix: Optional text that must immediately follow the selected span.

    Raises:
        ValueError: If the span is empty, missing, ambiguous, or mismatched.
    """
    if not exact_span:
        raise ValueError("exact_span must be a non-empty string")

    matches = _find_all_occurrences(document_text, exact_span)
    if not matches:
        raise ValueError("exact_span was not found in the document")

    if prefix is not None:
        matches = [
            start
            for start in matches
            if document_text[max(0, start - len(prefix)) : start] == prefix
        ]
    if suffix is not None:
        matches = [
            start
            for start in matches
            if document_text[start + len(exact_span) : start + len(exact_span) + len(suffix)]
            == suffix
        ]

    if occurrence is not None:
        if occurrence < 0:
            raise ValueError("occurrence must be zero or greater")
        if occurrence >= len(matches):
            raise ValueError(
                f"occurrence {occurrence} is out of range for {len(matches)} matching spans"
            )
        matches = [matches[occurrence]]

    if not matches:
        raise ValueError("no span match remained after applying disambiguation")
    if len(matches) > 1:
        raise ValueError(
            f"exact_span appears {len(matches)} times; provide occurrence, prefix, or suffix"
        )

    start = matches[0]
    end = start + len(exact_span)
    if document_text[start:end] != exact_span:
        raise ValueError("internal error: extracted span does not match exact_span")

    return start, end


def validate_entry(entry: dict[str, Any], repo_root: Path) -> None:
    """Validate one RAG evaluation dataset entry against the requested schema."""
    expected_keys = {"query", "targets", "expected_answer_points"}
    extra_keys = set(entry) - expected_keys
    missing_keys = expected_keys - set(entry)
    if missing_keys:
        raise ValueError(f"entry is missing required fields: {sorted(missing_keys)}")
    if extra_keys:
        raise ValueError(f"entry contains fields outside the requested schema: {sorted(extra_keys)}")

    query = entry.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    targets = entry.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("targets must be a non-empty list")

    expected_answer_points = entry.get("expected_answer_points")
    if not isinstance(expected_answer_points, list) or not expected_answer_points:
        raise ValueError("expected_answer_points must be a non-empty list")
    if not all(isinstance(point, str) and point.strip() for point in expected_answer_points):
        raise ValueError("expected_answer_points must contain only non-empty strings")

    for target_index, target in enumerate(targets, start=1):
        _validate_target(target, repo_root, target_index)


def load_reviewed_files(path: Path) -> set[str]:
    """Load reviewed Markdown paths from a line-oriented resume file."""
    if not path.exists():
        return set()

    reviewed_files = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if stripped_line:
            reviewed_files.add(stripped_line)
    return reviewed_files


def append_reviewed_file(path: Path, document_path: str) -> None:
    """Append a repo-relative Markdown path to the resume file if it is new."""
    reviewed_files = load_reviewed_files(path)
    if document_path in reviewed_files:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"{document_path}\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and require each non-empty line to be a JSON object."""
    rows: list[dict[str, Any]] = []
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


def normalized_query_key(query: str) -> str:
    """Normalize a query string for duplicate and near-duplicate checks."""
    normalized = query.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def span_preview(text: str, max_chars: int = 180) -> str:
    """Build a compact one-line preview for validation output."""
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= max_chars:
        return preview
    return f"{preview[: max_chars - 3]}..."


def _validate_target(target: Any, repo_root: Path, target_index: int) -> None:
    """Validate one target object and ensure its character span extracts text."""
    if not isinstance(target, dict):
        raise ValueError(f"target {target_index} must be an object")
    if set(target) != {"document_path", "char_span"}:
        raise ValueError(
            f"target {target_index} must contain exactly document_path and char_span"
        )

    document_path = target.get("document_path")
    if not isinstance(document_path, str) or not document_path.strip():
        raise ValueError(f"target {target_index} document_path must be a non-empty string")

    resolved_path = (repo_root / document_path).resolve()
    if not resolved_path.exists():
        raise ValueError(f"target {target_index} document does not exist: {document_path}")
    if not resolved_path.is_file():
        raise ValueError(f"target {target_index} document is not a file: {document_path}")

    char_span = target.get("char_span")
    if (
        not isinstance(char_span, list)
        or len(char_span) != 2
        or not all(isinstance(position, int) for position in char_span)
    ):
        raise ValueError(f"target {target_index} char_span must be two integers")

    start, end = char_span
    document_text = read_md(resolved_path)
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"target {target_index} char_span must satisfy 0 <= start < end")
    if end > len(document_text):
        raise ValueError(f"target {target_index} char_span ends beyond document length")
    if not document_text[start:end].strip():
        raise ValueError(f"target {target_index} char_span extracts empty text")


def _find_all_occurrences(document_text: str, exact_span: str) -> list[int]:
    """Return all starting character offsets for a substring."""
    matches: list[int] = []
    start = 0
    while True:
        index = document_text.find(exact_span, start)
        if index == -1:
            return matches
        matches.append(index)
        start = index + 1
