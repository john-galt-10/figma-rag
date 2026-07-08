"""Artifact helpers for generation evaluation outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .time import evaluation_now


def build_generation_output_paths(
    test_set_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Return details, summary, and optional JSONL output paths."""

    timestamp = evaluation_now().strftime("%Y%m%dT%H%M")
    label = slugify(test_set_path.stem)
    details_path = output_dir / f"generation_details_{label}_{timestamp}.parquet"
    summary_path = output_dir / f"generation_metrics_{label}_{timestamp}.json"
    jsonl_path = output_dir / f"generation_details_{label}_{timestamp}.jsonl"
    return details_path, summary_path, jsonl_path


def write_generation_details_parquet(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write per-query generation evaluation rows as a Parquet file."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_generation_details_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write per-query generation evaluation rows as newline-delimited JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def ensure_parquet_available() -> None:
    """Fail early when the required Parquet writer dependency is missing."""

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The 'pyarrow' package is required to write generation evaluation Parquet."
        ) from exc


def slugify(value: str) -> str:
    """Return a filesystem-friendly label."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug or "test-set"
