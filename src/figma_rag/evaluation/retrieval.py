"""Utilities for evaluating chunk retrieval against a mapped test set."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalQueryExample:
    """One test-set query with binary relevant chunk IDs."""

    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Aggregate and per-query retrieval metrics for one evaluation run."""

    metrics_by_k: dict[str, dict[str, float]]
    details: list[dict[str, Any]]


def load_retrieval_test_set(path: Path) -> list[RetrievalQueryExample]:
    """Load mapped retrieval examples from a JSONL test-set file."""

    examples: list[RetrievalQueryExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            examples.append(_parse_test_set_row(row, path, line_number))

    if not examples:
        raise ValueError(f"No retrieval examples found in {path}")

    return examples


def evaluate_retrieval_results(
    examples: Iterable[RetrievalQueryExample],
    retrieved_chunk_ids_by_query_id: dict[str, list[str]],
    top_k_values: Iterable[int],
    retrieved_component_ranks_by_query_id: (
        dict[str, list[dict[str, Any]]] | None
    ) = None,
) -> RetrievalEvaluation:
    """Compute retrieval metrics for every requested cutoff."""

    examples = list(examples)
    normalized_top_k_values = normalize_top_k_values(top_k_values)
    details: list[dict[str, Any]] = []
    rows_by_k: dict[int, list[dict[str, Any]]] = {
        top_k: [] for top_k in normalized_top_k_values
    }
    retrieved_component_ranks_by_query_id = (
        retrieved_component_ranks_by_query_id or {}
    )

    for example in examples:
        for top_k in normalized_top_k_values:
            row = evaluate_query_at_k(
                example=example,
                retrieved_chunk_ids=retrieved_chunk_ids_by_query_id.get(
                    example.query_id,
                    [],
                ),
                retrieved_component_ranks=retrieved_component_ranks_by_query_id.get(
                    example.query_id,
                    [],
                ),
                top_k=top_k,
            )
            details.append(row)
            rows_by_k[top_k].append(row)

    metrics_by_k = {
        str(top_k): summarize_query_metrics(rows_for_k)
        for top_k, rows_for_k in rows_by_k.items()
    }

    return RetrievalEvaluation(metrics_by_k=metrics_by_k, details=details)


def evaluate_query_at_k(
    example: RetrievalQueryExample,
    retrieved_chunk_ids: list[str],
    top_k: int,
    retrieved_component_ranks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute per-query retrieval metrics at one cutoff."""

    relevant_ids = set(example.relevant_chunk_ids)
    retrieved_at_k = retrieved_chunk_ids[:top_k]
    retrieved_component_ranks_at_k = (retrieved_component_ranks or [])[:top_k]
    relevant_retrieved_flags = [
        chunk_id in relevant_ids for chunk_id in retrieved_at_k
    ]
    relevant_retrieved_count = sum(relevant_retrieved_flags)
    first_relevant_rank = _first_relevant_rank(relevant_retrieved_flags)

    return {
        "query_id": example.query_id,
        "query": example.query,
        "k": top_k,
        "num_relevant": len(relevant_ids),
        "retrieved_chunk_ids": retrieved_at_k,
        "retrieved_component_ranks": retrieved_component_ranks_at_k,
        "relevant_chunk_ids": list(example.relevant_chunk_ids),
        "hit": first_relevant_rank is not None,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 0.0
        if first_relevant_rank is None
        else 1.0 / first_relevant_rank,
        "precision_at_k": relevant_retrieved_count / top_k,
        "recall_at_k": relevant_retrieved_count / len(relevant_ids),
        "average_precision_at_k": _average_precision_at_k(
            relevant_retrieved_flags=relevant_retrieved_flags,
            num_relevant=len(relevant_ids),
            top_k=top_k,
        ),
        "ndcg_at_k": _ndcg_at_k(
            relevant_retrieved_flags=relevant_retrieved_flags,
            num_relevant=len(relevant_ids),
            top_k=top_k,
        ),
    }


def summarize_query_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average per-query metrics into aggregate retrieval metrics."""

    if not rows:
        return {
            "hit_rate_at_k": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    return {
        "hit_rate_at_k": _mean(1.0 if row["hit"] else 0.0 for row in rows),
        "recall_at_k": _mean(row["recall_at_k"] for row in rows),
        "precision_at_k": _mean(row["precision_at_k"] for row in rows),
        "mrr_at_k": _mean(row["reciprocal_rank"] for row in rows),
        "map_at_k": _mean(row["average_precision_at_k"] for row in rows),
        "ndcg_at_k": _mean(row["ndcg_at_k"] for row in rows),
    }


def normalize_top_k_values(top_k_values: Iterable[int]) -> list[int]:
    """Return sorted unique positive retrieval cutoffs."""

    normalized_values = sorted(set(top_k_values))
    if not normalized_values:
        raise ValueError("At least one top-k value is required")
    if any(top_k <= 0 for top_k in normalized_values):
        raise ValueError("All top-k values must be greater than zero")
    return normalized_values


def set_reproducibility_seed(seed: int) -> dict[str, Any]:
    """Seed available random number generators and report applied settings."""

    seed_settings: dict[str, Any] = {
        "seed": seed,
        "python_random_seeded": True,
        "pythonhashseed_before": os.environ.get("PYTHONHASHSEED"),
        "pythonhashseed_after": str(seed),
        "numpy_seeded": False,
        "torch_seeded": False,
        "torch_cuda_seeded": False,
        "torch_deterministic_algorithms": False,
        "torch_cudnn_deterministic": False,
        "torch_cudnn_benchmark": None,
    }

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        seed_settings["numpy_error"] = "numpy is not installed"
    else:
        np.random.seed(seed)
        seed_settings["numpy_seeded"] = True

    try:
        import torch
    except ImportError:
        seed_settings["torch_error"] = "torch is not installed"
    else:
        torch.manual_seed(seed)
        seed_settings["torch_seeded"] = True

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seed_settings["torch_cuda_seeded"] = True

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            seed_settings["torch_deterministic_error"] = str(exc)
        else:
            seed_settings["torch_deterministic_algorithms"] = True

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            seed_settings["torch_cudnn_deterministic"] = True
            seed_settings["torch_cudnn_benchmark"] = False

    return seed_settings


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stabilize_retrieval_ties(results: list[Any]) -> list[Any]:
    """Sort exact-distance ties by chunk ID while preserving ranked groups."""

    stabilized_results = []
    current_group = []
    current_distance = None

    for result in results:
        if current_distance is None or result.distance == current_distance:
            current_group.append(result)
            current_distance = result.distance
            continue

        stabilized_results.extend(
            sorted(current_group, key=lambda item: item.chunk_id)
        )
        current_group = [result]
        current_distance = result.distance

    if current_group:
        stabilized_results.extend(
            sorted(current_group, key=lambda item: item.chunk_id)
        )

    return stabilized_results


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    """Write aggregate metrics as a compact JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_detailed_results_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-query retrieval results as a nested Parquet file."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "The 'pyarrow' package is required to write detailed Parquet results."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _parse_test_set_row(
    row: Any,
    path: Path,
    line_number: int,
) -> RetrievalQueryExample:
    """Validate and parse one mapped test-set row."""

    if not isinstance(row, dict):
        raise ValueError(f"Expected JSON object at {path}:{line_number}")

    query_id = row.get("query_id")
    query = row.get("query")
    chunks = row.get("chunks")

    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"Row {line_number} must include a non-empty query_id")
    if not isinstance(query, str) or not query:
        raise ValueError(f"Row {line_number} must include a non-empty query")
    if not isinstance(chunks, list):
        raise ValueError(f"Row {line_number} must include a chunks list")

    relevant_chunk_ids = []
    seen_chunk_ids = set()
    for chunk_index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Row {line_number} chunk {chunk_index} must be a JSON object"
            )

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(
                f"Row {line_number} chunk {chunk_index} must include a chunk_id"
            )
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            relevant_chunk_ids.append(chunk_id)

    if not relevant_chunk_ids:
        raise ValueError(f"Row {line_number} must include at least one relevant chunk")

    return RetrievalQueryExample(
        query_id=query_id,
        query=query,
        relevant_chunk_ids=tuple(relevant_chunk_ids),
    )


def _first_relevant_rank(relevant_retrieved_flags: list[bool]) -> int | None:
    """Return the one-based rank of the first relevant result, if present."""

    for index, is_relevant in enumerate(relevant_retrieved_flags, start=1):
        if is_relevant:
            return index
    return None


def _average_precision_at_k(
    relevant_retrieved_flags: list[bool],
    num_relevant: int,
    top_k: int,
) -> float:
    """Return average precision at K for binary relevance flags."""

    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_seen = 0
    for index, is_relevant in enumerate(relevant_retrieved_flags, start=1):
        if not is_relevant:
            continue
        relevant_seen += 1
        precision_sum += relevant_seen / index

    return precision_sum / min(num_relevant, top_k)


def _ndcg_at_k(
    relevant_retrieved_flags: list[bool],
    num_relevant: int,
    top_k: int,
) -> float:
    """Return normalized discounted cumulative gain at K for binary relevance."""

    if num_relevant == 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant_retrieved_flags, start=1)
        if is_relevant
    )
    ideal_result_count = min(num_relevant, top_k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_result_count + 1)
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean for an iterable of floats."""

    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
