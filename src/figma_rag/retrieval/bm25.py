"""Reusable BM25 retrieval for persisted documentation chunk indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chroma import RetrievalResult
from .filters import MetadataFilterSet


class BM25Retriever:
    """Retrieve keyword matches from a persisted LlamaIndex BM25 index."""

    def __init__(
        self,
        index_dir: Path,
        overfetch_multiplier: int = 10,
    ) -> None:
        if overfetch_multiplier <= 0:
            raise ValueError("overfetch_multiplier must be greater than zero")

        self.index_dir = index_dir
        self.overfetch_multiplier = overfetch_multiplier
        self.metadata = _load_index_metadata(index_dir)
        self._retriever = _load_bm25_retriever(index_dir, self.metadata)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilterSet = MetadataFilterSet(),
        metadata_filters_enabled: bool = True,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Return BM25 matches for a query with best-effort metadata post-filtering."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            raise ValueError("query must not be empty")

        candidate_count = top_k
        if metadata_filters_enabled or where:
            candidate_count = top_k * self.overfetch_multiplier
        _set_similarity_top_k(self._retriever, candidate_count)

        raw_results = self._retriever.retrieve(query)
        normalized = _normalize_results(raw_results)
        filtered = _filter_results(
            results=normalized,
            metadata_filters=metadata_filters,
            metadata_filters_enabled=metadata_filters_enabled,
            where=where,
        )
        return _rerank(filtered[:top_k])


def _load_bm25_retriever(index_dir: Path, metadata: dict):
    """Load a LlamaIndex BM25 retriever from its persisted directory."""

    if not index_dir.exists():
        raise ValueError(f"BM25 index directory does not exist: {index_dir}")
    if not index_dir.is_dir():
        raise ValueError(f"BM25 index path is not a directory: {index_dir}")

    try:
        from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever
    except ImportError as exc:
        raise RuntimeError(
            "The 'llama-index-retrievers-bm25' package is required to query a "
            "BM25 index. Install llama-index-core, llama-index-retrievers-bm25, "
            "and PyStemmer in the figma-navigator environment."
        ) from exc

    stemmer = _load_persisted_stemmer(metadata)
    try:
        if stemmer is not None:
            return LlamaBM25Retriever.from_persist_dir(
                str(index_dir),
                stemmer=stemmer,
                language=str(metadata.get("stemmer_language", "english")),
            )
        return LlamaBM25Retriever.from_persist_dir(str(index_dir))
    except AttributeError as exc:
        raise RuntimeError(
            "The installed llama-index-retrievers-bm25 package does not expose "
            "BM25Retriever.from_persist_dir, which is required to load the "
            "persisted BM25 index."
        ) from exc
    except TypeError:
        return LlamaBM25Retriever.from_persist_dir(str(index_dir))


def _load_persisted_stemmer(metadata: dict):
    """Load the PyStemmer stemmer described by persisted BM25 metadata."""

    if not metadata.get("stemming_enabled"):
        return None
    language = str(metadata.get("stemmer_language") or "english")
    try:
        import Stemmer
    except ImportError as exc:
        raise RuntimeError(
            "The 'PyStemmer' package is required to query this stemmed BM25 "
            "index. Install llama-index-core, llama-index-retrievers-bm25, and "
            "PyStemmer in the figma-navigator environment."
        ) from exc
    return Stemmer.Stemmer(language)


def _load_index_metadata(index_dir: Path) -> dict:
    """Load best-effort metadata for a persisted BM25 index."""

    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BM25 metadata JSON at {metadata_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BM25 metadata must be a JSON object: {metadata_path}")
    return payload


def _set_similarity_top_k(retriever: Any, top_k: int) -> None:
    """Set the LlamaIndex retriever candidate count for one query."""

    retriever.similarity_top_k = top_k


def _normalize_results(results: list[Any]) -> list[RetrievalResult]:
    """Adapt LlamaIndex node results to the shared RetrievalResult dataclass."""

    normalized: list[RetrievalResult] = []
    for rank, item in enumerate(results, start=1):
        node = getattr(item, "node", item)
        metadata = dict(getattr(node, "metadata", {}) or {})
        text = _node_text(node)
        score = getattr(item, "score", None)
        distance = -float(score) if score is not None else float(rank)
        chunk_id = str(metadata.get("chunk_id") or getattr(node, "node_id", ""))

        normalized.append(
            RetrievalResult(
                rank=rank,
                chunk_id=chunk_id,
                text=text,
                distance=distance,
                title=str(metadata.get("title", "")),
                section=_format_section(metadata.get("heading_path")),
                source_url=str(metadata.get("source_url", "")),
                metadata=metadata,
            )
        )
    return normalized


def _node_text(node: Any) -> str:
    """Return text content from a LlamaIndex node."""

    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        return str(get_content())
    return str(getattr(node, "text", ""))


def _filter_results(
    results: list[RetrievalResult],
    metadata_filters: MetadataFilterSet,
    metadata_filters_enabled: bool,
    where: dict | None,
) -> list[RetrievalResult]:
    """Apply Chroma-style metadata filters to BM25 results after retrieval."""

    filtered = results
    if metadata_filters_enabled:
        metadata_where = metadata_filters.to_chroma_where()
        if metadata_where:
            filtered = [
                result
                for result in filtered
                if _metadata_matches_where(result.metadata, metadata_where)
            ]
    if where:
        filtered = [
            result
            for result in filtered
            if _metadata_matches_where(result.metadata, where)
        ]
    return filtered


def _metadata_matches_where(metadata: dict, where: dict) -> bool:
    """Return whether metadata satisfies a Chroma-style where payload."""

    if "$and" in where:
        return all(_metadata_matches_where(metadata, clause) for clause in where["$and"])

    for field, condition in where.items():
        value = metadata.get(field)
        if isinstance(condition, dict):
            if not all(
                _metadata_matches_operator(value, op, expected)
                for op, expected in condition.items()
            ):
                return False
            continue
        if value != condition:
            return False
    return True


def _metadata_matches_operator(value: Any, operator: str, expected: Any) -> bool:
    """Evaluate one Chroma-style metadata operator."""

    if operator == "$eq":
        return value == expected
    if operator == "$ne":
        return value != expected
    if operator == "$in":
        return value in expected
    if operator in {"$lt", "$lte", "$gt", "$gte"}:
        return _compare_numeric(value, operator, expected)
    raise ValueError(f"Unsupported BM25 metadata filter operator: {operator}")


def _compare_numeric(value: Any, operator: str, expected: Any) -> bool:
    """Compare numeric metadata values for best-effort BM25 post-filtering."""

    try:
        numeric_value = float(value)
        numeric_expected = float(expected)
    except (TypeError, ValueError):
        return False

    if operator == "$lt":
        return numeric_value < numeric_expected
    if operator == "$lte":
        return numeric_value <= numeric_expected
    if operator == "$gt":
        return numeric_value > numeric_expected
    if operator == "$gte":
        return numeric_value >= numeric_expected
    raise ValueError(f"Unsupported numeric operator: {operator}")


def _rerank(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Return results with contiguous ranks after filtering."""

    return [
        RetrievalResult(
            rank=rank,
            chunk_id=result.chunk_id,
            text=result.text,
            distance=result.distance,
            title=result.title,
            section=result.section,
            source_url=result.source_url,
            metadata=result.metadata,
        )
        for rank, result in enumerate(results, start=1)
    ]


def _format_section(value: object) -> str:
    """Format stored heading path metadata for display."""

    if value is None:
        return "(document start)"

    try:
        heading_path = json.loads(str(value))
    except json.JSONDecodeError:
        heading_path = []

    if not heading_path:
        return "(document start)"
    return " > ".join(str(heading) for heading in heading_path)
