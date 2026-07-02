"""Strategies for aggregating ranked results from multiple retrievers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chroma import RetrievalResult

DEFAULT_AGGREGATION_STRATEGY = "weighted_rrf"


@dataclass(frozen=True)
class ComponentRetrievalResults:
    """Ranked results and aggregation weight for one retrieval component."""

    component_name: str
    results: list[RetrievalResult]
    weight: float = 1.0


class AggregationStrategy(Protocol):
    """Common interface implemented by retrieval aggregation strategies."""

    name: str

    def aggregate(
        self,
        component_results: list[ComponentRetrievalResults],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return combined results for multiple retrieval components."""


class WeightedReciprocalRankFusionStrategy:
    """Aggregate ranked lists with weighted Reciprocal Rank Fusion."""

    name = "weighted_rrf"

    def __init__(self, rank_constant: int = 60) -> None:
        if rank_constant < 0:
            raise ValueError("rank_constant must be greater than or equal to zero")
        self.rank_constant = rank_constant

    def aggregate(
        self,
        component_results: list[ComponentRetrievalResults],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse ranks with weighted RRF and return at most top_k results."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        fused_scores: dict[str, float] = {}
        best_results: dict[str, RetrievalResult] = {}
        best_contribution_by_chunk_id: dict[str, float] = {}
        provenance = _build_component_provenance(component_results)

        for component in component_results:
            if component.weight < 0:
                raise ValueError(
                    f"Component weight must be non-negative: {component.component_name}"
                )
            for fallback_rank, result in enumerate(component.results, start=1):
                rank = result.rank if result.rank > 0 else fallback_rank
                contribution = component.weight / (self.rank_constant + rank)
                fused_scores[result.chunk_id] = (
                    fused_scores.get(result.chunk_id, 0.0) + contribution
                )

                previous_contribution = best_contribution_by_chunk_id.get(
                    result.chunk_id,
                    float("-inf"),
                )
                if contribution > previous_contribution:
                    best_contribution_by_chunk_id[result.chunk_id] = contribution
                    best_results[result.chunk_id] = result

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )
        return _rerank_results(
            [
                _replace_score(
                    _with_component_provenance(best_results[chunk_id], provenance),
                    -fused_scores[chunk_id],
                )
                for chunk_id in ranked_chunk_ids[:top_k]
            ]
        )


class UnionAggregationStrategy:
    """Append component result lists and deduplicate by first seen chunk ID."""

    name = "union"

    def aggregate(
        self,
        component_results: list[ComponentRetrievalResults],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return the full deduplicated union without top-k truncation."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        provenance = _build_component_provenance(component_results)
        seen_chunk_ids = set()
        combined_results = []
        for component in component_results:
            for result in component.results:
                if result.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(result.chunk_id)
                combined_results.append(result)
        return _rerank_results(
            [
                _replace_score(
                    _with_component_provenance(result, provenance),
                    float(rank),
                )
                for rank, result in enumerate(combined_results, start=1)
            ]
        )


_STRATEGIES: dict[str, AggregationStrategy] = {
    WeightedReciprocalRankFusionStrategy.name: WeightedReciprocalRankFusionStrategy(),
    UnionAggregationStrategy.name: UnionAggregationStrategy(),
}


def available_aggregation_strategies() -> tuple[str, ...]:
    """Return stable aggregation strategy names accepted by the CLI."""

    return tuple(sorted(_STRATEGIES))


def get_aggregation_strategy(name: str) -> AggregationStrategy:
    """Return a registered aggregation strategy or raise a descriptive error."""

    try:
        return _STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(available_aggregation_strategies())
        raise ValueError(
            f"Unknown aggregation strategy '{name}'. Available strategies: {available}"
        ) from exc


def _rerank_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Return results with contiguous one-based ranks."""

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


def _replace_score(result: RetrievalResult, distance: float) -> RetrievalResult:
    """Return a result with a replacement distance score."""

    return RetrievalResult(
        rank=result.rank,
        chunk_id=result.chunk_id,
        text=result.text,
        distance=distance,
        title=result.title,
        section=result.section,
        source_url=result.source_url,
        metadata=result.metadata,
    )


def _build_component_provenance(
    component_results: list[ComponentRetrievalResults],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Index each chunk's original rank and distance by retrieval component."""

    provenance: dict[str, dict[str, dict[str, float | int]]] = {}
    for component in component_results:
        for fallback_rank, result in enumerate(component.results, start=1):
            rank = result.rank if result.rank > 0 else fallback_rank
            chunk_provenance = provenance.setdefault(
                result.chunk_id,
                {"ranks": {}, "distances": {}},
            )
            chunk_provenance["ranks"][component.component_name] = rank
            chunk_provenance["distances"][component.component_name] = result.distance
    return provenance


def _with_component_provenance(
    result: RetrievalResult,
    provenance: dict[str, dict[str, dict[str, float | int]]],
) -> RetrievalResult:
    """Return a result whose metadata includes per-component retrieval provenance."""

    metadata = dict(result.metadata)
    chunk_provenance = provenance.get(result.chunk_id, {"ranks": {}, "distances": {}})
    metadata["retrieval_component_ranks"] = dict(chunk_provenance["ranks"])
    metadata["retrieval_component_distances"] = dict(chunk_provenance["distances"])
    return RetrievalResult(
        rank=result.rank,
        chunk_id=result.chunk_id,
        text=result.text,
        distance=result.distance,
        title=result.title,
        section=result.section,
        source_url=result.source_url,
        metadata=metadata,
    )
