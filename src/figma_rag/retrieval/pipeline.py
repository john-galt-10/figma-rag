"""Composable retrieval pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Protocol

from .aggregation import (
    DEFAULT_AGGREGATION_STRATEGY,
    AggregationStrategy,
    ComponentRetrievalResults,
    get_aggregation_strategy,
)
from .bm25 import BM25Retriever
from .chroma import ChromaRetriever, RetrievalResult
from .filters import MetadataFilterSet

DEFAULT_RETRIEVAL_COMPONENTS = ("chroma", "bm25")
WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RetrievalRequest:
    """Input for one retrieval pipeline run."""

    query: str
    top_k: int
    metadata_filters: MetadataFilterSet = MetadataFilterSet()
    metadata_filters_enabled: bool = True
    raw_chroma_where: dict | None = None

    @property
    def chroma_where(self) -> dict | None:
        """Return the active Chroma metadata filter payload."""

        return self.metadata_where

    @property
    def metadata_where(self) -> dict | None:
        """Return active metadata filters as a Chroma-style where payload."""

        clauses = []
        if self.metadata_filters_enabled:
            metadata_where = self.metadata_filters.to_chroma_where()
            if metadata_where:
                clauses.append(metadata_where)
        if self.raw_chroma_where:
            clauses.append(self.raw_chroma_where)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}


class RetrievalComponent(Protocol):
    """Protocol implemented by retrieval components such as Chroma or BM25."""

    name: str

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Return ranked results for the request."""


class ChromaRetrievalComponent:
    """Retrieval component backed by a Chroma vector collection."""

    name = "chroma"

    def __init__(self, retriever: ChromaRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Retrieve vector matches from Chroma."""

        return self.retriever.retrieve(
            request.query,
            top_k=request.top_k,
            where=request.chroma_where,
        )


class BM25RetrievalComponent:
    """Retrieval component backed by a persisted BM25 keyword index."""

    name = "bm25"

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Retrieve keyword matches from BM25."""

        return self.retriever.retrieve(
            request.query,
            top_k=request.top_k,
            metadata_filters=request.metadata_filters,
            metadata_filters_enabled=request.metadata_filters_enabled,
            where=request.raw_chroma_where,
        )


class RetrievalPipeline:
    """Run enabled retrieval components for one request."""

    def __init__(
        self,
        components: list[RetrievalComponent],
        aggregation_strategy: AggregationStrategy | None = None,
        component_weights: dict[str, float] | None = None,
    ) -> None:
        if not components:
            raise ValueError("at least one retrieval component is required")
        self.components = components
        self.aggregation_strategy = aggregation_strategy or get_aggregation_strategy(
            DEFAULT_AGGREGATION_STRATEGY
        )
        if (
            component_weights is not None
            and self.aggregation_strategy.name != "weighted_rrf"
        ):
            raise ValueError("component weights are only supported with weighted_rrf")
        self.component_weights = normalize_component_weights(
            component_names=self.component_names,
            component_weights=component_weights,
        )

    @property
    def component_names(self) -> list[str]:
        """Return names of enabled retrieval components."""

        return [component.name for component in self.components]

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Run retrieval components and return ranked results."""

        if len(self.components) == 1:
            return self.components[0].retrieve(request)

        component_results = [
            component.retrieve(request) for component in self.components
        ]
        return aggregate_retrieval_results(
            component_results=component_results,
            component_names=self.component_names,
            top_k=request.top_k,
            aggregation_strategy=self.aggregation_strategy,
            component_weights=self.component_weights,
        )

    def to_description(self) -> dict:
        """Return a JSON-serializable description of the pipeline."""

        return {
            "components": self.component_names,
            "combination": "single_component"
            if len(self.components) == 1
            else "hybrid",
            "aggregation_strategy": self.aggregation_strategy.name,
            "component_weights": dict(self.component_weights)
            if len(self.components) > 1
            else None,
        }


def aggregate_retrieval_results(
    component_results: list[list[RetrievalResult]],
    component_names: list[str],
    top_k: int | None = None,
    aggregation_strategy: AggregationStrategy | None = None,
    component_weights: dict[str, float] | None = None,
) -> list[RetrievalResult]:
    """Combine results from multiple retrieval components."""

    if len(component_results) != len(component_names):
        raise ValueError("component_results and component_names must have same length")

    strategy = aggregation_strategy or get_aggregation_strategy(
        DEFAULT_AGGREGATION_STRATEGY
    )
    if component_weights is not None and strategy.name != "weighted_rrf":
        raise ValueError("component weights are only supported with weighted_rrf")
    weights = normalize_component_weights(
        component_names=component_names,
        component_weights=component_weights,
    )
    if top_k is None:
        top_k = max((len(results) for results in component_results), default=0)
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    return strategy.aggregate(
        component_results=[
            ComponentRetrievalResults(
                component_name=component_name,
                results=results,
                weight=weights[component_name],
            )
            for component_name, results in zip(component_names, component_results)
        ],
        top_k=top_k,
    )


def build_retrieval_pipeline(
    component_names: list[str] | None = None,
    chroma_retriever: ChromaRetriever | None = None,
    bm25_retriever: BM25Retriever | None = None,
    aggregation_strategy_name: str = DEFAULT_AGGREGATION_STRATEGY,
    component_weights: dict[str, float] | None = None,
) -> RetrievalPipeline:
    """Build a retrieval pipeline from enabled component names."""

    component_names = list(component_names or DEFAULT_RETRIEVAL_COMPONENTS)
    aggregation_strategy = get_aggregation_strategy(aggregation_strategy_name)
    components = []
    seen_names = set()
    for component_name in component_names:
        if component_name in seen_names:
            raise ValueError(f"Duplicate retrieval component: {component_name}")
        seen_names.add(component_name)

        if component_name == "chroma":
            if chroma_retriever is None:
                raise ValueError("chroma retriever is required when chroma is enabled")
            components.append(ChromaRetrievalComponent(chroma_retriever))
            continue

        if component_name == "bm25":
            if bm25_retriever is None:
                raise ValueError("BM25 retriever is required when bm25 is enabled")
            components.append(BM25RetrievalComponent(bm25_retriever))
            continue

        raise ValueError(f"Unsupported retrieval component: {component_name}")

    return RetrievalPipeline(
        components=components,
        aggregation_strategy=aggregation_strategy,
        component_weights=component_weights,
    )


def parse_component_weights(
    weight_specs: list[str] | None,
    component_names: list[str],
) -> dict[str, float] | None:
    """Parse repeatable component=weight CLI values into normalized weights."""

    if not weight_specs:
        return None

    weights: dict[str, float] = {}
    enabled_components = set(component_names)
    for spec in weight_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid component weight '{spec}'. Expected COMPONENT=WEIGHT."
            )
        component_name, raw_weight = (part.strip() for part in spec.split("=", 1))
        if not component_name:
            raise ValueError(
                f"Invalid component weight '{spec}'. Component name must not be empty."
            )
        if component_name in weights:
            raise ValueError(f"Duplicate component weight: {component_name}")
        if component_name not in enabled_components:
            available = ", ".join(component_names)
            raise ValueError(
                f"Component weight provided for disabled or unknown component "
                f"'{component_name}'. Enabled components: {available}"
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight for component '{component_name}': {raw_weight}"
            ) from exc
        weights[component_name] = weight

    return normalize_component_weights(
        component_names=component_names,
        component_weights=weights,
    )


def normalize_component_weights(
    component_names: list[str],
    component_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return validated weights that sum to one across enabled components."""

    if not component_names:
        raise ValueError("at least one retrieval component is required")

    if component_weights is None:
        equal_weight = 1.0 / len(component_names)
        return {component_name: equal_weight for component_name in component_names}

    component_name_set = set(component_names)
    weight_name_set = set(component_weights)
    unknown_names = sorted(weight_name_set - component_name_set)
    if unknown_names:
        raise ValueError(
            "Component weights include disabled or unknown components: "
            + ", ".join(unknown_names)
        )

    missing_names = sorted(component_name_set - weight_name_set)
    if missing_names:
        raise ValueError(
            "Component weights must be provided for every enabled component. "
            "Missing: "
            + ", ".join(missing_names)
        )

    normalized_weights = {}
    for component_name in component_names:
        weight = float(component_weights[component_name])
        if weight < 0:
            raise ValueError(
                f"Component weight must be non-negative: {component_name}={weight}"
            )
        normalized_weights[component_name] = weight

    weight_sum = sum(normalized_weights.values())
    if not isclose(
        weight_sum,
        1.0,
        rel_tol=WEIGHT_SUM_TOLERANCE,
        abs_tol=WEIGHT_SUM_TOLERANCE,
    ):
        raise ValueError(f"Component weights must sum to 1.0; got {weight_sum}")

    return normalized_weights
