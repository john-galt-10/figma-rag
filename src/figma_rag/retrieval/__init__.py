"""Retrieval utilities for local Figma documentation indexes."""

from .aggregation import (
    DEFAULT_AGGREGATION_STRATEGY,
    AggregationStrategy,
    ComponentRetrievalResults,
    WeightedReciprocalRankFusionStrategy,
    UnionAggregationStrategy,
    available_aggregation_strategies,
    get_aggregation_strategy,
)
from .bm25 import BM25Retriever
from .chroma import ChromaRetriever, RetrievalResult, resolve_collection_name
from .filters import MetadataFilter, MetadataFilterSet, parse_metadata_filter_set
from .pipeline import (
    BM25RetrievalComponent,
    ChromaRetrievalComponent,
    DEFAULT_RETRIEVAL_COMPONENTS,
    RetrievalComponent,
    RetrievalPipeline,
    RetrievalRequest,
    aggregate_retrieval_results,
    build_retrieval_pipeline,
    normalize_component_weights,
    parse_component_weights,
)

__all__ = [
    "BM25RetrievalComponent",
    "BM25Retriever",
    "ChromaRetrievalComponent",
    "ChromaRetriever",
    "AggregationStrategy",
    "ComponentRetrievalResults",
    "DEFAULT_AGGREGATION_STRATEGY",
    "DEFAULT_RETRIEVAL_COMPONENTS",
    "MetadataFilter",
    "MetadataFilterSet",
    "RetrievalComponent",
    "RetrievalPipeline",
    "RetrievalRequest",
    "RetrievalResult",
    "UnionAggregationStrategy",
    "WeightedReciprocalRankFusionStrategy",
    "aggregate_retrieval_results",
    "available_aggregation_strategies",
    "build_retrieval_pipeline",
    "get_aggregation_strategy",
    "normalize_component_weights",
    "parse_component_weights",
    "parse_metadata_filter_set",
    "resolve_collection_name",
]
