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
from .config import (
    RetrievalConfig,
    RetrievalOptions,
    build_configured_retrieval_pipeline,
    load_retrieval_config,
    parse_retrieval_config,
    resolve_retrieval_options,
)
from .filters import MetadataFilter, MetadataFilterSet, parse_metadata_filter_set
from .pipeline import (
    BM25RetrievalComponent,
    ChromaRetrievalComponent,
    DEFAULT_RETRIEVAL_COMPONENTS,
    Reranker,
    RetrievalComponent,
    RetrievalPipeline,
    RetrievalRequest,
    aggregate_retrieval_results,
    build_retrieval_pipeline,
    normalize_component_weights,
    parse_component_weights,
    resolve_candidate_k,
)
from .reranking import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankingResult,
)

__all__ = [
    "BM25RetrievalComponent",
    "BM25Retriever",
    "ChromaRetrievalComponent",
    "ChromaRetriever",
    "AggregationStrategy",
    "ComponentRetrievalResults",
    "DEFAULT_AGGREGATION_STRATEGY",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RETRIEVAL_COMPONENTS",
    "CrossEncoderReranker",
    "MetadataFilter",
    "MetadataFilterSet",
    "Reranker",
    "RerankingResult",
    "RetrievalComponent",
    "RetrievalConfig",
    "RetrievalOptions",
    "RetrievalPipeline",
    "RetrievalRequest",
    "RetrievalResult",
    "UnionAggregationStrategy",
    "WeightedReciprocalRankFusionStrategy",
    "aggregate_retrieval_results",
    "available_aggregation_strategies",
    "build_configured_retrieval_pipeline",
    "build_retrieval_pipeline",
    "get_aggregation_strategy",
    "load_retrieval_config",
    "normalize_component_weights",
    "parse_component_weights",
    "parse_metadata_filter_set",
    "parse_retrieval_config",
    "resolve_candidate_k",
    "resolve_collection_name",
    "resolve_retrieval_options",
]
