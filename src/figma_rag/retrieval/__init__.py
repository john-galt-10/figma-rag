"""Retrieval utilities for local Figma documentation indexes."""

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
)

__all__ = [
    "BM25RetrievalComponent",
    "BM25Retriever",
    "ChromaRetrievalComponent",
    "ChromaRetriever",
    "DEFAULT_RETRIEVAL_COMPONENTS",
    "MetadataFilter",
    "MetadataFilterSet",
    "RetrievalComponent",
    "RetrievalPipeline",
    "RetrievalRequest",
    "RetrievalResult",
    "aggregate_retrieval_results",
    "build_retrieval_pipeline",
    "parse_metadata_filter_set",
    "resolve_collection_name",
]
