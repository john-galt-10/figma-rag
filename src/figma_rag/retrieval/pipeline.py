"""Composable retrieval pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .bm25 import BM25Retriever
from .chroma import ChromaRetriever, RetrievalResult
from .filters import MetadataFilterSet

DEFAULT_RETRIEVAL_COMPONENTS = ("chroma", "bm25")


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

    def __init__(self, components: list[RetrievalComponent]) -> None:
        if not components:
            raise ValueError("at least one retrieval component is required")
        self.components = components

    @property
    def component_names(self) -> list[str]:
        """Return names of enabled retrieval components."""

        return [component.name for component in self.components]

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Run retrieval components and return ranked results."""

        if len(self.components) == 1:
            return self.components[0].retrieve(request)
        return aggregate_retrieval_results([], self.component_names)

    def to_description(self) -> dict:
        """Return a JSON-serializable description of the pipeline."""

        return {
            "components": self.component_names,
            "combination": "single_component"
            if len(self.components) == 1
            else "hybrid_placeholder",
        }


def aggregate_retrieval_results(
    component_results: list[list[RetrievalResult]],
    component_names: list[str],
) -> list[RetrievalResult]:
    """Combine results from multiple retrieval components."""

    # TODO: Implement hybrid result aggregation and score normalization.
    raise NotImplementedError(
        "Hybrid retrieval aggregation is not implemented yet for components: "
        + ", ".join(component_names)
    )


def build_retrieval_pipeline(
    component_names: list[str] | None = None,
    chroma_retriever: ChromaRetriever | None = None,
    bm25_retriever: BM25Retriever | None = None,
) -> RetrievalPipeline:
    """Build a retrieval pipeline from enabled component names."""

    component_names = list(component_names or DEFAULT_RETRIEVAL_COMPONENTS)
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

    return RetrievalPipeline(components=components)
