"""Composable retrieval pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chroma import ChromaRetriever, RetrievalResult
from .filters import MetadataFilterSet


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

        if len(self.components) != 1:
            raise ValueError("only one retrieval component is supported for now")
        return self.components[0].retrieve(request)

    def to_description(self) -> dict:
        """Return a JSON-serializable description of the pipeline."""

        return {
            "components": self.component_names,
            "combination": "single_component",
        }


def build_retrieval_pipeline(
    component_names: list[str],
    chroma_retriever: ChromaRetriever,
) -> RetrievalPipeline:
    """Build a retrieval pipeline from enabled component names."""

    components = []
    seen_names = set()
    for component_name in component_names:
        if component_name in seen_names:
            raise ValueError(f"Duplicate retrieval component: {component_name}")
        seen_names.add(component_name)

        if component_name == "chroma":
            components.append(ChromaRetrievalComponent(chroma_retriever))
            continue

        raise ValueError(f"Unsupported retrieval component: {component_name}")

    return RetrievalPipeline(components=components)
