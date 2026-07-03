"""Reusable Chroma retrieval for embedded documentation chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from figma_rag.embeddings import load_sentence_transformer
from figma_rag.indexing import build_default_collection_name


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved documentation chunk."""

    rank: int
    chunk_id: str
    text: str
    distance: float
    title: str
    section: str
    source_url: str
    metadata: dict


class ChromaRetriever:
    """Embed queries and retrieve nearest chunks from a Chroma collection."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        model_name: str,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self._model = load_sentence_transformer(model_name)
        self._collection = _load_collection(persist_dir, collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Return nearest chunks for a query, optionally filtered by metadata."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_args = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where

        results = self._collection.query(**query_args)
        return _normalize_results(results)


def resolve_collection_name(
    chunks_path: Path,
    model_name: str,
    collection_name: str | None = None,
) -> str:
    """Return the explicit collection name or infer the default one."""

    return collection_name or build_default_collection_name(chunks_path, model_name)


def _normalize_results(results: dict) -> list[RetrievalResult]:
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    normalized: list[RetrievalResult] = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        metadata = dict(metadata or {})
        normalized.append(
            RetrievalResult(
                rank=rank,
                chunk_id=str(chunk_id),
                text=str(document),
                distance=float(distance),
                title=str(metadata.get("title", "")),
                section=_format_section(metadata.get("heading_path")),
                source_url=str(metadata.get("source_url", "")),
                metadata=metadata,
            )
        )
    return normalized


def _format_section(value: object) -> str:
    if value is None:
        return "(document start)"

    try:
        heading_path = json.loads(str(value))
    except json.JSONDecodeError:
        heading_path = []

    if not heading_path:
        return "(document start)"
    return " > ".join(str(heading) for heading in heading_path)


def _load_collection(persist_dir: Path, collection_name: str):
    try:
        from chromadb import PersistentClient
    except ImportError as exc:
        raise RuntimeError(
            "The 'chromadb' package is required to query the persistent vector index"
        ) from exc

    client = PersistentClient(path=str(persist_dir))
    return client.get_collection(name=collection_name)
