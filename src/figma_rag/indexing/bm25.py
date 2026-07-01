"""Build a persistent BM25 index from retrieval chunks."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chroma import build_chunk_metadata, load_chunk_records


@dataclass(frozen=True)
class BM25IndexSummary:
    """Summary returned after building a BM25 index."""

    chunks_path: Path
    persist_dir: Path
    index_name: str
    index_dir: Path
    total_chunks: int
    stemming_enabled: bool
    stemmer_language: str | None
    recreated: bool


def build_bm25_index(
    chunks_path: Path,
    persist_dir: Path,
    index_name: str,
    stemming_enabled: bool = True,
    stemmer_language: str = "english",
    recreate: bool = False,
) -> BM25IndexSummary:
    """Build and persist a LlamaIndex BM25 retriever from chunk JSONL records."""

    chunks = load_chunk_records(chunks_path)
    index_dir = _resolve_index_dir(persist_dir=persist_dir, index_name=index_name)

    if recreate:
        _delete_index_dir(index_dir=index_dir, persist_dir=persist_dir)

    index_dir.mkdir(parents=True, exist_ok=True)
    nodes = [_build_text_node(record) for record in chunks]
    retriever = _build_bm25_retriever(
        nodes=nodes,
        stemming_enabled=stemming_enabled,
        stemmer_language=stemmer_language,
    )
    retriever.persist(str(index_dir))
    _write_metadata(
        path=index_dir / "metadata.json",
        chunks_path=chunks_path,
        index_name=index_name,
        total_chunks=len(chunks),
        stemming_enabled=stemming_enabled,
        stemmer_language=stemmer_language if stemming_enabled else None,
    )

    return BM25IndexSummary(
        chunks_path=chunks_path,
        persist_dir=persist_dir,
        index_name=index_name,
        index_dir=index_dir,
        total_chunks=len(chunks),
        stemming_enabled=stemming_enabled,
        stemmer_language=stemmer_language if stemming_enabled else None,
        recreated=recreate,
    )


def build_default_bm25_index_name(
    chunks_path: Path,
    stemming_enabled: bool = True,
    stemmer_language: str = "english",
) -> str:
    """Build a readable BM25 index name from the chunk artifact and current time."""

    chunk_label = _chunk_settings_slug(chunks_path)
    if stemming_enabled:
        stemmer_label = _slugify(stemmer_language)
        suffix = f"bm25_stemmed_{stemmer_label}"
    else:
        suffix = "bm25_raw"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    return _slugify(f"{chunk_label}_{suffix}_{timestamp}")


def _build_text_node(record: dict):
    """Convert one stored chunk record into a LlamaIndex text node."""

    try:
        from llama_index.core.schema import TextNode
    except ImportError as exc:
        raise RuntimeError(
            "The 'llama-index-core' package is required to build BM25 nodes. "
            "Install llama-index-core, llama-index-retrievers-bm25, and PyStemmer "
            "in the figma-navigator environment."
        ) from exc

    return TextNode(
        id_=str(record["chunk_id"]),
        text=str(record["text"]),
        metadata=build_chunk_metadata(record),
    )


def _build_bm25_retriever(
    nodes: list,
    stemming_enabled: bool,
    stemmer_language: str,
):
    """Create a LlamaIndex BM25 retriever without applying chunking."""

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError as exc:
        raise RuntimeError(
            "The 'llama-index-retrievers-bm25' package is required to build a "
            "BM25 index. Install llama-index-core, llama-index-retrievers-bm25, "
            "and PyStemmer in the figma-navigator environment."
        ) from exc

    stemmer = None
    if stemming_enabled:
        stemmer = _load_stemmer(stemmer_language)

    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,
        stemmer=stemmer,
        language=stemmer_language,
    )


def _load_stemmer(language: str):
    """Load a PyStemmer stemmer for the requested language."""

    try:
        import Stemmer
    except ImportError as exc:
        raise RuntimeError(
            "The 'PyStemmer' package is required when stemming is enabled. "
            "Install llama-index-core, llama-index-retrievers-bm25, and PyStemmer "
            "in the figma-navigator environment, or pass --disable-stemming."
        ) from exc

    try:
        return Stemmer.Stemmer(language)
    except Exception as exc:
        raise ValueError(f"Could not create PyStemmer stemmer for {language!r}") from exc


def _write_metadata(
    path: Path,
    chunks_path: Path,
    index_name: str,
    total_chunks: int,
    stemming_enabled: bool,
    stemmer_language: str | None,
) -> None:
    """Write a small metadata sidecar for the persisted BM25 index."""

    resolved_chunks_path = chunks_path.resolve()
    metadata = {
        "index_type": "bm25",
        "library": "llama-index-retrievers-bm25",
        "retrieval_intent": "keyword_search",
        "index_name": index_name,
        "source_chunks_path": resolved_chunks_path.as_posix(),
        "source_chunks_file": resolved_chunks_path.name,
        "total_chunks": total_chunks,
        "stemming_enabled": stemming_enabled,
        "stemmer_language": stemmer_language,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _delete_index_dir(index_dir: Path, persist_dir: Path) -> None:
    """Delete one persisted BM25 index directory after path safety checks."""

    if not index_dir.exists():
        return
    resolved_index_dir = index_dir.resolve()
    resolved_persist_dir = persist_dir.resolve()
    if resolved_index_dir == resolved_persist_dir:
        raise ValueError("refusing to delete the BM25 persist root")
    if resolved_persist_dir not in resolved_index_dir.parents:
        raise ValueError(
            f"refusing to delete {resolved_index_dir}; it is outside {resolved_persist_dir}"
        )
    if not index_dir.is_dir():
        raise ValueError(f"BM25 index path is not a directory: {index_dir}")
    shutil.rmtree(index_dir)


def _resolve_index_dir(persist_dir: Path, index_name: str) -> Path:
    """Return a safe index directory contained by the BM25 persistence root."""

    if not index_name.strip():
        raise ValueError("index_name must not be empty")
    index_dir = persist_dir / index_name
    resolved_index_dir = index_dir.resolve()
    resolved_persist_dir = persist_dir.resolve()
    if resolved_index_dir == resolved_persist_dir:
        raise ValueError("index_name must identify a child directory")
    if resolved_persist_dir not in resolved_index_dir.parents:
        raise ValueError(
            f"BM25 index directory must be inside {resolved_persist_dir}: "
            f"{resolved_index_dir}"
        )
    return index_dir


def _chunk_settings_slug(chunks_path: Path) -> str:
    stem = chunks_path.stem
    match = re.match(
        r"^chunks_(?P<strategy>.+?)_(?P<chunk_model>.+?)_"
        r"t(?P<max_tokens>\d+)_o(?P<overlap_tokens>\d+)_",
        stem,
    )
    if not match:
        return _slugify(stem)

    return _slugify(
        "{strategy}_{chunk_model}_t{max_tokens}_o{overlap_tokens}".format(
            **match.groupdict()
        )
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    slug = re.sub(r"\.{2,}", ".", slug)
    return slug or "index"
