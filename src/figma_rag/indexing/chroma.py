"""Build a persistent Chroma index from retrieval chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from figma_rag.embeddings import load_sentence_transformer

REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "text",
    "title",
    "source_url",
    "document_id",
    "chunk_index",
    "strategy",
    "token_count",
)

MAX_CHROMA_COLLECTION_NAME_LENGTH = 63


@dataclass(frozen=True)
class ChromaIndexSummary:
    """Summary returned after building or updating a Chroma collection."""

    chunks_path: Path
    persist_dir: Path
    collection_name: str
    model_name: str
    total_chunks: int
    inserted_vectors: int
    collection_count: int
    recreated: bool


def build_chroma_index(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    recreate: bool,
) -> ChromaIndexSummary:
    """Embed chunk records and upsert them into a persistent Chroma collection."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    chunks = load_chunk_records(chunks_path)
    model = load_sentence_transformer(model_name)
    client = _build_persistent_client(persist_dir)

    if recreate:
        _delete_collection_if_exists(client, collection_name)

    collection = _get_or_create_collection(
        client=client,
        collection_name=collection_name,
        metadata=_collection_metadata(chunks_path, model_name),
    )

    inserted_vectors = 0
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch in tqdm(
        _batched(chunks, batch_size),
        total=total_batches,
        desc="Indexing chunks",
        unit="batch",
    ):
        texts = [str(record["text"]) for record in batch]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        collection.upsert(
            ids=[str(record["chunk_id"]) for record in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[build_chunk_metadata(record) for record in batch],
        )
        inserted_vectors += len(batch)

    return ChromaIndexSummary(
        chunks_path=chunks_path,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        total_chunks=len(chunks),
        inserted_vectors=inserted_vectors,
        collection_count=collection.count(),
        recreated=recreate,
    )


def build_default_collection_name(chunks_path: Path, model_name: str) -> str:
    """Build a readable collection name from the chunk artifact and embedder."""

    chunk_settings = _chunk_settings_slug(chunks_path)
    embedding_model = _slugify(model_name.rstrip("/").rsplit("/", maxsplit=1)[-1])
    if embedding_model and embedding_model not in chunk_settings:
        collection_name = f"figma_{chunk_settings}_embed_{embedding_model}"
    else:
        collection_name = f"figma_{chunk_settings}"
    return _trim_collection_name(collection_name)


def load_chunk_records(path: Path) -> list[dict]:
    """Load and validate chunk JSONL records."""

    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            missing = [
                field for field in REQUIRED_CHUNK_FIELDS if record.get(field) is None
            ]
            if missing:
                raise ValueError(
                    f"Missing fields in {path} at line {line_number}: "
                    + ", ".join(missing)
                )
            if not str(record["chunk_id"]).strip():
                raise ValueError(f"Empty chunk_id in {path} at line {line_number}")
            if not str(record["text"]).strip():
                raise ValueError(f"Empty text in {path} at line {line_number}")
            records.append(record)

    if not records:
        raise ValueError(f"No chunk records found in {path}")
    return records


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


def _trim_collection_name(collection_name: str) -> str:
    if len(collection_name) <= MAX_CHROMA_COLLECTION_NAME_LENGTH:
        return collection_name
    trimmed = collection_name[:MAX_CHROMA_COLLECTION_NAME_LENGTH].rstrip(".-_")
    return trimmed or "figma_index"


def build_chunk_metadata(record: dict) -> dict:
    """Build the shared retrieval metadata payload for one chunk record."""

    return {
        "chunk_id": str(record["chunk_id"]),
        "document_id": str(record["document_id"]),
        "chunk_index": int(record["chunk_index"]),
        "strategy": str(record["strategy"]),
        "title": str(record["title"]),
        "heading_path": json.dumps(record.get("heading_path", []), ensure_ascii=True),
        "source_url": str(record["source_url"]),
        "source_type": str(record.get("source_type", "")),
        "product_area": str(record.get("product_area", "")),
        "product": str(record.get("product", "N.A.")),
        "topic": str(record.get("topic", "N.A.")),
        "processed_file_path": str(record.get("processed_file_path", "")),
        "token_count": int(record["token_count"]),
        "start_idx": int(record["char_span"][0]),
        "end_idx": int(record["char_span"][1])
    }


def _collection_metadata(chunks_path: Path, model_name: str) -> dict:
    resolved_chunks_path = chunks_path.resolve()
    return {
        "hnsw:space": "cosine",
        "embedding_model": model_name,
        "source_chunks_path": resolved_chunks_path.as_posix(),
        "source_chunks_file": resolved_chunks_path.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _build_persistent_client(persist_dir: Path):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "The 'chromadb' package is required to build the persistent vector index"
        ) from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _delete_collection_if_exists(client, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise


def _get_or_create_collection(client, collection_name: str, metadata: dict):
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise

    return client.create_collection(
        name=collection_name,
        metadata=metadata,
    )


def _batched(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
