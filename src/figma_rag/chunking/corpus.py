"""Strategy-independent orchestration for chunking a Markdown corpus."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from .base import DocumentMetadata, build_embedding_text
from .registry import get_strategy
from .tokenizer import HuggingFaceTokenizer

REQUIRED_FIELDS = (
    "document_id",
    "title",
    "source_url",
    "source_type",
    "product_area",
    "processed_file_path",
)


@dataclass(frozen=True)
class ChunkingFailure:
    document_id: str
    processed_file_path: str
    error: str


@dataclass(frozen=True)
class ChunkingSummary:
    strategy: str
    total_documents: int
    chunked_documents: int
    failed_documents: int
    total_chunks: int
    output_path: Path
    min_tokens: int
    median_tokens: float
    p90_tokens: int
    p95_tokens: int
    max_tokens: int
    failures: tuple[ChunkingFailure, ...]


def load_processed_manifest(path: Path) -> list[dict]:
    """Load and validate processed document metadata."""

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
            missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
            if missing:
                raise ValueError(
                    f"Missing fields in {path} at line {line_number}: "
                    + ", ".join(missing)
                )
            records.append(record)
    return records


def chunk_corpus(
    manifest_path: Path,
    output_path: Path,
    repository_root: Path,
    strategy_name: str,
    model_name: str,
    max_tokens: int,
    overlap_tokens: int,
) -> ChunkingSummary:
    """Chunk every processed document and replace the output JSONL."""

    strategy = get_strategy(strategy_name)
    tokenizer = HuggingFaceTokenizer(model_name)
    manifest = load_processed_manifest(manifest_path)
    output_records: list[dict] = []
    failures: list[ChunkingFailure] = []
    chunked_documents = 0

    for record in tqdm(manifest, desc="Chunking documents", unit="doc"):
        metadata = DocumentMetadata(
            document_id=str(record["document_id"]),
            title=str(record["title"]),
            source_url=str(record["source_url"]),
            source_type=str(record["source_type"]),
            product_area=str(record["product_area"]),
            processed_file_path=str(record["processed_file_path"]),
        )
        try:
            markdown_path = _resolve_path(metadata.processed_file_path, repository_root)
            document = markdown_path.read_text(encoding="utf-8")
            drafts = strategy.chunk(
                document,
                metadata,
                tokenizer,
                max_tokens,
                overlap_tokens,
            )
            if not drafts:
                raise ValueError("Strategy produced no chunks")

            document_records: list[dict] = []
            for chunk_index, draft in enumerate(drafts):
                text = build_embedding_text(
                    metadata.title, draft.heading_path, draft.content
                )
                token_count = tokenizer.count(text)
                if token_count > max_tokens:
                    raise ValueError(
                        f"Chunk {chunk_index} has {token_count} tokens; "
                        f"maximum is {max_tokens}"
                    )
                document_records.append(
                    {
                        "chunk_id": f"{metadata.document_id}::{chunk_index:04d}",
                        "document_id": metadata.document_id,
                        "chunk_index": chunk_index,
                        "strategy": strategy.name,
                        "title": metadata.title,
                        "heading_path": list(draft.heading_path),
                        "content": draft.content,
                        "char_span": list(draft.char_span) if draft.char_span else None,
                        "text": text,
                        "token_count": token_count,
                        "source_url": metadata.source_url,
                        "source_type": metadata.source_type,
                        "product_area": metadata.product_area,
                        "processed_file_path": metadata.processed_file_path,
                    }
                )
            output_records.extend(document_records)
            chunked_documents += 1
        except Exception as exc:
            failures.append(
                ChunkingFailure(
                    document_id=metadata.document_id,
                    processed_file_path=metadata.processed_file_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    _write_jsonl(output_path, output_records)
    counts = [int(record["token_count"]) for record in output_records]
    return ChunkingSummary(
        strategy=strategy.name,
        total_documents=len(manifest),
        chunked_documents=chunked_documents,
        failed_documents=len(failures),
        total_chunks=len(output_records),
        output_path=output_path,
        min_tokens=min(counts, default=0),
        median_tokens=statistics.median(counts) if counts else 0,
        p90_tokens=_percentile(counts, 0.90),
        p95_tokens=_percentile(counts, 0.95),
        max_tokens=max(counts, default=0),
        failures=tuple(failures),
    )


def _resolve_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]
