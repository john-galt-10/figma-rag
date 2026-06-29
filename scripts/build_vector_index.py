"""Entrypoint for building a persistent Chroma vector index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.indexing import build_chroma_index, build_default_collection_name

DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_gte-modernbert-base_t600_o60_20260622-1836.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed Figma documentation chunks into a persistent Chroma index."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the chunk JSONL file to index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "figma_docs" / "chroma",
        help="Directory where Chroma will persist the vector collection.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Name of the Chroma collection to create or update. By default, a "
            "name is built from the chunking artifact and embedding model."
        ),
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
#        default="Alibaba-NLP/gte-modernbert-base",
        help="Sentence Transformers model used to embed chunk text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of chunks to embed and upsert per batch.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete only the selected collection before rebuilding it. Older "
            "collections with different names are left untouched."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    collection_name = args.collection_name or build_default_collection_name(
        chunks_path=args.chunks_path,
        model_name=args.model,
    )
    summary = build_chroma_index(
        chunks_path=args.chunks_path,
        persist_dir=args.persist_dir,
        collection_name=collection_name,
        model_name=args.model,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )

    print(f"Chunks path: {summary.chunks_path.as_posix()}")
    print(f"Persist directory: {summary.persist_dir.as_posix()}")
    print(f"Collection: {summary.collection_name}")
    print(f"Embedding model: {summary.model_name}")
    print(f"Recreated collection: {summary.recreated}")
    print(f"Chunks read: {summary.total_chunks}")
    print(f"Vectors upserted: {summary.inserted_vectors}")
    print(f"Collection count: {summary.collection_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
