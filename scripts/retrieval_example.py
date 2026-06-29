"""Minimal example for retrieving Figma documentation chunks from Chroma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.retrieval import ChromaRetriever, resolve_collection_name

DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_gte-modernbert-base_t600_o60_20260622-1836.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple semantic retrieval query against local Chroma."
    )
    parser.add_argument(
        "query",
        help="Question or search text to retrieve relevant documentation chunks for.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=(
            "Chunk JSONL used only to infer the default collection name. Use the "
            "same path used when building the vector index."
        ),
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "figma_docs" / "chroma",
        help="Directory containing the persistent Chroma database.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Name of the Chroma collection to query. By default, a name is built "
            "from the chunking artifact and embedding model."
        ),
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
#        default="Alibaba-NLP/gte-modernbert-base",
        help="Sentence Transformers model used to embed the query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest chunks to return.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    collection_name = resolve_collection_name(
        chunks_path=args.chunks_path,
        model_name=args.model,
        collection_name=args.collection_name,
    )

    print(f"Query: {args.query}")
    print(f"Collection: {collection_name}")
    print(f"Embedding model: {args.model}")
    print()

    retriever = ChromaRetriever(
        persist_dir=args.persist_dir,
        collection_name=collection_name,
        model_name=args.model,
    )
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for result in results:
        preview = " ".join(result.text.split())[:500]

        print(f"{result.rank}. {result.title}")
        print(f"   Chunk: {result.chunk_id}")
        print(f"   Section: {result.section}")
        print(f"   Distance: {result.distance:.4f}")
        print(f"   Source: {result.source_url}")
        print(f"   Metadata: {result.metadata}")
        print(f"   Preview: {preview}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
