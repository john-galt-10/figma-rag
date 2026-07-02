"""Minimal example for retrieving Figma documentation chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.retrieval import (
    BM25Retriever,
    ChromaRetriever,
    DEFAULT_RETRIEVAL_COMPONENTS,
    RetrievalRequest,
    build_retrieval_pipeline,
    parse_metadata_filter_set,
    resolve_collection_name,
)

DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_gte-modernbert-base_t600_o60_20260622-1836.jsonl"
)
DEFAULT_METADATA_FILTERS = [
    "token_count>30",
    # "product=figma-design-or-general",
]
DEFAULT_TOPIC_FILTER = {"topic": {"$in": ["Figma Design", "Administration", "Help", "Community", "Work across Figma", "Get Started"]}}
DEFAULT_BM25_INDEX_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "bm25"
    / "hierarchical_bge-small-en-v1.5_t320_o40_bm25_stemmed_english_20260701t1733"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple retrieval query against local indexes."
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
        default="hierarchical-bge-w-product",
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
        "--bm25-index-dir",
        type=Path,
        default=DEFAULT_BM25_INDEX_DIR,
        help="Directory containing the persisted BM25 index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest chunks to return.",
    )
    parser.add_argument(
        "--metadata-filter",
        action="append",
        default=DEFAULT_METADATA_FILTERS.copy(),
        metavar="FILTER",
        help=(
            "Metadata filter to apply before vector ranking. Supports =, !=, <, "
            "<=, >, and >=. Can be repeated and filters are combined with AND. "
            "Defaults to token_count>30 and product=figma-design-or-general. "
            'Examples: --metadata-filter source_type=help_center --metadata-filter "token_count<80".'
        ),
    )
    parser.add_argument(
        "--disable-metadata-filters",
        action="store_true",
        help="Parse metadata filters but do not apply them during retrieval.",
    )
    parser.add_argument(
        "--disable-topic-filter",
        action="store_true",
        help=(
            "Do not restrict retrieval to the default Figma Design and "
            "Administration topics."
        ),
    )
    parser.add_argument(
        "--retrieval-component",
        action="append",
        choices=["chroma", "bm25"],
        default=None,
        help=(
            "Retrieval component to enable. Can be repeated. Defaults to chroma "
            "and bm25 together, which currently raises the hybrid TODO error."
        ),
    )
    return parser


def _describe_topic_filter(topic_filter: dict, enabled: bool = True) -> dict:
    """Return a JSON-serializable description of the default topic filter."""

    return {
        "enabled": enabled,
        "field": "topic",
        "operator": "in",
        "values": topic_filter["topic"]["$in"],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    try:
        metadata_filters = parse_metadata_filter_set(args.metadata_filter)
    except ValueError as exc:
        parser.error(str(exc))

    metadata_filters_enabled = not args.disable_metadata_filters
    topic_filter = None if args.disable_topic_filter else DEFAULT_TOPIC_FILTER
    retrieval_components = args.retrieval_component or list(DEFAULT_RETRIEVAL_COMPONENTS)

    collection_name = None
    if "chroma" in retrieval_components:
        collection_name = resolve_collection_name(
            chunks_path=args.chunks_path,
            model_name=args.model,
            collection_name=args.collection_name,
        )

    print(f"Query: {args.query}")
    if collection_name:
        print(f"Collection: {collection_name}")
        print(f"Embedding model: {args.model}")
    if "bm25" in retrieval_components:
        print(f"BM25 index directory: {args.bm25_index_dir.as_posix()}")
    print(f"Pipeline: {', '.join(retrieval_components)}")
    print(f"Metadata filters: {metadata_filters.to_description(metadata_filters_enabled)}")
    print(
        "Topic filter: "
        f"{_describe_topic_filter(DEFAULT_TOPIC_FILTER, enabled=not args.disable_topic_filter)}"
    )
    print()

    chroma_retriever = None
    if "chroma" in retrieval_components:
        chroma_retriever = ChromaRetriever(
            persist_dir=args.persist_dir,
            collection_name=str(collection_name),
            model_name=args.model,
        )
    bm25_retriever = None
    if "bm25" in retrieval_components:
        bm25_retriever = BM25Retriever(index_dir=args.bm25_index_dir)
    try:
        pipeline = build_retrieval_pipeline(
            component_names=retrieval_components,
            chroma_retriever=chroma_retriever,
            bm25_retriever=bm25_retriever,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        results = pipeline.retrieve(
            RetrievalRequest(
                query=args.query,
                top_k=args.top_k,
                metadata_filters=metadata_filters,
                metadata_filters_enabled=metadata_filters_enabled,
                raw_chroma_where=topic_filter,
            )
        )
    except NotImplementedError as exc:
        parser.error(str(exc))

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
