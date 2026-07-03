"""Minimal example for retrieving Figma documentation chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colorama import Fore, Style, init

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.retrieval import (
    BM25Retriever,
    ChromaRetriever,
    CrossEncoderReranker,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVAL_COMPONENTS,
    RetrievalRequest,
    available_aggregation_strategies,
    build_retrieval_pipeline,
    parse_component_weights,
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
        default="hierarchical-bge-w-topic",
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
        "--candidate-k",
        type=int,
        default=None,
        help=(
            "Number of candidates retrieved by each retriever before reranking. "
            "Defaults to --top-k * 5 when reranking is enabled and is ignored "
            "when reranking is disabled."
        ),
    )
    parser.add_argument(
        "--disable-reranking",
        action="store_true",
        help="Disable cross-encoder reranking and return the retrieval ranking directly.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Sentence Transformers CrossEncoder model used for reranking.",
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
            "and bm25 together."
        ),
    )
    parser.add_argument(
        "--aggregation-strategy",
        choices=available_aggregation_strategies(),
        default="weighted_rrf",
        help=(
            "Strategy used to combine multiple retrieval components. "
            "Defaults to weighted_rrf."
        ),
    )
    parser.add_argument(
        "--component-weight",
        action="append",
        default=None,
        metavar="COMPONENT=WEIGHT",
        help=(
            "Weighted RRF component weight. Can be repeated once per enabled "
            "component, and weights must sum to 1.0. "
            "Example: --component-weight chroma=0.5 --component-weight bm25=0.5."
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


def print_retrieval_settings(
    query: str,
    collection_name: str | None,
    model_name: str,
    bm25_index_dir: Path,
    retrieval_components: list[str],
    aggregation_strategy: str,
    top_k: int,
    reranking_enabled: bool,
    reranker_model: str,
    effective_candidate_k: int | None,
    component_weights: dict[str, float] | None,
    metadata_filter_description: list[dict],
    topic_filter_description: dict,
) -> None:
    """Print retrieval settings in cyan for terminal readability."""

    lines = [f"Query: {query}"]
    if collection_name:
        lines.extend(
            [
                f"Collection: {collection_name}",
                f"Embedding model: {model_name}",
            ]
        )
    if "bm25" in retrieval_components:
        lines.append(f"BM25 index directory: {bm25_index_dir.as_posix()}")
    lines.extend(
        [
            f"Pipeline: {', '.join(retrieval_components)}",
            f"Aggregation strategy: {aggregation_strategy}",
            f"Final top-k: {top_k}",
            f"Reranking enabled: {reranking_enabled}",
        ]
    )
    if reranking_enabled:
        lines.extend(
            [
                f"Reranker model: {reranker_model}",
                f"Candidate-k per retriever: {effective_candidate_k}",
            ]
        )
    else:
        lines.append("Candidate-k per retriever: ignored")
    if component_weights is not None:
        lines.append(f"Component weights: {component_weights}")
    lines.extend(
        [
            f"Metadata filters: {metadata_filter_description}",
            f"Topic filter: {topic_filter_description}",
        ]
    )
    print(Fore.CYAN + "\n".join(lines) + Style.RESET_ALL)
    print()


def print_retrieved_chunk(result) -> None:
    """Print one retrieved chunk block with separate metadata and preview colors."""

    preview = " ".join(result.text.split())[:500]
    rerank_score = result.metadata.get("rerank_score")
    main_lines = [
        f"{result.rank}. {result.title}",
        f"   Chunk: {result.chunk_id}",
        f"   Section: {result.section}",
    ]
    if rerank_score is not None:
        main_lines.append(f"   Rerank score: {float(rerank_score):.4f}")
    main_lines.extend(
        [
            f"   Distance: {result.distance:.4f}",
            f"   Source: {result.source_url}",
        ]
    )
    print(Fore.LIGHTYELLOW_EX + "\n".join(main_lines) + Style.RESET_ALL)
    print(Fore.LIGHTBLACK_EX + f"   Metadata: {result.metadata}" + Style.RESET_ALL)
    print(Fore.WHITE + f"   Preview: {preview}" + Style.RESET_ALL)
    print()


def main() -> int:
    init(autoreset=True)
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
    reranking_enabled = not args.disable_reranking
    if reranking_enabled and args.candidate_k is not None and args.candidate_k <= 0:
        raise ValueError("--candidate-k must be greater than zero")
    effective_candidate_k = (
        None
        if not reranking_enabled
        else args.candidate_k or args.top_k * 5
    )
    if args.aggregation_strategy == "union":
        if args.component_weight:
            print(
                "Warning: --component-weight values are ignored because "
                "--aggregation-strategy is union.",
                file=sys.stderr,
            )
        component_weights = None
    elif args.aggregation_strategy == "weighted_rrf":
        try:
            component_weights = parse_component_weights(
                args.component_weight,
                retrieval_components,
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.component_weight:
            parser.error(
                "--component-weight is only supported with weighted_rrf aggregation"
            )
        component_weights = None

    collection_name = None
    if "chroma" in retrieval_components:
        collection_name = resolve_collection_name(
            chunks_path=args.chunks_path,
            model_name=args.model,
            collection_name=args.collection_name,
        )

    print_retrieval_settings(
        query=args.query,
        collection_name=collection_name,
        model_name=args.model,
        bm25_index_dir=args.bm25_index_dir,
        retrieval_components=retrieval_components,
        aggregation_strategy=args.aggregation_strategy,
        top_k=args.top_k,
        reranking_enabled=reranking_enabled,
        reranker_model=args.reranker_model,
        effective_candidate_k=effective_candidate_k,
        component_weights=component_weights,
        metadata_filter_description=metadata_filters.to_description(
            metadata_filters_enabled
        ),
        topic_filter_description=_describe_topic_filter(
            DEFAULT_TOPIC_FILTER,
            enabled=not args.disable_topic_filter,
        ),
    )

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
    reranker = (
        CrossEncoderReranker(model_name=args.reranker_model)
        if reranking_enabled
        else None
    )
    try:        
        pipeline = build_retrieval_pipeline(
            component_names=retrieval_components,
            chroma_retriever=chroma_retriever,
            bm25_retriever=bm25_retriever,
            aggregation_strategy_name=args.aggregation_strategy,
            component_weights=component_weights,
            reranker=reranker,
        )
    except ValueError as exc:
        parser.error(str(exc))

    results = pipeline.retrieve(
        RetrievalRequest(
            query=args.query,
            top_k=args.top_k,
            candidate_k=effective_candidate_k,
            metadata_filters=metadata_filters,
            metadata_filters_enabled=metadata_filters_enabled,
            raw_chroma_where=topic_filter,
        )
    )
    if pipeline.last_reranking_result:
        latency_ms = pipeline.last_reranking_result.latency_seconds * 1000
        print(
            "Reranking latency: "
            f"{latency_ms:.2f} ms for "
            f"{pipeline.last_reranking_result.candidate_count} candidates"
        )
        print()

    for result in results:
        print_retrieved_chunk(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
