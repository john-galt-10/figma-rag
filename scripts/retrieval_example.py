"""Minimal example for retrieving Figma documentation chunks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from colorama import Fore, Style, init

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.retrieval import (
    RetrievalConfig,
    RetrievalOptions,
    RetrievalRequest,
    available_aggregation_strategies,
    build_configured_retrieval_pipeline,
    load_retrieval_config,
    parse_component_weights,
    resolve_retrieval_options,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("generate_answer_config.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple retrieval query against local indexes."
    )
    parser.add_argument(
        "query",
        help="Question or search text to retrieve relevant documentation chunks for.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config whose retrieval section defines the retrieval pipeline.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help=(
            "Chunk JSONL used only to infer the default collection name. "
            "Defaults to retrieval.chunks_path from the YAML config."
        ),
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the persistent Chroma database. Defaults to "
            "retrieval.chroma_persist_dir from the YAML config."
        ),
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Name of the Chroma collection to query. Defaults to the YAML config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Sentence Transformers model used to embed the query. Defaults to "
            "retrieval.embedding_model from the YAML config."
        ),
    )
    parser.add_argument(
        "--bm25-index-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the persisted BM25 index. Defaults to "
            "retrieval.bm25_index_dir from the YAML config."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of nearest chunks to return. Defaults to retrieval.top_k.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help=(
            "Number of candidates retrieved by each retriever before reranking. "
            "Defaults to retrieval.candidate_k, then top_k * 5 when reranking "
            "is enabled."
        ),
    )
    parser.add_argument(
        "--disable-reranking",
        action="store_true",
        help="Disable cross-encoder reranking for this run.",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Sentence Transformers CrossEncoder model used for reranking.",
    )
    parser.add_argument(
        "--metadata-filter",
        action="append",
        default=None,
        metavar="FILTER",
        help=(
            "Metadata filter to apply before ranking. Can be repeated. Explicit "
            "values replace retrieval.metadata_filters from the YAML config."
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
        help="Do not apply retrieval.topic_filter from the YAML config.",
    )
    parser.add_argument(
        "--retrieval-component",
        action="append",
        choices=["chroma", "bm25"],
        default=None,
        help=(
            "Retrieval component to enable. Can be repeated. Explicit values "
            "replace retrieval.components from the YAML config."
        ),
    )
    parser.add_argument(
        "--aggregation-strategy",
        choices=available_aggregation_strategies(),
        default=None,
        help="Strategy used to combine multiple retrieval components.",
    )
    parser.add_argument(
        "--component-weight",
        action="append",
        default=None,
        metavar="COMPONENT=WEIGHT",
        help=(
            "Weighted RRF component weight. Can be repeated once per enabled "
            "component, and weights must sum to 1.0."
        ),
    )
    return parser


def apply_cli_overrides(
    config: RetrievalConfig,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> RetrievalConfig:
    """Return the YAML retrieval config with explicitly supplied CLI overrides."""

    updates = {}
    if args.chunks_path is not None:
        updates["chunks_path"] = args.chunks_path
    if args.persist_dir is not None:
        updates["chroma_persist_dir"] = args.persist_dir
    if args.collection_name is not None:
        updates["collection_name"] = args.collection_name
    if args.model is not None:
        updates["embedding_model"] = args.model
    if args.bm25_index_dir is not None:
        updates["bm25_index_dir"] = args.bm25_index_dir
    if args.top_k is not None:
        if args.top_k <= 0:
            parser.error("--top-k must be greater than zero")
        updates["top_k"] = args.top_k
    if args.candidate_k is not None:
        if args.candidate_k <= 0:
            parser.error("--candidate-k must be greater than zero")
        updates["candidate_k"] = args.candidate_k
    if args.disable_reranking:
        updates["reranking_enabled"] = False
    if args.reranker_model is not None:
        updates["reranker_model"] = args.reranker_model
    if args.metadata_filter is not None:
        updates["metadata_filters"] = args.metadata_filter
    if args.disable_metadata_filters:
        updates["metadata_filters_enabled"] = False
    if args.disable_topic_filter:
        updates["topic_filter_enabled"] = False
    if args.retrieval_component is not None:
        updates["components"] = args.retrieval_component
    if args.aggregation_strategy is not None:
        updates["aggregation_strategy"] = args.aggregation_strategy

    effective_config = replace(config, **updates)
    if args.component_weight is not None:
        if effective_config.aggregation_strategy != "weighted_rrf":
            parser.error(
                "--component-weight is only supported with weighted_rrf aggregation"
            )
        try:
            component_weights = parse_component_weights(
                args.component_weight,
                effective_config.components,
            )
        except ValueError as exc:
            parser.error(str(exc))
        effective_config = replace(effective_config, component_weights=component_weights)
    elif args.retrieval_component is not None or args.aggregation_strategy is not None:
        effective_config = replace(effective_config, component_weights=None)

    return effective_config


def _describe_topic_filter(topic_filter: dict | None, enabled: bool = True) -> dict:
    """Return a JSON-serializable description of the default topic filter."""

    if not topic_filter:
        return {"enabled": enabled, "where": None}
    return {
        "enabled": enabled,
        "where": topic_filter,
    }


def print_retrieval_settings(
    query: str,
    config_path: Path,
    config: RetrievalConfig,
    options: RetrievalOptions,
) -> None:
    """Print retrieval settings in cyan for terminal readability."""

    lines = [
        f"Query: {query}",
        f"Config: {config_path}",
        f"Retrieval components: {', '.join(config.components)}",
        f"Aggregation strategy: {config.aggregation_strategy}",
        f"Component weights: {config.component_weights}",
        f"Collection: {config.collection_name}",
        f"Embedding model: {config.embedding_model}",
        f"Chroma persist directory: {config.chroma_persist_dir.as_posix()}",
        f"BM25 index directory: {config.bm25_index_dir.as_posix()}",
        f"Final top-k: {options.top_k}",
        f"Reranking enabled: {config.reranking_enabled}",
    ]
    if config.reranking_enabled:
        lines.extend(
            [
                f"Reranker model: {config.reranker_model}",
                f"Candidate-k per retriever: {options.candidate_k}",
            ]
        )
    else:
        lines.append("Candidate-k per retriever: ignored")
    lines.extend(
        [
            f"Metadata filters enabled: {options.metadata_filters_enabled}",
            f"Metadata filters: {options.metadata_filters.to_description(options.metadata_filters_enabled)}",
            f"Topic filter: {_describe_topic_filter(options.topic_filter, config.topic_filter_enabled)}",
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

    try:
        config = load_retrieval_config(args.config_path, REPO_ROOT)
        config = apply_cli_overrides(config, args, parser)
        options = resolve_retrieval_options(config)
    except ValueError as exc:
        parser.error(str(exc))

    print_retrieval_settings(
        query=args.query,
        config_path=args.config_path,
        config=config,
        options=options,
    )

    try:
        pipeline = build_configured_retrieval_pipeline(config)
    except ValueError as exc:
        parser.error(str(exc))

    results = pipeline.retrieve(
        RetrievalRequest(
            query=args.query,
            top_k=options.top_k,
            candidate_k=options.candidate_k,
            metadata_filters=options.metadata_filters,
            metadata_filters_enabled=options.metadata_filters_enabled,
            raw_chroma_where=options.topic_filter,
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
