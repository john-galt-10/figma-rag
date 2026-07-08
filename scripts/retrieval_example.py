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
    RetrievalConfig,
    RetrievalOptions,
    RetrievalRequest,
    build_configured_retrieval_pipeline,
    load_retrieval_config,
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
    return parser


def _describe_topic_filter(topic_filter: dict, enabled: bool = True) -> dict:
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
