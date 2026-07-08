"""Generate a grounded answer from the local Figma RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colorama import Fore, Style, init

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.generation import (  # noqa: E402
    AnswerGenerationConfig,
    build_answer_generation_pipeline,
    load_answer_generation_config,
    resolve_generation_retrieval_options,
)
from figma_rag.retrieval import RetrievalResult  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).with_name("generate_answer_config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for answer generation."""

    parser = argparse.ArgumentParser(
        description="Generate a grounded answer with the configured RAG pipeline."
    )
    parser.add_argument(
        "query",
        help="Question to answer using retrieved Figma documentation chunks.",
    )
    return parser


def print_request_options(
    query: str,
    config: AnswerGenerationConfig,
    candidate_k: int | None,
) -> None:
    """Print retrieval and generation request options in cyan."""

    retrieval_config = config.retrieval
    generation_config = config.generation
    lines = [
        "Answer generation request options:",
        f"Query: {query}",
        f"Provider: {generation_config.provider}",
        f"Generation model: {generation_config.model}",
        f"Base URL: {generation_config.base_url}",
        f"API key environment variable: {generation_config.api_key_env}",
        f"Temperature: {generation_config.temperature}",
        f"Top-p: {generation_config.top_p}",
        f"Max tokens: {generation_config.max_tokens}",
        f"Retrieval components: {', '.join(retrieval_config.components)}",
        f"Aggregation strategy: {retrieval_config.aggregation_strategy}",
        f"Component weights: {retrieval_config.component_weights}",
        f"Embedding model: {retrieval_config.embedding_model}",
        f"Collection: {retrieval_config.collection_name}",
        f"BM25 index directory: {retrieval_config.bm25_index_dir.as_posix()}",
        f"Final top-k: {retrieval_config.top_k}",
        f"Candidate-k per retriever: {candidate_k}",
        f"Reranking enabled: {retrieval_config.reranking_enabled}",
        f"Reranker model: {retrieval_config.reranker_model}",
        f"Metadata filters enabled: {retrieval_config.metadata_filters_enabled}",
        f"Metadata filters: {retrieval_config.metadata_filters}",
        f"Topic filter enabled: {retrieval_config.topic_filter_enabled}",
        f"Topic filter: {retrieval_config.topic_filter_where}",
    ]
    print(Fore.CYAN + "\n".join(lines) + Style.RESET_ALL)
    print()


def print_retrieved_chunk(result: RetrievalResult) -> None:
    """Print one retrieved chunk in grey for visual separation from the answer."""

    preview = " ".join(result.text.split())[:700]
    rerank_score = result.metadata.get("rerank_score")
    lines = [
        f"{result.rank}. {result.title}",
        f"   Chunk: {result.chunk_id}",
        f"   Section: {result.section}",
    ]
    if rerank_score is not None:
        lines.append(f"   Rerank score: {float(rerank_score):.4f}")
    lines.extend(
        [
            f"   Distance: {result.distance:.4f}",
            f"   Source: {result.source_url}",
            f"   Preview: {preview}",
        ]
    )
    print(Fore.LIGHTBLACK_EX + "\n".join(lines) + Style.RESET_ALL)
    print()


def print_truncation_warning(
    results: list[RetrievalResult],
    max_chunk_chars: int,
) -> None:
    """Warn when retrieved chunks will be truncated in the generation prompt."""

    truncated_chunks = [
        result
        for result in results
        if len(" ".join(result.text.split())) > max_chunk_chars
    ]
    if not truncated_chunks:
        return

    lines = [
        "Warning: some retrieved chunks exceed prompt.max_chunk_chars and will be "
        "truncated before answer generation.",
        f"Limit: {max_chunk_chars} characters per chunk",
        "Truncated chunks: "
        + ", ".join(
            f"{result.rank}. {result.chunk_id}" for result in truncated_chunks
        ),
    ]
    print(Fore.YELLOW + "\n".join(lines) + Style.RESET_ALL)
    print()


def main() -> int:
    """Retrieve context, generate an answer, and print the full request trace."""

    init(autoreset=True)
    parser = build_parser()
    args = parser.parse_args()

    config = load_answer_generation_config(DEFAULT_CONFIG_PATH, REPO_ROOT)
    retrieval_options = resolve_generation_retrieval_options(config)

    print_request_options(args.query, config, retrieval_options.candidate_k)

    pipeline = build_answer_generation_pipeline(config)
    generation_result = pipeline.generate(args.query)
    results = generation_result.retrieved_chunks

    print(Fore.LIGHTBLACK_EX + "Retrieved chunks:" + Style.RESET_ALL)
    print()
    for result in results:
        print_retrieved_chunk(result)

    print_truncation_warning(results, config.prompt.max_chunk_chars)

    print(generation_result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
