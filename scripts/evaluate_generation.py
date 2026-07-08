"""Evaluate generated RAG answers with an LLM-as-judge workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from colorama import Fore, Style, init

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.evaluation import (  # noqa: E402
    build_generation_output_paths,
    build_generation_summary_payload,
    ensure_parquet_available,
    generation_rows_to_dicts,
    load_retrieval_test_set,
    RetrievalQueryExample,
    run_generation_evaluation,
    write_generation_details_jsonl,
    write_generation_details_parquet,
    write_metrics_json,
)
from figma_rag.generation import (  # noqa: E402
    AnswerGenerationPipeline,
    load_answer_generation_config,
)

DEFAULT_TEST_SET_PATH = (
    REPO_ROOT
    / "data"
    / "eval"
    / "retrieval_test"
    / "golden_set_manual_and_codex_relevant_chunks_hierarchical_bge-small-en-v1.5_20260630-1601.jsonl"
)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("generate_answer_config.yaml")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "generation_test" / "test_results"


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for generation evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate generated RAG answers with an LLM-as-judge workflow."
    )
    parser.add_argument(
        "--test-set-path",
        type=Path,
        default=DEFAULT_TEST_SET_PATH,
        help="Mapped retrieval test-set JSONL containing query and chunks fields.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config used by scripts/generate_answer.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generation judge artifacts are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples from the test set.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Sleep interval between evaluated samples.",
    )
    parser.add_argument(
        "--save-jsonl",
        action="store_true",
        help="Also write per-query records as newline-delimited JSON.",
    )
    return parser


def main() -> int:
    """Run generation, judge each answer, and write evaluation artifacts."""

    init(autoreset=True)
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be zero or greater")

    ensure_parquet_available()

    config = load_answer_generation_config(args.config_path, REPO_ROOT)
    examples = load_retrieval_test_set(args.test_set_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    run = run_generation_evaluation(
        examples=examples,
        config=config,
        test_set_path=args.test_set_path,
        delay_seconds=args.delay_seconds,
        settings_callback=lambda pipeline: print_evaluation_settings(
            test_set_path=args.test_set_path,
            config_path=args.config_path,
            output_dir=args.output_dir,
            query_count=len(examples),
            pipeline=pipeline,
        ),
        progress_callback=print_progress,
    )

    details_path, summary_path, jsonl_path = build_generation_output_paths(
        test_set_path=args.test_set_path,
        output_dir=args.output_dir,
    )
    serializable_rows = generation_rows_to_dicts(run.rows)
    write_generation_details_parquet(details_path, serializable_rows)
    summary_payload = build_generation_summary_payload(
        rows=run.rows,
        test_set_path=args.test_set_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        details_path=details_path,
        jsonl_path=jsonl_path if args.save_jsonl else None,
        config=config,
        pipeline_description=run.pipeline_description,
        candidate_k=run.candidate_k,
    )
    write_metrics_json(summary_path, summary_payload)

    if args.save_jsonl:
        write_generation_details_jsonl(jsonl_path, serializable_rows)
        print(f"Wrote per-query JSONL details to {jsonl_path}")

    print(f"Wrote per-query Parquet details to {details_path}")
    print(f"Wrote aggregate judge metrics to {summary_path}")
    print_summary(summary_payload)
    return 0


def print_evaluation_settings(
    test_set_path: Path,
    config_path: Path,
    output_dir: Path,
    query_count: int,
    pipeline: AnswerGenerationPipeline,
) -> None:
    """Print the main generation evaluation settings in cyan."""

    generation_config = pipeline.config.generation
    judge_config = pipeline.config.judge
    lines = [
        "Evaluating generated answers with LLM-as-judge settings:",
        f"Test set: {test_set_path}",
        f"Config: {config_path}",
        f"Output directory: {output_dir}",
        f"Query count: {query_count}",
        f"Generation provider: {generation_config.provider}",
        f"Generation model: {generation_config.model}",
        f"Judge provider: {judge_config.provider}",
        f"Judge model: {judge_config.model}",
        f"Pipeline: {pipeline.retrieval_pipeline.to_description()}",
        f"Candidate-k per retriever: {pipeline.retrieval_options.candidate_k}",
    ]
    print(Fore.CYAN + "\n".join(lines) + Style.RESET_ALL)
    print()


def print_progress(
    index: int,
    total: int,
    example: RetrievalQueryExample,
) -> None:
    """Print a compact progress line for one evaluation example."""

    print(
        Fore.LIGHTBLACK_EX
        + f"[{index}/{total}] Evaluating {example.query_id}: {example.query}"
        + Style.RESET_ALL
    )


def print_summary(summary_payload: dict[str, Any]) -> None:
    """Print a compact aggregate score summary."""

    average_scores = summary_payload["average_scores"]
    lines = ["Average judge scores:"]
    for metric_name, score in average_scores.items():
        formatted_score = "n/a" if score is None else f"{score:.3f}"
        lines.append(f"{metric_name}: {formatted_score}")
    lines.append(f"Error counts: {summary_payload['error_counts']}")
    print()
    print(Fore.CYAN + "\n".join(lines) + Style.RESET_ALL)


if __name__ == "__main__":
    raise SystemExit(main())
