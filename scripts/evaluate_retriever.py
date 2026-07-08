"""Entrypoint for evaluating retrieval quality on a mapped test set."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from colorama import Fore, Style, init
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.evaluation import (  # noqa: E402
    evaluate_retrieval_results,
    load_retrieval_test_set,
    normalize_top_k_values,
    set_reproducibility_seed,
    sha256_file,
    stabilize_retrieval_ties,
    write_detailed_results_parquet,
    write_metrics_json,
)
from figma_rag.retrieval import (  # noqa: E402
    BM25Retriever,
    ChromaRetriever,
    RetrievalConfig,
    RetrievalOptions,
    RetrievalRequest,
    build_configured_retrieval_pipeline,
    load_retrieval_config,
    resolve_retrieval_options,
)

DEFAULT_TEST_SET_PATH = (
    REPO_ROOT
    / "data"
    / "eval"
    / "retrieval_test"
    # / "golden_set_manual_2_complete_20260701_1753_relevant_chunks_hierarchical_bge-small-en-v1.5_20260629-1709.jsonl"
    / "golden_set_manual_and_codex_relevant_chunks_hierarchical_bge-small-en-v1.5_20260630-1601.jsonl"
)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("generate_answer_config.yaml")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "retrieval_test" / "test_results"
TEST_SET_FILENAME_PATTERN = re.compile(
    r"^.+?_relevant_chunks_(?P<label>.+?)_(?P<timestamp>\d{8}-\d{4})$"
)
METRIC_COLUMNS = [
    "hit_rate_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "map_at_k",
    "ndcg_at_k",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for retrieval evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate retrievers against a mapped retrieval test set."
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
        help="YAML config whose retrieval section defines the retrieval pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more retrieval cutoffs to evaluate. Defaults to retrieval.top_k "
            "from the YAML config when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where metrics and optional detailed results are written.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Write detailed per-query results as a Parquet file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for available random number generators.",
    )
    return parser


def build_output_paths(
    test_set_path: Path,
    top_k_values: list[int],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Build traceable output paths for aggregate and detailed artifacts."""

    test_set_label, test_set_timestamp = _artifact_label_and_timestamp(test_set_path)
    current_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")

    top_k_label = "k" + "-".join(str(top_k) for top_k in top_k_values)
    artifact_label = f"{test_set_label}_{test_set_timestamp}_{top_k_label}"

    return (
        output_dir / f"retrieval_metrics_{artifact_label}_{current_timestamp}.json",
        output_dir / f"retrieval_details_{artifact_label}_{current_timestamp}.parquet",
    )


def _describe_topic_filter(topic_filter: dict | None, enabled: bool = True) -> dict:
    """Return a JSON-serializable description of the default topic filter."""

    if not topic_filter:
        return {"enabled": enabled, "where": None}
    return {
        "enabled": enabled,
        "where": topic_filter,
    }


def print_evaluation_settings(
    pipeline_description: dict,
    config_path: Path,
    config: RetrievalConfig,
    options: RetrievalOptions,
    top_k_values: list[int],
) -> None:
    """Print the evaluation settings block in green for terminal readability."""

    lines = [
        "Evaluating Retriever performance w/ the following settings:",
        f"Config: {config_path}",
        f"Pipeline: {pipeline_description}",
        f"Final top-k values: {top_k_values}",
        f"Retrieval components: {', '.join(config.components)}",
        f"Aggregation strategy: {config.aggregation_strategy}",
        f"Component weights: {config.component_weights}",
        f"Collection: {config.collection_name}",
        f"Embedding model: {config.embedding_model}",
        f"Chroma persist directory: {config.chroma_persist_dir.as_posix()}",
        f"BM25 index directory: {config.bm25_index_dir.as_posix()}",
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
            f"Metadata filters: {options.metadata_filters.to_description(options.metadata_filters_enabled)}",
            f"Topic filter: {_describe_topic_filter(options.topic_filter, config.topic_filter_enabled)}",
        ]
    )
    print(Fore.CYAN + "\n".join(lines) + Style.RESET_ALL)


def format_metrics_table(metrics_by_k: dict[str, dict[str, float]]) -> str:
    """Return a compact display-only metrics table rounded to 3 decimals."""

    rows = []
    for top_k, metrics in sorted(metrics_by_k.items(), key=lambda item: int(item[0])):
        row = {"k": int(top_k)}
        row.update(
            {
                metric_name: f"{metrics[metric_name]:.3f}"
                for metric_name in METRIC_COLUMNS
            }
        )
        rows.append(row)

    return tabulate(rows, headers="keys", tablefmt="grid")


def main() -> int:
    """Run retrieval evaluation and write comparison-friendly artifacts."""

    init(autoreset=True)
    parser = build_parser()
    args = parser.parse_args()
    print(f"Evaluating the retriever performance on: {args.test_set_path}")
    seed_settings = set_reproducibility_seed(args.seed)

    try:
        config = load_retrieval_config(args.config_path, REPO_ROOT)
    except ValueError as exc:
        parser.error(str(exc))

    top_k_values = normalize_top_k_values(args.top_k or [config.top_k])
    max_top_k = max(top_k_values)
    try:
        options = resolve_retrieval_options(config, top_k=max_top_k)
    except ValueError as exc:
        parser.error(str(exc))

    test_set_examples = load_retrieval_test_set(args.test_set_path)

    try:
        pipeline = build_configured_retrieval_pipeline(config)
    except ValueError as exc:
        parser.error(str(exc))

    chroma_retriever = _component_retriever(pipeline, "chroma")
    bm25_retriever = _component_retriever(pipeline, "bm25")

    retrieved_chunk_ids_by_query_id = {}
    retrieved_component_ranks_by_query_id = {}
    reranking_latencies_seconds: list[float] = []

    print_evaluation_settings(
        pipeline_description=pipeline.to_description(),
        config_path=args.config_path,
        config=config,
        options=options,
        top_k_values=top_k_values,
    )

    for example in test_set_examples:
        request = RetrievalRequest(
            query=example.query,
            top_k=max_top_k,
            candidate_k=options.candidate_k,
            metadata_filters=options.metadata_filters,
            metadata_filters_enabled=options.metadata_filters_enabled,
            raw_chroma_where=options.topic_filter,
        )
        results = stabilize_retrieval_ties(pipeline.retrieve(request))
        if pipeline.last_reranking_result:
            reranking_latencies_seconds.append(
                pipeline.last_reranking_result.latency_seconds
            )
        retrieved_chunk_ids_by_query_id[example.query_id] = [
            result.chunk_id for result in results
        ]
        retrieved_component_ranks_by_query_id[example.query_id] = (
            _component_rank_details(results, pipeline.component_names)
        )

    representative_request = RetrievalRequest(
        query="metadata",
        top_k=max_top_k,
        metadata_filters=options.metadata_filters,
        metadata_filters_enabled=options.metadata_filters_enabled,
        raw_chroma_where=options.topic_filter,
    )

    evaluation = evaluate_retrieval_results(
        examples=test_set_examples,
        retrieved_chunk_ids_by_query_id=retrieved_chunk_ids_by_query_id,
        retrieved_component_ranks_by_query_id=retrieved_component_ranks_by_query_id,
        top_k_values=top_k_values,
    )
    metrics_path, details_path = build_output_paths(
        test_set_path=args.test_set_path,
        top_k_values=top_k_values,
        output_dir=args.output_dir,
    )
    metrics_payload = {
        "metadata": {
            "test_set_path": args.test_set_path.as_posix(),
            "test_set_path_resolved": args.test_set_path.resolve().as_posix(),
            "test_set_sha256": sha256_file(args.test_set_path),
            "config_path": args.config_path.as_posix(),
            "config_path_resolved": args.config_path.resolve().as_posix(),
            "persist_dir": config.chroma_persist_dir.as_posix(),
            "persist_dir_resolved": config.chroma_persist_dir.resolve().as_posix(),
            "bm25_index_dir": config.bm25_index_dir.as_posix(),
            "bm25_index_dir_resolved": config.bm25_index_dir.resolve().as_posix(),
            "output_dir": args.output_dir.as_posix(),
            "output_dir_resolved": args.output_dir.resolve().as_posix(),
            "collection_name": config.collection_name,
            "collection": _collection_metadata(chroma_retriever)
            if chroma_retriever
            else None,
            "bm25_index": _bm25_index_metadata(bm25_retriever)
            if bm25_retriever
            else None,
            "model": config.embedding_model,
            "retrieval_pipeline": pipeline.to_description(),
            "reranking": _reranking_metadata(
                enabled=config.reranking_enabled,
                model_name=config.reranker_model,
                top_k_values=top_k_values,
                candidate_k=options.candidate_k,
                latencies_seconds=reranking_latencies_seconds,
            ),
            "metadata_filters": options.metadata_filters.to_description(
                enabled=options.metadata_filters_enabled
            ),
            "topic_filter": _describe_topic_filter(
                options.topic_filter,
                enabled=config.topic_filter_enabled,
            ),
            "chroma_where": representative_request.chroma_where,
            "top_k_values": top_k_values,
            "max_top_k_retrieved": max_top_k,
            "query_count": len(test_set_examples),
            "seed": args.seed,
            "seed_settings": seed_settings,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics_by_k": evaluation.metrics_by_k,
    }

    write_metrics_json(metrics_path, metrics_payload)
    print(f"Wrote aggregate metrics to {metrics_path}")

    if args.save_details:
        write_detailed_results_parquet(details_path, evaluation.details)
        print(f"Wrote detailed per-query results to {details_path}")

    print()
    print(format_metrics_table(evaluation.metrics_by_k))
    return 0


def _artifact_label_and_timestamp(test_set_path: Path) -> tuple[str, str]:
    """Extract a readable label and timestamp from a mapped test-set filename."""

    stem = test_set_path.stem
    match = TEST_SET_FILENAME_PATTERN.match(stem)
    if match:
        return _slugify(match.group("label")), match.group("timestamp")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return _slugify(stem), timestamp


def _component_retriever(pipeline, component_name: str):
    """Return the underlying retriever for a named pipeline component, if present."""

    for component in pipeline.components:
        if component.name == component_name:
            return component.retriever
    return None


def _slugify(value: str) -> str:
    """Return a filesystem-friendly label."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug or "test-set"


def _component_rank_details(results: list, component_names: list[str]) -> list[dict]:
    """Build Parquet-friendly per-component rank details for retrieved chunks."""

    details = []
    for result in results:
        ranks = result.metadata.get("retrieval_component_ranks")
        distances = result.metadata.get("retrieval_component_distances")
        if not isinstance(ranks, dict):
            ranks = (
                {
                    component_names[0]: result.metadata.get(
                        "reranking_original_rank",
                        result.rank,
                    )
                }
                if len(component_names) == 1
                else {}
            )
        else:
            ranks = dict(ranks)
        if not isinstance(distances, dict):
            distances = (
                {
                    component_names[0]: result.metadata.get(
                        "reranking_original_distance",
                        result.distance,
                    )
                }
                if len(component_names) == 1
                else {}
            )
        else:
            distances = dict(distances)
        if "rerank_score" in result.metadata:
            ranks["reranker"] = result.rank
            distances["reranker"] = result.distance
        details.append(
            {
                "chunk_id": result.chunk_id,
                "ranks": ranks,
                "distances": distances,
            }
        )
    return details


def _reranking_metadata(
    enabled: bool,
    model_name: str,
    top_k_values: list[int],
    candidate_k: int | None,
    latencies_seconds: list[float],
) -> dict:
    """Return JSON-friendly reranking configuration and latency summary."""

    return {
        "enabled": enabled,
        "model": model_name if enabled else None,
        "top_k_values": top_k_values,
        "candidate_k": candidate_k,
        "latency_seconds": _latency_summary(latencies_seconds),
    }


def _latency_summary(latencies_seconds: list[float]) -> dict:
    """Summarize per-query reranking latencies."""

    if not latencies_seconds:
        return {
            "total": 0.0,
            "average": None,
            "min": None,
            "max": None,
            "count": 0,
        }

    return {
        "total": sum(latencies_seconds),
        "average": sum(latencies_seconds) / len(latencies_seconds),
        "min": min(latencies_seconds),
        "max": max(latencies_seconds),
        "count": len(latencies_seconds),
    }


def _collection_metadata(retriever: ChromaRetriever) -> dict:
    """Return best-effort metadata for the Chroma collection used in evaluation."""

    collection = retriever._collection
    metadata = {
        "name": retriever.collection_name,
        "count": None,
        "metadata": None,
    }

    try:
        metadata["count"] = collection.count()
    except Exception as exc:
        metadata["count_error"] = str(exc)

    try:
        metadata["metadata"] = dict(collection.metadata or {})
    except Exception as exc:
        metadata["metadata_error"] = str(exc)

    return metadata


def _bm25_index_metadata(retriever: BM25Retriever) -> dict:
    """Return best-effort metadata for the BM25 index used in evaluation."""

    return {
        "index_dir": retriever.index_dir.as_posix(),
        "index_dir_resolved": retriever.index_dir.resolve().as_posix(),
        "metadata": retriever.metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())
