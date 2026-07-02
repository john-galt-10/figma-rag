"""Entrypoint for evaluating retrieval quality on a mapped test set."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    write_detailed_results_csv,
    write_metrics_json,
)
from figma_rag.retrieval import (  # noqa: E402
    BM25Retriever,
    ChromaRetriever,
    DEFAULT_RETRIEVAL_COMPONENTS,
    RetrievalRequest,
    build_retrieval_pipeline,
    parse_metadata_filter_set,
)

DEFAULT_TEST_SET_PATH = (
    REPO_ROOT
    / "data"
    / "eval"
    / "retrieval_test"
    / "golden_set_manual_2_complete_20260701_1753_relevant_chunks_hierarchical_bge-small-en-v1.5_20260629-1709.jsonl"
)
DEFAULT_PERSIST_DIR = REPO_ROOT / "data" / "processed" / "figma_docs" / "chroma"
DEFAULT_BM25_INDEX_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "bm25"
    / "hierarchical_bge-small-en-v1.5_t320_o40_bm25_stemmed_english_20260701t1733"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "retrieval_test" / "test_results"
DEFAULT_METADATA_FILTERS = [
    "token_count>30",
    # "product=figma-design-or-general",
]
DEFAULT_TOPIC_FILTER = {"topic": {"$in": ["Figma Design", "Administration", "Help", "Community", "Work across Figma", "Get Started"]}}
TEST_SET_FILENAME_PATTERN = re.compile(
    r"^.+?_relevant_chunks_(?P<label>.+?)_(?P<timestamp>\d{8}-\d{4})$"
)


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
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory containing the persistent Chroma database.",
    )
    parser.add_argument(
        "--collection-name",
        required=False,
        default="hierarchical-bge-w-topic",
        help="Name of the Chroma collection to query.",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence Transformers model used to embed each query.",
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
        nargs="+",
        default=[1, 3, 5, 9, 15, 20],
        help="One or more retrieval cutoffs to evaluate, such as --top-k 1 3 5 10.",
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
        help="Write one CSV row per query per k for debugging retrieval behavior.",
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
        output_dir / f"retrieval_details_{artifact_label}_{current_timestamp}.csv",
    )


def _describe_topic_filter(topic_filter: dict, enabled: bool = True) -> dict:
    """Return a JSON-serializable description of the default topic filter."""

    return {
        "enabled": enabled,
        "field": "topic",
        "operator": "in",
        "values": topic_filter["topic"]["$in"],
    }


def main() -> int:
    """Run retrieval evaluation and write comparison-friendly artifacts."""

    parser = build_parser()
    args = parser.parse_args()
    seed_settings = set_reproducibility_seed(args.seed)
    top_k_values = normalize_top_k_values(args.top_k)

    try:
        metadata_filters = parse_metadata_filter_set(args.metadata_filter)
    except ValueError as exc:
        parser.error(str(exc))

    metadata_filters_enabled = not args.disable_metadata_filters
    topic_filter = None if args.disable_topic_filter else DEFAULT_TOPIC_FILTER
    retrieval_components = args.retrieval_component or list(DEFAULT_RETRIEVAL_COMPONENTS)

    test_set_examples = load_retrieval_test_set(args.test_set_path)

    chroma_retriever = None
    if "chroma" in retrieval_components:
        chroma_retriever = ChromaRetriever(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
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

    max_top_k = max(top_k_values)
    retrieved_chunk_ids_by_query_id = {}

    print("+-----------------------------------------------------------------------+")
    print("Evaluating Retriever performance w/ the following settings:")
    print(f"Pipeline: {pipeline.to_description()}")
    print(f"Metadata filters: {[fil.to_description() for fil in metadata_filters.filters]}")
    print(
        "Topic filter: "
        f"{_describe_topic_filter(DEFAULT_TOPIC_FILTER, enabled=not args.disable_topic_filter)}"
    )
    print("+-----------------------------------------------------------------------+")
    
    for example in test_set_examples:
        request = RetrievalRequest(
            query=example.query,
            top_k=max_top_k,
            metadata_filters=metadata_filters,
            metadata_filters_enabled=metadata_filters_enabled,
            raw_chroma_where=topic_filter,
        )
        try:
            results = stabilize_retrieval_ties(pipeline.retrieve(request))
        except NotImplementedError as exc:
            parser.error(str(exc))
        retrieved_chunk_ids_by_query_id[example.query_id] = [
            result.chunk_id for result in results
        ]

    representative_request = RetrievalRequest(
        query="metadata",
        top_k=max_top_k,
        metadata_filters=metadata_filters,
        metadata_filters_enabled=metadata_filters_enabled,
        raw_chroma_where=topic_filter,
    )

    evaluation = evaluate_retrieval_results(
        examples=test_set_examples,
        retrieved_chunk_ids_by_query_id=retrieved_chunk_ids_by_query_id,
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
            "persist_dir": args.persist_dir.as_posix(),
            "persist_dir_resolved": args.persist_dir.resolve().as_posix(),
            "bm25_index_dir": args.bm25_index_dir.as_posix(),
            "bm25_index_dir_resolved": args.bm25_index_dir.resolve().as_posix(),
            "output_dir": args.output_dir.as_posix(),
            "output_dir_resolved": args.output_dir.resolve().as_posix(),
            "collection_name": args.collection_name,
            "collection": _collection_metadata(chroma_retriever)
            if chroma_retriever
            else None,
            "bm25_index": _bm25_index_metadata(bm25_retriever)
            if bm25_retriever
            else None,
            "model": args.model,
            "retrieval_pipeline": pipeline.to_description(),
            "metadata_filters": metadata_filters.to_description(
                enabled=metadata_filters_enabled
            ),
            "topic_filter": _describe_topic_filter(
                DEFAULT_TOPIC_FILTER,
                enabled=not args.disable_topic_filter,
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
        write_detailed_results_csv(details_path, evaluation.details)
        print(f"Wrote detailed per-query results to {details_path}")

    print(json.dumps(evaluation.metrics_by_k, indent=2))
    return 0


def _artifact_label_and_timestamp(test_set_path: Path) -> tuple[str, str]:
    """Extract a readable label and timestamp from a mapped test-set filename."""

    stem = test_set_path.stem
    match = TEST_SET_FILENAME_PATTERN.match(stem)
    if match:
        return _slugify(match.group("label")), match.group("timestamp")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return _slugify(stem), timestamp


def _slugify(value: str) -> str:
    """Return a filesystem-friendly label."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug or "test-set"


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
