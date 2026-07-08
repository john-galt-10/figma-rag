"""Evaluation utilities for Figma RAG experiments."""

from .artifacts import (
    build_generation_output_paths,
    ensure_parquet_available,
    write_generation_details_jsonl,
    write_generation_details_parquet,
)
from .generation import (
    GenerationEvaluationRow,
    GenerationEvaluationRun,
    RetrievedChunkRecord,
    build_generation_summary_payload,
    generation_rows_to_dicts,
    run_generation_evaluation,
)
from .judge import (
    MetricScore,
    OpenAICompatibleStructuredJudge,
    RAGEvaluation,
    build_structured_judge,
)
from .retrieval import (
    RetrievalEvaluation,
    RetrievalQueryExample,
    evaluate_retrieval_results,
    load_retrieval_test_set,
    normalize_top_k_values,
    set_reproducibility_seed,
    sha256_file,
    stabilize_retrieval_ties,
    write_detailed_results_parquet,
    write_metrics_json,
)

__all__ = [
    "GenerationEvaluationRow",
    "GenerationEvaluationRun",
    "MetricScore",
    "OpenAICompatibleStructuredJudge",
    "RAGEvaluation",
    "RetrievedChunkRecord",
    "RetrievalEvaluation",
    "RetrievalQueryExample",
    "build_generation_output_paths",
    "build_generation_summary_payload",
    "build_structured_judge",
    "ensure_parquet_available",
    "evaluate_retrieval_results",
    "generation_rows_to_dicts",
    "load_retrieval_test_set",
    "normalize_top_k_values",
    "run_generation_evaluation",
    "set_reproducibility_seed",
    "sha256_file",
    "stabilize_retrieval_ties",
    "write_detailed_results_parquet",
    "write_generation_details_jsonl",
    "write_generation_details_parquet",
    "write_metrics_json",
]
