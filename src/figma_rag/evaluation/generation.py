"""Dataset-level evaluation for generated RAG answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from figma_rag.generation import (
    AnswerGenerationConfig,
    AnswerGenerationPipeline,
    GenerationModelConfig,
    build_answer_generation_pipeline,
)
from figma_rag.retrieval import RetrievalResult

from .judge import (
    JUDGE_PROMPT,
    OpenAICompatibleStructuredJudge,
    RAGEvaluation,
    build_structured_judge,
    copy_pydantic_model,
    pydantic_model_to_dict,
    build_chat_prompt
)
from .rag_dataset import load_jsonl
from .retrieval import RetrievalQueryExample, sha256_file
from .time import evaluation_now


@dataclass(frozen=True)
class GenerationEvaluationRun:
    """In-memory result of one generation evaluation run."""

    rows: list["GenerationEvaluationRow"]
    pipeline_description: dict[str, Any]
    candidate_k: int | None


class RetrievedChunkRecord(BaseModel):
    """Serializable trace for one chunk retrieved before answer generation."""

    rank: int
    chunk_id: str
    title: str
    section: str
    source_url: str
    distance: float
    rerank_score: float | None = None
    text: str
    text_char_count: int
    metadata_json: str


class GenerationEvaluationRow(BaseModel):
    """One per-query generation evaluation record."""

    query_id: str
    query: str
    query_type: str | None = None
    answer_type: str | None = None
    expected_answer_points: list[str]
    relevant_chunk_ids: list[str]
    generated_answer: str | None = None
    retrieved_chunks: list[RetrievedChunkRecord]
    judge_evaluation: RAGEvaluation | None = None
    retrieval_error: str | None = None
    generation_error: str | None = None
    judge_error: str | None = None


def run_generation_evaluation(
    examples: list[RetrievalQueryExample],
    config: AnswerGenerationConfig,
    test_set_path: Path,
    settings_callback: Callable[[AnswerGenerationPipeline], None] | None = None,
    progress_callback: Callable[[int, int, RetrievalQueryExample], None] | None = None,
) -> GenerationEvaluationRun:
    """Evaluate generated answers for a loaded retrieval test set."""

    query_metadata = load_query_metadata(test_set_path)
    pipeline = build_answer_generation_pipeline(config)
    judge = build_structured_judge(config.judge)
    if settings_callback:
        settings_callback(pipeline)

    rows: list[GenerationEvaluationRow] = []
    for index, example in enumerate(examples, start=1):
        if progress_callback:
            progress_callback(index, len(examples), example)
        rows.append(
            evaluate_example(
                example=example,
                query_metadata=query_metadata.get(example.query_id, {}),
                pipeline=pipeline,
                judge=judge,
                max_chunk_chars=config.prompt.max_chunk_chars,
            )
        )

    return GenerationEvaluationRun(
        rows=rows,
        pipeline_description=pipeline.retrieval_pipeline.to_description(),
        candidate_k=pipeline.retrieval_options.candidate_k,
    )


def evaluate_example(
    example: RetrievalQueryExample,
    query_metadata: dict[str, Any],
    pipeline: AnswerGenerationPipeline,
    judge: OpenAICompatibleStructuredJudge,
    max_chunk_chars: int,
) -> GenerationEvaluationRow:
    """Run retrieval, answer generation, and judge evaluation for one query."""

    base_row = GenerationEvaluationRow(
        query_id=example.query_id,
        query=example.query,
        query_type=optional_string(query_metadata.get("query_type")),
        answer_type=optional_string(query_metadata.get("answer_type")),
        expected_answer_points=string_list(query_metadata.get("expected_answer_points")),
        relevant_chunk_ids=list(example.relevant_chunk_ids),
        retrieved_chunks=[],
    )

    try:
        results = pipeline.retrieve(example.query)
    except Exception as exc:
        return copy_pydantic_model(base_row, {"retrieval_error": str(exc)})

    retrieved_chunks = [
        build_retrieved_chunk_record(result, max_chunk_chars)
        for result in results
    ]

    try:
        generated_answer = pipeline.generate_with_context(example.query, results)
    except Exception as exc:
        return copy_pydantic_model(
            base_row,
            {
                "retrieved_chunks": retrieved_chunks,
                "generation_error": str(exc),
            },
        )

    try:
        judge_messages = build_judge_messages(
            query=example.query,
            answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            query_metadata=query_metadata,
        )
        judge_evaluation = judge.judge(judge_messages)
    except Exception as exc:
        return copy_pydantic_model(
            base_row,
            {
                "generated_answer": generated_answer,
                "retrieved_chunks": retrieved_chunks,
                "judge_error": str(exc),
            },
        )

    return copy_pydantic_model(
        base_row,
        {
            "generated_answer": generated_answer,
            "retrieved_chunks": retrieved_chunks,
            "judge_evaluation": judge_evaluation,
        },
    )


def build_retrieved_chunk_record(
    result: RetrievalResult,
    max_chunk_chars: int,
) -> RetrievedChunkRecord:
    """Build a compact serializable trace for one retrieved chunk."""

    normalized_text = " ".join(result.text.split())
    text_char_count = len(normalized_text)
    if len(normalized_text) > max_chunk_chars:
        normalized_text = normalized_text[: max_chunk_chars - 3].rstrip() + "..."

    rerank_score = result.metadata.get("rerank_score")
    return RetrievedChunkRecord(
        rank=result.rank,
        chunk_id=result.chunk_id,
        title=result.title,
        section=result.section,
        source_url=result.source_url,
        distance=float(result.distance),
        rerank_score=None if rerank_score is None else float(rerank_score),
        text=normalized_text,
        text_char_count=text_char_count,
        metadata_json=json.dumps(result.metadata, ensure_ascii=False, default=str),
    )


def build_judge_messages(
    query: str,
    answer: str,
    retrieved_chunks: list[RetrievedChunkRecord],
    query_metadata: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the judge prompt messages and include all evaluation inputs."""

    payload = {
        "query": query,
        "answer": answer,
        "retrieved_chunks": [
            pydantic_model_to_dict(chunk) for chunk in retrieved_chunks
        ],
        "expected_answer_points": query_metadata.get("expected_answer_points", []),
        "query_type": query_metadata.get("query_type"),
        "answer_type": query_metadata.get("answer_type"),
    }

    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": build_chat_prompt(payload)},
    ]


def build_generation_summary_payload(
    rows: list[GenerationEvaluationRow],
    test_set_path: Path,
    config_path: Path,
    output_dir: Path,
    details_path: Path,
    jsonl_path: Path | None,
    config: AnswerGenerationConfig,
    pipeline_description: dict[str, Any],
    candidate_k: int | None,
) -> dict[str, Any]:
    """Build a JSON summary with aggregate scores and run metadata."""

    successful_rows = [row for row in rows if row.judge_evaluation is not None]
    failed_rows = [row for row in rows if row.judge_evaluation is None]
    return {
        "metadata": {
            "test_set_path": test_set_path.as_posix(),
            "test_set_path_resolved": test_set_path.resolve().as_posix(),
            "test_set_sha256": sha256_file(test_set_path),
            "config_path": config_path.as_posix(),
            "config_path_resolved": config_path.resolve().as_posix(),
            "output_dir": output_dir.as_posix(),
            "output_dir_resolved": output_dir.resolve().as_posix(),
            "details_path": details_path.as_posix(),
            "jsonl_path": None if jsonl_path is None else jsonl_path.as_posix(),
            "query_count": len(rows),
            "successful_judgment_count": len(successful_rows),
            "failed_judgment_count": len(failed_rows),
            "created_at": evaluation_now().isoformat(timespec="seconds"),
            "generation": model_config_metadata(config.generation),
            "judge": model_config_metadata(config.judge),
            "retrieval": {
                "pipeline": pipeline_description,
                "top_k": config.retrieval.top_k,
                "candidate_k": candidate_k,
                "metadata_filters": config.retrieval.metadata_filters,
                "metadata_filters_enabled": config.retrieval.metadata_filters_enabled,
                "topic_filter_enabled": config.retrieval.topic_filter_enabled,
                "topic_filter_where": config.retrieval.topic_filter_where,
            },
        },
        "average_scores": average_judge_scores(successful_rows),
        "error_counts": error_counts(rows),
    }


def model_config_metadata(config: GenerationModelConfig) -> dict[str, Any]:
    """Return JSON-friendly metadata for one configured model request."""

    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "extra_options": config.extra_options,
    }


def generation_rows_to_dicts(
    rows: list[GenerationEvaluationRow],
) -> list[dict[str, Any]]:
    """Convert generation evaluation rows to JSON-friendly dictionaries."""

    return [pydantic_model_to_dict(row) for row in rows]


def load_query_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load optional query metadata from the raw JSONL test set rows."""

    metadata_by_query_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        query_id = row.get("query_id")
        if isinstance(query_id, str) and query_id:
            metadata_by_query_id[query_id] = row
    return metadata_by_query_id


def average_judge_scores(
    rows: list[GenerationEvaluationRow],
) -> dict[str, float | None]:
    """Return average scores for all successfully judged rows."""

    metric_names = ["context_relevance", "groundedness", "answer_relevance"]
    if not rows:
        return {metric_name: None for metric_name in metric_names}

    averages: dict[str, float | None] = {}
    for metric_name in metric_names:
        scores = [
            getattr(row.judge_evaluation, metric_name).score
            for row in rows
            if row.judge_evaluation is not None
        ]
        averages[metric_name] = sum(scores) / len(scores) if scores else None
    return averages


def error_counts(rows: list[GenerationEvaluationRow]) -> dict[str, int]:
    """Count retrieval, generation, and judge errors."""

    return {
        "retrieval_error": sum(1 for row in rows if row.retrieval_error),
        "generation_error": sum(1 for row in rows if row.generation_error),
        "judge_error": sum(1 for row in rows if row.judge_error),
    }


def string_list(value: object) -> list[str]:
    """Return a clean string list from a loosely typed metadata value."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def optional_string(value: object) -> str | None:
    """Return a string value when available."""

    if isinstance(value, str):
        return value
    return None
