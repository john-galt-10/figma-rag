"""Evaluate generated RAG answers with an LLM-as-judge workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colorama import Fore, Style, init
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.evaluation import (  # noqa: E402
    load_retrieval_test_set,
    sha256_file,
    write_metrics_json,
)
from figma_rag.evaluation.rag_dataset import load_jsonl  # noqa: E402
from figma_rag.generation import (  # noqa: E402
    GenerationModelConfig,
    build_grounded_messages,
    build_model_provider,
    load_answer_generation_config,
)
from figma_rag.retrieval import (  # noqa: E402
    RetrievalRequest,
    RetrievalResult,
    parse_metadata_filter_set,
)
from generate_answer import build_pipeline  # noqa: E402

DEFAULT_TEST_SET_PATH = (
    REPO_ROOT
    / "data"
    / "eval"
    / "retrieval_test"
    / "golden_set_manual_and_codex_relevant_chunks_hierarchical_bge-small-en-v1.5_20260630-1601.jsonl"
)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("generate_answer_config.yaml")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "generation_test" / "test_results"
JUDGE_PROMPT = ""


class MetricScore(BaseModel):
    """One judge metric with a short explanation and a 1-to-5 score."""

    explanation: str = Field(
        ...,
        description="Explanation for the assigned metric score.",
    )
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Integer score from 1 to 5.",
    )

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"


class RAGEvaluation(BaseModel):
    """Structured LLM-as-judge output for the RAG triad."""

    context_relevance: MetricScore = Field(
        ...,
        description="Evaluation of whether retrieved context is relevant to the query.",
    )
    groundedness: MetricScore = Field(
        ...,
        description="Evaluation of whether the answer is grounded in retrieved context.",
    )
    answer_relevance: MetricScore = Field(
        ...,
        description="Evaluation of whether the answer addresses the query.",
    )

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"


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


class OpenAICompatibleStructuredJudge:
    """Run structured judge requests through an OpenAI-compatible API."""

    def __init__(self, config: GenerationModelConfig) -> None:
        """Store model settings for repeated judge calls."""

        self.config = config
        self._client = None

    def judge(self, messages: list[dict[str, str]]) -> RAGEvaluation:
        """Return a validated structured RAG evaluation for one judge prompt."""

        request_options = _build_structured_request_options(
            messages=messages,
            config=self.config,
            response_model=RAGEvaluation,
        )
        response = self._get_client().chat.completions.create(**request_options)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge model returned an empty response")
        return _validate_pydantic_json(RAGEvaluation, content)

    def _get_client(self):
        """Return a cached OpenAI client configured from the generation config."""

        if self._client is not None:
            return self._client

        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "The 'python-dotenv' package is required to load credentials from .env."
            ) from exc

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for the openai generation provider."
            ) from exc

        load_dotenv()
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Add {self.config.api_key_env}=... to .env or "
                "export it in the current environment."
            )

        self._client = OpenAI(base_url=self.config.base_url, api_key=api_key)
        return self._client


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

    _ensure_parquet_available()

    config = load_answer_generation_config(args.config_path, REPO_ROOT)
    retrieval_config = config.retrieval
    metadata_filters = parse_metadata_filter_set(retrieval_config.metadata_filters)
    candidate_k = (
        None
        if not retrieval_config.reranking_enabled
        else retrieval_config.candidate_k or retrieval_config.top_k * 5
    )
    topic_filter = (
        retrieval_config.topic_filter_where
        if retrieval_config.topic_filter_enabled
        else None
    )

    examples = load_retrieval_test_set(args.test_set_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    query_metadata = _load_query_metadata(args.test_set_path)

    pipeline = build_pipeline(config)
    answer_provider = build_model_provider(config.generation.provider)
    judge = _build_structured_judge(config.judge)

    print_evaluation_settings(
        test_set_path=args.test_set_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        query_count=len(examples),
        pipeline_description=pipeline.to_description(),
        generation_config=config.generation,
        judge_config=config.judge,
        candidate_k=candidate_k,
    )

    rows: list[GenerationEvaluationRow] = []
    for index, example in enumerate(examples, start=1):
        print(
            Fore.LIGHTBLACK_EX
            + f"[{index}/{len(examples)}] Evaluating {example.query_id}: {example.query}"
            + Style.RESET_ALL
        )
        row = evaluate_example(
            example=example,
            query_metadata=query_metadata.get(example.query_id, {}),
            pipeline=pipeline,
            answer_provider=answer_provider,
            judge=judge,
            retrieval_request=RetrievalRequest(
                query=example.query,
                top_k=retrieval_config.top_k,
                candidate_k=candidate_k,
                metadata_filters=metadata_filters,
                metadata_filters_enabled=retrieval_config.metadata_filters_enabled,
                raw_chroma_where=topic_filter,
            ),
            max_chunk_chars=config.prompt.max_chunk_chars,
            prompt_config=config.prompt,
            generation_config=config.generation,
        )
        rows.append(row)

    details_path, summary_path, jsonl_path = build_output_paths(
        test_set_path=args.test_set_path,
        output_dir=args.output_dir,
    )
    serializable_rows = [_model_to_dict(row) for row in rows]
    write_details_parquet(details_path, serializable_rows)
    summary_payload = build_summary_payload(
        rows=rows,
        test_set_path=args.test_set_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        details_path=details_path,
        jsonl_path=jsonl_path if args.save_jsonl else None,
        config=config,
        pipeline_description=pipeline.to_description(),
        candidate_k=candidate_k,
    )
    write_metrics_json(summary_path, summary_payload)

    if args.save_jsonl:
        write_details_jsonl(jsonl_path, serializable_rows)
        print(f"Wrote per-query JSONL details to {jsonl_path}")

    print(f"Wrote per-query Parquet details to {details_path}")
    print(f"Wrote aggregate judge metrics to {summary_path}")
    print_summary(summary_payload)
    return 0


def evaluate_example(
    example: Any,
    query_metadata: dict[str, Any],
    pipeline: Any,
    answer_provider: Any,
    judge: OpenAICompatibleStructuredJudge,
    retrieval_request: RetrievalRequest,
    max_chunk_chars: int,
    prompt_config: Any,
    generation_config: GenerationModelConfig,
) -> GenerationEvaluationRow:
    """Run retrieval, answer generation, and judge evaluation for one query."""

    base_row = GenerationEvaluationRow(
        query_id=example.query_id,
        query=example.query,
        query_type=_optional_string(query_metadata.get("query_type")),
        answer_type=_optional_string(query_metadata.get("answer_type")),
        expected_answer_points=_string_list(query_metadata.get("expected_answer_points")),
        relevant_chunk_ids=list(example.relevant_chunk_ids),
        retrieved_chunks=[],
    )

    try:
        results = pipeline.retrieve(retrieval_request)
    except Exception as exc:
        return _copy_model(base_row, {"retrieval_error": str(exc)})

    retrieved_chunks = [
        build_retrieved_chunk_record(result, max_chunk_chars)
        for result in results
    ]

    try:
        answer_messages = build_grounded_messages(
            example.query,
            results,
            prompt_config,
        )
        generated_answer = answer_provider.generate(answer_messages, generation_config)
    except Exception as exc:
        return _copy_model(
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
        return _copy_model(
            base_row,
            {
                "generated_answer": generated_answer,
                "retrieved_chunks": retrieved_chunks,
                "judge_error": str(exc),
            },
        )

    return _copy_model(
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
        "retrieved_chunks": [_model_to_dict(chunk) for chunk in retrieved_chunks],
        "expected_answer_points": query_metadata.get("expected_answer_points", []),
        "query_type": query_metadata.get("query_type"),
        "answer_type": query_metadata.get("answer_type"),
    }
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_output_paths(test_set_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Return details, summary, and optional JSONL output paths."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    label = _slugify(test_set_path.stem)
    details_path = output_dir / f"generation_details_{label}_{timestamp}.parquet"
    summary_path = output_dir / f"generation_metrics_{label}_{timestamp}.json"
    jsonl_path = output_dir / f"generation_details_{label}_{timestamp}.jsonl"
    return details_path, summary_path, jsonl_path


def build_summary_payload(
    rows: list[GenerationEvaluationRow],
    test_set_path: Path,
    config_path: Path,
    output_dir: Path,
    details_path: Path,
    jsonl_path: Path | None,
    config: Any,
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
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generation": _model_config_metadata(config.generation),
            "judge": _model_config_metadata(config.judge),
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
        "average_scores": _average_judge_scores(successful_rows),
        "error_counts": _error_counts(rows),
    }


def print_evaluation_settings(
    test_set_path: Path,
    config_path: Path,
    output_dir: Path,
    query_count: int,
    pipeline_description: dict[str, Any],
    generation_config: GenerationModelConfig,
    judge_config: GenerationModelConfig,
    candidate_k: int | None,
) -> None:
    """Print the main generation evaluation settings in cyan."""

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
        f"Pipeline: {pipeline_description}",
        f"Candidate-k per retriever: {candidate_k}",
        "Judge prompt: empty string placeholder",
    ]
    print(Fore.CYAN + "\n".join(lines) + Style.RESET_ALL)
    print()


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


def write_details_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-query generation evaluation rows as a Parquet file."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_details_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-query generation evaluation rows as newline-delimited JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def _build_structured_judge(
    config: GenerationModelConfig,
) -> OpenAICompatibleStructuredJudge:
    """Return the structured judge implementation for the configured provider."""

    if config.provider.strip().lower() != "openai":
        raise ValueError(
            f"Unsupported judge provider '{config.provider}'. "
            "Only 'openai' is implemented."
        )
    return OpenAICompatibleStructuredJudge(config)


def _model_config_metadata(config: GenerationModelConfig) -> dict[str, Any]:
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


def _build_structured_request_options(
    messages: list[dict[str, str]],
    config: GenerationModelConfig,
    response_model: type[BaseModel],
) -> dict[str, Any]:
    """Build OpenAI-compatible chat options with JSON-schema response format."""

    request_options: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
    }
    optional_options = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    for key, value in optional_options.items():
        if value is not None:
            request_options[key] = value
    request_options.update(config.extra_options)
    request_options["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": _pydantic_json_schema(response_model),
            "strict": True,
        },
    }
    return request_options


def _pydantic_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema for Pydantic v1 or v2 models."""

    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def _validate_pydantic_json(model: type[BaseModel], raw_json: str):
    """Validate a JSON string against a Pydantic v1 or v2 model."""

    if hasattr(model, "model_validate_json"):
        return model.model_validate_json(raw_json)
    return model.parse_raw(raw_json)


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a Pydantic v1 or v2 model to a JSON-friendly dictionary."""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _copy_model(model: BaseModel, update: dict[str, Any]) -> Any:
    """Copy a Pydantic v1 or v2 model with selected field updates."""

    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)
    return model.copy(update=update)


def _load_query_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load optional query metadata from the raw JSONL test set rows."""

    metadata_by_query_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        query_id = row.get("query_id")
        if isinstance(query_id, str) and query_id:
            metadata_by_query_id[query_id] = row
    return metadata_by_query_id


def _average_judge_scores(rows: list[GenerationEvaluationRow]) -> dict[str, float | None]:
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


def _error_counts(rows: list[GenerationEvaluationRow]) -> dict[str, int]:
    """Count retrieval, generation, and judge errors."""

    return {
        "retrieval_error": sum(1 for row in rows if row.retrieval_error),
        "generation_error": sum(1 for row in rows if row.generation_error),
        "judge_error": sum(1 for row in rows if row.judge_error),
    }


def _ensure_parquet_available() -> None:
    """Fail early when the required Parquet writer dependency is missing."""

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The 'pyarrow' package is required to write generation evaluation Parquet."
        ) from exc


def _string_list(value: object) -> list[str]:
    """Return a clean string list from a loosely typed metadata value."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _optional_string(value: object) -> str | None:
    """Return a string value when available."""

    if isinstance(value, str):
        return value
    return None


def _slugify(value: str) -> str:
    """Return a filesystem-friendly label."""

    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug or "test-set"


if __name__ == "__main__":
    raise SystemExit(main())
