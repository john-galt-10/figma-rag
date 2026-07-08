"""Structured LLM-as-judge models and OpenAI-compatible client."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from figma_rag.generation import GenerationModelConfig

# JUDGE_PROMPT = """
# You are evaluating a RAG system.

# Use only the retrieved context and the question.
# Do not use outside knowledge.
# Do not reward an answer for being factually correct if it is not supported by the retrieved context.

# Scoring scale:
# 1 = very poor
# 2 = poor
# 3 = acceptable but with important issues
# 4 = good with minor issues
# 5 = excellent

# Definitions:

# context_relevance:
# Whether the retrieved context aligns with the user's query. Consider the relevance of the information to the query's intent and the appropriateness of the context in providing a coherent and useful response

# groundedness:
# How faithful the generated answer is to the retrieved context. Consider the factual accuracy and reliability of the answer, ensuring it is grounded in the retrieved information. Penalize passages of the answer that are not found in the retrieved context.


# answer_relevance:
# How well the generated answer addresses the user's original query. Consider the helpfulness and on-point nature of the answer, aligning with the user's intent and providing valuable insights.

# """ 

JUDGE_PROMPT = """
You are evaluating a RAG system.

Use only the retrieved context and the question.
Do not use outside knowledge.
Do not reward an answer for being factually correct if it is not supported by the retrieved context.

Scoring scale:
1 = very poor
2 = poor
3 = acceptable but with important issues
4 = good with minor issues
5 = excellent

Definitions:

context_relevance:
Evaluate ONLY the retrieved context, not the generated answer.
Question: Does the retrieved context contain information that is useful and sufficient for answering the user's question?
Use the expected answer points to check whether the context contains the key information needed for a complete answer.
Score lower if the context is off-topic, incomplete, too vague, missing key expected points, or contains mostly irrelevant material.
Do not consider whether the generated answer used the context well.

groundedness:
Evaluate ONLY the generated answer against the retrieved context.
Question: Are all factual claims in the generated answer directly supported by the retrieved context?
Score lower for unsupported claims, contradictions, exaggerations, or details added beyond the retrieved context.
Do not penalize the answer for omitting expected answer points.
Do not use expected answer points as evidence that the answer is ungrounded.
A missing expected point is an answer relevance issue, not a groundedness issue.
Only penalize groundedness when the answer says something that is unsupported by, contradicted by, or stronger than the retrieved context.

answer_relevance:
Evaluate how well the generated answer addresses the user question.
Question: Does the answer respond directly, completely, and helpfully to the user's question?
Use the expected answer points to judge completeness.
Score lower if the answer is incomplete, vague, evasive, overly broad, or misses important expected points that should have been included.
Do not give credit for claims that are not supported by the retrieved context, even if they match the expected answer points.

""" 

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

    context_relevance: MetricScore
    groundedness: MetricScore
    answer_relevance: MetricScore

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"


class OpenAICompatibleStructuredJudge:
    """Run structured judge requests through an OpenAI-compatible API."""

    def __init__(self, config: GenerationModelConfig) -> None:
        """Store model settings for repeated judge calls."""

        self.config = config
        self._client = None

    def judge(self, messages: list[dict[str, str]]) -> RAGEvaluation:
        """Return a validated structured RAG evaluation for one judge prompt."""

        request_options = build_structured_request_options(
            messages=messages,
            config=self.config,
            response_model=RAGEvaluation,
        )
        response = self._get_client().chat.completions.create(**request_options)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge model returned an empty response")
        return validate_pydantic_json(RAGEvaluation, content)

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


def build_structured_judge(
    config: GenerationModelConfig,
) -> OpenAICompatibleStructuredJudge:
    """Return the structured judge implementation for the configured provider."""

    if config.provider.strip().lower() != "openai":
        raise ValueError(
            f"Unsupported judge provider '{config.provider}'. "
            "Only 'openai' is implemented."
        )
    return OpenAICompatibleStructuredJudge(config)


def build_structured_request_options(
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
            "schema": pydantic_json_schema(response_model),
            "strict": True,
        },
    }
    return request_options


def pydantic_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema for Pydantic v1 or v2 models."""

    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def validate_pydantic_json(model: type[BaseModel], raw_json: str):
    """Validate a JSON string against a Pydantic v1 or v2 model."""

    if hasattr(model, "model_validate_json"):
        return model.model_validate_json(raw_json)
    return model.parse_raw(raw_json)


def pydantic_model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a Pydantic v1 or v2 model to a JSON-friendly dictionary."""

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def copy_pydantic_model(model: BaseModel, update: dict[str, Any]) -> Any:
    """Copy a Pydantic v1 or v2 model with selected field updates."""

    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)
    return model.copy(update=update)

def build_chat_prompt(payload):
    retrieved_context = "\n".join(
        f"Chunk {chunk_rank + 1}:\n{chunk['text']}"
        for chunk_rank, chunk in enumerate(payload["retrieved_chunks"])
    )

    expected_points = "\n".join(
        str(answer_point)
        for answer_point in payload["expected_answer_points"]
    )

    return f"""
Query:
{payload["query"]}

Retrieved Context:
{retrieved_context}

Generated Answer:
{payload["answer"]}

Expected Answer Points:
{expected_points}
"""