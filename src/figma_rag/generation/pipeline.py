"""YAML-configured answer generation pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from figma_rag.retrieval import (
    RetrievalPipeline,
    RetrievalOptions as GenerationRetrievalOptions,
    RetrievalRequest,
    RetrievalResult,
    build_configured_retrieval_pipeline as build_retrieval_pipeline_from_config,
    resolve_retrieval_options,
)

from .config import AnswerGenerationConfig
from .prompting import build_grounded_messages
from .providers import ModelProvider, build_model_provider


@dataclass(frozen=True)
class AnswerGenerationResult:
    """Output of one grounded answer generation request."""

    query: str
    retrieved_chunks: list[RetrievalResult]
    answer: str


class AnswerGenerationPipeline:
    """Run retrieval, prompt construction, and model generation for one query."""

    def __init__(
        self,
        config: AnswerGenerationConfig,
        retrieval_pipeline: RetrievalPipeline,
        provider: ModelProvider,
        retrieval_options: GenerationRetrievalOptions,
    ) -> None:
        """Store the configured retrieval pipeline and model provider."""

        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.provider = provider
        self.retrieval_options = retrieval_options

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve grounded context chunks for one query."""

        request = RetrievalRequest(
            query=query,
            top_k=self.retrieval_options.top_k,
            candidate_k=self.retrieval_options.candidate_k,
            metadata_filters=self.retrieval_options.metadata_filters,
            metadata_filters_enabled=self.retrieval_options.metadata_filters_enabled,
            raw_chroma_where=self.retrieval_options.topic_filter,
        )
        return self.retrieval_pipeline.retrieve(request)

    def generate(self, query: str) -> AnswerGenerationResult:
        """Retrieve context and generate one grounded answer."""

        retrieved_chunks = self.retrieve(query)
        answer = self.generate_with_context(query, retrieved_chunks)
        return AnswerGenerationResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
        )

    def generate_with_context(
        self,
        query: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> str:
        """Generate an answer from already retrieved context chunks."""

        messages = build_grounded_messages(query, retrieved_chunks, self.config.prompt)

        return self.provider.generate(messages, self.config.generation)

    def to_description(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the configured pipeline."""

        return {
            "retrieval_pipeline": self.retrieval_pipeline.to_description(),
            "retrieval_options": {
                "top_k": self.retrieval_options.top_k,
                "candidate_k": self.retrieval_options.candidate_k,
                "metadata_filters_enabled": (
                    self.retrieval_options.metadata_filters_enabled
                ),
                "metadata_filters": self.config.retrieval.metadata_filters,
                "topic_filter": self.retrieval_options.topic_filter,
            },
            "generation": {
                "provider": self.config.generation.provider,
                "model": self.config.generation.model,
            },
        }


def build_answer_generation_pipeline(
    config: AnswerGenerationConfig,
) -> AnswerGenerationPipeline:
    """Build the complete answer generation pipeline from YAML config."""

    return AnswerGenerationPipeline(
        config=config,
        retrieval_pipeline=build_retrieval_pipeline_from_config(config.retrieval),
        provider=build_model_provider(config.generation.provider),
        retrieval_options=resolve_generation_retrieval_options(config),
    )


def build_configured_retrieval_pipeline(
    config: AnswerGenerationConfig,
) -> RetrievalPipeline:
    """Build the retrieval pipeline configured for answer generation."""

    return build_retrieval_pipeline_from_config(config.retrieval)


def resolve_generation_retrieval_options(
    config: AnswerGenerationConfig,
) -> GenerationRetrievalOptions:
    """Resolve YAML retrieval settings into request-time options."""

    return resolve_retrieval_options(config.retrieval)
