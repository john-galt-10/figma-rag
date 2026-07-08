"""Answer generation helpers for the Figma RAG pipeline."""

from .config import (
    AnswerGenerationConfig,
    GenerationModelConfig,
    PromptConfig,
    RetrievalConfig,
    load_answer_generation_config,
)
from .pipeline import (
    AnswerGenerationPipeline,
    AnswerGenerationResult,
    GenerationRetrievalOptions,
    build_answer_generation_pipeline,
    build_configured_retrieval_pipeline,
    resolve_generation_retrieval_options,
)
from .prompting import build_grounded_messages
from .providers import ModelProvider, OpenAICompatibleProvider, build_model_provider

__all__ = [
    "AnswerGenerationPipeline",
    "AnswerGenerationConfig",
    "AnswerGenerationResult",
    "GenerationModelConfig",
    "GenerationRetrievalOptions",
    "ModelProvider",
    "OpenAICompatibleProvider",
    "PromptConfig",
    "RetrievalConfig",
    "build_answer_generation_pipeline",
    "build_configured_retrieval_pipeline",
    "build_grounded_messages",
    "build_model_provider",
    "load_answer_generation_config",
    "resolve_generation_retrieval_options",
]
