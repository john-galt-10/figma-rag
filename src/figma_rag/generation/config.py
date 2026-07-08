"""Configuration loading for the answer generation example."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from figma_rag.retrieval import RetrievalConfig, parse_retrieval_config


@dataclass(frozen=True)
class GenerationModelConfig:
    """Settings for one model-provider request."""

    provider: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeConfig:
    """Settings for one judge model request and its system prompt."""

    provider: str
    model: str
    base_url: str
    api_key_env: str
    system_prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptConfig:
    """Settings for formatting retrieved chunks into a generation prompt."""

    system_prompt: str
    max_chunk_chars: int


@dataclass(frozen=True)
class AnswerGenerationConfig:
    """Complete YAML-backed configuration for answer generation."""

    retrieval: RetrievalConfig
    generation: GenerationModelConfig
    judge: JudgeConfig
    prompt: PromptConfig


def load_answer_generation_config(path: Path, repo_root: Path) -> AnswerGenerationConfig:
    """Load and validate answer generation settings from a YAML file."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "The 'PyYAML' package is required to load answer generation config."
        ) from exc

    if not path.exists():
        raise ValueError(f"Generation config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Generation config must be a YAML mapping: {path}")

    generation = _parse_generation_config(payload.get("generation"))
    judge = _parse_judge_config(payload.get("judge"))

    return AnswerGenerationConfig(
        retrieval=parse_retrieval_config(payload.get("retrieval"), repo_root),
        generation=generation,
        judge=judge,
        prompt=_parse_prompt_config(payload.get("prompt")),
    )


def _parse_generation_config(value: object) -> GenerationModelConfig:
    """Parse and validate the generation section of the YAML config."""

    return _parse_model_config(value, "generation")


def _parse_judge_config(value: object) -> JudgeConfig:
    """Parse and validate the required judge section of the YAML config."""

    section = _require_mapping(value, "judge")
    model_config = _parse_model_config(
        section,
        "judge",
        additional_known_keys={"system_prompt"},
    )
    return JudgeConfig(
        provider=model_config.provider,
        model=model_config.model,
        base_url=model_config.base_url,
        api_key_env=model_config.api_key_env,
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        max_tokens=model_config.max_tokens,
        extra_options=model_config.extra_options,
        system_prompt=_require_string(
            section.get("system_prompt"),
            "judge.system_prompt",
        ),
    )


def _parse_model_config(
    value: object,
    section_name: str,
    additional_known_keys: set[str] | None = None,
) -> GenerationModelConfig:
    """Parse and validate one model-provider section of the YAML config."""

    section = _require_mapping(value, section_name)
    known_keys = {
        "provider",
        "model",
        "base_url",
        "api_key_env",
        "temperature",
        "top_p",
        "max_tokens",
    }
    if additional_known_keys is not None:
        known_keys.update(additional_known_keys)
    extra_options = {
        str(key): option_value
        for key, option_value in section.items()
        if key not in known_keys
    }

    return GenerationModelConfig(
        provider=_require_string(section.get("provider"), f"{section_name}.provider"),
        model=_require_string(section.get("model"), f"{section_name}.model"),
        base_url=_require_string(section.get("base_url"), f"{section_name}.base_url"),
        api_key_env=_require_string(
            section.get("api_key_env"),
            f"{section_name}.api_key_env",
        ),
        temperature=_optional_float(
            section.get("temperature"),
            f"{section_name}.temperature",
        ),
        top_p=_optional_float(section.get("top_p"), f"{section_name}.top_p"),
        max_tokens=_optional_positive_int(
            section.get("max_tokens"),
            f"{section_name}.max_tokens",
        ),
        extra_options=extra_options,
    )


def _parse_prompt_config(value: object) -> PromptConfig:
    """Parse and validate the prompt section of the YAML config."""

    section = _require_mapping(value, "prompt")
    return PromptConfig(
        system_prompt=_require_string(section.get("system"), "prompt.system"),
        max_chunk_chars=_require_positive_int(
            section.get("max_chunk_chars"),
            "prompt.max_chunk_chars",
        ),
    )


def _require_mapping(value: object, field_name: str) -> dict:
    """Return a required mapping field or raise a readable error."""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_string(value: object, field_name: str) -> str:
    """Return a required non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: object, field_name: str) -> int:
    """Return a required positive integer field."""

    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    """Return an optional positive integer field."""

    if value is None:
        return None
    return _require_positive_int(value, field_name)


def _optional_float(value: object, field_name: str) -> float | None:
    """Return an optional numeric field as a float."""

    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)

