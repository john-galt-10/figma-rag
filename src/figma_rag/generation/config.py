"""Configuration loading for the answer generation example."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalConfig:
    """Settings used to build and run the retrieval pipeline."""

    chunks_path: Path
    chroma_persist_dir: Path
    collection_name: str | None
    embedding_model: str
    bm25_index_dir: Path
    top_k: int
    candidate_k: int | None
    reranking_enabled: bool
    reranker_model: str
    metadata_filters: list[str]
    metadata_filters_enabled: bool
    topic_filter_enabled: bool
    topic_filter_where: dict | None
    components: list[str]
    aggregation_strategy: str
    component_weights: dict[str, float] | None


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
class PromptConfig:
    """Settings for formatting retrieved chunks into a generation prompt."""

    system_prompt: str
    max_chunk_chars: int


@dataclass(frozen=True)
class AnswerGenerationConfig:
    """Complete YAML-backed configuration for answer generation."""

    retrieval: RetrievalConfig
    generation: GenerationModelConfig
    judge: GenerationModelConfig
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
    judge = _parse_judge_config(payload.get("judge"), generation)

    return AnswerGenerationConfig(
        retrieval=_parse_retrieval_config(payload.get("retrieval"), repo_root),
        generation=generation,
        judge=judge,
        prompt=_parse_prompt_config(payload.get("prompt")),
    )


def _parse_retrieval_config(value: object, repo_root: Path) -> RetrievalConfig:
    """Parse and validate the retrieval section of the YAML config."""

    section = _require_mapping(value, "retrieval")
    reranking = _optional_mapping(section.get("reranking"), "retrieval.reranking") or {}
    topic_filter = (
        _optional_mapping(section.get("topic_filter"), "retrieval.topic_filter") or {}
    )

    top_k = _require_positive_int(section.get("top_k"), "retrieval.top_k")
    candidate_k = _optional_positive_int(section.get("candidate_k"), "retrieval.candidate_k")
    metadata_filters = _string_list(
        section.get("metadata_filters", []),
        "retrieval.metadata_filters",
    )
    components = _string_list(section.get("components"), "retrieval.components")
    if not components:
        raise ValueError("retrieval.components must contain at least one component")

    return RetrievalConfig(
        chunks_path=_resolve_repo_path(
            _require_string(section.get("chunks_path"), "retrieval.chunks_path"),
            repo_root,
        ),
        chroma_persist_dir=_resolve_repo_path(
            _require_string(section.get("chroma_persist_dir"), "retrieval.chroma_persist_dir"),
            repo_root,
        ),
        collection_name=_optional_string(
            section.get("collection_name"),
            "retrieval.collection_name",
        ),
        embedding_model=_require_string(
            section.get("embedding_model"),
            "retrieval.embedding_model",
        ),
        bm25_index_dir=_resolve_repo_path(
            _require_string(section.get("bm25_index_dir"), "retrieval.bm25_index_dir"),
            repo_root,
        ),
        top_k=top_k,
        candidate_k=candidate_k,
        reranking_enabled=bool(reranking.get("enabled", True)),
        reranker_model=_require_string(
            reranking.get("model"),
            "retrieval.reranking.model",
        ),
        metadata_filters=metadata_filters,
        metadata_filters_enabled=bool(section.get("metadata_filters_enabled", True)),
        topic_filter_enabled=bool(topic_filter.get("enabled", True)),
        topic_filter_where=_optional_mapping(
            topic_filter.get("where"),
            "retrieval.topic_filter.where",
        ),
        components=components,
        aggregation_strategy=_require_string(
            section.get("aggregation_strategy"),
            "retrieval.aggregation_strategy",
        ),
        component_weights=_optional_float_mapping(
            section.get("component_weights"),
            "retrieval.component_weights",
        ),
    )


def _parse_generation_config(value: object) -> GenerationModelConfig:
    """Parse and validate the generation section of the YAML config."""

    return _parse_model_config(value, "generation")


def _parse_judge_config(
    value: object,
    generation_config: GenerationModelConfig,
) -> GenerationModelConfig:
    """Parse the optional judge section or reuse generation settings."""

    if value is None:
        return generation_config
    return _parse_model_config(value, "judge")


def _parse_model_config(value: object, section_name: str) -> GenerationModelConfig:
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


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    """Resolve relative config paths from the repository root."""

    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _require_mapping(value: object, field_name: str) -> dict:
    """Return a required mapping field or raise a readable error."""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _optional_mapping(value: object, field_name: str) -> dict | None:
    """Return an optional mapping field or raise a readable error."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_string(value: object, field_name: str) -> str:
    """Return a required non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, field_name: str) -> str | None:
    """Return an optional non-empty string field."""

    if value is None:
        return None
    return _require_string(value, field_name)


def _string_list(value: object, field_name: str) -> list[str]:
    """Return a list of strings from a YAML sequence."""

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    parsed = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain only non-empty strings")
        parsed.append(item.strip())
    return parsed


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


def _optional_float_mapping(value: object, field_name: str) -> dict[str, float] | None:
    """Return an optional mapping of names to float values."""

    if value is None:
        return None
    mapping = _require_mapping(value, field_name)
    parsed = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, int | float):
            raise ValueError(f"{field_name}.{key} must be numeric")
        parsed[key.strip()] = float(item)
    return parsed
