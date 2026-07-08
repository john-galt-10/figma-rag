"""YAML-backed configuration helpers for retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Retriever
from .chroma import ChromaRetriever, resolve_collection_name
from .filters import MetadataFilterSet, parse_metadata_filter_set
from .pipeline import RetrievalPipeline, build_retrieval_pipeline
from .reranking import CrossEncoderReranker


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
class RetrievalOptions:
    """Resolved request-time options used by retrieval scripts."""

    top_k: int
    candidate_k: int | None
    metadata_filters: MetadataFilterSet
    metadata_filters_enabled: bool
    topic_filter: dict | None


def load_retrieval_config(path: Path, repo_root: Path) -> RetrievalConfig:
    """Load and validate the retrieval section from a shared YAML config file."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "The 'PyYAML' package is required to load retrieval config."
        ) from exc

    if not path.exists():
        raise ValueError(f"Retrieval config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Retrieval config must be a YAML mapping: {path}")

    return parse_retrieval_config(payload.get("retrieval"), repo_root)


def parse_retrieval_config(value: object, repo_root: Path) -> RetrievalConfig:
    """Parse and validate a retrieval config section."""

    section = _require_mapping(value, "retrieval")
    reranking = _optional_mapping(section.get("reranking"), "retrieval.reranking") or {}
    topic_filter = (
        _optional_mapping(section.get("topic_filter"), "retrieval.topic_filter") or {}
    )

    top_k = _require_positive_int(section.get("top_k"), "retrieval.top_k")
    candidate_k = _optional_positive_int(
        section.get("candidate_k"),
        "retrieval.candidate_k",
    )
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
            _require_string(
                section.get("chroma_persist_dir"),
                "retrieval.chroma_persist_dir",
            ),
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


def build_configured_retrieval_pipeline(config: RetrievalConfig) -> RetrievalPipeline:
    """Build a retrieval pipeline from YAML-backed retrieval settings."""

    chroma_retriever = None
    if "chroma" in config.components:
        collection_name = resolve_collection_name(
            chunks_path=config.chunks_path,
            model_name=config.embedding_model,
            collection_name=config.collection_name,
        )
        chroma_retriever = ChromaRetriever(
            persist_dir=config.chroma_persist_dir,
            collection_name=str(collection_name),
            model_name=config.embedding_model,
        )

    bm25_retriever = None
    if "bm25" in config.components:
        bm25_retriever = BM25Retriever(index_dir=config.bm25_index_dir)

    reranker = (
        CrossEncoderReranker(model_name=config.reranker_model)
        if config.reranking_enabled
        else None
    )

    return build_retrieval_pipeline(
        component_names=config.components,
        chroma_retriever=chroma_retriever,
        bm25_retriever=bm25_retriever,
        aggregation_strategy_name=config.aggregation_strategy,
        component_weights=config.component_weights,
        reranker=reranker,
    )


def resolve_retrieval_options(
    config: RetrievalConfig,
    top_k: int | None = None,
) -> RetrievalOptions:
    """Resolve YAML retrieval settings into request-time options."""

    resolved_top_k = config.top_k if top_k is None else top_k
    if resolved_top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    candidate_k = (
        None
        if not config.reranking_enabled
        else config.candidate_k or resolved_top_k * 5
    )
    topic_filter = config.topic_filter_where if config.topic_filter_enabled else None

    return RetrievalOptions(
        top_k=resolved_top_k,
        candidate_k=candidate_k,
        metadata_filters=parse_metadata_filter_set(config.metadata_filters),
        metadata_filters_enabled=config.metadata_filters_enabled,
        topic_filter=topic_filter,
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
