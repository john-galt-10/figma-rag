"""Sentence Transformers model loading helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODEL_PATHS_ENV_VAR = "FIGMA_RAG_MODEL_PATHS_JSON"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sentence_transformer(model_name: str):
    """Load a Sentence Transformers model, preferring configured local weights."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The 'sentence-transformers' package is required to load embedding models"
        ) from exc

    return _load_configured_model(
        model_name=model_name,
        model_kind="embedding model",
        model_factory=SentenceTransformer,
    )


def load_cross_encoder(model_name: str):
    """Load a Sentence Transformers cross-encoder, preferring local weights."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "The 'sentence-transformers' package is required to rerank chunks"
        ) from exc

    return _load_configured_model(
        model_name=model_name,
        model_kind="reranker model",
        model_factory=CrossEncoder,
    )


def _load_configured_model(model_name: str, model_kind: str, model_factory):
    """Load a model from a configured local path or its canonical model ID."""

    _load_dotenv_settings()
    model_reference = model_name.strip()
    if not model_reference:
        raise ValueError("model_name must not be empty")

    local_model_path = _mapped_model_path(model_reference)
    if local_model_path is not None and local_model_path.is_dir():
        print(
            f"Loading local {model_kind} for {model_reference}: {local_model_path}",
            file=sys.stderr,
        )
        return model_factory(str(local_model_path), trust_remote_code=True)

    if local_model_path is not None:
        print(
            f"Local {model_kind} path not found for {model_reference}: "
            f"{local_model_path}. Loading from Hugging Face/cache.",
            file=sys.stderr,
        )
    else:
        print(
            f"No local {model_kind} mapping found for {model_reference}. "
            "Loading from Hugging Face/cache.",
            file=sys.stderr,
        )

    return model_factory(model_reference, trust_remote_code=True)


def _load_dotenv_settings() -> None:
    """Load repo-local dotenv settings for model path mappings."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "The 'python-dotenv' package is required to load local model settings"
        ) from exc

    load_dotenv(dotenv_path=REPO_ROOT / ".env")


def _mapped_model_path(model_name: str) -> Path | None:
    """Return the mapped local path for a model ID, if configured."""

    model_paths = _load_model_path_mapping()
    mapped_path = model_paths.get(model_name)
    if not mapped_path:
        return None
    return Path(mapped_path).expanduser()


def _load_model_path_mapping() -> dict[str, str]:
    """Load model ID to local path mappings from the dotenv environment."""

    raw_mapping = os.getenv(MODEL_PATHS_ENV_VAR, "").strip()
    if not raw_mapping:
        return {}

    try:
        mapping = json.loads(raw_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{MODEL_PATHS_ENV_VAR} must be a JSON object mapping model IDs to paths"
        ) from exc

    if not isinstance(mapping, dict):
        raise ValueError(
            f"{MODEL_PATHS_ENV_VAR} must be a JSON object mapping model IDs to paths"
        )

    return {
        str(model_id): str(model_path)
        for model_id, model_path in mapping.items()
        if str(model_id).strip() and str(model_path).strip()
    }
