"""Model-provider interfaces and OpenAI-compatible generation provider."""

from __future__ import annotations

import os
from typing import Protocol

from .config import GenerationModelConfig


class ModelProvider(Protocol):
    """Protocol implemented by generation model providers."""

    def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationModelConfig,
    ) -> str:
        """Return generated answer text for the provided chat messages."""


class OpenAICompatibleProvider:
    """Generate answers through an OpenAI-compatible chat completions API."""

    def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationModelConfig,
    ) -> str:
        """Call the configured OpenAI-compatible provider and return answer text."""

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
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Add {config.api_key_env}=... to .env or "
                "export it in the current environment."
            )

        client = OpenAI(base_url=config.base_url, api_key=api_key)
        request_options = _build_request_options(messages, config)
        response = client.chat.completions.create(**request_options)
        answer = response.choices[0].message.content
        return answer or ""


def build_model_provider(provider_name: str) -> ModelProvider:
    """Return the configured model provider implementation."""

    normalized_name = provider_name.strip().lower()
    if normalized_name == "openai":
        return OpenAICompatibleProvider()
    raise ValueError(
        f"Unsupported generation provider '{provider_name}'. "
        "Only 'openai' is implemented."
    )


def _build_request_options(
    messages: list[dict[str, str]],
    config: GenerationModelConfig,
) -> dict:
    """Build chat-completions request options from YAML-backed config."""

    request_options = {
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
    return request_options
