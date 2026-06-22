"""Tokenizer adapter used to enforce model-aware chunk limits."""

from __future__ import annotations

from typing import Any


class HuggingFaceTokenizer:
    """Load only a model tokenizer, without loading embedding weights."""

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for model-aware chunking"
            ) from exc

        self.model_name = model_name
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def split(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [
            self._tokenizer.decode(
                token_ids[start : start + max_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for start in range(0, len(token_ids), max_tokens)
            if token_ids[start : start + max_tokens]
        ]
