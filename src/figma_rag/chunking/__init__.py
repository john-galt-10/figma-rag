"""Extensible strategies for chunking retrieval documents."""

from .corpus import ChunkingSummary, chunk_corpus
from .registry import available_strategies, get_strategy

__all__ = [
    "ChunkingSummary",
    "available_strategies",
    "chunk_corpus",
    "get_strategy",
]
