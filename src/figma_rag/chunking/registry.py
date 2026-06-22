"""Registry of available document chunking strategies."""

from __future__ import annotations

from .base import ChunkingStrategy
from .hierarchical import HierarchicalChunkingStrategy

_STRATEGIES: dict[str, ChunkingStrategy] = {
    HierarchicalChunkingStrategy.name: HierarchicalChunkingStrategy(),
}


def available_strategies() -> tuple[str, ...]:
    """Return stable strategy names accepted by the CLI."""

    return tuple(sorted(_STRATEGIES))


def get_strategy(name: str) -> ChunkingStrategy:
    """Return a registered strategy or raise a descriptive error."""

    try:
        return _STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(available_strategies())
        raise ValueError(
            f"Unknown chunking strategy '{name}'. Available strategies: {available}"
        ) from exc
