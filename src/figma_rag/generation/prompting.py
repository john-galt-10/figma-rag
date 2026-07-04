"""Prompt construction for grounded answer generation."""

from __future__ import annotations

from figma_rag.retrieval import RetrievalResult

from .config import PromptConfig


def build_grounded_messages(
    query: str,
    chunks: list[RetrievalResult],
    config: PromptConfig,
) -> list[dict[str, str]]:
    """Build a simple chat prompt grounded in retrieved documentation chunks."""

    return [
        {
            "role": "system",
            "content": config.system_prompt,
        },
        {
            "role": "user",
            "content": _build_user_prompt(query, chunks, config.max_chunk_chars),
        },
    ]


def _build_user_prompt(
    query: str,
    chunks: list[RetrievalResult],
    max_chunk_chars: int,
) -> str:
    """Format the user query and retrieved chunks for the generator model."""

    context_blocks = [
        _format_chunk_for_prompt(index, chunk, max_chunk_chars)
        for index, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_blocks) if context_blocks else "No chunks retrieved."
    return (
        "Answer the question using only the retrieved Figma documentation context. "
        "If the context is insufficient, say what is missing instead of guessing.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}"
    )


def _format_chunk_for_prompt(
    index: int,
    chunk: RetrievalResult,
    max_chunk_chars: int,
) -> str:
    """Format one retrieved chunk with a stable citation marker."""

    text = " ".join(chunk.text.split())
    if len(text) > max_chunk_chars:
        text = text[: max_chunk_chars - 3].rstrip() + "..."

    return (
        f"[{index}] Title: {chunk.title}\n"
        f"Section: {chunk.section}\n"
        f"Source: {chunk.source_url}\n"
        # f"Chunk ID: {chunk.chunk_id}\n"
        f"Text: {text}"
    )
