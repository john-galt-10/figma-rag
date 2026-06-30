"""Shared interfaces and data structures for document chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata shared by every chunk produced from a document."""

    document_id: str
    title: str
    source_url: str
    source_type: str
    product_area: str
    topic: str
    processed_file_path: str


@dataclass(frozen=True)
class ChunkDraft:
    """Strategy output before common IDs and metadata are attached."""

    content: str
    heading_path: tuple[str, ...]
    char_span: tuple[int, int] | None = None


class Tokenizer(Protocol):
    """Minimal tokenizer behavior required by chunking strategies."""

    def count(self, text: str) -> int:
        """Return the encoded length including model special tokens."""

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Split text into pieces no larger than the requested token count."""


class ChunkingStrategy(Protocol):
    """Common interface implemented by all chunking strategies."""

    name: str

    def chunk(
        self,
        document: str,
        metadata: DocumentMetadata,
        tokenizer: Tokenizer,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[ChunkDraft]:
        """Split one Markdown document into strategy-specific drafts."""


def build_embedding_text(
    title: str,
    heading_path: tuple[str, ...],
    content: str,
) -> str:
    """Build the text that will be embedded and used for token limits."""

    context = [f"Title: {title}"]
    if heading_path:
        context.append(f"Section: {' > '.join(heading_path)}")
    return "\n".join(context) + "\n\n" + content.strip()
