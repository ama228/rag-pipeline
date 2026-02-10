"""Document chunking strategies for RAG pipelines.

Splits documents into overlapping chunks suitable for embedding
and retrieval. Supports recursive splitting by separators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A piece of a document with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (words * 1.3 is close enough for most models)."""
        return int(len(self.text.split()) * 1.3)

    def __len__(self) -> int:
        return len(self.text)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Simple character-level chunking with overlap.

    Good enough for most use cases. Use RecursiveChunker if you
    need smarter boundary detection.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    if not text.strip():
        return []

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_slice = text[start:end]

        # try to break on whitespace if we're mid-word
        if end < len(text) and not text[end - 1].isspace():
            last_space = chunk_text_slice.rfind(" ")
            if last_space > chunk_size // 2:
                end = start + last_space + 1
                chunk_text_slice = text[start:end]

        stripped = chunk_text_slice.strip()
        if stripped:
            chunks.append(Chunk(
                text=stripped,
                index=idx,
                start_char=start,
                end_char=end,
            ))
            idx += 1
        start = end - chunk_overlap

    return chunks


class RecursiveChunker:
    """Recursively split text using a hierarchy of separators.

    Tries paragraph breaks first, then sentences, then words.
    This gives much better chunks than naive character splitting
    because it respects document structure.

    Similar approach to LangChain's RecursiveCharacterTextSplitter
    but without the dependency.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks respecting separator hierarchy."""
        if not text.strip():
            return []

        raw_chunks = self._recursive_split(text, 0)
        merged = self._merge_small_chunks(raw_chunks)

        result = []
        offset = 0
        for i, piece in enumerate(merged):
            start = text.find(piece[:50], offset)
            if start == -1:
                start = offset
            result.append(Chunk(
                text=piece,
                index=i,
                start_char=start,
                end_char=start + len(piece),
                metadata=metadata or {},
            ))
            offset = start + len(piece) - self.chunk_overlap

        return result

    def _recursive_split(self, text: str, sep_idx: int) -> list[str]:
        """Split using current separator, recurse on oversized pieces."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if sep_idx >= len(self.separators):
            # no more separators, force split
            return [text[i:i + self.chunk_size].strip()
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                    if text[i:i + self.chunk_size].strip()]

        sep = self.separators[sep_idx]
        parts = text.split(sep)

        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    result.append(part.strip())
            else:
                result.extend(self._recursive_split(part, sep_idx + 1))

        return result

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge tiny chunks together up to chunk_size."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for piece in chunks[1:]:
            combined = current + " " + piece
            if len(combined) <= self.chunk_size:
                current = combined
            else:
                merged.append(current)
                current = piece

        if current:
            merged.append(current)

        return merged
