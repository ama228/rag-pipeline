"""Tests for document chunking."""

import pytest
from rag_pipeline.chunker import chunk_text, RecursiveChunker, Chunk


class TestSimpleChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello world", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].index == 0

    def test_splits_at_size(self):
        text = "word " * 100  # 500 chars
        chunks = chunk_text(text.strip(), chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_overlap(self):
        text = "a " * 200
        chunks = chunk_text(text.strip(), chunk_size=50, chunk_overlap=10)
        # with overlap, chunks should share some content
        assert len(chunks) >= 3

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_overlap_too_large(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=10, chunk_overlap=10)

    def test_chunk_metadata(self):
        chunks = chunk_text("some text here that is long enough to chunk", chunk_size=20, chunk_overlap=0)
        for c in chunks:
            assert c.start_char >= 0
            assert c.end_char > c.start_char

    def test_token_estimate(self):
        c = Chunk(text="hello world this is a test", index=0, start_char=0, end_char=26)
        # 6 words * 1.3 = 7.8 -> 7
        assert c.token_estimate == 7


class TestRecursiveChunker:
    def test_small_text_no_split(self):
        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.split("short text")
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_splits_on_paragraphs(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) >= 2

    def test_splits_on_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) >= 2

    def test_merges_small_chunks(self):
        text = "A.\n\nB.\n\nC."
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.split(text)
        # should merge small chunks together
        assert len(chunks) <= 2

    def test_metadata_passed_through(self):
        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.split("hello world", metadata={"source": "test.txt"})
        assert chunks[0].metadata["source"] == "test.txt"

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.split("") == []
        assert chunker.split("   ") == []

    def test_custom_separators(self):
        text = "a|b|c|d|e|f|g|h"
        chunker = RecursiveChunker(chunk_size=5, chunk_overlap=0, separators=["|"])
        chunks = chunker.split(text)
        assert len(chunks) >= 2
