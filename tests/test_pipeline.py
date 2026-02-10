"""Tests for the full RAG pipeline."""

import pytest
from rag_pipeline import RAGPipeline


def fake_embed(text: str) -> list[float]:
    """Dumb embedding: just count certain keywords.

    Not real embeddings obviously, but good enough for testing
    the pipeline wiring.
    """
    words = text.lower().split()
    return [
        words.count("python") + words.count("code") + words.count("programming"),
        words.count("cooking") + words.count("food") + words.count("recipe"),
        words.count("music") + words.count("song") + words.count("band"),
        len(words) / 100,  # length signal
    ]


def fake_generate(prompt: str) -> str:
    """Fake LLM that just echoes back part of the context."""
    if "Context:" in prompt:
        ctx = prompt.split("Context:")[1].split("Question:")[0].strip()
        return f"Based on the context: {ctx[:100]}"
    return "I don't know."


class TestPipelineIngestion:
    def test_ingest_single_doc(self):
        pipe = RAGPipeline(embed_fn=fake_embed)
        chunks = pipe.ingest("doc1", "Python is a great programming language for beginners.")
        assert len(chunks) >= 1
        assert pipe.doc_count == 1
        assert pipe.chunk_count >= 1

    def test_ingest_long_doc(self):
        pipe = RAGPipeline(embed_fn=fake_embed, chunk_size=50, chunk_overlap=10)
        text = "Python programming " * 50  # long enough to need multiple chunks
        chunks = pipe.ingest("doc1", text)
        assert len(chunks) > 1
        assert pipe.chunk_count == len(chunks)

    def test_ingest_batch(self):
        pipe = RAGPipeline(embed_fn=fake_embed)
        total = pipe.ingest_batch([
            {"id": "a", "text": "Python code"},
            {"id": "b", "text": "Cooking recipes"},
        ])
        assert total >= 2
        assert pipe.doc_count == 2

    def test_ingest_with_metadata(self):
        pipe = RAGPipeline(embed_fn=fake_embed)
        chunks = pipe.ingest("doc1", "Python code", metadata={"source": "wiki"})
        assert chunks[0].metadata["source"] == "wiki"


class TestPipelineRetrieval:
    def setup_method(self):
        self.pipe = RAGPipeline(embed_fn=fake_embed, generate_fn=fake_generate)
        self.pipe.ingest("py", "Python is a programming language used for code and scripts")
        self.pipe.ingest("cook", "Cooking food and making recipes is fun")
        self.pipe.ingest("music", "Playing music and listening to songs by a band")

    def test_retrieve_relevant(self):
        results = self.pipe.retrieve("Python programming code")
        assert len(results) >= 1
        # python doc should be most relevant
        assert "python" in results[0]["text"].lower() or "programming" in results[0]["text"].lower()

    def test_retrieve_top_k(self):
        results = self.pipe.retrieve("something", top_k=2)
        assert len(results) <= 2


class TestPipelineQuery:
    def test_full_query(self):
        pipe = RAGPipeline(embed_fn=fake_embed, generate_fn=fake_generate)
        pipe.ingest("doc1", "Python is used for web development and data science")

        result = pipe.query("What is Python used for?")
        assert "answer" in result
        assert "prompt" in result
        assert len(result["answer"]) > 0

    def test_query_with_sources(self):
        pipe = RAGPipeline(embed_fn=fake_embed, generate_fn=fake_generate)
        pipe.ingest("doc1", "Python programming language")

        result = pipe.query("What is Python?", return_sources=True)
        assert "sources" in result
        assert len(result["sources"]) >= 1

    def test_query_without_generate_fn(self):
        pipe = RAGPipeline(embed_fn=fake_embed)
        pipe.ingest("doc1", "hello world")

        with pytest.raises(RuntimeError, match="No generate_fn"):
            pipe.query("test")

    def test_custom_prompt_template(self):
        template = "Context: {context}\nQ: {question}\nA:"
        pipe = RAGPipeline(
            embed_fn=fake_embed,
            generate_fn=fake_generate,
            prompt_template=template,
        )
        pipe.ingest("doc1", "Python is great")
        result = pipe.query("What?")
        assert "Q:" in result["prompt"]
