"""End-to-end RAG pipeline.

Wires together: document loading -> chunking -> embedding -> storage -> retrieval -> generation.
"""

from __future__ import annotations

from typing import Any, Callable

from .chunker import RecursiveChunker, Chunk
from .vectorstore import VectorStore
from .retriever import Retriever


class RAGPipeline:
    """Full RAG pipeline from documents to answers.

    Handles the entire flow:
    1. Ingest documents (chunk + embed + store)
    2. Query (retrieve + build prompt + generate)

    You provide the embed_fn and generate_fn - this works
    with any LLM provider (OpenAI, Anthropic, local models).

    Example
    -------
    >>> pipe = RAGPipeline(
    ...     embed_fn=my_embedding_function,
    ...     generate_fn=my_llm_function,
    ... )
    >>> pipe.ingest("doc1", "Python is a programming language...")
    >>> answer = pipe.query("What is Python?")
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Answer the question based on the context below. "
        "If the context doesn't contain enough info, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        generate_fn: Callable[[str], str] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_context_tokens: int = 4096,
        prompt_template: str | None = None,
        rerank_fn: Callable[[str, list[dict]], list[dict]] | None = None,
    ):
        self._embed_fn = embed_fn
        self._generate_fn = generate_fn
        self._chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._store = VectorStore(metric="cosine")
        self._retriever = Retriever(
            store=self._store,
            embed_fn=embed_fn,
            rerank_fn=rerank_fn,
            max_context_tokens=max_context_tokens,
        )
        self._template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self._doc_count = 0
        self._chunk_count = 0

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def store(self) -> VectorStore:
        """Direct access to the vector store if you need it."""
        return self._store

    def ingest(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk, embed, and store a document.

        Returns the list of chunks created.
        """
        meta = metadata or {}
        meta["source_doc"] = doc_id

        chunks = self._chunker.split(text, metadata=meta)

        for chunk in chunks:
            chunk_id = f"{doc_id}::chunk_{chunk.index}"
            embedding = self._embed_fn(chunk.text)
            self._store.add(
                doc_id=chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata={**chunk.metadata, "chunk_index": chunk.index},
            )

        self._doc_count += 1
        self._chunk_count += len(chunks)

        return chunks

    def ingest_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """Ingest multiple documents.

        Each dict needs: 'id', 'text'. Optional: 'metadata'.
        Returns total chunks created.
        """
        total = 0
        for doc in documents:
            chunks = self.ingest(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata"),
            )
            total += len(chunks)
        return total

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        return self._retriever.retrieve(query, top_k=top_k, min_score=min_score)

    def query(
        self,
        question: str,
        top_k: int = 5,
        return_sources: bool = False,
    ) -> dict[str, Any]:
        """Full RAG: retrieve context, build prompt, generate answer.

        Returns dict with 'answer', 'prompt', and optionally 'sources'.
        """
        if self._generate_fn is None:
            raise RuntimeError("No generate_fn provided - can't generate answers")

        context = self._retriever.build_context(question, top_k=top_k)

        prompt = self._template.format(
            context=context,
            question=question,
        )

        answer = self._generate_fn(prompt)

        result: dict[str, Any] = {
            "answer": answer,
            "prompt": prompt,
        }

        if return_sources:
            result["sources"] = self._retriever.retrieve(question, top_k=top_k)

        return result
