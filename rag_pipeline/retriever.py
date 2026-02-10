"""Retrieval strategies for RAG.

Wraps the vector store with reranking, deduplication,
and context window management.
"""

from __future__ import annotations

from typing import Any, Callable

from .vectorstore import VectorStore


class Retriever:
    """High-level retrieval interface with optional reranking.

    Handles the full retrieval flow:
    1. Embed the query
    2. Vector search for candidates
    3. Optional reranking
    4. Deduplication
    5. Context window fitting
    """

    def __init__(
        self,
        store: VectorStore,
        embed_fn: Callable[[str], list[float]],
        rerank_fn: Callable[[str, list[dict]], list[dict]] | None = None,
        max_context_tokens: int = 4096,
    ):
        self._store = store
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._max_tokens = max_context_tokens
        self._stats = {"queries": 0, "docs_retrieved": 0, "docs_after_rerank": 0}

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        filter_metadata: dict | None = None,
        fetch_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k:
            Number of final results to return.
        min_score:
            Minimum similarity threshold.
        filter_metadata:
            Filter documents by metadata key-value pairs.
        fetch_k:
            How many candidates to fetch before reranking.
            Defaults to top_k * 3 if reranker is set.
        """
        self._stats["queries"] += 1

        query_embedding = self._embed_fn(query)

        if fetch_k is None:
            fetch_k = top_k * 3 if self._rerank_fn else top_k

        candidates = self._store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )
        self._stats["docs_retrieved"] += len(candidates)

        if self._rerank_fn and candidates:
            candidates = self._rerank_fn(query, candidates)
            self._stats["docs_after_rerank"] += len(candidates)

        # deduplicate by text content
        seen = set()
        unique = []
        for doc in candidates:
            text_hash = hash(doc["text"][:200])
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(doc)

        # fit within context window
        fitted = self._fit_context(unique, top_k)

        return fitted

    def _fit_context(
        self,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Keep adding docs until we'd exceed the token budget."""
        result = []
        total_tokens = 0

        for doc in docs[:top_k]:
            # rough token estimate
            doc_tokens = int(len(doc["text"].split()) * 1.3)
            if total_tokens + doc_tokens > self._max_tokens:
                break
            result.append(doc)
            total_tokens += doc_tokens

        return result

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Retrieve docs and join them into a single context string.

        Ready to drop into an LLM prompt.
        """
        docs = self.retrieve(query, top_k=top_k)
        return separator.join(d["text"] for d in docs)
