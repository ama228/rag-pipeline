"""In-memory vector store with cosine similarity search.

No external dependencies - uses pure numpy-style math.
For production use FAISS, Pinecone, Chroma, etc. This is
intentionally minimal to show the core concepts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A stored document with its embedding vector."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class VectorStore:
    """In-memory vector store with similarity search.

    Stores document embeddings and supports nearest-neighbor
    retrieval using cosine similarity or euclidean distance.

    For real workloads you'd swap this for FAISS or Chroma.
    The interface stays the same though.
    """

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")
        self._docs: dict[str, Document] = {}
        self._metric = metric
        self._stats = {"inserts": 0, "queries": 0, "deletes": 0}

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def add(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        """Add a document to the store. Overwrites if id exists."""
        self._docs[doc_id] = Document(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._stats["inserts"] += 1

    def add_batch(
        self,
        docs: list[dict[str, Any]],
    ) -> int:
        """Add multiple documents at once.

        Each dict needs: id, text, embedding. Optional: metadata.
        Returns count of documents added.
        """
        count = 0
        for doc in docs:
            self.add(
                doc_id=doc["id"],
                text=doc["text"],
                embedding=doc["embedding"],
                metadata=doc.get("metadata", {}),
            )
            count += 1
        return count

    def get(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Remove a document. Returns True if it existed."""
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._stats["deletes"] += 1
            return True
        return False

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float | None = None,
        filter_metadata: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Find the most similar documents to a query embedding.

        Parameters
        ----------
        query_embedding:
            The vector to search against.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum similarity score threshold (cosine)
            or maximum distance threshold (euclidean).
        filter_metadata:
            Only include documents where metadata contains
            all these key-value pairs.

        Returns
        -------
        List of dicts with 'id', 'text', 'score', 'metadata' keys,
        sorted by relevance (best first).
        """
        self._stats["queries"] += 1

        candidates = self._docs.values()

        if filter_metadata:
            candidates = [
                d for d in candidates
                if all(d.metadata.get(k) == v for k, v in filter_metadata.items())
            ]

        scored = []
        for doc in candidates:
            if self._metric == "cosine":
                score = cosine_similarity(query_embedding, doc.embedding)
            else:
                # for euclidean, lower = better, so we negate for sorting
                score = -euclidean_distance(query_embedding, doc.embedding)

            if min_score is not None:
                if self._metric == "cosine" and score < min_score:
                    continue
                if self._metric == "euclidean" and (-score) > min_score:
                    continue

            scored.append({
                "id": doc.id,
                "text": doc.text,
                "score": score if self._metric == "cosine" else -score,
                "metadata": doc.metadata,
            })

        scored.sort(key=lambda x: x["score"], reverse=(self._metric == "cosine"))
        if self._metric == "euclidean":
            scored.sort(key=lambda x: x["score"])

        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        """Persist the store to a JSON file."""
        data = {
            "metric": self._metric,
            "documents": [
                {
                    "id": d.id,
                    "text": d.text,
                    "embedding": d.embedding,
                    "metadata": d.metadata,
                }
                for d in self._docs.values()
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a store from a JSON file."""
        data = json.loads(Path(path).read_text())
        store = cls(metric=data.get("metric", "cosine"))
        for doc in data["documents"]:
            store.add(doc["id"], doc["text"], doc["embedding"], doc.get("metadata", {}))
        return store
