"""Tests for vector store."""

import json
import pytest
from rag_pipeline.vectorstore import VectorStore, cosine_similarity, euclidean_distance


class TestSimilarityFunctions:
    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_cosine_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_euclidean_same_point(self):
        assert euclidean_distance([1, 2], [1, 2]) == 0.0

    def test_euclidean_known(self):
        assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)


class TestVectorStore:
    def test_add_and_get(self):
        store = VectorStore()
        store.add("doc1", "hello world", [1.0, 0.0, 0.0])
        doc = store.get("doc1")
        assert doc is not None
        assert doc.text == "hello world"

    def test_add_overwrites(self):
        store = VectorStore()
        store.add("doc1", "v1", [1.0, 0.0])
        store.add("doc1", "v2", [0.0, 1.0])
        assert store.get("doc1").text == "v2"

    def test_delete(self):
        store = VectorStore()
        store.add("doc1", "hello", [1.0])
        assert store.delete("doc1") is True
        assert store.get("doc1") is None
        assert store.delete("doc1") is False

    def test_len_and_contains(self):
        store = VectorStore()
        store.add("a", "text", [1.0])
        store.add("b", "text", [0.0])
        assert len(store) == 2
        assert "a" in store
        assert "c" not in store

    def test_add_batch(self):
        store = VectorStore()
        count = store.add_batch([
            {"id": "a", "text": "hello", "embedding": [1.0, 0.0]},
            {"id": "b", "text": "world", "embedding": [0.0, 1.0]},
        ])
        assert count == 2
        assert len(store) == 2


class TestSearch:
    def setup_method(self):
        self.store = VectorStore()
        # doc about python
        self.store.add("python", "Python programming language", [0.9, 0.1, 0.0])
        # doc about java
        self.store.add("java", "Java programming language", [0.7, 0.3, 0.0])
        # doc about cooking
        self.store.add("cooking", "How to make pasta", [0.0, 0.1, 0.9])

    def test_search_returns_relevant(self):
        # query close to python/java
        results = self.store.search([0.8, 0.2, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "python"  # most similar

    def test_search_top_k(self):
        results = self.store.search([0.5, 0.5, 0.5], top_k=1)
        assert len(results) == 1

    def test_search_min_score(self):
        results = self.store.search([0.9, 0.1, 0.0], top_k=10, min_score=0.95)
        # only python should be above 0.95
        assert all(r["score"] >= 0.95 for r in results)

    def test_search_metadata_filter(self):
        self.store.add("py2", "Python 2", [0.85, 0.15, 0.0], metadata={"version": 2})
        self.store.add("py3", "Python 3", [0.88, 0.12, 0.0], metadata={"version": 3})

        results = self.store.search(
            [0.9, 0.1, 0.0],
            top_k=10,
            filter_metadata={"version": 3},
        )
        assert all(r["metadata"]["version"] == 3 for r in results)

    def test_euclidean_metric(self):
        store = VectorStore(metric="euclidean")
        store.add("a", "close", [1.0, 0.0])
        store.add("b", "far", [10.0, 10.0])

        results = store.search([1.0, 0.1], top_k=2)
        assert results[0]["id"] == "a"  # closer

    def test_stats_tracking(self):
        self.store.search([0.5, 0.5, 0.5])
        self.store.search([0.5, 0.5, 0.5])
        assert self.store.stats["queries"] == 2
        assert self.store.stats["inserts"] == 3


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        store = VectorStore()
        store.add("doc1", "hello", [1.0, 0.0], metadata={"k": "v"})

        path = tmp_path / "store.json"
        store.save(path)

        loaded = VectorStore.load(path)
        assert len(loaded) == 1
        doc = loaded.get("doc1")
        assert doc.text == "hello"
        assert doc.metadata == {"k": "v"}
