# rag-pipeline

Minimal RAG (Retrieval-Augmented Generation) pipeline in Python. No heavyweight dependencies - just the core concepts implemented cleanly.

Built this to understand how RAG actually works under the hood instead of just calling LangChain APIs.

## What's in here

- **Chunker** - recursive text splitter that respects paragraph/sentence boundaries
- **VectorStore** - in-memory vector store with cosine similarity and metadata filtering
- **Retriever** - handles embedding, search, reranking, dedup, and context window fitting
- **RAGPipeline** - wires everything together: ingest docs -> query -> get answers

## Quick start

```python
from rag_pipeline import RAGPipeline

# bring your own embedding + generation functions
pipe = RAGPipeline(
    embed_fn=my_embed_function,       # str -> list[float]
    generate_fn=my_llm_function,      # str -> str
    chunk_size=512,
    max_context_tokens=4096,
)

# ingest documents
pipe.ingest("doc1", open("paper.txt").read(), metadata={"source": "arxiv"})
pipe.ingest("doc2", open("notes.md").read())

# query
result = pipe.query("How does attention work?", return_sources=True)
print(result["answer"])
print(result["sources"])
```

## Chunking

```python
from rag_pipeline import RecursiveChunker

chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.split(long_document)
# splits on paragraphs first, then sentences, then words
```

## Vector search

```python
from rag_pipeline import VectorStore

store = VectorStore(metric="cosine")
store.add("id1", "some text", embedding_vector, metadata={"topic": "ml"})

results = store.search(query_vector, top_k=5, filter_metadata={"topic": "ml"})
store.save("index.json")
```

## Hybrid search

v0.2.0 added hybrid retrieval that combines vector similarity with keyword matching:

```python
results = store.hybrid_search(
    query_embedding=embed("attention mechanism"),
    query_text="attention mechanism",
    vector_weight=0.7,
    keyword_weight=0.3,
)
```

Works way better than pure vector search for technical queries where exact terms matter.

## Why not LangChain?

LangChain is great but it's a black box for learning. Building this from scratch helped me understand:
- How chunking strategy affects retrieval quality
- Why overlap matters for context continuity
- How metadata filtering works at the vector level
- The tradeoffs in reranking vs just using top-k
- How hybrid search outperforms pure vector search

## Tests

```bash
pytest -v
```
