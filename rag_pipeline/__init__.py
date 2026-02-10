"""rag-pipeline - minimal RAG system with chunking, embeddings and retrieval."""

from .chunker import RecursiveChunker, chunk_text
from .vectorstore import VectorStore
from .retriever import Retriever
from .pipeline import RAGPipeline

__version__ = "0.1.0"
