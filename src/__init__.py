"""
Movie Plot RAG System - A lightweight Retrieval-Augmented Generation system.
"""
from .data_loader import DataLoader
from .chunker import TextChunker
from .vector_store import VectorStore
from .llm_client import get_llm_client, LLMClient
from .rag_pipeline import RAGPipeline

__version__ = "1.0.0"

__all__ = [
    "DataLoader",
    "TextChunker",
    "VectorStore",
    "LLMClient",
    "get_llm_client",
    "RAGPipeline",
]
