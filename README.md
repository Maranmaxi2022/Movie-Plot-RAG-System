# Movie Plot RAG System

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions about movie plots using semantic search and large language models.

## Overview

This system demonstrates a complete RAG pipeline that:
- Loads movie plots from the Wikipedia Movie Plots dataset
- Chunks text into ~300 word segments for better retrieval
- Creates semantic embeddings using Sentence Transformers
- Stores vectors in FAISS for fast similarity search
- Retrieves top-k most relevant chunks for queries
- Generates natural language answers using OpenAI GPT models
- Returns structured JSON output with answer, contexts, and reasoning