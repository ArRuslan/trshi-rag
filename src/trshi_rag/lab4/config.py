from __future__ import annotations

import os
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.0.111:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "bonsai-2:2b")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "google/embeddinggemma-300m")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
EMBEDDING_QUERY_PREFIX = os.environ.get("EMBEDDING_QUERY_PREFIX", "task: search result | query: ")

CORPUS_PATH = Path(os.environ.get("RAG_CORPUS", "data/processed.jsonl"))
CHROMA_DIR = Path(os.environ.get("RAG_CHROMA_DIR", "data/chromadb"))
COLLECTION = os.environ.get("RAG_COLLECTION", "docs")

BM25_CACHE = Path(os.environ.get("RAG_BM25_CACHE", "data/lab4/bm25-cache.pkl"))
RRF_K = 60
