from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from trshi_rag.lab4 import config
from trshi_rag.lab4.retrieval import Retrieved

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    with config.CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                corpus[row["global_id"]] = row
    return corpus


class BM25Retriever:
    def __init__(self) -> None:
        self._corpus = load_corpus()
        self._ids, self._tokens = self._load_tokens()
        self._bm25 = BM25Okapi(self._tokens)

    def _load_tokens(self) -> tuple[list[str], list[list[str]]]:
        cache_path = config.BM25_CACHE
        meta = {
            "mtime": config.CORPUS_PATH.stat().st_mtime,
            "size": config.CORPUS_PATH.stat().st_size,
        }
        if cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("meta") == meta:
                return cached["ids"], cached["tokens"]

        ids = list(self._corpus)
        tokens = [tokenize(self._corpus[chunk_id]["text"]) for chunk_id in ids]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"meta": meta, "ids": ids, "tokens": tokens}, f)
        return ids, tokens

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        order = np.argsort(-scores)[:k]
        out: list[Retrieved] = []
        for idx in order:
            chunk_id = self._ids[int(idx)]
            row = self._corpus[chunk_id]
            out.append(Retrieved(
                id=chunk_id,
                document=row["document"],
                section=row["section"],
                text=row["text"],
                score=float(scores[int(idx)]),
            ))

        return out
