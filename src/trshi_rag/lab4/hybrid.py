from __future__ import annotations

from trshi_rag.lab4 import config
from trshi_rag.lab4.dense import DenseRetriever
from trshi_rag.lab4.retrieval import Retrieved
from trshi_rag.lab4.sparse import BM25Retriever


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever) -> None:
        self.dense = dense
        self.sparse = sparse

    def search(self, query: str, k: int = 5, pool: int = 20) -> list[Retrieved]:
        dense_hits = self.dense.search(query, pool)
        sparse_hits = self.sparse.search(query, pool)
        scores = rrf_fuse([[hit.id for hit in dense_hits], [hit.id for hit in sparse_hits]])
        ordered = sorted(scores.items(), key=lambda item: -item[1])[:k]

        by_id = {hit.id: hit for hit in [*dense_hits, *sparse_hits]}
        out: list[Retrieved] = []
        for chunk_id, score in ordered:
            hit = by_id[chunk_id]
            out.append(Retrieved(id=hit.id, document=hit.document, section=hit.section, text=hit.text, score=score))

        return out
