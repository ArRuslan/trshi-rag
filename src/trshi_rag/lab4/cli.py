from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trshi_rag.lab4 import config
from trshi_rag.lab4.dense import DenseRetriever
from trshi_rag.lab4.generate import generate
from trshi_rag.lab4.hybrid import HybridRetriever
from trshi_rag.lab4.retrieval import Retrieved
from trshi_rag.lab4.sparse import BM25Retriever


def _source_dict(source: Retrieved, n: int, score_name: str) -> dict[str, object]:
    return {
        "n": n,
        "global_id": source.id,
        "document": source.document,
        "section": source.section,
        score_name: round(source.score, 4),
        "text": source.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("-q", "--question", dest="query_opt")
    parser.add_argument("--mode", choices=("dense", "sparse", "hybrid", "compare"), default="hybrid")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--host", default=config.OLLAMA_HOST)
    parser.add_argument("--no-query-prefix", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    question = args.query_opt or args.query
    if not question:
        parser.error("Either positional argument or --question is required")

    started = time.perf_counter()
    dense = DenseRetriever(query_prefix="" if args.no_query_prefix else None)
    sparse = BM25Retriever() if args.mode in ("sparse", "hybrid", "compare") else None

    dense_hits = []
    sparse_hits = []
    hybrid_hits = []

    if args.mode == "dense":
        dense_hits = dense.search(question, args.top_k)
        hits, score_name = dense_hits, "distance"
    elif args.mode == "sparse":
        assert sparse is not None
        sparse_hits = sparse.search(question, args.top_k)
        hits, score_name = sparse_hits, "bm25"
    elif args.mode == "hybrid":
        assert sparse is not None
        hybrid_hits = HybridRetriever(dense=dense, sparse=sparse).search(question, args.top_k, pool=args.pool)
        hits, score_name = hybrid_hits, "rrf"
    else:
        assert sparse is not None
        dense_hits = dense.search(question, args.pool)
        sparse_hits = sparse.search(question, args.pool)
        hybrid_hits = HybridRetriever(dense=dense, sparse=sparse).search(question, args.top_k, pool=args.pool)
        hits, score_name = hybrid_hits, "rrf"

    retrieval_ms = round((time.perf_counter() - started) * 1000, 1)

    result: dict[str, object] = {"question": question, "mode": args.mode}

    if args.mode == "compare":
        result["retrieval"] = {
            "dense": [_source_dict(h, i, "distance") for i, h in enumerate(dense_hits, 1)],
            "sparse": [_source_dict(h, i, "bm25") for i, h in enumerate(sparse_hits, 1)],
            "hybrid": [_source_dict(h, i, "rrf") for i, h in enumerate(hybrid_hits, 1)],
        }
    else:
        result["sources"] = [_source_dict(h, i, score_name) for i, h in enumerate(hits, 1)]

    result["retrieval_ms"] = retrieval_ms
    result["model"] = args.model

    if args.retrieval_only:
        out = result
    else:
        generation = generate(question, hits, host=args.host, model=args.model)
        result["generation"] = generation
        out = result

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
