from __future__ import annotations

import argparse
from typing import cast

from chromadb import PersistentClient
from chromadb.api.client import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, required=True)
    parser.add_argument("-k", "--n-results", type=int, default=5)
    parser.add_argument("--db", type=str, default="data/chromadb")
    parser.add_argument("--collection", type=str, default="docs")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--device", type=str, default="cpu")
    ns = parser.parse_args()

    with cast(Client, PersistentClient(ns.db)) as client:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=ns.model, device=ns.device)
        collection = client.get_collection(ns.collection, embedding_function=embed_fn)

        results = collection.query(query_texts=[ns.query], n_results=ns.n_results)

    for idx, dist, meta in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
        print(f"{idx}  dist={dist:.4f}  {meta['document']} | {meta['section']}")


if __name__ == "__main__":
    main()
