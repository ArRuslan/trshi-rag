from typing import cast

from chromadb import PersistentClient
from chromadb.api.client import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def main() -> None:
    with cast(Client, PersistentClient("data/chromadb")) as client:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name="google/embeddinggemma-300m", device="cuda")
        collection = client.get_collection("docs", embedding_function=embed_fn)

        results = collection.query(
            query_texts=["In which python version was itertools.batched function added?"],
            n_results=2,
        )

        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            print(f"[{dist:.3f}] {doc}")


if __name__ == "__main__":
    main()
