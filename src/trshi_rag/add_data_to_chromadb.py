import json
import os
from pathlib import Path
from typing import cast

from chromadb import PersistentClient
from chromadb.api.client import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

from trshi_rag.process_parts import RagPartProcessed


def main() -> None:
    in_dir = Path("data/parts_processed")

    with cast(Client, PersistentClient("data/chromadb")) as client:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name="google/embeddinggemma-300m", device="cuda")
        collection = client.get_or_create_collection("docs", embedding_function=embed_fn)

        for doc in tqdm(os.listdir(in_dir)):
            documents_to_add = []
            ids_to_add = []

            with open(in_dir / doc) as f:
                parts: list[RagPartProcessed] = json.load(f)
                if not parts:
                    continue

                for idx, part in enumerate(parts):
                    ids_to_add.append(f"{part['document']}-{idx}")
                    documents_to_add.append(part["text"])

            collection.add(documents=documents_to_add, ids=ids_to_add)


if __name__ == "__main__":
    main()
