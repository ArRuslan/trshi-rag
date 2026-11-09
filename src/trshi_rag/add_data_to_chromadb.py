from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from chromadb import PersistentClient
from chromadb.api.client import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

from trshi_rag.process_parts import RagPartProcessed

BATCH_SIZE = 1000


def build_metadata(part: RagPartProcessed) -> dict[str, str | int]:
    sym = part.get("symbol") or {}
    return {
        "document": part["document"],
        "section": part["section"],
        "part_index": part["part_index"],
        "parts_total": part["parts_total"],
        "tokens": part["tokens"],
        "global_id": part["global_id"],
        "symbol_type": sym.get("type", ""),
        "module": sym.get("module", ""),
        "class_": sym.get("class_", ""),
        "function": sym.get("function", ""),
        "attribute": sym.get("attribute", ""),
        "other": sym.get("other", ""),
        "signatures": "; ".join(sym.get("signatures") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--db", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--collection", type=str, default="docs")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    parts: list[RagPartProcessed] = []
    for file in sorted(args.input.glob("*.json")):
        parts.extend(json.loads(file.read_text(encoding="utf-8")))

    with cast(Client, PersistentClient(str(args.db))) as client:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=args.model, device=args.device)
        collection = client.get_or_create_collection(args.collection, embedding_function=embed_fn)

        for start in tqdm(range(0, len(parts), BATCH_SIZE)):
            batch = parts[start:start + BATCH_SIZE]
            collection.add(
                ids=[part["global_id"] for part in batch],
                documents=[part["text"] for part in batch],
                metadatas=[build_metadata(part) for part in batch],
            )


if __name__ == "__main__":
    main()
