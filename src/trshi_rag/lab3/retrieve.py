from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from chromadb import PersistentClient
from chromadb.api.client import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DEFAULT_QUESTIONS = Path("reports/lab-3/eval-questions.json")
DEFAULT_OUT = Path("reports/lab-3/retrieval-results.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="?", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-n", "--n-results", type=int, default=10)
    parser.add_argument("--db", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--collection", type=str, default="docs")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    print(f"Questions: {len(questions)}")

    with cast(Client, PersistentClient(str(args.db))) as client:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=args.model, device=args.device)
        collection = client.get_collection(args.collection, embedding_function=embed_fn)

        results = collection.query(
            query_texts=[q["question"] for q in questions],
            n_results=min(args.n_results, collection.count()),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for question, ids in zip(questions, results["ids"]):
            out.write(json.dumps(
                {"id": question["id"], "question": question["question"], "retrieved": ids},
                ensure_ascii=False,
            ))
            out.write("\n")


if __name__ == "__main__":
    main()
