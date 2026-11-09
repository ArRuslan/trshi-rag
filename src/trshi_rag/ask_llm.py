import argparse
from typing import cast

import ollama
from chromadb import PersistentClient
from chromadb.api.client import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from ollama import Options


class ArgumentsNamespace(argparse.Namespace):
    model: str
    question: str
    no_think: bool
    stream: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bonsai-2:4b")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--no-think", action="store_true", default=False)
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args(namespace=ArgumentsNamespace())

    with cast(Client, PersistentClient("data/chromadb")) as client:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name="google/embeddinggemma-300m", device="cpu")
        collection = client.get_collection("docs", embedding_function=embed_fn)
        results = collection.query(query_texts=[args.question], n_results=4)

    resp = ollama.chat(
        model=args.model,
        stream=args.stream,
        think=not args.no_think,
        options=Options(num_ctx=16 * 1024),
        messages=[{
            "role": "user",
            "content": "\n".join([
                "User asked question:",
                f"{args.question}",
                "",
                f"Top {len(results['documents'])} found documents:",
                *(
                    f"Answer #{num}:\n{doc}"
                    for num, doc in enumerate(results["documents"][0], start=1)
                ),
                "",
                "If found documents have actual answer (information/function/class/method/etc.) "
                "for user's questions - respond using the information from document, "
                "otherwise - explicitly tell user that nothing was found and DO NOT give any other answers, alternatives, files or examples.",
                "Answer MUST contain source file and section."
                "DO NOT just recite whole document."
                "Keep answer short, concise and understandable for user."
                "If necessary - provide short code example."
            ]),
        }],
    )

    if args.stream:
        in_thinking = False

        for chunk in resp:
            if chunk.message.thinking and not in_thinking:
                in_thinking = True
                print("Thinking:\n", end='')

            if chunk.message.thinking:
                print(chunk.message.thinking, end='')
            elif chunk.message.content:
                if in_thinking:
                    print("\n\nAnswer:\n", end='')
                    in_thinking = False
                print(chunk.message.content, end='')
    else:
        print(resp.message.content)


if __name__ == "__main__":
    main()
