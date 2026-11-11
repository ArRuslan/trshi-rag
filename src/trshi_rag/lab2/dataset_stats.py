from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "be", "as",
    "for", "this", "that", "if", "on", "it", "by", "with", "from", "at", "not",
    "can", "will", "was", "its", "no", "when", "which", "than", "you", "may",
    "return", "returns", "using", "use", "used", "value", "values", "one",
    "see", "also", "new", "function", "type", "class", "none", "true", "false",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line]

    documents = collections.Counter(r["document"] for r in rows)
    words = [len(r["text"].split()) for r in rows]
    chars = [len(r["text"]) for r in rows]
    tokens = [r["tokens"] for r in rows] if "tokens" in rows[0] else None

    counter = collections.Counter(w.lower() for r in rows for w in re.findall(r"[A-Za-z]+", r["text"]))
    top20 = counter.most_common(20)
    no_stopwords = {w: c for w, c in counter.items() if w not in STOPWORDS}
    top20_content = collections.Counter(no_stopwords).most_common(20)

    def med(values: list[int], p: float) -> int:
        ordered = sorted(values)
        return ordered[int(p / 100 * (len(ordered) - 1))]

    stats: dict = {
        "file": str(args.input),
        "documents": len(documents),
        "chunks": len(rows),
        "chars_total": sum(chars),
        "chunk_size": {
            "chars": {"min": min(chars), "median": med(chars, 50), "mean": round(sum(chars) / len(chars), 1), "max": max(chars)},
            "words": {"min": min(words), "median": med(words, 50), "mean": round(sum(words) / len(words), 1), "max": max(words)},
            "tokens": {"min": min(tokens), "median": med(tokens, 50), "mean": round(sum(tokens) / len(tokens), 1), "max": max(tokens)} if tokens else None,
        },
        "top20_words": top20,
        "top20_words_no_stopwords": top20_content,
        "distinct_word_tokens": len(counter),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
