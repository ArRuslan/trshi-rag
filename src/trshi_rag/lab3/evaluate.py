from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("No relevant chunks")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    for q in questions:
        if not q.get("relevant"):
            raise ValueError(f"Question {q.get('id')} doesn't have relevant chunks")
    return questions


def load_results(path: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = row.get("id") or row.get("query_id")
        retrieved = row.get("retrieved") or row.get("ids") or row.get("documents")
        if query_id is None or retrieved is None:
            raise ValueError(f"Expected id/retrieved, got: {sorted(row)}")
        results[query_id] = [str(x) for x in retrieved]
    return results


def evaluate(
    questions: list[dict[str, Any]],
    results: dict[str, list[str]],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    per_question = []
    for q in questions:
        retrieved = results.get(q["id"], [])
        relevant = set(q["relevant"])
        row: dict[str, Any] = {
            "id": q["id"],
            "question": q.get("question", ""),
            "relevant": len(relevant),
            "retrieved": len(retrieved),
        }
        for k in ks:
            row[f"recall@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
        per_question.append(row)

    missing = sorted({q["id"] for q in questions} - set(results))
    overall = {
        f"recall@{k}": round(sum(row[f"recall@{k}"] for row in per_question) / len(per_question), 4)
        for k in ks
    }
    return {
        "questions": len(per_question),
        "missing_results": missing,
        "overall": overall,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("-k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    results = load_results(args.results)
    report = evaluate(questions, results, tuple(args.k))

    if report["missing_results"]:
        print("No results for:", ", ".join(report["missing_results"]))

    print(f"Questions: {report['questions']}")
    for k, value in report["overall"].items():
        print(f"{k}: {value:.4f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
