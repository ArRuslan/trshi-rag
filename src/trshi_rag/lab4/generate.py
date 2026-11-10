from __future__ import annotations

import re
import time

import ollama
from ollama import Options

from trshi_rag.lab4 import config
from trshi_rag.lab4.retrieval import Retrieved

CITATION_RE = re.compile(r"\[(\d+)\]")
NO_ANSWER = "NO_ANSWER"
UNCERTAIN_ANSWER = "Failed to get answer with correct sources."


def build_messages(question: str, sources: list[Retrieved]) -> list[dict[str, str]]:
    blocks = []
    for n, source in enumerate(sources, start=1):
        blocks.append(
            f"[{n}] document: {source.document} | section: {source.section}\n{source.text}"
        )
    system = (
        "You are a Python documentation assistant. Answer ONLY from the SOURCES below.\n"
        "Rules:\n"
        f"1. Cite every statement with source numbers in square brackets, e.g. [1] or [2][3].\n"
        f"2. Use only numbers from 1 to {len(sources)}.\n"
        f"3. If the sources do not contain the answer, reply exactly {NO_ANSWER}.\n"
        "4. Do not invent functions, parameters or behaviour that are not in the sources.\n"
        "5. Answer in the same language as the question. Be concise; add a short code example only if useful or asked."
    )
    user = "SOURCES\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    seen: list[int] = []
    invalid: list[int] = []
    for raw in CITATION_RE.findall(text):
        number = int(raw)
        if 1 <= number <= n_sources:
            if number not in seen:
                seen.append(number)
        elif number not in invalid:
            invalid.append(number)
    return seen, invalid


def _is_no_answer(text: str) -> bool:
    return text.strip().upper().startswith(NO_ANSWER)


def _chat(host: str, model: str, messages: list[dict[str, str]]) -> str:
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=messages,
        options=Options(num_ctx=16384, temperature=0.0),
    )
    return response.message.content or ""


def generate(
    question: str,
    sources: list[Retrieved],
    host: str = config.OLLAMA_HOST,
    model: str = config.OLLAMA_MODEL,
    max_retries: int = 1,
) -> dict[str, object]:
    if not sources:
        return {
            "answer": "No relevant documents were found.",
            "status": "no_answer",
            "citations": [],
            "retry_used": False,
            "raw_answer": "",
        }

    messages = build_messages(question, sources)
    started = time.perf_counter()
    text = _chat(host, model, messages)
    retry_used = False

    for _ in range(max_retries):
        valid, invalid = parse_citations(text, len(sources))
        if _is_no_answer(text) or (valid and not invalid):
            break
        retry_used = True
        messages = [
            *messages,
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "Your previous answer violated the citation contract "
                    f"(valid numbers: {valid or 'none'}, invalid: {invalid or 'none'}). "
                    f"Rewrite it using ONLY numbers 1..{len(sources)} in square brackets, "
                    f"or reply exactly {NO_ANSWER} if the sources are insufficient."
                ),
            },
        ]
        text = _chat(host, model, messages)

    valid, invalid = parse_citations(text, len(sources))
    if not text or _is_no_answer(text):
        status = "no_answer"
        answer = f"Sources don't contain answer for the question ({NO_ANSWER})."
        cited: list[int] = []
    elif valid and not invalid:
        status = "ok"
        answer = text.strip()
        cited = valid
    else:
        status = "uncertain"
        answer = UNCERTAIN_ANSWER
        cited = []

    return {
        "answer": answer,
        "status": status,
        "citations": [sources[n - 1].id for n in cited],
        "cited_numbers": cited,
        "retry_used": retry_used,
        "raw_answer": text.strip(),
        "generation_ms": round((time.perf_counter() - started) * 1000, 1),
    }
