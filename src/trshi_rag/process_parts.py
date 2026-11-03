from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypedDict

from sentence_transformers import SentenceTransformer

from trshi_rag.docs_to_json import RagPart, RagPartSymbol

DEFAULT_MODEL = "google/embeddinggemma-300m"
SOFT_LIMIT = 768
HARD_LIMIT = 1024
OVERLAP_RATIO = 0.10

SENT_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Unit:
    text: str
    tokens: int
    atomic: bool


@dataclass
class Fragment:
    body: str
    tokens: int
    index: int
    total: int


class RagPartProcessed(TypedDict):
    document: str
    section: str
    symbol: RagPartSymbol
    text: str
    tokens: int
    part_index: int
    parts_total: int


class ArgumentsNamespace(argparse.Namespace):
    input: Path
    out: Path
    model: str
    soft: int
    hard: int
    overlap: float


class TokenCounter:
    def __init__(self, model_name_or_path: str) -> None:
        self.model = SentenceTransformer(model_name_or_path)
        self.tokenizer = self.model.tokenizer

    def count(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return len(ids)

    def unit(self, text: str, atomic: bool = False) -> Unit:
        n = self.count(text)
        return Unit(text, n, atomic)


def build_units(counter: TokenCounter, text: str) -> list[Unit]:
    units: list[Unit] = []
    for seg in text.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        if SENT_BREAK.search(seg) and len(seg) > 1:
            for piece in SENT_BREAK.split(seg):
                piece = piece.strip()
                if piece:
                    units.append(counter.unit(piece))
        else:
            units.append(counter.unit(seg, atomic=True))
    return units


def body_join(units: list[Unit]) -> str:
    out = ""
    prev_atomic = False
    for unit in units:
        if out:
            out += "\n" if (unit.atomic or prev_atomic) else " "
        out += unit.text
        prev_atomic = unit.atomic

    return out


def render_header(part: RagPart, part_no: int = 0, total: int = 1) -> str:
    sym = part["symbol"]
    tiles = [f"File: {part['document']}"]
    section = part["section"]
    if section and sym is not None and section in sym["signatures"]:
        tiles.append(f"Section: {section}")
    if sym:
        tiles.append(f"Symbol type: {sym['type']}")
        key: Literal["module", "class_", "function", "attribute", "other"]
        for key, label in (
            ("module", "Module"),
            ("class_", "Class"),
            ("function", "Function"),
            ("attribute", "Attribute"),
            ("other", "Id"),
        ):
            val = sym[key]
            if val:
                tiles.append(f"{label}: {val}")
        for sig in sym.get("signatures") or []:
            tiles.append(f"Signature: {sig}")
    if total > 1:
        tiles.append(f"Fragment: {part_no + 1}/{total}")
    return "\n".join(tiles)


def build_fragment(units: list[Unit], header: str, counter: TokenCounter, index: int, total: int) -> Fragment:
    assert units
    body = body_join(units)
    full = f"{header}\n\n{body}" if header else body
    return Fragment(full, counter.count(full), index, total)


def tail_overlap(units: list[Unit], ratio: float) -> list[Unit]:
    if not units:
        return []
    limit = max(8, round(sum(u.tokens for u in units) * ratio))
    carry: list[Unit] = []
    total_tokens = 0
    for u in reversed(units):
        if total_tokens + u.tokens > limit and carry:
            break
        carry.append(replace(u))
        total_tokens += u.tokens
    carry.reverse()
    return carry


def split_text(unit: Unit, counter: TokenCounter, soft: int, hard: int, ratio: float) -> list[Unit]:
    words = unit.text.split(" ")
    wtks = [max(1, counter.count(w + " ")) for w in words]

    pieces: list[list[str]] = []
    buf: list[str] = []
    buf_tk = 0
    for w, wtk in zip(words, wtks):
        if buf and buf_tk + wtk > hard:
            pieces.append(buf)
            buf, buf_tk = [], 0
        if not buf and wtk > hard:
            pieces.append([w])
            continue
        buf.append(w)
        buf_tk += wtk
        if buf_tk >= soft:
            pieces.append(buf)
            buf, buf_tk = [], 0
    if buf:
        pieces.append(buf)

    out: list[Unit] = []
    carry_words: list[str] = []
    for piece in pieces:
        piece_txt = " ".join(piece)
        text = (" ".join(carry_words) + " " + piece_txt).strip() if carry_words else piece_txt
        out.append(counter.unit(text))

        limit = max(8, round(counter.count(piece_txt) * ratio))
        take: list[str] = []
        acc = 0
        for w in reversed(piece):
            wtk = counter.count(w + " ")
            if acc + wtk > limit and take:
                break
            take.insert(0, w)
            acc += wtk
        carry_words = take
    return out


def word_tail_unit(piece: Unit, counter: TokenCounter, ratio: float) -> Unit:
    words = piece.text.split(" ")
    limit = max(8, round(piece.tokens * ratio))
    take: list[str] = []
    total_tokens = 0
    for word in reversed(words):
        wtk = max(1, counter.count(word + " "))
        if total_tokens + wtk > limit and take:
            break
        take.insert(0, word)
        total_tokens += wtk
    return counter.unit(" ".join(take), atomic=True) if take else piece


def pack_units(units: list[Unit], counter: TokenCounter, soft: int, hard: int, ratio: float) -> list[list[Unit]]:
    bags: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_tk = 0
    fresh = 0
    carry: list[Unit] = []

    def close() -> None:
        nonlocal cur, cur_tk, fresh, carry
        if fresh == 0:
            cur, cur_tk = [], 0
            return
        bags.append(list(cur))
        carry = tail_overlap(cur, ratio)
        cur = list(carry)
        cur_tk = sum(u.tokens for u in carry)
        fresh = 0

    for unit in units:
        if unit.tokens > hard:
            if fresh:
                close()
            else:
                cur, cur_tk = [], 0
            pieces = split_text(unit, counter, soft, hard, ratio)
            for piece in pieces:
                bags.append([piece])
            tail = word_tail_unit(pieces[-1], counter, ratio)
            cur, cur_tk, fresh = [tail], tail.tokens, 0
            continue

        if fresh and cur_tk + unit.tokens > hard:
            close()

        cur.append(unit)
        cur_tk += unit.tokens
        fresh += 1

        if cur_tk >= soft:
            close()

    if fresh:
        close()
    return bags


def process_part(counter: TokenCounter, part: RagPart, soft: int, hard: int, ratio: float) -> list[RagPartProcessed]:
    header = render_header(part)
    header_tk = counter.count(header)
    soft_body = max(64, soft - header_tk)
    hard_body = max(soft_body + 16, hard - header_tk)

    units = build_units(counter, part["text"])
    if not units:
        return []

    body_tk = sum(u.tokens for u in units)
    if body_tk <= soft_body:
        bags = [units]
    else:
        bags = pack_units(units, counter, soft_body, hard_body, ratio)

    out: list[RagPartProcessed] = []
    total = len(bags)
    for bi, bag in enumerate(bags):
        frag = build_fragment(bag, header, counter, bi, total)
        if frag.tokens > hard:
            raise ValueError(
                f"fragment exceeds hard limit: {frag.tokens} > {hard} ({part['document']} / {part['section'][:40]})"
            )

        part: RagPartProcessed = {
            "document": part["document"],
            "section": part["section"],
            "symbol": part["symbol"],
            "text": frag.body,
            "tokens": frag.tokens,
            "part_index": bi,
            "parts_total": total,
        }
        out.append(part)

    return out


def process_file(counter: TokenCounter, path_in: Path, path_out: Path, soft: int, hard: int, ratio: float) -> None:
    raw: list[RagPart] = json.loads(path_in.read_text(encoding="utf-8"))
    out_parts: list[RagPartProcessed] = []
    for part in raw:
        out_parts.extend(process_part(counter, part, soft, hard, ratio))

    path_out.parent.mkdir(parents=True, exist_ok=True)

    with open(path_out, "w") as f:
        json.dump(out_parts, f, indent=2)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--soft", type=int, default=SOFT_LIMIT)
    parser.add_argument("--hard", type=int, default=HARD_LIMIT)
    parser.add_argument("--overlap", type=float, default=OVERLAP_RATIO)
    ns = parser.parse_args(argv, namespace=ArgumentsNamespace())

    if ns.soft >= ns.hard:
        sys.exit(f"soft ({ns.soft}) must be < hard ({ns.hard})")
    if not 0 <= ns.overlap < 0.5:
        sys.exit("--overlap must be in [0, 0.5)")

    counter = TokenCounter(ns.model)
    inputs = sorted(ns.input.glob("*.json"))
    if not inputs:
        sys.exit(f"no input files found: {ns.input}")

    for f in inputs:
        target = (ns.out / f.name) if ns.out.suffix == "" else ns.out
        process_file(counter, f, target, ns.soft, ns.hard, ns.overlap)


if __name__ == "__main__":
    main()