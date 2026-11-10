from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Retrieved:
    id: str
    document: str
    section: str
    text: str
    score: float
