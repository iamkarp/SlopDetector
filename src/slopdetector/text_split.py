"""Splits raw text into paragraph or line units, then sentence units for drill-down."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


@dataclass
class Unit:
    index: int
    text: str
    start_line: int
    end_line: int


def split_paragraphs(text: str) -> list[Unit]:
    lines = text.splitlines()
    units: list[Unit] = []
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            if buf:
                units.append(Unit(len(units), "\n".join(buf).strip(), start + 1, i))
                buf = []
            start = i + 1
        else:
            if not buf:
                start = i
            buf.append(line)
    if buf:
        units.append(Unit(len(units), "\n".join(buf).strip(), start + 1, len(lines)))
    return [u for u in units if u.text]


def split_lines(text: str) -> list[Unit]:
    units = []
    for i, line in enumerate(text.splitlines()):
        if line.strip():
            units.append(Unit(len(units), line.strip(), i + 1, i + 1))
    return units


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_units(text: str, granularity: str) -> list[Unit]:
    if granularity == "line":
        return split_lines(text)
    return split_paragraphs(text)
