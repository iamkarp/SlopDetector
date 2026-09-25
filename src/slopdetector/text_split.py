"""Splits raw text into paragraph or line units, then sentence units for drill-down."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Two alternatives, not one \s* for both cases:
# - before a DIGIT, still require real whitespace (\s+). Otherwise "3.14"
#   or "$5.10" would split into "3." + "14" — a decimal's period has no
#   space before the next digit either, so digits need the stricter rule.
# - before an uppercase letter or quote, \s* (including zero) is safe: a
#   decimal number is never immediately followed by a capital letter with
#   no space. This is what fixes a real bug — pasted/scraped text with no
#   space after sentence-ending punctuation ("sidewalk.It is important")
#   used to glue three distinct sentences into one drill-down "sentence"
#   because the old \s+-only regex refused to split without a space.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[0-9])|(?<=[.!?])\s*(?=[A-Z\"'“])")


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


def split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Like split_sentences, but also returns each sentence's [start, end)
    character offset into the (stripped) input, so callers can map a
    sentence back to a line number instead of inheriting the whole unit's
    line range."""
    stripped = text.strip()
    if not stripped:
        return []
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(stripped):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(stripped)))
    out = []
    for start, end in spans:
        sent = stripped[start:end].strip()
        if sent:
            out.append((sent, start, end))
    return out


_HEADING_ONLY_RE = re.compile(r"^#{1,6}\s+\S.*$")


def is_non_prose_unit(text: str) -> bool:
    """True for a unit that's pure Markdown/LaTeX structure, not prose — a
    section heading, or a display-math block with no surrounding sentence.

    Real bug found scanning a real manuscript: JEV's reason question
    flagged a raw LaTeX display equation (two \\text{} macros joined by
    \\qquad) as formulaic_construction, and section headings occasionally
    drew a flag/reason too — neither is prose an AI-slop judgment applies
    to. Callers should still run Stage A rules on these (e.g.
    register_title_case_headings is designed for exactly this case) and
    only skip the LLM judgment (Stage B).
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" not in stripped and _HEADING_ONLY_RE.match(stripped):
        return True
    if stripped.startswith("\\[") and stripped.endswith("\\]"):
        return True
    return False


def split_units(text: str, granularity: str) -> list[Unit]:
    if granularity == "line":
        return split_lines(text)
    return split_paragraphs(text)
