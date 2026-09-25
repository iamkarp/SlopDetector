from __future__ import annotations

import re

from .base import Detector, Hit, register

# Ported from slopmonster/tools/deslop.py _root_pattern: strip common suffixes,
# then reattach the inflection family, so "elevate" also catches "elevates"/"elevating".
_STRIP_SUFFIXES = ("ed", "ing", "ly", "e")
_REATTACH = ("", "e", "es", "ed", "ing", "ion", "ions", "ional", "ive", "al", "ally", "s", "ly", "ness")


def _root(word: str) -> str:
    w = word.lower()
    for suf in _STRIP_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _root_pattern(word: str) -> re.Pattern:
    root = _root(word)
    alternation = "|".join(sorted({re.escape(root + tail) for tail in _REATTACH}, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


class _WordlistBase(Detector):
    stemmed: bool

    def match(self, text: str, node: dict) -> list[Hit]:
        data = node["data"]
        words: list[str] = data["words"]
        hits: list[Hit] = []
        for word in words:
            rx = _root_pattern(word) if self.stemmed else re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            for m in rx.finditer(text):
                hits.append(
                    Hit(
                        pattern_id=node["id"],
                        label=node.get("label", node["id"]),
                        span=(m.start(), m.end()),
                        detail=f"{word!r} ({m.group(0)!r})",
                        weight=node.get("weight", 1.0),
                    )
                )
        return hits


@register("wordlist_stemmed")
class WordlistStemmedDetector(_WordlistBase):
    """Root-matched: node.data.words, e.g. 'elevate' also flags 'elevates'/'elevating'."""

    stemmed = True


@register("wordlist_exact")
class WordlistExactDetector(_WordlistBase):
    """Exact phrase match only, for words with an honest literal sense (e.g. 'crafted', 'journey')."""

    stemmed = False
