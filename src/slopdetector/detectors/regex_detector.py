from __future__ import annotations

import re

from .base import Detector, Hit, register


@register("regex")
class RegexDetector(Detector):
    """node.data: {pattern: <regex str>, flags: ["I"] optional, label: str}"""

    def match(self, text: str, node: dict) -> list[Hit]:
        data = node["data"]
        flags = 0
        for f in data.get("flags", []):
            flags |= getattr(re, f)
        rx = re.compile(data["pattern"], flags)
        hits = []
        for m in rx.finditer(text):
            hits.append(
                Hit(
                    pattern_id=node["id"],
                    label=node.get("label", node["id"]),
                    span=(m.start(), m.end()),
                    detail=m.group(0),
                    weight=node.get("weight", 1.0),
                )
            )
        return hits


@register("regex_min_count")
class RegexMinCountDetector(Detector):
    """Fires only when a regex occurs >= data.min_count times in the unit
    (e.g. two+ consecutive 'Not X.' sentences, two+ em-dashes in one window)."""

    def match(self, text: str, node: dict) -> list[Hit]:
        data = node["data"]
        flags = 0
        for f in data.get("flags", []):
            flags |= getattr(re, f)
        rx = re.compile(data["pattern"], flags)
        matches = list(rx.finditer(text))
        min_count = data.get("min_count", 2)
        if len(matches) >= min_count:
            return [
                Hit(
                    pattern_id=node["id"],
                    label=node.get("label", node["id"]),
                    span=(matches[0].start(), matches[-1].end()),
                    detail=f"{len(matches)} occurrences",
                    weight=node.get("weight", 1.0),
                )
            ]
        return []
