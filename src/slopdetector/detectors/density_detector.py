from __future__ import annotations

import re

from .base import Detector, Hit, register, should_skip

_WORD_RE = re.compile(r"\b\w+\b")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


@register("density")
class DensityDetector(Detector):
    """node.data: {pattern, flags, per_1000_bands: [[max_density, label, weight], ...]}
    Runs a sub-regex, counts hits, divides by unit word count / 1000, picks a band.
    Mirrors book-forge's minor/moderate/strong density weighting (-0.125/-0.25/-0.50).
    """

    def match(self, text: str, node: dict) -> list[Hit]:
        data = node["data"]
        if should_skip(text, data):
            return []
        flags = 0
        for f in data.get("flags", []):
            flags |= getattr(re, f)
        rx = re.compile(data["pattern"], flags)
        matches = list(rx.finditer(text))
        if not matches:
            return []
        wc = max(word_count(text), 1)
        density = len(matches) / (wc / 1000.0)
        for max_density, label, weight in data.get("bands", [[0.5, "minor", 0.125], [1.0, "moderate", 0.25]]):
            if density <= max_density:
                return [
                    Hit(
                        pattern_id=node["id"],
                        label=f"{node.get('label', node['id'])} ({label})",
                        span=(matches[0].start(), matches[-1].end()),
                        detail=f"{len(matches)} hits, {density:.2f}/1000 words",
                        weight=weight,
                    )
                ]
        # Above the highest band ceiling: use the last band's label with "strong" weight
        last_label, last_weight = data.get("bands", [[None, "strong", 0.5]])[-1][1:]
        return [
            Hit(
                pattern_id=node["id"],
                label=f"{node.get('label', node['id'])} (strong)",
                span=(matches[0].start(), matches[-1].end()),
                detail=f"{len(matches)} hits, {density:.2f}/1000 words",
                weight=data.get("strong_weight", 0.5),
            )
        ]
