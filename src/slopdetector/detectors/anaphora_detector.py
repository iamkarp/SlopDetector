from __future__ import annotations

import re

from .base import Detector, Hit, register

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LEAD_WORDS = re.compile(r"^\s*(\w+\s+\w+)")


@register("anaphora_run")
class AnaphoraRunDetector(Detector):
    """Flags 3+ consecutive sentences sharing the same first two words
    (excessive parallelism), per book-forge narrative pattern #8."""

    def match(self, text: str, node: dict) -> list[Hit]:
        sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s]
        run_lead = None
        run_len = 0
        min_run = node["data"].get("min_run", 3)
        for s in sentences:
            m = _LEAD_WORDS.match(s)
            lead = m.group(1).lower() if m else None
            if lead and lead == run_lead:
                run_len += 1
            else:
                run_lead = lead
                run_len = 1
            if run_len >= min_run:
                return [
                    Hit(
                        pattern_id=node["id"],
                        label=node.get("label", node["id"]),
                        detail=f"{run_len} consecutive sentences opening with {run_lead!r}",
                        weight=node.get("weight", 1.0),
                    )
                ]
        return []
