from __future__ import annotations

import re

from .base import Detector, Hit, register

# New detector, not present in any source skill: AI drafts skew heavily toward
# expanded forms ("they do not" instead of "they don't"). Flag units whose
# contraction ratio falls well below a human-conversational baseline, scaled
# by the configured formality (a formal essay legitimately avoids contractions).
_PAIRS = [
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
    ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"), ("were not", "weren't"),
    ("cannot", "can't"), ("can not", "can't"), ("could not", "couldn't"),
    ("will not", "won't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
    ("has not", "hasn't"), ("have not", "haven't"), ("had not", "hadn't"),
    ("it is", "it's"), ("that is", "that's"), ("there is", "there's"),
    ("they are", "they're"), ("we are", "we're"), ("you are", "you're"), ("I am", "I'm"),
    ("I will", "I'll"), ("we will", "we'll"), ("you will", "you'll"), ("they will", "they'll"),
    ("I have", "I've"), ("we have", "we've"), ("you have", "you've"), ("they have", "they've"),
    ("let us", "let's"),
]

# Below this ratio in casual/neutral text, flag as suspiciously formal-avoidant.
_FORMALITY_FLOOR = {"casual": 0.6, "neutral": 0.35, "formal": 0.0}


@register("contraction_ratio")
class ContractionRatioDetector(Detector):
    def match(self, text: str, node: dict) -> list[Hit]:
        expanded = 0
        contracted = 0
        details = []
        for exp, con in _PAIRS:
            exp_hits = len(re.findall(rf"\b{re.escape(exp)}\b", text, re.IGNORECASE))
            con_hits = len(re.findall(re.escape(con), text, re.IGNORECASE))
            expanded += exp_hits
            contracted += con_hits
            if exp_hits:
                details.append(f"{exp!r}x{exp_hits}")
        total = expanded + contracted
        if total < node["data"].get("min_opportunities", 3):
            return []  # not enough contractable forms in this unit to judge
        ratio = contracted / total
        formality = node.get("_formality", "neutral")
        floor = _FORMALITY_FLOOR.get(formality, _FORMALITY_FLOOR["neutral"])
        if ratio < floor:
            return [
                Hit(
                    pattern_id=node["id"],
                    label=node.get("label", node["id"]),
                    detail=f"contraction ratio {ratio:.2f} < floor {floor:.2f} for formality={formality}; "
                    f"expanded forms: {', '.join(details)}",
                    weight=node.get("weight", 1.0),
                )
            ]
        return []
