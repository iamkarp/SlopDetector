"""The JEV judge question. Jev is an OpenRouter "decisions" model: it takes
typed questions (noul/choice/score) against a state object, not a chat
completion. A "noul" question returns a native 0-1 probability, which is
exactly the "is this slop?" signal this tool wants — no JSON-mode parsing,
no prompt-enforced schema, no repair pass needed.

Includes the taxonomy names for patterns Stage A cannot mechanically detect
(book-forge's "REQUIRES HUMAN OR MODEL REVIEW" list), so Jev's own judgment
covers what regex structurally can't: forced symmetry, empty metaphors,
pseudo-philosophical closings, overly smooth transitions, explanatory
extension, emotional control demonstration, authoritative description,
philosophical asides, clean dialogue, thematic echo chamber, graduated reveal.
"""

from __future__ import annotations

NON_MECHANICAL_TAXONOMY = (
    "forced symmetry (balanced clause structure that reads engineered, not felt), "
    "empty metaphors (figurative language with no concrete referent), "
    "pseudo-philosophical closings (a sentence gesturing at meaning without earning it), "
    "overly smooth transitions (paragraph joins that erase the friction a human writer would leave), "
    "explanatory extension (the sentence explains its own significance instead of trusting the reader), "
    "emotional control demonstration (a character's composure narrated as an achievement), "
    "authoritative description (confident specificity simulating expertise rather than conveying it), "
    "philosophical asides (a generalization dropped in for weight, not story reasons), "
    "clean dialogue (exchanges too tidy/efficient for how people actually talk), "
    "thematic echo chamber (every beat visibly serves one stated theme), "
    "graduated reveal (information paced out in a suspiciously tidy escalation)"
)

QUESTION_KEY = "is_ai_slop"


def build_decision_body(model: str, text: str, rule_hits_summary: str, pattern_taxonomy: str | None = None) -> dict:
    taxonomy = pattern_taxonomy or NON_MECHANICAL_TAXONOMY
    return {
        "model": model,
        "state": {
            "paragraph": text,
            "rule_based_hits_already_found": rule_hits_summary or "(none)",
        },
        "questions": {
            QUESTION_KEY: {
                "type": "noul",
                "instructions": (
                    "Given the paragraph and the rule-based hits already found in it, is this "
                    "text AI-generated slop (LLM-written or heavily AI-smoothed prose), as opposed "
                    "to text written by a human without AI assistance? Weigh density and clustering "
                    "of tells, not the mere presence of one common word — a single instance of a "
                    "common word is not a tell, a cluster of distinct tells in one short paragraph "
                    "is. Also weigh tells no mechanical rule can catch: " + taxonomy + "."
                ),
                "criteria": {
                    "true": (
                        "The paragraph shows AI-slop characteristics: clustered AI-tell vocabulary, "
                        "formulaic sentence constructions, empty rhetorical flourishes, or other "
                        "machine-smoothed patterns — especially when several distinct tells co-occur."
                    ),
                    "false": (
                        "The paragraph reads as human-written: specific concrete detail, natural "
                        "rhythm variance, no clustering of AI tells even if one isolated common "
                        "word appears."
                    ),
                },
            }
        },
    }
