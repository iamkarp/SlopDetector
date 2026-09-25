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
REASON_KEY_SUFFIX = "__reason"

# Small, human-meaningful, non-overlapping diagnosis categories — not a
# restatement of the pattern graph's internal ids. A "choice" question
# returns a full probability distribution across these, not just a top
# pick, so this gives soft multi-label diagnosis (read the probabilities
# dict) even though only one category is reported as "choice".
REASON_CATEGORIES = {
    "vocabulary": (
        "AI-tell words or phrases (e.g. delve, seamless, testament, 'in today's "
        "fast-paced world', empty buzzwords) cluster in this paragraph."
    ),
    "formulaic_construction": (
        "The paragraph relies on formulaic sentence shapes (not just X but Y, "
        "rule-of-three lists, self-answering questions, negative parallelism) "
        "rather than the vocabulary itself."
    ),
    "rhythm_uniformity": (
        "Sentences run at a monotonous, uniform length and cadence — a string "
        "of short declaratives or repeated openers — rather than natural human "
        "rhythm variance."
    ),
    "register_or_tone": (
        "The tone is promotional, sycophantic, or has chatbot residue "
        "(disclaimers, 'I hope this helps', generic positive conclusions) "
        "rather than a grounded authorial voice."
    ),
    "unsupported_or_vague_content": (
        "The content is hedgy, vague, or unsupported (filler phrases, invented "
        "statistics, weasel-word attribution) rather than concrete and specific."
    ),
    "not_slop": (
        "None of the above meaningfully apply — the paragraph reads as "
        "genuinely human-written with concrete, specific detail."
    ),
}


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


def build_batch_decision_body(
    model: str,
    items: list[tuple[str, str, str]],  # (question_key, text, rule_hits_summary)
    pattern_taxonomy: str | None = None,
    include_reason: bool = True,
) -> dict:
    """Batched form of build_decision_body: N paragraphs judged as N
    independent 'noul' questions in one request (plus, if include_reason,
    one companion 'choice' question per paragraph naming the primary
    diagnosis category — see REASON_CATEGORIES). Jev's `state` object is
    shared across every question in the request, so the taxonomy and the
    reason criteria — the genuinely identical, repeated blobs — go there
    once instead of being repeated N times. Each paragraph's own text stays
    embedded directly in ITS OWN question's `instructions`, not in shared
    state, so there is no ambiguity about which text a given question is
    judging even though all questions are answered from the same request
    context.
    """
    taxonomy = pattern_taxonomy or NON_MECHANICAL_TAXONOMY
    criteria = {
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
    }
    state = {"ai_tell_taxonomy": taxonomy}
    if include_reason:
        state["diagnosis_categories"] = REASON_CATEGORIES
    questions = {}
    for key, text, rule_hits_summary in items:
        questions[key] = {
            "type": "noul",
            "instructions": (
                f"This question judges ONE specific paragraph only — other questions in this "
                f"batch judge different, unrelated paragraphs; do not let them influence each "
                f"other. The paragraph for THIS question:\n\"\"\"\n{text}\n\"\"\"\n\n"
                f"Rule-based hits already found in THIS paragraph: {rule_hits_summary or '(none)'}\n\n"
                "Is this specific paragraph AI-generated slop (LLM-written or heavily "
                "AI-smoothed prose), as opposed to text written by a human without AI "
                "assistance? Weigh density and clustering of tells, not the mere presence of "
                "one common word. Also weigh tells no mechanical rule can catch, described in "
                "state.ai_tell_taxonomy."
            ),
            "criteria": criteria,
        }
        if include_reason:
            questions[key + REASON_KEY_SUFFIX] = {
                "type": "choice",
                "instructions": (
                    f"For the SAME specific paragraph judged by question '{key}' above "
                    f"(reproduced here so this question is self-contained too):\n\"\"\"\n{text}\n\"\"\"\n\n"
                    "If this paragraph shows any AI-slop characteristics at all, which single "
                    "category in state.diagnosis_categories best explains why? If it reads as "
                    "clean, genuinely human-written text, choose not_slop."
                ),
                "criteria": REASON_CATEGORIES,
            }
    return {
        "model": model,
        "state": state,
        "questions": questions,
    }
