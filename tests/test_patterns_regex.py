"""Regression cases ported from slopmonster/tools/test_deslop.py: edge cases
the regexes/word-lists must handle correctly (true positives that must fire,
false positives that must NOT fire)."""

from __future__ import annotations

from slopdetector.config import Config
from slopdetector.graph import load_nodes
from slopdetector.rules import score_unit

NODES = load_nodes()


def hit_ids(text: str, granularity: str = "paragraph") -> set[str]:
    _, hits = score_unit(text, granularity, Config(), NODES)
    return {h.pattern_id for h in hits}


def test_stemmed_vocab_catches_inflections():
    assert "vocab_ai_words_stemmed" in hit_ids("This will elevate your workflow.")
    assert "vocab_ai_words_stemmed" in hit_ids("This elevates your workflow.")
    assert "vocab_ai_words_stemmed" in hit_ids("We are elevating the entire team.")


def test_exact_vocab_does_not_over_stem():
    # "crafted" is exact-match on purpose (also has an honest literal sense).
    assert "vocab_ai_words_exact" in hit_ids("This blog post was crafted with care.")


def test_literal_sense_words_do_not_false_positive_on_stemmed_list():
    # "harness" as a literal object, "landscape" as literal geography — these
    # live in vocab_ai_words_exact via specific phrases, not as bare stems,
    # so a plain sentence about an actual landscape must not trip the stemmed list.
    hits = hit_ids("The mountain landscape stretched for miles under a clear sky.")
    assert "vocab_ai_words_stemmed" not in hits


def test_contracted_negation_fires_equally_to_uncontracted():
    a = hit_ids("It isn't just a tool, but a lifestyle.")
    b = hit_ids("It is not just a tool, but a lifestyle.")
    assert "not_just_but" in a
    assert "not_just_but" in b


def test_tricolon_oxford_fires():
    assert "tricolon_oxford" in hit_ids("The plan was bold, ambitious, and daring.")


def test_tricolon_short_service_list_does_not_overfire_generic_rule():
    # book-forge explicitly designs this to score clean.
    hits = hit_ids("We offer inspection, repair and replacement.")
    assert "tricolon_oxford" not in hits


def test_proof_noun_word_boundary_does_not_false_positive():
    hits = hit_ids("Please check the sitemaps and projectors before the event.")
    assert "invented_proof_number" not in hits


def test_proof_does_not_reach_across_a_full_stop():
    hits = hit_ids("Don't Make Me Think, 2000. It's a classic.")
    assert "invented_proof_number" not in hits


def test_proof_fires_on_real_invented_claim():
    assert "invented_proof_number" in hit_ids("Trusted by 10,000+ happy customers worldwide.")


def test_em_dash_hard_zero_fires_on_single_dash_under_surface_gate():
    # Default profile (prose-advisory) tolerates one dash — see
    # test_prose_advisory_profile_tolerates_a_single_em_dash below.
    _, hits = score_unit("This is bold — truly bold.", "paragraph", Config(profile="surface-gate"), NODES)
    assert "em_dash_hard_zero" in {h.pattern_id for h in hits}


def test_em_dash_cluster_requires_two_under_prose_advisory():
    assert "em_dash_cluster" not in hit_ids("This is bold — truly bold.")
    assert "em_dash_cluster" in hit_ids("This — bold — is also — unusual.")


def test_double_hyphen_as_dash_detected():
    assert "double_hyphen_as_dash" in hit_ids("This is bold -- truly bold.")


def test_missing_contractions_needs_minimum_opportunities():
    # Only one contractable form present: not enough signal, must not fire.
    hits = hit_ids("It is fine.", granularity="paragraph")
    assert "missing_contractions" not in hits


def test_missing_contractions_fires_on_casual_text_with_no_contractions():
    text = "They do not want to go. It is not possible. We cannot help you today."
    _, hits = score_unit(text, "paragraph", Config(formality="casual"), NODES)
    ids = {h.pattern_id for h in hits}
    assert "missing_contractions" in ids


def test_missing_contractions_does_not_fire_for_formal_register():
    text = "They do not want to go. It is not possible. We cannot help you today."
    _, hits = score_unit(text, "paragraph", Config(formality="formal"), NODES)
    ids = {h.pattern_id for h in hits}
    assert "missing_contractions" not in ids


def test_anaphora_run_fires_on_three_repeated_openers():
    text = "She walked away slowly. She walked away without looking. She walked away and did not turn back."
    assert "narrative_excessive_parallelism" in hit_ids(text)


def test_anaphora_run_does_not_fire_on_two():
    text = "She walked away slowly. She walked away without looking. He stayed behind."
    assert "narrative_excessive_parallelism" not in hit_ids(text)


def test_surface_gate_profile_uses_zero_tolerance_em_dash_rule():
    _, hits = score_unit("Bold — truly bold.", "paragraph", Config(profile="surface-gate"), NODES)
    ids = {h.pattern_id for h in hits}
    assert "em_dash_hard_zero" in ids
    assert "em_dash_cluster" not in ids  # would double-count the same dash rule


def test_prose_advisory_profile_tolerates_a_single_em_dash():
    _, hits = score_unit("Bold — truly bold.", "paragraph", Config(profile="prose-advisory"), NODES)
    ids = {h.pattern_id for h in hits}
    assert "em_dash_hard_zero" not in ids
    assert "em_dash_cluster" not in ids  # only one dash present, cluster needs 2+


def test_prose_advisory_profile_still_flags_a_dash_cluster():
    _, hits = score_unit("This — bold — is also — unusual.", "paragraph", Config(profile="prose-advisory"), NODES)
    ids = {h.pattern_id for h in hits}
    assert "em_dash_cluster" in ids
    assert "em_dash_hard_zero" not in ids


FINGERPRINT_TEXT = "She keeps looking at the door. He holds still and remains quiet, almost as if he knows."


def test_fingerprint_scores_zero_weight_with_no_genre_specified():
    score, hits = score_unit(FINGERPRINT_TEXT, "paragraph", Config(genre=None), NODES)
    ids = {h.pattern_id for h in hits}
    assert "fingerprint_b_continuation_holding" in ids  # detector still fires...
    assert score == 0.0  # ...but contributes nothing without an explicit fiction genre


def test_fingerprint_scores_zero_weight_for_prescriptive_nf():
    score, _ = score_unit(FINGERPRINT_TEXT, "paragraph", Config(genre="prescriptive-nf"), NODES)
    assert score == 0.0


def test_fingerprint_scores_full_weight_for_literary_fiction():
    score, _ = score_unit(FINGERPRINT_TEXT, "paragraph", Config(genre="literary-fiction"), NODES)
    assert score > 0.0


def test_curly_quotes_skips_footnote_definition_paragraphs():
    text = '[^zillow]: Zillow Group, “Form 8-K,” filed 2 November 2021.'
    hits = hit_ids(text)
    assert "register_curly_quotes" not in hits


def test_curly_quotes_skips_numbered_bibliography_paragraphs():
    text = '1. Murphy, Kevin P. *Probabilistic Machine Learning.* MIT Press, 2022. 2. Bishop, Christopher. “Pattern Recognition.” Springer, 2006.'
    hits = hit_ids(text)
    assert "register_curly_quotes" not in hits


def test_curly_quotes_still_fires_on_ordinary_prose():
    text = 'She said, “I am not sure about this,” and left the room.'
    assert "register_curly_quotes" in hit_ids(text)


def test_genre_suppressed_hit_weight_is_zeroed_not_just_omitted_from_score():
    # rules.score_unit mutates Hit.weight to the resolved (genre-adjusted)
    # value so downstream consumers (the JEV prompt context, the report)
    # can tell "detected but suppressed" apart from "detected and counted".
    _, hits = score_unit(FINGERPRINT_TEXT, "paragraph", Config(genre="prescriptive-nf"), NODES)
    fingerprint_hits = [h for h in hits if h.pattern_id == "fingerprint_b_continuation_holding"]
    assert fingerprint_hits, "detector should still fire and be reported"
    assert all(h.weight == 0.0 for h in fingerprint_hits)


def test_hits_summary_excludes_zero_weight_hits_from_jev_context():
    from slopdetector.pipeline import _hits_summary

    _, hits = score_unit(FINGERPRINT_TEXT, "paragraph", Config(genre="prescriptive-nf"), NODES)
    summary = _hits_summary(hits)
    assert "continuation" not in summary.lower()


def test_hits_summary_still_includes_nonzero_weight_hits():
    from slopdetector.pipeline import _hits_summary

    _, hits = score_unit(FINGERPRINT_TEXT, "paragraph", Config(genre="literary-fiction"), NODES)
    summary = _hits_summary(hits)
    assert "continuation" in summary.lower()


def test_lazy_extremes_does_not_fire_on_hyphenated_technical_terms():
    # Real false positive found scanning a published-quality ML manuscript:
    # "the always-answer baseline" is the author's own defined term, not a
    # hedge-free overclaim.
    text = "Overall test accuracy supplies the always-answer baseline for comparison."
    assert "lazy_extremes" not in hit_ids(text)


def test_lazy_extremes_still_fires_on_bare_quantifiers():
    text = "This always works and every input gets the same treatment, no matter what."
    assert "lazy_extremes" in hit_ids(text)


def test_semicolon_density_downweighted_for_prescriptive_nf():
    # Textbook-correct semicolon usage from a real manuscript: two related
    # independent clauses. Should barely register for formal nonfiction.
    text = (
        "Its accuracy is high; the scores overstate the observed correctness "
        "rate. Risk concerns known probabilities; ambiguity concerns the "
        "uncertainty surrounding those probabilities. Chapter 8 examines "
        "that transfer problem; Chapter 4 first shows how a predictive "
        "model can represent input-dependent variation."
    )
    score_nf, hits_nf = score_unit(text, "paragraph", Config(genre="prescriptive-nf"), NODES)
    score_default, hits_default = score_unit(text, "paragraph", Config(genre=None), NODES)
    nf_semicolon = next(h for h in hits_nf if h.pattern_id == "semicolon_density")
    default_semicolon = next(h for h in hits_default if h.pattern_id == "semicolon_density")
    assert nf_semicolon.weight < default_semicolon.weight
