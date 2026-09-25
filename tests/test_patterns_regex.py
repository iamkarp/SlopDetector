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


def test_em_dash_hard_zero_fires_on_single_dash():
    assert "em_dash_hard_zero" in hit_ids("This is bold — truly bold.")


def test_em_dash_cluster_requires_two():
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
