from __future__ import annotations

from slopdetector.config import Config
from slopdetector.pipeline import scan_text

SLOP_PARAGRAPH = (
    "In today's fast-paced world, we need to delve into the seamless, robust "
    "solution that will elevate your journey. It's not just a product, it's a "
    "game-changing paradigm shift. Whether you're a beginner or an expert, this "
    "cutting-edge platform will empower you."
)

CLEAN_PARAGRAPH = (
    "I walked to the store yesterday because we ran out of milk. The clerk "
    "rang me up without much conversation. Outside it had started to rain."
)


def test_every_paragraph_gets_a_recorded_probability_even_when_clean():
    cfg = Config(use_llm=False)
    result = scan_text(CLEAN_PARAGRAPH, cfg)
    assert len(result["units"]) == 1
    unit = result["units"][0]
    assert "probability" in unit
    assert unit["tier"] == "clean"


def test_flagged_paragraph_drills_down_into_sentence_children():
    cfg = Config(use_llm=False, threshold=0.5)
    result = scan_text(SLOP_PARAGRAPH, cfg)
    unit = result["units"][0]
    assert unit["tier"] == "flag"
    assert len(unit["children"]) >= 2
    for child in unit["children"]:
        assert child["granularity"] == "sentence"
        assert "probability" in child


def test_clean_paragraph_does_not_drill_down():
    cfg = Config(use_llm=False, threshold=0.5)
    result = scan_text(CLEAN_PARAGRAPH, cfg)
    unit = result["units"][0]
    assert unit["children"] == []


def test_multi_paragraph_document_scores_each_paragraph_independently():
    text = f"{SLOP_PARAGRAPH}\n\n{CLEAN_PARAGRAPH}"
    cfg = Config(use_llm=False, threshold=0.5)
    result = scan_text(text, cfg)
    assert len(result["units"]) == 2
    assert result["units"][0]["tier"] == "flag"
    assert result["units"][1]["tier"] == "clean"


def test_line_granularity_splits_by_line_not_paragraph():
    text = f"{SLOP_PARAGRAPH}\n{CLEAN_PARAGRAPH}"
    cfg = Config(use_llm=False, granularity="line")
    result = scan_text(text, cfg)
    assert result["config"]["granularity"] == "line"
    assert len(result["units"]) == 2


def test_summary_counts_match_unit_tiers():
    text = f"{SLOP_PARAGRAPH}\n\n{CLEAN_PARAGRAPH}"
    cfg = Config(use_llm=False, threshold=0.5)
    result = scan_text(text, cfg)
    counts = result["summary"]["tier_counts"]
    assert counts["flag"] + counts["watch"] + counts["clean"] == len(result["units"])
    assert counts["flag"] == 1
    assert counts["clean"] == 1


def test_sentence_children_get_distinct_accurate_line_offsets():
    # Multi-line paragraph: sentence 2 starts on line 2, not line 1.
    multiline = "In today's fast-paced world, we delve into it.\nIt's not just a product, but a lifestyle, too."
    cfg = Config(use_llm=False, threshold=0.3, granularity="paragraph")
    result = scan_text(multiline, cfg)
    unit = result["units"][0]
    assert unit["start_line"] == 1
    assert unit["end_line"] == 2
    assert len(unit["children"]) >= 2
    # children must not all inherit the parent's full [1,2] range identically
    lines = {(c["start_line"], c["end_line"]) for c in unit["children"]}
    assert len(lines) > 1


def test_gate_passed_true_by_default_prose_advisory():
    cfg = Config(use_llm=False, threshold=0.5, profile="prose-advisory")
    result = scan_text(SLOP_PARAGRAPH, cfg)
    assert result["summary"]["gate_passed"] is True


def test_gate_passed_false_under_surface_gate_when_flagged():
    cfg = Config(use_llm=False, threshold=0.5, profile="surface-gate")
    result = scan_text(SLOP_PARAGRAPH, cfg)
    assert result["summary"]["tier_counts"]["flag"] >= 1
    assert result["summary"]["gate_passed"] is False
