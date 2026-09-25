"""Batched-decisions-call behavior: multiple units per JEV request, dedup of
identical text, correct usage attribution, and a clean fallback to one call
per unit at batch_size=1. judge_batch and load_api_key are faked here so
these tests never hit the network.
"""

from __future__ import annotations

import pytest

import slopdetector.pipeline as pipeline_module
from slopdetector.config import Config
from slopdetector.pipeline import scan_text

PARA_A = "Alpha paragraph with some distinctive content here to score."
PARA_B = "Bravo paragraph, a different one, also distinctive on its own."
PARA_C = "Charlie paragraph rounds out the trio with its own unique text."

PROB_MAP = {PARA_A: 0.9, PARA_B: 0.2, PARA_C: 0.5}


def _fake_judge_batch_factory(calls: list):
    def fake_judge_batch(
        items, *, model, api_key, pattern_taxonomy="", include_reason=True, timeout=60, max_retries=3
    ):
        calls.append([key for key, _text, _hits in items])
        answers = {}
        for key, text, _hits in items:
            prob = PROB_MAP.get(text, 0.5)
            reason = {"category": "not_slop", "confidence": 0.9, "probabilities": {}} if include_reason else None
            answers[key] = {"probability": prob, "rationale": "fake", "reason": reason}
        usage = {
            "input_tokens": 100 * len(items),
            "output_tokens": 10 * len(items),
            "cost": 0.001 * len(items),
        }
        return {"answers": answers, "usage": usage}

    return fake_judge_batch


@pytest.fixture
def fake_llm(monkeypatch):
    calls: list = []
    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", _fake_judge_batch_factory(calls))
    return calls


def test_multiple_paragraphs_land_in_one_batched_call(fake_llm):
    text = f"{PARA_A}\n\n{PARA_B}\n\n{PARA_C}"
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, threshold=0.99)  # threshold high: skip drill-down
    result = scan_text(text, cfg)

    assert len(fake_llm) == 1  # all 3 paragraphs in a single request
    assert len(fake_llm[0]) == 3
    assert result["summary"]["usage_totals"]["billed_calls"] == 1
    assert result["summary"]["usage_totals"]["fresh_units"] == 3
    probs = {u["excerpt"]: u["probability"] for u in result["units"]}
    assert probs[PARA_A] == pytest.approx(0.9)
    assert probs[PARA_B] == pytest.approx(0.2)


def test_batch_size_one_falls_back_to_one_call_per_unit(fake_llm):
    text = f"{PARA_A}\n\n{PARA_B}\n\n{PARA_C}"
    cfg = Config(use_llm=True, batch_size=1, cache_dir=None, threshold=0.99)
    result = scan_text(text, cfg)

    assert len(fake_llm) == 3  # one request per paragraph
    assert all(len(c) == 1 for c in fake_llm)
    assert result["summary"]["usage_totals"]["billed_calls"] == 3
    assert result["summary"]["usage_totals"]["fresh_units"] == 3


def test_usage_is_exact_only_for_a_true_batch_of_one(fake_llm):
    text = f"{PARA_A}\n\n{PARA_B}"
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, threshold=0.99)
    result = scan_text(text, cfg)
    # Both paragraphs shared one 2-item batch call: usage isn't attributable
    # to either one individually, so per-unit usage must be None.
    for u in result["units"]:
        assert u["usage"] is None

    cfg_single = Config(use_llm=True, batch_size=1, cache_dir=None, threshold=0.99)
    result_single = scan_text(PARA_A, cfg_single)
    # A genuine batch of 1: usage IS attributable to that one paragraph.
    assert result_single["units"][0]["usage"] is not None
    assert result_single["units"][0]["usage"]["input_tokens"] == 100


def test_duplicate_paragraph_text_only_judged_once(fake_llm):
    text = f"{PARA_A}\n\n{PARA_A}\n\n{PARA_B}"  # PARA_A appears twice, verbatim
    # threshold=0.99 avoids drill-down noise in the call count; each PARA_*
    # is single-sentence anyway (nothing to drill into), but be explicit.
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, threshold=0.99)
    result = scan_text(text, cfg)

    assert len(fake_llm) == 1
    assert len(fake_llm[0]) == 2  # 2 unique texts sent, not 3
    assert result["summary"]["unit_count"] == 3  # still 3 paragraphs in the output
    probs = [u["probability"] for u in result["units"]]
    assert probs[0] == probs[1] == pytest.approx(0.9)  # both PARA_A copies got the same answer


def test_cached_units_never_reach_judge_batch(fake_llm, tmp_path):
    # threshold=0.99: PARA_A is single-sentence anyway (nothing to drill
    # into), but keep this explicit rather than relying on that.
    cfg = Config(use_llm=True, batch_size=10, cache_dir=str(tmp_path), threshold=0.99)
    scan_text(PARA_A, cfg)  # first run: real (fake) call, populates cache
    assert len(fake_llm) == 1

    result2 = scan_text(PARA_A, cfg)  # second run: should be a pure cache hit
    assert len(fake_llm) == 1  # no new call
    assert result2["units"][0]["score_source"] == "llm-cached"
    assert result2["summary"]["usage_totals"]["billed_calls"] == 0
    assert result2["summary"]["usage_totals"]["cached_calls"] == 1


def test_stage_c_drilldown_units_are_pooled_into_the_batch_count(fake_llm):
    # A flagged paragraph with 2+ sentences triggers Stage C; those sentence
    # judgments should also flow through judge_batch (pooled with any other
    # flagged paragraphs' sentences), not bypass batching.
    flagged_text = f"{PARA_A} {PARA_B}"  # 2 sentences, will score 0.9 -> flags at low threshold
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, threshold=0.3)
    result = scan_text(flagged_text, cfg)

    assert len(fake_llm) == 2  # one call for the paragraph pass, one for the sentence pass
    assert len(result["units"][0]["children"]) == 2
    assert result["summary"]["usage_totals"]["fresh_units"] == 1 + 2  # 1 paragraph + 2 sentences


def test_reason_is_included_by_default(fake_llm):
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, threshold=0.99)
    result = scan_text(PARA_A, cfg)
    assert result["units"][0]["reason"] == {"category": "not_slop", "confidence": 0.9, "probabilities": {}}


def test_no_reason_flag_skips_it_and_is_passed_through_to_judge_batch(fake_llm, monkeypatch):
    seen_include_reason = []
    original = pipeline_module.judge_batch

    def spy(*args, **kwargs):
        seen_include_reason.append(kwargs.get("include_reason"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "judge_batch", spy)
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, threshold=0.99, include_reason=False)
    result = scan_text(PARA_A, cfg)
    assert seen_include_reason == [False]
    assert result["units"][0]["reason"] is None


def test_cached_reason_is_served_on_cache_hit(fake_llm, tmp_path):
    cfg = Config(use_llm=True, batch_size=10, cache_dir=str(tmp_path), threshold=0.99)
    scan_text(PARA_A, cfg)
    result2 = scan_text(PARA_A, cfg)
    assert result2["units"][0]["score_source"] == "llm-cached"
    assert result2["units"][0]["reason"] == {"category": "not_slop", "confidence": 0.9, "probabilities": {}}


def test_raw_noul_flows_through_fresh_and_cached_paths(monkeypatch, tmp_path):
    # judge_batch (the real one, with reconciliation) is faked here at a
    # lower level than fake_llm's fixture, returning a probability that
    # already differs from raw_noul, to check ScoredUnit/to_dict plumb both
    # through rather than reconcile_probability's math itself (that's
    # covered in test_openrouter_client.py).
    calls = []

    def fake_judge_batch(items, *, model, api_key, pattern_taxonomy="", include_reason=True, timeout=60, max_retries=3):
        calls.append(items)
        answers = {
            key: {"probability": 0.2, "raw_noul": 0.6, "rationale": "fake", "reason": None} for key, _t, _h in items
        }
        usage = {"input_tokens": 10, "output_tokens": 1, "cost": 0.0001}
        return {"answers": answers, "usage": usage}

    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", fake_judge_batch)

    cfg = Config(use_llm=True, batch_size=10, cache_dir=str(tmp_path), threshold=0.99)
    result = scan_text(PARA_A, cfg)
    assert result["units"][0]["probability"] == pytest.approx(0.2)
    assert result["units"][0]["raw_noul"] == pytest.approx(0.6)

    result2 = scan_text(PARA_A, cfg)  # cache hit
    assert result2["units"][0]["score_source"] == "llm-cached"
    assert result2["units"][0]["probability"] == pytest.approx(0.2)
    assert result2["units"][0]["raw_noul"] == pytest.approx(0.6)
    assert len(calls) == 1  # second run never called judge_batch


def _fake_judge_batch_with_reason(probability, not_slop):
    def fake(items, *, model, api_key, pattern_taxonomy="", include_reason=True, timeout=60, max_retries=3):
        reason = {
            "category": "not_slop" if not_slop > 0.5 else "formulaic_construction",
            "confidence": max(not_slop, 1 - not_slop),
            "probabilities": {"not_slop": not_slop},
        }
        answers = {
            key: {"probability": probability, "raw_noul": probability, "rationale": "fake", "reason": reason}
            for key, _t, _h in items
        }
        usage = {"input_tokens": 10, "output_tokens": 1, "cost": 0.0001}
        return {"answers": answers, "usage": usage}

    return fake


def test_low_confidence_flag_is_gated_down_to_watch(monkeypatch):
    # probability crosses the 0.55 default threshold, but the reason
    # question's not_slop sits right at 0.5 (a coin flip) -> certainty is 0,
    # well below the default min_flag_confidence of 0.3 -> demoted.
    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", _fake_judge_batch_with_reason(0.59, 0.5))

    cfg = Config(use_llm=True, batch_size=10, cache_dir=None)
    result = scan_text(PARA_A, cfg)
    unit = result["units"][0]
    assert unit["probability"] == pytest.approx(0.59)
    assert unit["tier"] == "watch"
    assert unit["confidence_gated"] is True
    assert result["summary"]["confidence_gated_count"] == 1
    assert result["summary"]["tier_counts"]["flag"] == 0


def test_high_confidence_flag_is_not_gated(monkeypatch):
    # not_slop very low (0.02) -> certainty near 1.0, well above the floor.
    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", _fake_judge_batch_with_reason(0.9, 0.02))

    cfg = Config(use_llm=True, batch_size=10, cache_dir=None)
    result = scan_text(PARA_A, cfg)
    unit = result["units"][0]
    assert unit["tier"] == "flag"
    assert unit["confidence_gated"] is False
    assert result["summary"]["confidence_gated_count"] == 0


def test_gated_unit_does_not_trigger_sentence_drilldown(monkeypatch):
    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", _fake_judge_batch_with_reason(0.59, 0.5))

    text = f"{PARA_A} {PARA_B}"  # multi-sentence, would normally drill down if trusted as a flag
    cfg = Config(use_llm=True, batch_size=10, cache_dir=None)
    result = scan_text(text, cfg)
    unit = result["units"][0]
    assert unit["confidence_gated"] is True
    assert unit["children"] == []  # gated: not trusted enough to spend calls localizing it


def test_min_flag_confidence_zero_disables_gating(monkeypatch):
    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", _fake_judge_batch_with_reason(0.59, 0.5))

    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, min_flag_confidence=0.0)
    result = scan_text(PARA_A, cfg)
    unit = result["units"][0]
    assert unit["tier"] == "flag"
    assert unit["confidence_gated"] is False


def test_gating_falls_back_to_plain_threshold_without_reason(monkeypatch):
    # include_reason=False -> reason is None -> gating can't evaluate,
    # falls back to the plain threshold comparison (never gates blindly).
    def fake(items, *, model, api_key, pattern_taxonomy="", include_reason=True, timeout=60, max_retries=3):
        answers = {
            key: {"probability": 0.9, "raw_noul": 0.9, "rationale": "fake", "reason": None} for key, _t, _h in items
        }
        return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0}}

    monkeypatch.setattr(pipeline_module, "load_api_key", lambda env_file=None: "fake-key")
    monkeypatch.setattr(pipeline_module, "judge_batch", fake)

    cfg = Config(use_llm=True, batch_size=10, cache_dir=None, include_reason=False)
    result = scan_text(PARA_A, cfg)
    unit = result["units"][0]
    assert unit["tier"] == "flag"
    assert unit["confidence_gated"] is False
