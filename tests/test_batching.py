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
    def fake_judge_batch(items, *, model, api_key, pattern_taxonomy="", timeout=60, max_retries=3):
        calls.append([key for key, _text, _hits in items])
        answers = {}
        for key, text, _hits in items:
            prob = PROB_MAP.get(text, 0.5)
            answers[key] = {"probability": prob, "rationale": "fake"}
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
