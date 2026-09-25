from __future__ import annotations

import os

import pytest

from slopdetector.llm.openrouter_client import OpenRouterAuthError, load_api_key, reconcile_probability


def test_load_api_key_raises_clear_error_when_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("openrouter_api_key", raising=False)
    with pytest.raises(OpenRouterAuthError, match="bring your own"):
        load_api_key()


def test_load_api_key_reads_standard_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    assert load_api_key() == "sk-test-123"


def test_load_api_key_reads_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("openrouter_api_key", raising=False)
    env_file = tmp_path / "test.env"
    env_file.write_text("openrouter_api_key=sk-from-file\n")
    assert load_api_key(str(env_file)) == "sk-from-file"


def test_load_api_key_never_logs_or_returns_placeholder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-value")
    key = load_api_key()
    assert key == "sk-secret-value"
    assert "secret" not in os.environ.get("SLOPDETECTOR_LOG", "")


def test_reconcile_probability_falls_back_to_bare_noul_without_reason():
    blended, raw = reconcile_probability(0.62, None)
    assert blended == 0.62
    assert raw == 0.62


def test_reconcile_probability_falls_back_when_no_not_slop_key():
    reason = {"category": "vocabulary", "confidence": 0.8, "probabilities": {"vocabulary": 0.8}}
    blended, raw = reconcile_probability(0.62, reason)
    assert blended == 0.62
    assert raw == 0.62


def test_reconcile_probability_averages_noul_with_one_minus_not_slop():
    # Real case from the session: noul=0.30, reason 93% confident not_slop.
    # Expected: (0.30 + (1 - 0.93)) / 2 = (0.30 + 0.07) / 2 = 0.185
    reason = {"category": "not_slop", "confidence": 0.93, "probabilities": {"not_slop": 0.93}}
    blended, raw = reconcile_probability(0.30, reason)
    assert blended == pytest.approx(0.185)
    assert raw == 0.30


def test_reconcile_probability_agrees_when_signals_already_agree():
    # noul says slop-suspicious, reason is confidently NOT not_slop (i.e.
    # confidently something else) -> blend should barely move.
    reason = {"category": "vocabulary", "confidence": 0.9, "probabilities": {"not_slop": 0.05, "vocabulary": 0.9}}
    blended, raw = reconcile_probability(0.9, reason)
    assert blended == pytest.approx((0.9 + 0.95) / 2)
    assert raw == 0.9


def test_reconcile_probability_clamped_to_valid_range():
    reason = {"category": "not_slop", "confidence": 1.0, "probabilities": {"not_slop": 1.0}}
    blended, raw = reconcile_probability(0.0, reason)
    assert 0.0 <= blended <= 1.0
    assert raw == 0.0
