from __future__ import annotations

import os

import pytest

from slopdetector.llm.openrouter_client import OpenRouterAuthError, load_api_key


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
