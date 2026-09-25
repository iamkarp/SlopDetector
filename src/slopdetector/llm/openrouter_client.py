"""BYOK OpenRouter client for Jev-style "decisions" models. The key is read
once per call, from the environment only, never logged, never persisted,
never bundled with this tool. Sharing this repository shares zero secrets —
every user supplies their own OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
import time

import requests

DECISIONS_API_URL = "https://openrouter.ai/api/alpha/decisions"

_ENV_VAR_CANDIDATES = ("OPENROUTER_API_KEY", "openrouter_api_key")


class OpenRouterAuthError(RuntimeError):
    pass


class OpenRouterModelError(RuntimeError):
    pass


def load_api_key(env_file: str | None = None) -> str:
    if env_file and os.path.isfile(env_file):
        for line in open(env_file, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in _ENV_VAR_CANDIDATES and v.strip():
                return v.strip()
    for var in _ENV_VAR_CANDIDATES:
        val = os.environ.get(var)
        if val:
            return val
    raise OpenRouterAuthError(
        "No OpenRouter API key found. Set OPENROUTER_API_KEY in your environment "
        "(or pass --env-file pointing at a file with one), then retry. "
        "This tool never ships a key — bring your own."
    )


def judge_batch(
    items: list[tuple[str, str, str]],  # (key, text, rule_hits_summary)
    *,
    model: str,
    api_key: str,
    pattern_taxonomy: str = "",
    timeout: int = 60,
    max_retries: int = 3,
) -> dict:
    """Judges 1+ paragraphs in ONE decisions call — each as its own
    independent 'noul' question, sharing one request's fixed overhead (the
    ~500-token AI-tell taxonomy is sent once in `state`, not once per
    paragraph). Returns {"answers": {key: {"probability", "rationale"}},
    "usage": {...}}. `usage` is for the WHOLE request as OpenRouter reports
    it — real metering, not an estimate — and is only attributable to a
    single paragraph when len(items) == 1; batching trades that per-unit
    cost attribution for real, measured token savings in aggregate.

    Raises OpenRouterModelError with a clear message (naming a fallback) on
    repeated failure or a malformed/missing answer — never fails silently
    and never returns a partial result for some keys but not others.
    """
    from .prompts import build_batch_decision_body

    if not items:
        return {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}}

    body = build_batch_decision_body(model, items, pattern_taxonomy or None)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(DECISIONS_API_URL, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            time.sleep(2**attempt)
            continue

        if resp.status_code == 401:
            raise OpenRouterAuthError("OpenRouter rejected the API key (401). Check OPENROUTER_API_KEY.")
        if resp.status_code == 404:
            raise OpenRouterModelError(
                f"Model '{model}' not found on OpenRouter (404). "
                "Pass --model with a known-good decisions-capable slug as a fallback, "
                "or use --no-llm for rules-only scoring."
            )
        if resp.status_code == 429:
            time.sleep(2**attempt + 1)
            continue
        if resp.status_code >= 500:
            last_err = RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:200]}")
            time.sleep(2**attempt)
            continue
        if resp.status_code != 200:
            raise OpenRouterModelError(f"OpenRouter decisions call failed ({resp.status_code}): {resp.text[:300]}")

        try:
            payload = resp.json()
            raw_answers = payload["answers"]
        except (ValueError, KeyError, TypeError) as e:
            raise OpenRouterModelError(f"Unexpected decisions response: {resp.text[:300]}") from e

        answers: dict[str, dict] = {}
        for key, _text, rule_hits_summary in items:
            ans = raw_answers.get(key)
            if not isinstance(ans, dict) or "noul" not in ans:
                raise OpenRouterModelError(
                    f"Batched decisions response is missing an answer for '{key}': {resp.text[:300]}"
                )
            probability = max(0.0, min(1.0, float(ans["noul"])))
            rationale = f"Jev noul={probability:.3f} (batched, {len(items)} paragraph(s) in request)" + (
                f"; rule hits: {rule_hits_summary[:150]}" if rule_hits_summary else ""
            )
            answers[key] = {"probability": probability, "rationale": rationale}

        raw_usage = payload.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("input_tokens", 0),
            "output_tokens": raw_usage.get("output_tokens", 0),
            "cost": raw_usage.get("cost", 0.0),
        }
        return {"answers": answers, "usage": usage}

    raise OpenRouterModelError(f"OpenRouter decisions call failed after {max_retries} attempts: {last_err}")


def judge_paragraph(
    text: str,
    *,
    model: str,
    api_key: str,
    rule_hits_summary: str = "",
    pattern_taxonomy: str = "",
    timeout: int = 30,
    max_retries: int = 3,
) -> dict:
    """Single-paragraph convenience wrapper over judge_batch (a batch of
    one) — kept for simple library/script use. Returns {"probability",
    "rationale", "usage"}; usage is exact here since the batch has one item.
    """
    result = judge_batch(
        [("q", text, rule_hits_summary)],
        model=model,
        api_key=api_key,
        pattern_taxonomy=pattern_taxonomy,
        timeout=timeout,
        max_retries=max_retries,
    )
    answer = result["answers"]["q"]
    return {"probability": answer["probability"], "rationale": answer["rationale"], "usage": result["usage"]}
