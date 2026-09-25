from __future__ import annotations

_TIER_MARK = {"clean": " ", "watch": "~", "flag": "!"}


def _render_unit(u: dict, depth: int = 0) -> list[str]:
    indent = "  " * depth
    mark = _TIER_MARK.get(u["tier"], "?")
    excerpt = u["excerpt"].replace("\n", " ")
    if len(excerpt) > 100:
        excerpt = excerpt[:97] + "..."
    lines = [f"{indent}- [{mark}] `{u['id']}` p={u['probability']:.2f} ({u['tier']}): {excerpt}"]
    for h in u["rule_hits"][:5]:
        lines.append(f"{indent}    - hit: {h['label']}")
    for child in u["children"]:
        lines.extend(_render_unit(child, depth + 1))
    return lines


def render(result: dict) -> str:
    s = result["summary"]
    cfg = result["config"]
    out = [
        f"# SlopDetector report{' — ' + result['document'] if 'document' in result else ''}",
        "",
        f"Model: `{cfg['model']}` (used: {cfg['used_llm']}) | Granularity: {cfg['granularity']} | "
        f"Threshold: {cfg['threshold']} | Genre: {cfg['genre'] or '-'} | Profile: {cfg['profile']}",
        "",
        f"Units: {s['unit_count']} | Mean probability: {s['mean_probability']:.2f} | "
        f"Clean: {s['tier_counts']['clean']} Watch: {s['tier_counts']['watch']} Flag: {s['tier_counts']['flag']}",
        "",
        "Top patterns: " + (", ".join(f"{p['pattern_id']}({p['count']})" for p in s["top_patterns"]) or "none"),
        "",
        f"Usage this run: {s['usage_totals']['fresh_units']} unit(s) judged fresh via "
        f"{s['usage_totals']['billed_calls']} API call(s) (batch size {cfg['batch_size']}), "
        f"{s['usage_totals']['cached_calls']} cache hit(s) | "
        f"{s['usage_totals']['input_tokens']} input tokens, {s['usage_totals']['output_tokens']} output tokens | "
        f"${s['usage_totals']['cost_usd']:.6f}",
        "",
        "## Units",
        "",
    ]
    for u in result["units"]:
        out.extend(_render_unit(u))
    return "\n".join(out)
