"""Three-stage tree pipeline:
  Stage A (rules, free, every unit)      -> deterministic hits + rule_score
  Stage B (JEV call, every unit)         -> probability, always recorded
  Stage C (drill-down, conditional)      -> re-run A+B per sentence, as children
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .config import Config
from .detectors import Hit
from .graph import load_nodes
from .llm import DiskCache, OpenRouterAuthError, OpenRouterModelError, judge_paragraph, load_api_key
from .rules import rule_score_to_pseudo_probability, score_unit
from .text_split import split_sentences, split_units


def _tier(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "flag"
    if probability >= threshold * 0.6:
        return "watch"
    return "clean"


def _hits_summary(hits: list[Hit]) -> str:
    if not hits:
        return ""
    lines = [f"- {h.label}: {h.detail}" for h in hits[:12]]
    if len(hits) > 12:
        lines.append(f"- ...and {len(hits) - 12} more hits")
    return "\n".join(lines)


@dataclass
class ScoredUnit:
    id: str
    text: str
    start_line: int
    end_line: int
    granularity: str
    rule_hits: list[Hit]
    probability: float
    tier: str
    source: str  # "llm" | "rules-only"
    children: list["ScoredUnit"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "excerpt": self.text[:240],
            "start_line": self.start_line,
            "end_line": self.end_line,
            "granularity": self.granularity,
            "probability": round(self.probability, 4),
            "tier": self.tier,
            "score_source": self.source,
            "rule_hits": [
                {"pattern_id": h.pattern_id, "label": h.label, "detail": h.detail, "weight": h.weight}
                for h in self.rule_hits
            ],
            "children": [c.to_dict() for c in self.children],
        }


def _score_one(
    text: str,
    start_line: int,
    end_line: int,
    granularity: str,
    unit_id: str,
    config: Config,
    nodes: list[dict],
    api_key: str | None,
    cache: DiskCache | None,
) -> ScoredUnit:
    rule_score, hits = score_unit(text, granularity, config, nodes)

    if config.use_llm and api_key:
        cached = cache.get(config.model, text) if cache else None
        if cached:
            probability, source = cached["probability"], "llm-cached"
        else:
            try:
                result = judge_paragraph(
                    text,
                    model=config.model,
                    api_key=api_key,
                    rule_hits_summary=_hits_summary(hits),
                )
                probability, source = result["probability"], "llm"
                if cache:
                    cache.set(config.model, text, result)
            except (OpenRouterAuthError, OpenRouterModelError):
                raise
    else:
        probability, source = rule_score_to_pseudo_probability(rule_score), "rules-only"

    return ScoredUnit(
        id=unit_id,
        text=text,
        start_line=start_line,
        end_line=end_line,
        granularity=granularity,
        rule_hits=hits,
        probability=probability,
        tier=_tier(probability, config.threshold),
        source=source,
    )


def scan_text(text: str, config: Config | None = None) -> dict:
    config = config or Config()
    nodes = load_nodes(config.extra_node_dirs)
    units = split_units(text, config.granularity)

    api_key = None
    if config.use_llm:
        # Deliberately not caught here: if the caller wants LLM scoring
        # (use_llm=True, the default) but no key is configured, that's a
        # setup error, not a soft-degrade condition — let it propagate so
        # the CLI can refuse clearly instead of silently returning
        # rules-only numbers that look like JEV numbers. Pass --no-llm to
        # opt into rules-only mode explicitly instead.
        api_key = load_api_key(config.env_file)

    cache = DiskCache(config.cache_dir) if config.use_llm else None

    scored: list[ScoredUnit] = [None] * len(units)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {
            pool.submit(
                _score_one,
                u.text,
                u.start_line,
                u.end_line,
                config.granularity,
                f"{config.granularity}-{u.index}",
                config,
                nodes,
                api_key,
                cache,
            ): u.index
            for u in units
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            scored[idx] = fut.result()

    # Stage C: drill down on flagged units.
    for unit in scored:
        if unit.probability < config.threshold:
            continue
        sentences = split_sentences(unit.text)
        if len(sentences) < 2:
            continue
        children = []
        for i, sent in enumerate(sentences):
            child = _score_one(
                sent,
                unit.start_line,
                unit.end_line,
                "sentence",
                f"{unit.id}-s{i}",
                config,
                nodes,
                api_key,
                cache,
            )
            children.append(child)
        unit.children = children

    doc_probs = [u.probability for u in scored]
    mean_prob = sum(doc_probs) / len(doc_probs) if doc_probs else 0.0
    tier_counts = {"clean": 0, "watch": 0, "flag": 0}
    for u in scored:
        tier_counts[u.tier] += 1

    pattern_freq: dict[str, int] = {}
    for u in scored:
        for h in u.rule_hits:
            pattern_freq[h.pattern_id] = pattern_freq.get(h.pattern_id, 0) + 1
    top_patterns = sorted(pattern_freq.items(), key=lambda kv: -kv[1])[:10]

    return {
        "config": {
            "model": config.model,
            "granularity": config.granularity,
            "threshold": config.threshold,
            "genre": config.genre,
            "profile": config.profile,
            "used_llm": config.use_llm and api_key is not None,
        },
        "summary": {
            "unit_count": len(scored),
            "mean_probability": round(mean_prob, 4),
            "tier_counts": tier_counts,
            "top_patterns": [{"pattern_id": pid, "count": c} for pid, c in top_patterns],
        },
        "units": [u.to_dict() for u in scored],
    }


def scan_document(path: str, config: Config | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    result = scan_text(text, config)
    result["document"] = path
    return result
