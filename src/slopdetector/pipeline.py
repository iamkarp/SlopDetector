"""Three-stage tree pipeline:
  Stage A (rules, free, every unit)      -> deterministic hits + rule_score
  Stage B (JEV call, every unit)         -> probability, always recorded,
                                             batched config.batch_size units
                                             per call to cut fixed overhead
  Stage C (drill-down, conditional)      -> re-run A+B per sentence, as
                                             children; every flagged
                                             paragraph's sentences are pooled
                                             into one batched dispatch pass
                                             across the whole document
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .config import Config
from .detectors import Hit
from .graph import load_nodes
from .llm import DiskCache, cache_key, judge_batch, load_api_key
from .rules import rule_score_to_pseudo_probability, score_unit
from .text_split import split_sentences_with_spans, split_units


def _tier(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "flag"
    if probability >= threshold * 0.6:
        return "watch"
    return "clean"


def _hits_summary(hits: list[Hit]) -> str:
    # Only pass JEV hits that actually counted toward the rule score in this
    # context. A pattern zeroed out for the active genre (e.g. a fiction
    # fingerprint under --genre prescriptive-nf) still gets *detected* and
    # still shows up in the report for transparency, but it should not also
    # get whispered to JEV as "evidence" once we've decided it doesn't apply
    # here — that would just reintroduce the genre mismatch through the
    # back door of the prompt instead of the rule_score.
    meaningful = [h for h in hits if h.weight > 0]
    if not meaningful:
        return ""
    lines = [f"- {h.label}: {h.detail}" for h in meaningful[:12]]
    if len(meaningful) > 12:
        lines.append(f"- ...and {len(meaningful) - 12} more hits")
    return "\n".join(lines)


@dataclass
class ScoredUnit:
    id: str
    text: str
    start_line: int
    end_line: int
    granularity: str
    rule_hits: list[Hit]
    probability: float = 0.0
    tier: str = "clean"
    source: str = "rules-only"  # "llm" | "llm-cached" | "rules-only"
    # Real OpenRouter usage for the exact call that scored THIS unit — only
    # set when that call judged this unit alone (batch of 1). Once batched,
    # usage is real but only attributable at the batch level (see
    # summary.usage_totals), not to any single paragraph, so this stays
    # None rather than guess a split.
    usage: dict | None = None
    # {"category", "confidence", "probabilities"} from Jev's companion
    # "choice" question — see prompts.REASON_CATEGORIES. None when
    # config.include_reason is False, or in rules-only/cached-without-it mode.
    reason: dict | None = None
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
            "usage": self.usage,
            "reason": self.reason,
            "rule_hits": [
                {"pattern_id": h.pattern_id, "label": h.label, "detail": h.detail, "weight": h.weight}
                for h in self.rule_hits
            ],
            "children": [c.to_dict() for c in self.children],
        }


def _judge_many(
    unit_specs: list[tuple[str, str, int, int]],  # (uid, text, start_line, end_line)
    granularity: str,
    config: Config,
    nodes: list[dict],
    api_key: str | None,
    cache: DiskCache | None,
) -> tuple[dict[str, ScoredUnit], list[dict]]:
    """Scores every unit in unit_specs (Stage A always; Stage B if
    config.use_llm and a key is available). Returns (ScoredUnit per uid,
    list of real usage records — one per API request actually made).

    Identical text across units (a repeated paragraph, a sentence that also
    stands alone) is judged once: units are grouped by content hash before
    dispatch, so duplicates never cost a second API call. Unique pending
    items are then chunked into config.batch_size-sized decisions calls and
    dispatched with up to config.concurrency requests in flight at once.
    """
    rule_results = {uid: score_unit(text, granularity, config, nodes) for uid, text, _, _ in unit_specs}
    results: dict[str, ScoredUnit] = {}
    batch_usages: list[dict] = []

    if not (config.use_llm and api_key):
        for uid, text, start_line, end_line in unit_specs:
            rule_score, hits = rule_results[uid]
            probability = rule_score_to_pseudo_probability(rule_score)
            results[uid] = ScoredUnit(
                id=uid,
                text=text,
                start_line=start_line,
                end_line=end_line,
                granularity=granularity,
                rule_hits=hits,
                probability=probability,
                tier=_tier(probability, config.threshold),
                source="rules-only",
            )
        return results, batch_usages

    pending_by_key: dict[str, tuple[str, str]] = {}  # content hash -> (text, rule_hits_summary)
    uid_to_key: dict[str, str] = {}
    for uid, text, start_line, end_line in unit_specs:
        _, hits = rule_results[uid]
        key = cache_key(config.model, text)
        uid_to_key[uid] = key
        cached = cache.get(config.model, text) if cache else None
        if cached:
            results[uid] = ScoredUnit(
                id=uid,
                text=text,
                start_line=start_line,
                end_line=end_line,
                granularity=granularity,
                rule_hits=hits,
                probability=cached["probability"],
                tier=_tier(cached["probability"], config.threshold),
                source="llm-cached",
                reason=cached.get("reason"),
            )
        else:
            pending_by_key.setdefault(key, (text, _hits_summary(hits)))

    if pending_by_key:
        pending_items = list(pending_by_key.items())
        batches = [pending_items[i : i + config.batch_size] for i in range(0, len(pending_items), config.batch_size)]

        def run_batch(batch: list[tuple[str, tuple[str, str]]]) -> tuple[list, dict]:
            items = [(key, text, hits) for key, (text, hits) in batch]
            return batch, judge_batch(
                items, model=config.model, api_key=api_key, include_reason=config.include_reason
            )

        answers_by_key: dict[str, dict] = {}
        usage_by_key: dict[str, dict | None] = {}
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            futures = [pool.submit(run_batch, b) for b in batches]
            for fut in as_completed(futures):
                batch, batch_result = fut.result()
                batch_usages.append(batch_result["usage"])
                exact_usage = batch_result["usage"] if len(batch) == 1 else None
                for key, answer in batch_result["answers"].items():
                    answers_by_key[key] = answer
                    usage_by_key[key] = exact_usage
                    if cache:
                        cache.set(config.model, pending_by_key[key][0], answer)

        for uid, text, start_line, end_line in unit_specs:
            if uid in results:
                continue  # resolved from cache above
            key = uid_to_key[uid]
            answer = answers_by_key[key]
            _, hits = rule_results[uid]
            results[uid] = ScoredUnit(
                id=uid,
                text=text,
                start_line=start_line,
                end_line=end_line,
                granularity=granularity,
                rule_hits=hits,
                probability=answer["probability"],
                tier=_tier(answer["probability"], config.threshold),
                source="llm",
                usage=usage_by_key[key],
                reason=answer.get("reason"),
            )

    return results, batch_usages


def _count_unit_sources(units: list[ScoredUnit]) -> tuple[int, int]:
    """(fresh_llm_units, cached_llm_units), walking children too."""
    fresh = cached = 0

    def walk(u: ScoredUnit) -> None:
        nonlocal fresh, cached
        if u.source == "llm":
            fresh += 1
        elif u.source == "llm-cached":
            cached += 1
        for c in u.children:
            walk(c)

    for u in units:
        walk(u)
    return fresh, cached


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

    unit_specs = [(f"{config.granularity}-{u.index}", u.text, u.start_line, u.end_line) for u in units]
    stage_b_results, batch_usages = _judge_many(unit_specs, config.granularity, config, nodes, api_key, cache)
    scored = [stage_b_results[uid] for uid, _text, _sl, _el in unit_specs]

    # Stage C: pool every flagged paragraph's sentences into ONE batched
    # dispatch pass across the whole document (not a loop per paragraph),
    # so drill-down gets the same batching win, and runs concurrently
    # instead of the sequential-per-paragraph shape this used to have.
    sentence_specs: list[tuple[str, str, int, int]] = []
    sentence_owner: dict[str, ScoredUnit] = {}
    for unit in scored:
        if unit.probability < config.threshold:
            continue
        sentence_spans = split_sentences_with_spans(unit.text)
        if len(sentence_spans) < 2:
            continue
        for i, (sent, span_start, span_end) in enumerate(sentence_spans):
            sent_start_line = unit.start_line + unit.text[:span_start].count("\n")
            sent_end_line = unit.start_line + unit.text[:span_end].count("\n")
            suid = f"{unit.id}-s{i}"
            sentence_specs.append((suid, sent, sent_start_line, sent_end_line))
            sentence_owner[suid] = unit

    if sentence_specs:
        stage_c_results, stage_c_usages = _judge_many(sentence_specs, "sentence", config, nodes, api_key, cache)
        batch_usages.extend(stage_c_usages)
        for suid, _text, _sl, _el in sentence_specs:
            sentence_owner[suid].children.append(stage_c_results[suid])

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

    input_tokens = sum(u.get("input_tokens", 0) for u in batch_usages)
    output_tokens = sum(u.get("output_tokens", 0) for u in batch_usages)
    cost = sum(u.get("cost", 0.0) for u in batch_usages)
    fresh_units, cached_units = _count_unit_sources(scored)
    usage_totals = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "billed_calls": len(batch_usages),  # actual HTTP requests made
        "fresh_units": fresh_units,  # units judged fresh this run (can be >> billed_calls when batched)
        "cached_calls": cached_units,
    }

    # profile.gates=True (surface-gate) means any flagged unit fails the
    # document, matching book-forge's 5/5-required marketing-copy gate;
    # profile.gates=False (prose-advisory/marketing) never fails, matching
    # book-forge's advisory-only manuscript-prose behavior.
    gate_passed = not (config.profile_settings().get("gates") and tier_counts["flag"] > 0)

    return {
        "config": {
            "model": config.model,
            "granularity": config.granularity,
            "threshold": config.threshold,
            "genre": config.genre,
            "profile": config.profile,
            "batch_size": config.batch_size,
            "used_llm": config.use_llm and api_key is not None,
        },
        "summary": {
            "unit_count": len(scored),
            "mean_probability": round(mean_prob, 4),
            "tier_counts": tier_counts,
            "gate_passed": gate_passed,
            "top_patterns": [{"pattern_id": pid, "count": c} for pid, c in top_patterns],
            "usage_totals": usage_totals,
        },
        "units": [u.to_dict() for u in scored],
    }


def scan_document(path: str, config: Config | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    result = scan_text(text, config)
    result["document"] = path
    return result
