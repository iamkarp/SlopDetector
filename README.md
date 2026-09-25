# SlopDetector

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![BYOK](https://img.shields.io/badge/OpenRouter%20key-bring%20your%20own-informational)

Extensible AI-slop probability scorer. Scans a document paragraph by
paragraph (or line by line), always records a probability for every unit,
and drills down into sentence-level children for any paragraph that flags.

Two-stage scoring per unit:

1. **Rule graph** (free, instant): regex/word-list/structural checks against
   an extensible pattern catalog (`src/slopdetector/graph/nodes/*.yaml`),
   merged and deduplicated from four prior slop-detection systems: vocab
   tells, sentence-shape constructions, punctuation cadence, rule-of-three
   rhythm, invented social proof, non-prose register patterns, narrative
   patterns, fiction phrase fingerprints, filler/hedging, and missing
   contractions.
2. **JEV judge** (OpenRouter model, default `typesafe/jev-1.13`, swap with
   `--model`): given the rule hits as context, returns a calibrated
   probability + one-sentence rationale for the unit.

Paragraphs scoring at or above `--threshold` (default 0.55) get split into
sentences and re-scored, attached as a `children` tree under the parent —
so a flagged paragraph tells you exactly which sentence is driving it.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Add your own OpenRouter key (BYOK)

This tool never ships an API key and never logs or persists one. Without a
key it still runs (`--no-llm`, rules-only, see below) — a key is only
needed for the JEV probability call.

1. Get a key at [openrouter.ai/keys](https://openrouter.ai/keys) (sign up,
   create a key, add credit — Jev decisions calls are billed per call, a
   few cents covers a whole chapter).
2. Give it to the tool, any one of these:
   - **Export it for the session:**
     ```bash
     export OPENROUTER_API_KEY=sk-or-...
     ```
   - **Put it in a `.env` file in this repo** (gitignored, never committed):
     ```bash
     cp .env.example .env
     # edit .env, set OPENROUTER_API_KEY=sk-or-...
     ```
   - **Point at a key file anywhere else**, e.g. one you already use for
     other tools:
     ```bash
     slopdetector manuscript.md --env-file /path/to/your/openrouter.env
     ```
     (accepts either `OPENROUTER_API_KEY=...` or `openrouter_api_key=...`)
3. Run it. No key configured and `--no-llm` not passed → the tool refuses
   with a clear error naming exactly what's missing, it never silently
   falls back or spends someone else's credit.

Sharing this repository with someone else shares zero secrets — they get
their own key the same way. `.env` is gitignored; `.slopdetector-cache/`
(cached Jev responses, also gitignored) never contains a key either.

## Usage

```bash
slopdetector manuscript.md --genre literary-fiction --format markdown
slopdetector --text "In today's fast-paced world, we delve into..." --no-llm
```

Key flags: `--granularity paragraph|line`, `--threshold 0.0-1.0`,
`--genre`, `--profile prose-advisory|surface-gate|marketing`,
`--formality formal|neutral|casual` (gates the missing-contractions check),
`--model <openrouter-slug>`, `--no-llm` (rules-only, no API call),
`--format json|markdown`, `--report FILE`,
`--batch-size N` (default 10 — N paragraphs judged per JEV call instead of
one call each; `--batch-size 1` reverts to one call per unit),
`--concurrency N` (default 6 — batched calls in flight at once).

Batching is a real, measured win, not a guess: the same 10-paragraph test
document that cost 25 API calls / 18.4K input tokens / $0.00078 at
`--batch-size 1` cost 3 calls / 9.0K input tokens / $0.00038 at the default
`--batch-size 10` — about half the cost, same probabilities (see
`summary.usage_totals` in the JSON output, or the "Usage this run" line in
markdown). The trade-off: `units[].usage` (per-paragraph token/cost) is
only exact when that paragraph's call judged it alone; once batched, cost
is real but only attributable at the batch level.

## Extending the pattern graph

Add a new pattern with zero code changes: drop a YAML file (or a node) into
`src/slopdetector/graph/nodes/`. Each node:

```yaml
patterns:
  - id: my_new_pattern
    category: my_category
    detector: regex            # or wordlist_stemmed / wordlist_exact / density / contraction_ratio / anaphora_run
    label: "human-readable name"
    weight: 0.6
    source: "where this came from"
    data:
      pattern: '\bmy regex\b'
      flags: [IGNORECASE]
```

To add a genuinely new *kind* of check (not expressible as regex/word-list),
add `src/slopdetector/detectors/<name>.py` implementing the `Detector`
protocol (`match(text, node) -> list[Hit]`) and decorate it with
`@register("<name>")`; import it in `detectors/__init__.py`. Reference it
from a node's `detector:` field.

## Both Claude Code and Codex

This package is the whole tool — `skills/claude-code/SKILL.md` and
`skills/codex/AGENTS_SNIPPET.md` are thin invocation docs, not separate
implementations. Same binary, same JSON contract, either caller.

## Tests

```bash
pytest
```

Regression cases in `tests/test_patterns_regex.py` are ported from
slopmonster's `test_deslop.py` (stemming edge cases, literal-sense words
that must not false-positive, proof-noun word-boundary cases, tricolon
discriminator).
