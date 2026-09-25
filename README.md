# SlopDetector

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

## Bring your own key (BYOK)

This tool never ships an API key and never logs or persists one. Set your
own before running:

```bash
cp .env.example .env   # then fill in OPENROUTER_API_KEY=
export OPENROUTER_API_KEY=sk-...
```

Sharing this repository with someone else shares zero secrets — they
supply their own key the same way. `.env` is gitignored.

## Usage

```bash
slopdetector manuscript.md --genre literary-fiction --format markdown
slopdetector --text "In today's fast-paced world, we delve into..." --no-llm
```

Key flags: `--granularity paragraph|line`, `--threshold 0.0-1.0`,
`--genre`, `--profile prose-advisory|surface-gate|marketing`,
`--formality formal|neutral|casual` (gates the missing-contractions check),
`--model <openrouter-slug>`, `--no-llm` (rules-only, no API call),
`--format json|markdown`, `--report FILE`.

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
