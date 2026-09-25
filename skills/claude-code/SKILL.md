---
name: slop-detector
description: >
  Score a document's AI-slop probability paragraph by paragraph, with
  sentence-level drill-down on flagged paragraphs. Backed by an extensible
  rule graph (vocab, constructions, punctuation, rhythm, invented proof,
  register, narrative, fiction fingerprints, filler/hedging, missing
  contractions) plus an OpenRouter judge model (JEV by default). Use when
  asked to check text for AI slop, AI tells, or "does this sound like AI".
---

# slop-detector

Standalone CLI, installed separately from this skill (see repo README:
`/Users/jasonkarpeles/Documents/SlopDetector`). This file only documents
how to call it from Claude Code.

## Invocation

```bash
slopdetector <path-to-file> [--granularity paragraph|line] [--threshold 0.55] \
  [--genre literary-fiction|memoir|commercial-fiction|prescriptive-nf] \
  [--profile prose-advisory|surface-gate|marketing] \
  [--formality formal|neutral|casual] [--format json|markdown] [--no-llm]
```

Requires `OPENROUTER_API_KEY` in the caller's environment (BYOK — never
ships with this tool, never shared when this repo is shared). Without it,
or with `--no-llm`, the tool falls back to rules-only scoring (still
returns a probability for every unit, just without the JEV call).

## Reading the output

JSON: `summary.mean_probability`, `summary.tier_counts`, and `units[]`
(one entry per paragraph/line, always present, never suppressed — each has
`probability`, `tier` (`clean`/`watch`/`flag`), `rule_hits`, and `children`
for any sentence-level drill-down on flagged paragraphs). Markdown mode
renders the same tree indented, for direct human reading.

## When to reach for this

Any time you're asked to check prose, marketing copy, or a manuscript
chapter for AI tells, and want a calibrated probability per paragraph
rather than a binary judgment. For a single line/phrase or a quick eyeball
check, the existing `slopmonster`/`humanizer` skills may be faster; this
tool is for a full-document scan with a persistent, extensible pattern
catalog and a real probability record.
