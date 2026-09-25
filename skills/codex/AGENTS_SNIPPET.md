# slop-detector (Codex usage note)

Standalone CLI at `/Users/jasonkarpeles/Documents/SlopDetector`, same tool
Claude Code uses (see `skills/claude-code/SKILL.md` for the full contract).
No Codex-specific integration exists or is needed — it's a plain shell
command:

```bash
slopdetector <path-to-file> [--granularity paragraph|line] [--threshold 0.55] \
  [--genre literary-fiction|memoir|commercial-fiction|prescriptive-nf] \
  [--profile prose-advisory|surface-gate|marketing] \
  [--format json|markdown] [--no-llm]
```

Needs `OPENROUTER_API_KEY` set in the shell environment (bring your own
key; this tool never bundles one). Reads a file or `--text "..."`, prints
a JSON (or `--format markdown`) report with a probability per paragraph
and, for flagged paragraphs, a `children` array scoring each sentence.
