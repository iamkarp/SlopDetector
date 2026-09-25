from __future__ import annotations

import argparse
import sys

from .config import Config, GENRE_BANDS, PROFILES
from .llm import OpenRouterAuthError, OpenRouterModelError
from .pipeline import scan_document, scan_text
from .report import json_report, markdown_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slopdetector", description="Extensible AI-slop probability scorer.")
    p.add_argument("path", nargs="?", help="File to scan. Omit and use --text for inline input.")
    p.add_argument("--text", help="Scan this literal string instead of a file.")
    p.add_argument("--granularity", choices=["paragraph", "line"], default="paragraph")
    p.add_argument("--threshold", type=float, default=0.55, help="Probability >= threshold triggers sentence drill-down.")
    p.add_argument("--model", default=None, help="OpenRouter model slug (default: config.DEFAULT_MODEL, JEV).")
    p.add_argument("--genre", choices=list(GENRE_BANDS), default=None)
    p.add_argument("--profile", choices=list(PROFILES), default="prose-advisory")
    p.add_argument("--formality", choices=["formal", "neutral", "casual"], default="neutral")
    p.add_argument("--no-llm", action="store_true", help="Rules-only mode, skip the JEV call entirely.")
    p.add_argument("--env-file", default=None, help="Load OPENROUTER_API_KEY from this dotenv-style file.")
    p.add_argument("--concurrency", type=int, default=6, help="Concurrent JEV requests in flight at once.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Units judged per JEV decisions call (1 = one call per unit, matches pre-batching behavior).",
    )
    p.add_argument("--format", choices=["json", "markdown"], default="json")
    p.add_argument("--report", metavar="FILE", help="Write the report to this file instead of stdout.")
    p.add_argument("--node-dir", action="append", default=[], help="Extra directory of pattern YAML files (repeatable).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.path and not args.text:
        print("error: provide a file path or --text", file=sys.stderr)
        return 2

    config = Config(
        granularity=args.granularity,
        threshold=args.threshold,
        genre=args.genre,
        profile=args.profile,
        formality=args.formality,
        use_llm=not args.no_llm,
        env_file=args.env_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        extra_node_dirs=args.node_dir,
    )
    if args.model:
        config.model = args.model

    try:
        if args.text:
            result = scan_text(args.text, config)
        else:
            result = scan_document(args.path, config)
    except (OpenRouterAuthError, OpenRouterModelError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    rendered = json_report.render(result) if args.format == "json" else markdown_report.render(result)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(rendered)
    else:
        print(rendered)

    # profile=surface-gate fails the run (exit 1) if any unit flagged,
    # matching book-forge's 5/5-required convention for non-prose surfaces.
    # prose-advisory/marketing profiles never gate (exit 0 regardless).
    return 0 if result["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
