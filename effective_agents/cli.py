"""Command line: `effective-agents <demo> [options]` (or `python -m effective_agents`)."""

from __future__ import annotations

import argparse
import sys

from . import llm, trace
from .demos import DEMOS

EPILOG = "Run `effective-agents <demo> --help` for a demo's options."


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider",
        choices=llm.PROVIDERS,
        help="Model provider (default: Gemini if only GEMINI_API_KEY is set, otherwise Claude).",
    )
    common.add_argument("--model", help="Model to use (default: the provider's default model).")
    common.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help=f"How hard the model thinks (default: {llm.DEFAULT_EFFORT}).",
    )
    common.add_argument("--quiet", action="store_true", help="Hide the step-by-step trace.")
    common.add_argument(
        "--verbose", action="store_true", help="Show full progress notes and untruncated tool output."
    )

    parser = argparse.ArgumentParser(
        prog="effective-agents",
        description="The patterns from Anthropic's 'Building effective agents', as runnable demos.",
        epilog=EPILOG,
    )
    sub = parser.add_subparsers(dest="demo", metavar="<demo>")
    for name, module in DEMOS.items():
        demo_parser = sub.add_parser(
            name,
            parents=[common],
            help=module.SUMMARY,
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(demo_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help()
        return 0

    if args.provider:
        llm.PROVIDER = args.provider
    if args.model:
        llm.MODEL = args.model
    trace.enabled = not args.quiet
    if args.verbose:
        trace.full_notes = True
        trace.max_chars = 0

    try:
        trace.log("info", "model", f"{llm.provider_name()} · {llm.resolve_model()}")
        DEMOS[args.demo].main(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        message = llm.friendly_error(exc)
        if message is None:
            raise
        print(f"error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
