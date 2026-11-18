"""Prompt-chaining demo: product notes -> selling points -> marketing copy -> translation.

This is the article's first chaining example (write marketing copy, then
translate it), with two programmatic gates in between. If a gate fails, the
chain stops instead of passing bad output along.
"""

from __future__ import annotations

import re

from .. import trace
from ..workflows import GateFailed, Step, chain

SUMMARY = "Prompt chaining: product notes -> selling points -> copy -> translation, with gates."

DEFAULT_PRODUCT = (
    "AquaPulse smart water bottle. 750 ml, double-wall stainless steel, keeps drinks cold for 24 hours. "
    "A ring of light in the lid glows when it's time to drink, based on your activity and the local "
    "weather. Syncs with phone health apps over Bluetooth. Battery lasts 3 weeks; charges over USB-C. "
    "Dishwasher-safe body. Price $49."
)

MAX_WORDS = 90


def has_three_bullets(text: str) -> str | None:
    bullets = [line for line in text.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+", line)]
    return None if len(bullets) >= 3 else f"expected at least 3 bullet points, got {len(bullets)}"


def short_enough(text: str) -> str | None:
    words = len(text.split())
    return None if words <= MAX_WORDS else f"copy is {words} words; the limit is {MAX_WORDS}"


def build_steps(language: str) -> list[Step]:
    return [
        Step(
            "List the three most compelling selling points for a busy office worker. "
            "Reply with a bulleted list only, one point per line, each starting with '- '.",
            gate=has_three_bullets,
            name="selling points",
        ),
        Step(
            "Write upbeat marketing copy of at most 70 words built on these selling points. "
            "End with a short call to action. Reply with the copy only.",
            gate=short_enough,
            name="marketing copy",
        ),
        Step(
            f"Translate this marketing copy into {language}. Keep the tone, the brevity and the call "
            "to action. Reply with the translation only.",
            name=f"translate to {language}",
        ),
    ]


def add_arguments(parser) -> None:
    parser.add_argument("--product", default=DEFAULT_PRODUCT, help="Product notes to start from.")
    parser.add_argument("--language", default="Spanish", help="Language to translate the copy into.")


def main(args) -> None:
    try:
        result = chain(args.product, build_steps(args.language), effort=args.effort)
    except GateFailed as exc:
        trace.show("chain stopped", f"{exc}\n\nOutput that failed the check:\n{exc.output}")
        return
    trace.show(f"marketing copy ({args.language})", result)
