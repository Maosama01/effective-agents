"""Evaluator-optimizer demo: literary translation with a critic (the article's example).

A translator call drafts a translation, an evaluator call checks it against
explicit criteria and gives feedback, and the translator revises until the
evaluator passes it or the round limit is reached.
"""

from __future__ import annotations

from .. import trace
from ..workflows import optimize

SUMMARY = "Evaluator-optimizer: translate a passage, critique it, revise until it passes."

PASSAGE = (
    "Grandma never threw anything away. Her pantry was a time capsule, her advice came with a side of "
    "soup, and if you left her house hungry, that was on you. She liked to say a watched pot never boils, "
    "but an unwatched one boils over, so keep one eye on the stove and the other on your life."
)

CRITERIA = """- Reads as natural, idiomatic {language} that a native speaker would publish; no awkward literal renderings.
- Idioms and wordplay ("came with a side of soup", "that was on you", "a watched pot never boils") use {language} equivalents that keep both the meaning and the humor.
- Keeps the warm, wry tone and the punchy rhythm of the last sentence.
- Adds nothing and leaves nothing out."""


def add_arguments(parser) -> None:
    parser.add_argument("--language", default="French", help="Target language.")
    parser.add_argument("--text", default=PASSAGE, help="The passage to translate.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum generate/evaluate rounds.")


def main(args) -> None:
    task = f"Translate this passage into {args.language} for a literary magazine.\n\n<passage>\n{args.text}\n</passage>"
    result = optimize(task, CRITERIA.format(language=args.language), max_rounds=args.max_rounds)
    status = "passed review" if result.passed else f"best effort after {len(result.rounds)} rounds"
    trace.show(f"translation ({status}, {len(result.rounds)} round(s))", result.output)
