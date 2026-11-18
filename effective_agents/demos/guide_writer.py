"""Orchestrator-workers demo: write a guide whose sections the orchestrator chooses.

The orchestrator reads the task and decides which sections it needs (these
change with the task, which is what separates this from plain
parallelization), workers write the sections in parallel, and a final call
combines them into one document.
"""

from __future__ import annotations

from .. import trace
from ..workflows import orchestrate

SUMMARY = "Orchestrator-workers: plan subtasks for a writing task, run workers, synthesize."

DEFAULT_TASK = (
    "Write a practical one-page guide for a 6-person startup deciding whether to split their monolithic "
    "web app into microservices."
)


def add_arguments(parser) -> None:
    parser.add_argument("--task", default=DEFAULT_TASK, help="The writing or research task.")
    parser.add_argument("--max-subtasks", type=int, default=4, help="Upper limit on worker subtasks.")


def main(args) -> None:
    result = orchestrate(args.task, max_subtasks=args.max_subtasks)
    plan = "\n".join(f"{n}. {st.title}" for n, st in enumerate(result.plan.subtasks, 1))
    trace.show("orchestrator plan", f"{result.plan.approach}\n\n{plan}")
    trace.show("final document", result.final)
