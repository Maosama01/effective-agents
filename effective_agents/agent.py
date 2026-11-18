"""Agents: LLMs that direct their own process and tool use.

The article's key observation is that agents are usually simple: an LLM using
tools in a loop, guided by feedback from its environment. So ``Agent`` is a
thin layer over ``AugmentedLLM`` that adds what the article says agents need:

* ground truth from the environment at each step (tool results);
* human checkpoints before risky actions (``approve``);
* a stopping condition (``max_steps``);
* transparency: the agent states its plan and its progress notes are shown.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable

from . import trace
from .augmented import AugmentedLLM, StepLimitReached
from .tools import Approver, Tool

AGENT_GUIDELINES = """

How to work:
- Before your first action, state your plan in one or two sentences.
- Use your tools to get ground truth. Check each result before the next step; never assume an action worked.
- If something fails, read the error, adjust, and try a different approach.
- If you are blocked, or a decision belongs to the user, stop and ask them.
- When you are done, finish with a short summary: what you did, what you found, and anything left for the user."""


def ask_human(tool: Tool, args: dict[str, Any]) -> bool:
    """Checkpoint: show the pending action and ask the person at the terminal."""
    print(f"\n⚠  The agent wants to run `{tool.name}`:", file=sys.stderr)
    for key, value in args.items():
        print(f"     {key}: {value}", file=sys.stderr)
    try:
        answer = input("   Allow it? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def auto_approve(tool: Tool, args: dict[str, Any]) -> bool:
    trace.log("info", "auto-approved", tool.name)
    return True


class Agent(AugmentedLLM):
    def __init__(
        self,
        system: str,
        tools: Iterable[Tool],
        *,
        model: str | None = None,
        effort: str | None = None,
        max_steps: int = 30,
        max_tokens: int = 64000,
        approve: Approver | None = ask_human,
        guidelines: str = AGENT_GUIDELINES,
    ) -> None:
        super().__init__(
            system.rstrip() + guidelines,
            tools,
            model=model,
            effort=effort,
            max_steps=max_steps,
            max_tokens=max_tokens,
            approve=approve,
            show_progress=True,
        )

    def run_task(self, task: str) -> str:
        """Work on a task until done, blocked, or out of steps."""
        trace.log("step", "task", task)
        try:
            return self.run(task)
        except StepLimitReached as exc:
            return f"[stopped] {exc}"
