"""Workflow: prompt chaining.

Split a task into a fixed sequence of steps; each LLM call works on the
previous call's output. Between steps, plain code can check ("gate") the
intermediate result and stop the chain early if it has gone off track.

Use it when a task splits cleanly into fixed subtasks: you trade some latency
for accuracy by making each call an easier job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .. import trace
from ..llm import call_llm

# A gate inspects a step's output and returns a problem description,
# or None when the output is fine.
Gate = Callable[[str], "str | None"]


@dataclass
class Step:
    prompt: str
    gate: Gate | None = None
    name: str = ""


class GateFailed(RuntimeError):
    def __init__(self, step: int, name: str, problem: str, output: str) -> None:
        super().__init__(f"Step {step} ({name or 'unnamed'}) failed its check: {problem}")
        self.step, self.name, self.problem, self.output = step, name, problem, output


def chain(input_text: str, steps: list[Step | str], **llm_options: Any) -> str:
    """Run the steps in order, feeding each output into the next step."""
    result = input_text
    for number, step in enumerate(steps, 1):
        if isinstance(step, str):
            step = Step(step)
        trace.log("step", f"step {number}{': ' + step.name if step.name else ''}")
        result = call_llm(f"{step.prompt}\n\n<input>\n{result}\n</input>", **llm_options)
        trace.log("result", "output", result)

        if step.gate:
            problem = step.gate(result)
            if problem:
                trace.log("error", "gate", problem)
                raise GateFailed(number, step.name, problem, result)
            trace.log("ok", "gate", "passed")
    return result
