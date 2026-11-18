"""Workflow: evaluator-optimizer.

One LLM call produces a response; another evaluates it against clear criteria
and gives feedback; the generator revises; repeat until it passes or a round
limit is hit. Much like a writer revising a draft with an editor.

Use it when the criteria are clear and feedback measurably improves the result:
a human's feedback would help, and an LLM can give that kind of feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from .. import trace
from ..llm import call_llm, call_structured


class Evaluation(BaseModel):
    verdict: Literal["PASS", "NEEDS_IMPROVEMENT"]
    feedback: str = Field(
        description="Specific, actionable feedback. If PASS, briefly say why it meets the criteria."
    )


@dataclass
class Round:
    attempt: str
    evaluation: Evaluation


@dataclass
class OptimizationResult:
    output: str
    passed: bool
    rounds: list[Round] = field(default_factory=list)


GENERATOR_PROMPT = """{task}

Your response must meet these criteria:
{criteria}

Reply with the response only, no commentary.{revision}"""

REVISION_PROMPT = """

Your previous attempt did not pass review. Revise it to address the feedback.

<previous_attempt>
{attempt}
</previous_attempt>

<feedback>
{feedback}
</feedback>"""

EVALUATOR_PROMPT = """You are a demanding reviewer. Evaluate the response against every criterion. Return PASS only if all criteria are fully met; otherwise return NEEDS_IMPROVEMENT with concrete feedback.

<task>
{task}
</task>

<criteria>
{criteria}
</criteria>

<response>
{attempt}
</response>"""


def optimize(task: str, criteria: str, max_rounds: int = 3) -> OptimizationResult:
    result = OptimizationResult(output="", passed=False)
    revision = ""
    for number in range(1, max_rounds + 1):
        trace.log("step", f"round {number}", "generating")
        attempt = call_llm(GENERATOR_PROMPT.format(task=task, criteria=criteria, revision=revision))
        trace.log("result", "attempt", attempt)

        evaluation = call_structured(
            EVALUATOR_PROMPT.format(task=task, criteria=criteria, attempt=attempt), Evaluation
        )
        trace.log("ok" if evaluation.verdict == "PASS" else "error", evaluation.verdict, evaluation.feedback)

        result.rounds.append(Round(attempt, evaluation))
        result.output = attempt
        if evaluation.verdict == "PASS":
            result.passed = True
            break
        revision = REVISION_PROMPT.format(attempt=attempt, feedback=evaluation.feedback)
    return result
