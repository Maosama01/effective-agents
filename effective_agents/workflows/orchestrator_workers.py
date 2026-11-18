"""Workflow: orchestrator-workers.

A central LLM (the orchestrator) looks at the task and decides what the
subtasks are, worker LLMs carry them out in parallel, and a final call
synthesizes their results.

It looks like parallelization, but the subtasks are not fixed in code: the
orchestrator chooses them for each input. Use it for complex tasks where you
can't predict the subtasks ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from .. import trace
from ..llm import call_llm, call_structured
from .parallelization import run_parallel


class Subtask(BaseModel):
    title: str = Field(description="Short name for the subtask.")
    instructions: str = Field(description="Self-contained instructions a worker can follow alone.")


class Plan(BaseModel):
    approach: str = Field(description="One or two sentences on how the task is being split.")
    subtasks: list[Subtask]


@dataclass
class OrchestratorResult:
    plan: Plan
    worker_outputs: list[str]
    final: str


ORCHESTRATOR_PROMPT = """You are coordinating a team of writers and analysts. Break this task into between 2 and {max_subtasks} independent subtasks that can be done in parallel by workers who cannot see each other's work. Choose the subtasks that this particular task needs.

<task>
{task}
</task>"""

WORKER_PROMPT = """You are one worker on a larger task. Do only your subtask, thoroughly and concisely. Do not write an introduction or conclusion for the overall task.

<overall_task>
{task}
</overall_task>

<your_subtask title="{title}">
{instructions}
</your_subtask>"""

SYNTHESIS_PROMPT = """Combine the workers' results below into one finished response to the task. Remove overlap, keep it coherent, and fill any obvious gaps.

<task>
{task}
</task>

{results}"""


def orchestrate(task: str, max_subtasks: int = 5) -> OrchestratorResult:
    trace.log("step", "orchestrator", "planning subtasks")
    plan = call_structured(
        ORCHESTRATOR_PROMPT.format(task=task, max_subtasks=max_subtasks), Plan, effort="high"
    )
    plan.subtasks = plan.subtasks[:max_subtasks]
    trace.log("plan", "approach", plan.approach)
    for number, subtask in enumerate(plan.subtasks, 1):
        trace.log("plan", f"subtask {number}", subtask.title)

    trace.log("step", "workers", f"running {len(plan.subtasks)} in parallel")
    outputs = run_parallel(
        [
            lambda st=subtask: call_llm(
                WORKER_PROMPT.format(task=task, title=st.title, instructions=st.instructions)
            )
            for subtask in plan.subtasks
        ]
    )

    trace.log("step", "synthesizer", "combining results")
    results = "\n\n".join(
        f'<worker_result title="{st.title}">\n{out}\n</worker_result>'
        for st, out in zip(plan.subtasks, outputs)
    )
    final = call_llm(SYNTHESIS_PROMPT.format(task=task, results=results))
    return OrchestratorResult(plan, outputs, final)
