"""Workflow: parallelization.

Run several LLM calls at the same time and combine their results in code.
Two variations:

* Sectioning: split a task into independent parts that run in parallel,
  e.g. one call answers a user while another screens the request (guardrails).
* Voting: run the same check several times, or with several different
  prompts, and aggregate the votes for a more confident result.

Use it when parts can run concurrently for speed, or when several attempts or
perspectives give a better answer. Separate, focused calls usually beat one
call juggling every consideration.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field

from .. import trace
from ..llm import call_llm, call_structured

T = TypeVar("T")


def run_parallel(jobs: list[Callable[[], T]], max_workers: int = 8) -> list[T]:
    """Run zero-argument functions concurrently; results keep the input order."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        futures = [pool.submit(job) for job in jobs]
        return [future.result() for future in futures]


def sectioning(input_text: str, sections: dict[str, str], **llm_options: Any) -> dict[str, str]:
    """Apply a different prompt to the same input for each section, in parallel."""
    names = list(sections)
    outputs = run_parallel(
        [
            lambda prompt=sections[name]: call_llm(
                f"{prompt}\n\n<input>\n{input_text}\n</input>", **llm_options
            )
            for name in names
        ]
    )
    return dict(zip(names, outputs))


class Vote(BaseModel):
    flagged: bool = Field(description="True if you found the problem you were asked to look for.")
    reason: str = Field(description="One or two sentences explaining the vote.")


@dataclass
class VoteResult:
    flagged: bool
    votes: list[Vote]
    threshold: int

    @property
    def count(self) -> int:
        return sum(vote.flagged for vote in self.votes)


def vote(content: str, voter_prompts: list[str], threshold: int = 1, **llm_options: Any) -> VoteResult:
    """Ask each voter prompt about the content in parallel; flag at >= threshold votes.

    A low threshold catches more problems (more false positives); a high one
    flags only when voters agree (more false negatives).
    """
    votes = run_parallel(
        [
            lambda prompt=prompt: call_structured(
                f"{prompt}\n\n<content>\n{content}\n</content>", Vote, **llm_options
            )
            for prompt in voter_prompts
        ]
    )
    for number, ballot in enumerate(votes, 1):
        trace.log("error" if ballot.flagged else "ok", f"voter {number}", ballot.reason)
    result = VoteResult(sum(v.flagged for v in votes) >= threshold, votes, threshold)
    trace.log("step", "tally", f"{result.count}/{len(votes)} flagged (threshold {threshold})")
    return result
