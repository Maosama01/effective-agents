"""Building block: the augmented LLM.

An LLM enhanced with retrieval, tools and memory. Retrieval and long-term
memory are given to the model as tools, so the model itself decides when to
search, which tool to call and what to remember. The conversation history kept
on the instance is its short-term memory.

Under the hood this is the standard tool-use loop, written out by hand so you
can see every step:

    send history -> model answers or asks for tools -> run tools ->
    append results -> repeat until the model answers.
"""

from __future__ import annotations

from typing import Any, Iterable

from . import trace
from .llm import DEFAULT_MAX_TOKENS, start_conversation
from .tools import Approver, Tool, run_tool


class StepLimitReached(RuntimeError):
    """The loop hit max_steps before the model finished."""


class AugmentedLLM:
    def __init__(
        self,
        system: str,
        tools: Iterable[Tool] = (),
        *,
        model: str | None = None,
        effort: str | None = None,
        max_steps: int = 10,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        approve: Approver | None = None,
        show_progress: bool = False,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.approve = approve
        self.conversation = start_conversation(
            system,
            self.tools.values(),
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )

    @property
    def messages(self) -> list[Any]:
        """The full conversation history, in the provider's own format."""
        return self.conversation.messages

    def run(self, user_input: str) -> str:
        """Add a user message and loop until the model gives a final answer."""
        self.conversation.add_user(user_input)

        for _ in range(self.max_steps):
            turn = self.conversation.next()  # one model call
            trace.turn(turn)

            if turn.paused:  # a server-side tool paused; let it continue
                continue
            if not turn.tool_calls:
                return turn.text

            results = []
            for call in turn.tool_calls:
                result = run_tool(self.tools, call, self.approve)
                trace.tool_result(result)
                results.append(result)
            self.conversation.add_tool_results(results)

        raise StepLimitReached(f"Stopped after {self.max_steps} model calls without a final answer.")


def chat(llm: AugmentedLLM, greeting: str = "") -> None:
    """A minimal terminal chat loop around any AugmentedLLM (or Agent)."""
    if greeting:
        print(greeting)
    print("(type 'quit' to exit)")
    while True:
        try:
            user_input = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input.lower() in {"quit", "exit", "q"}:
            return
        if not user_input:
            continue
        try:
            answer = llm.run(user_input)
        except StepLimitReached as exc:
            answer = str(exc)
        print(f"\nassistant › {answer}")
