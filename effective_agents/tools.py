"""Tools: how an LLM acts on the world.

Appendix 2 of the article ("Prompt engineering your tools") argues that tool
definitions deserve as much care as prompts. In practice that means:

* write the description like a docstring for a new teammate: what the tool
  does, when to use it, input formats, and edge cases;
* choose parameters that make mistakes hard ("poka-yoke"), e.g. require
  absolute paths instead of relative ones;
* return clear, actionable error messages so the model can recover.

A Tool is provider-neutral; each provider turns it into its own format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .llm import ToolCall, ToolResult

# Called before a tool marked needs_approval runs. Return True to allow it.
Approver = Callable[["Tool", dict[str, Any]], bool]


class ToolError(Exception):
    """Raise from a tool to send a helpful error message back to the model."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]  # name -> JSON Schema; all are required
    function: Callable[..., Any]
    needs_approval: bool = False  # pause for a human before running (a checkpoint)

    def schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input."""
        return {"type": "object", "properties": self.parameters, "required": list(self.parameters)}


def run_tool(tools: dict[str, Tool], call: ToolCall, approve: Approver | None = None) -> ToolResult:
    """Execute one tool call and return its result.

    Failures are returned to the model as errors rather than raised: the error
    text is the environment feedback the model needs to correct itself.
    """
    tool = tools.get(call.name)
    if tool is None:
        return ToolResult(call, f"Unknown tool {call.name!r}. Available: {', '.join(tools)}", True)

    if tool.needs_approval and approve is not None and not approve(tool, call.input):
        return ToolResult(
            call,
            "The user declined this action, so it was not performed. "
            "Do not retry it; ask the user how they would like to proceed.",
            True,
        )

    try:
        output = tool.function(**call.input)
    except ToolError as exc:
        return ToolResult(call, str(exc), True)
    except Exception as exc:  # report unexpected failures to the model too
        return ToolResult(call, f"{type(exc).__name__}: {exc}", True)
    return ToolResult(call, str(output) if output is not None else "Done.")
