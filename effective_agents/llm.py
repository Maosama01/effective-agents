"""The one door to the model.

Everything else in the project goes through these functions and never touches
an SDK directly. That keeps every request easy to find and inspect (the
article's transparency principle), and lets the same patterns run on more than
one model provider. Each provider lives in ``effective_agents/providers/`` and
implements the same three things:

    complete(prompt)               -> text
    structured(prompt, Model)      -> a validated Pydantic object
    Conversation                   -> a tool-use conversation, one Turn per model call
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Iterable, Protocol, TypeVar

from pydantic import BaseModel

PROVIDERS = ("claude", "gemini")

# Which provider to use. Unset: Gemini if only a Gemini key is set, else Claude.
PROVIDER = os.environ.get("EFFECTIVE_AGENTS_PROVIDER")
# Model overrides. Unset: the provider's own defaults.
MODEL = os.environ.get("EFFECTIVE_AGENTS_MODEL")
SMALL_MODEL = os.environ.get("EFFECTIVE_AGENTS_SMALL_MODEL")
# How hard the model thinks: low, medium, high, xhigh or max.
DEFAULT_EFFORT = os.environ.get("EFFECTIVE_AGENTS_EFFORT", "medium")
DEFAULT_MAX_TOKENS = 16000

# Pass model=SMALL to use the provider's smaller, cheaper model.
SMALL = "small"

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    """The model declined the request, or its answer was blocked."""


class MissingCredentials(RuntimeError):
    """No API key was found for the provider."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    is_error: bool = False


@dataclass
class Turn:
    """What came back from one model call in a tool-use conversation."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # progress notes, shown in the trace
    paused: bool = False  # a server-side tool paused; call next() again to resume


class Conversation(Protocol):
    messages: list[Any]  # the provider's own history, only ever appended to

    def add_user(self, text: str) -> None: ...
    def add_tool_results(self, results: list[ToolResult]) -> None: ...
    def next(self) -> Turn: ...


def provider_name() -> str:
    if PROVIDER:
        if PROVIDER not in PROVIDERS:
            raise ValueError(f"Unknown provider {PROVIDER!r}; choose one of {', '.join(PROVIDERS)}.")
        return PROVIDER
    has_gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    has_claude = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    return "gemini" if has_gemini and not has_claude else "claude"


def get_provider() -> ModuleType:
    return importlib.import_module(f"{__package__}.providers.{provider_name()}")


def resolve_model(model: str | None = None) -> str:
    provider = get_provider()
    if model == SMALL:
        return SMALL_MODEL or provider.SMALL_MODEL
    return model or MODEL or provider.DEFAULT_MODEL


def call_llm(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """One prompt in, one text answer out. The workhorse of every workflow."""
    return get_provider().complete(
        prompt,
        system=system,
        model=resolve_model(model),
        effort=effort or DEFAULT_EFFORT,
        max_tokens=max_tokens,
    )


def call_structured(
    prompt: str,
    output: type[T],
    *,
    system: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> T:
    """Ask for an answer that fits a Pydantic model, and return it validated.

    Routers, screens, voters, planners and evaluators use this so plain code
    can act on what they decide.
    """
    return get_provider().structured(
        prompt,
        output,
        system=system,
        model=resolve_model(model),
        effort=effort or DEFAULT_EFFORT,
        max_tokens=max_tokens,
    )


def start_conversation(
    system: str,
    tools: Iterable[Any] = (),
    *,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    show_progress: bool = False,
) -> Conversation:
    """Start a tool-use conversation (used by AugmentedLLM and Agent)."""
    return get_provider().Conversation(
        system,
        list(tools),
        model=resolve_model(model),
        effort=effort or DEFAULT_EFFORT,
        max_tokens=max_tokens,
        show_progress=show_progress,
    )


def friendly_error(exc: BaseException) -> str | None:
    """A one-line explanation of a known error, or None for anything else."""
    if isinstance(exc, (RefusalError, MissingCredentials)):
        return str(exc)
    for name in PROVIDERS:
        message = importlib.import_module(f"{__package__}.providers.{name}").friendly_error(exc)
        if message:
            return message
    return None
