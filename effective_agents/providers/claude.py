"""Claude, through the official Anthropic SDK."""

from __future__ import annotations

import sys
import threading
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from ..llm import MissingCredentials, RefusalError, ToolCall, ToolResult, Turn

DEFAULT_MODEL = "claude-opus-5-5"
SMALL_MODEL = "claude-haiku-5-5"

# If a safety classifier declines a request on these models, the API can retry
# it on a fallback model inside the same call instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

# On these models, the short notes written between tool calls come back as
# "progress update" thinking blocks; display="updates" makes their text visible.
PROGRESS_BETA = "thinking-display-updates-2026-08-18"
_PROGRESS_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-sonnet-5-5"}

T = TypeVar("T", bound=BaseModel)

_client: Any = None
_client_lock = threading.Lock()


def get_client() -> Any:
    """The shared client. Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile."""
    global _client
    # Locked: parallel workflows call this from several threads at once.
    with _client_lock:
        if _client is None:
            client = anthropic.Anthropic()
            if not any(getattr(client, name, None) for name in ("api_key", "auth_token", "credentials")):
                raise MissingCredentials(
                    "No Claude API credentials found. Set ANTHROPIC_API_KEY "
                    "(export ANTHROPIC_API_KEY=sk-ant-...) or run `ant auth login`."
                )
            _client = client
        return _client


def _base_request(model: str, effort: str, max_tokens: int, betas: list[str] | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "output_config": {"effort": effort}}
    betas = list(betas or [])
    if model in _FALLBACK_MODELS:
        betas.append(FALLBACK_BETA)
        request["fallbacks"] = "default"
    if betas:
        request["betas"] = betas
    return request


def _check(message: Any) -> None:
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise RefusalError(f"Claude declined this request (category: {category}).")
    if message.stop_reason == "max_tokens":
        print("warning: response hit max_tokens and was cut off", file=sys.stderr)


def _send(request: dict[str, Any]) -> Any:
    # Streamed under the hood so a long answer never hits an HTTP timeout.
    with get_client().beta.messages.stream(**request) as stream:
        message = stream.get_final_message()
    _check(message)
    return message


def _text(message: Any) -> str:
    return "".join(block.text for block in message.content if block.type == "text").strip()


def complete(prompt: str, *, system: str | None, model: str, effort: str, max_tokens: int) -> str:
    request = _base_request(model, effort, max_tokens)
    request["messages"] = [{"role": "user", "content": prompt}]
    if system:
        request["system"] = system
    return _text(_send(request))


def structured(
    prompt: str, output: type[T], *, system: str | None, model: str, effort: str, max_tokens: int
) -> T:
    request = _base_request(model, effort, max_tokens)
    request["messages"] = [{"role": "user", "content": prompt}]
    if system:
        request["system"] = system
    response = get_client().beta.messages.parse(output_format=output, **request)
    _check(response)
    if response.parsed_output is None:
        raise ValueError("Claude did not return output matching the schema.")
    return response.parsed_output


def tool_spec(tool: Any) -> dict[str, Any]:
    # strict=True: the API guarantees the model's arguments match the schema.
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {**tool.schema(), "additionalProperties": False},
        "strict": True,
    }


class Conversation:
    """A tool-use conversation in Anthropic's message format."""

    def __init__(
        self, system: str, tools: list[Any], *, model: str, effort: str, max_tokens: int, show_progress: bool
    ) -> None:
        progress = show_progress and model in _PROGRESS_MODELS
        self._request = _base_request(model, effort, max_tokens, [PROGRESS_BETA] if progress else None)
        if system:
            self._request["system"] = system
        if tools:
            # Built once, so every request sends byte-identical tool definitions.
            # That keeps the prompt cache, and earlier thinking blocks, valid.
            self._request["tools"] = [tool_spec(tool) for tool in tools]
        if progress:
            self._request["thinking"] = {"type": "adaptive", "display": "updates"}
        self._request["cache_control"] = {"type": "ephemeral"}  # each call reuses the cached history
        self.messages: list[dict[str, Any]] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        blocks = []
        for result in results:
            block = {"type": "tool_result", "tool_use_id": result.call.id, "content": result.content}
            if result.is_error:
                block["is_error"] = True
            blocks.append(block)
        # All results for one turn go back together in a single message.
        self.messages.append({"role": "user", "content": blocks})

    def next(self) -> Turn:
        message = _send({**self._request, "messages": self.messages})
        # Keep the full content unchanged, thinking blocks included: the
        # history must only ever grow, never be edited.
        self.messages.append({"role": "assistant", "content": message.content})
        calls = []
        if message.stop_reason == "tool_use":
            calls = [ToolCall(b.id, b.name, b.input) for b in message.content if b.type == "tool_use"]
        notes = [b.thinking for b in message.content if b.type == "thinking" and b.thinking.strip()]
        return Turn(_text(message), calls, notes, paused=message.stop_reason == "pause_turn")


def friendly_error(exc: BaseException) -> str | None:
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Claude API rejected the key. Check ANTHROPIC_API_KEY."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Wait a moment and try again."
    if isinstance(exc, anthropic.APIStatusError):
        return f"The Claude API returned {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check your network connection."
    if isinstance(exc, anthropic.AnthropicError):
        return str(exc)
    return None
