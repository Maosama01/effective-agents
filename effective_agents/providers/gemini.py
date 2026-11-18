"""Gemini, through Google's official Gen AI SDK (google-genai)."""

from __future__ import annotations

import itertools
import os
import sys
import threading
from typing import Any, TypeVar

import httpx2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ..llm import MissingCredentials, RefusalError, ToolCall, ToolResult, Turn

DEFAULT_MODEL = "gemini-3.8-flash"
SMALL_MODEL = "gemini-3.5-flash-lite"

# The project's effort levels mapped to Gemini thinking levels. Not every model
# accepts XHIGH, so the top levels all map to HIGH.
_THINKING_LEVELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}

REQUEST_TIMEOUT_SECONDS = 120

# Finish reasons that mean the answer was blocked rather than completed.
_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "MODEL_ARMOR"}

T = TypeVar("T", bound=BaseModel)

_client: Any = None
_client_lock = threading.Lock()


def get_client() -> Any:
    """The shared client. It reads the key from GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    global _client
    # Locked: parallel workflows call this from several threads at once, and a
    # second client replacing the first would close it mid-request.
    with _client_lock:
        if _client is None:
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                raise MissingCredentials(
                    "No Gemini API key found. Set GEMINI_API_KEY (export GEMINI_API_KEY=...)."
                )
            # Without a timeout the SDK waits forever on a stuck request. With
            # one, a stalled request (common when a model is overloaded) times
            # out and is retried, as are 429s and 5xx errors. The backoff
            # (2, 4, 8, 16, 32s) spans a full minute, so per-minute rate
            # limits have time to reset.
            _client = genai.Client(
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_SECONDS * 1000,  # milliseconds
                    retry_options=types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=40.0),
                )
            )
        return _client


def _config(*, system: str | None, effort: str, max_tokens: int, show_progress: bool = False, **extra: Any):
    return types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(
            thinking_level=_THINKING_LEVELS.get(effort, "MEDIUM"),
            include_thoughts=show_progress,  # thought summaries, shown in the trace
        ),
        **extra,
    )


def _name(reason: Any) -> str:
    return getattr(reason, "value", None) or str(reason)


def _generate(model: str, contents: Any, config: Any) -> Any:
    response = get_client().models.generate_content(model=model, contents=contents, config=config)
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise RefusalError(f"Gemini blocked the prompt ({_name(feedback.block_reason)}).")
    if not response.candidates:
        raise RefusalError("Gemini returned no answer.")
    reason = _name(response.candidates[0].finish_reason)
    if reason in _BLOCKED:
        raise RefusalError(f"Gemini stopped the answer ({reason}).")
    if reason == "MAX_TOKENS":
        print("warning: response hit max_tokens and was cut off", file=sys.stderr)
    return response


def _parts(response: Any) -> list[Any]:
    content = response.candidates[0].content
    return list(content.parts or []) if content else []


def _text(parts: list[Any]) -> str:
    return "".join(part.text for part in parts if part.text and not part.thought).strip()


def complete(prompt: str, *, system: str | None, model: str, effort: str, max_tokens: int) -> str:
    response = _generate(model, prompt, _config(system=system, effort=effort, max_tokens=max_tokens))
    return _text(_parts(response))


def structured(
    prompt: str, output: type[T], *, system: str | None, model: str, effort: str, max_tokens: int
) -> T:
    config = _config(
        system=system,
        effort=effort,
        max_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=output,
    )
    response = _generate(model, prompt, config)
    if isinstance(response.parsed, output):
        return response.parsed
    return output.model_validate_json(_text(_parts(response)))


def declaration(tool: Any) -> Any:
    schema = tool.schema()
    if not schema["properties"]:  # a tool that takes no arguments
        return types.FunctionDeclaration(name=tool.name, description=tool.description)
    return types.FunctionDeclaration(name=tool.name, description=tool.description, parameters_json_schema=schema)


class Conversation:
    """A tool-use conversation in Gemini's content format."""

    def __init__(
        self, system: str, tools: list[Any], *, model: str, effort: str, max_tokens: int, show_progress: bool
    ) -> None:
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = [types.Tool(function_declarations=[declaration(tool) for tool in tools])]
            # We run the tools ourselves, in the loop you can read in augmented.py.
            extra["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        self._config = _config(
            system=system, effort=effort, max_tokens=max_tokens, show_progress=show_progress, **extra
        )
        self._model = model
        self._local_ids = itertools.count(1)
        self._gemini_ids: dict[str, str | None] = {}
        self.messages: list[Any] = []

    def add_user(self, text: str) -> None:
        self.messages.append(types.Content(role="user", parts=[types.Part(text=text)]))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=self._gemini_ids.get(result.call.id),
                    name=result.call.name,
                    response={"error": result.content} if result.is_error else {"result": result.content},
                )
            )
            for result in results
        ]
        # All results for one turn go back together in a single message.
        self.messages.append(types.Content(role="user", parts=parts))

    def next(self) -> Turn:
        response = _generate(self._model, self.messages, self._config)
        content = response.candidates[0].content or types.Content(role="model", parts=[])
        # Keep the model's turn exactly as returned: its thought signatures
        # must be sent back unchanged for multi-step tool use to work.
        self.messages.append(content)
        parts = list(content.parts or [])

        calls = []
        for part in parts:
            if part.function_call:
                call_id = part.function_call.id or f"call_{next(self._local_ids)}"
                self._gemini_ids[call_id] = part.function_call.id
                calls.append(ToolCall(call_id, part.function_call.name, dict(part.function_call.args or {})))
        notes = [part.text for part in parts if part.thought and part.text]
        return Turn(_text(parts), calls, notes)


def friendly_error(exc: BaseException) -> str | None:
    if isinstance(exc, errors.ClientError):
        if exc.code == 429:
            return f"Gemini rate limit or quota reached, even after retrying for a minute: {exc.message}"
        return f"The Gemini API returned {exc.code}: {exc.message}"
    if isinstance(exc, errors.APIError):
        hint = " Try again shortly, or pick another model with --model." if exc.code == 503 else ""
        return f"Gemini server error {exc.code}: {exc.message}{hint}"
    if isinstance(exc, (httpx2.TimeoutException, httpx2.ConnectError)):
        return "Gemini did not respond in time, even after retries. Try again, or pick another model with --model."
    return None
