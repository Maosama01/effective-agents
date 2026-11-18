"""A stand-in for the Anthropic client, so tests run offline and for free.

Responses are real SDK message objects, so the code under test reads them
exactly as it would read live API responses.
"""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace
from typing import Any, Callable, Literal, get_args, get_origin

from anthropic.types.beta import BetaMessage
from google.genai import types as gtypes
from pydantic import BaseModel

_ids = itertools.count(1)


def tool_use(name: str, input: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": id or f"toolu_{next(_ids)}", "name": name, "input": input}


def message(*blocks: str | dict[str, Any], stop_reason: str | None = None) -> BetaMessage:
    content = [{"type": "text", "text": b} if isinstance(b, str) else b for b in blocks]
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
    return BetaMessage.model_validate(
        {
            "id": f"msg_{next(_ids)}",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
    )


def parsed(obj: BaseModel, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(parsed_output=obj, stop_reason=stop_reason, stop_details=None, content=[])


def fake_instance(model: type[BaseModel]) -> BaseModel:
    """Build a minimal valid instance of any Pydantic model (first Literal option, etc.)."""

    def value(tp: Any) -> Any:
        origin = get_origin(tp)
        if origin is Literal:
            return get_args(tp)[0]
        if origin is list:
            return [value(get_args(tp)[0]), value(get_args(tp)[0])]
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            return fake_instance(tp)
        if tp is bool:
            return True
        if tp in (int, float):
            return 1
        return "fake text"

    return model(**{name: value(field.annotation) for name, field in model.model_fields.items()})


Handler = Callable[[str, dict[str, Any]], Any]


class _Stream:
    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        return self._message


class _Messages:
    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, kind: str, kwargs: dict[str, Any]) -> Any:
        with self._lock:
            # Snapshot the message list: callers keep appending to it.
            self.calls.append((kind, {**kwargs, "messages": list(kwargs.get("messages", []))}))
        return self._handler(kind, kwargs)

    def stream(self, **kwargs: Any) -> _Stream:
        return _Stream(self._record("stream", kwargs))

    def parse(self, **kwargs: Any) -> Any:
        return self._record("parse", kwargs)


class FakeClient:
    def __init__(self, handler: Handler) -> None:
        self.beta = SimpleNamespace(messages=_Messages(handler))

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return self.beta.messages.calls


def scripted(*responses: Any) -> Handler:
    """Return the responses in order. A callable response is called with the request kwargs."""
    queue = list(responses)
    lock = threading.Lock()

    def handler(kind: str, kwargs: dict[str, Any]) -> Any:
        with lock:
            if not queue:
                raise AssertionError(f"Unexpected extra {kind} call")
            response = queue.pop(0)
        return response(kwargs) if callable(response) else response

    return handler


def prompt_of(kwargs: dict[str, Any]) -> str:
    """The text of the first user message in a request (Claude or Gemini)."""
    if "contents" in kwargs:  # Gemini
        contents = kwargs["contents"]
        return contents if isinstance(contents, str) else contents[0].parts[0].text
    content = kwargs["messages"][0]["content"]
    return content if isinstance(content, str) else str(content)


# --- Gemini -------------------------------------------------------------------

def gemini_response(*parts: str | gtypes.Part, finish: str = "STOP", block_reason: str | None = None):
    """A real GenerateContentResponse. Strings become text parts."""
    content = gtypes.Content(role="model", parts=[gtypes.Part(text=p) if isinstance(p, str) else p for p in parts])
    feedback = gtypes.GenerateContentResponsePromptFeedback(block_reason=block_reason) if block_reason else None
    candidates = [] if block_reason else [gtypes.Candidate(content=content, finish_reason=finish)]
    return gtypes.GenerateContentResponse(candidates=candidates, prompt_feedback=feedback)


def gemini_call(name: str, args: dict[str, Any], id: str | None = None, signature: bytes = b"sig") -> gtypes.Part:
    """A function-call part, carrying a thought signature like real Gemini responses do."""
    return gtypes.Part(function_call=gtypes.FunctionCall(name=name, args=args, id=id), thought_signature=signature)


def gemini_thought(text: str) -> gtypes.Part:
    return gtypes.Part(text=text, thought=True)


class _GeminiModels:
    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        with self._lock:
            contents = kwargs["contents"]
            self.calls.append(("generate", {**kwargs, "contents": contents if isinstance(contents, str) else list(contents)}))
        return self._handler("generate", kwargs)


class FakeGemini:
    def __init__(self, handler: Handler) -> None:
        self.models = _GeminiModels(handler)

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return self.models.calls
