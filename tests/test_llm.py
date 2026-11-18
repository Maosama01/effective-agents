"""The provider-neutral facade, and the Claude provider behind it."""

import pytest
from pydantic import BaseModel

from effective_agents import llm
from effective_agents.providers import claude
from fakes import message, parsed, scripted


def test_call_llm_sends_model_effort_and_fallbacks(fake):
    client = fake(scripted(message("Hello there")))

    assert llm.call_llm("Hi", system="Be nice") == "Hello there"

    kind, request = client.calls[0]
    assert kind == "stream"
    assert request["model"] == claude.DEFAULT_MODEL == "claude-opus-5-5"
    assert request["output_config"] == {"effort": llm.DEFAULT_EFFORT}
    assert request["system"] == "Be nice"
    assert request["messages"] == [{"role": "user", "content": "Hi"}]
    assert request["fallbacks"] == "default"
    assert claude.FALLBACK_BETA in request["betas"]


def test_small_model_alias_and_no_fallbacks_where_unsupported(fake):
    client = fake(scripted(message("ok")))

    llm.call_llm("Hi", model=llm.SMALL, effort="low")

    request = client.calls[0][1]
    assert request["model"] == claude.SMALL_MODEL
    assert "fallbacks" not in request
    assert "betas" not in request
    assert request["output_config"] == {"effort": "low"}


def test_text_skips_thinking_blocks():
    msg = message({"type": "thinking", "thinking": "", "signature": "sig"}, "Answer", stop_reason="end_turn")
    assert claude._text(msg) == "Answer"


def test_refusal_raises(fake):
    fake(scripted(message("", stop_reason="refusal")))
    with pytest.raises(llm.RefusalError):
        llm.call_llm("Hi")


def test_call_structured_returns_validated_object(fake):
    class Answer(BaseModel):
        value: int

    client = fake(scripted(parsed(Answer(value=42))))

    assert llm.call_structured("What is 6 * 7?", Answer).value == 42
    kind, request = client.calls[0]
    assert kind == "parse"
    assert request["output_format"] is Answer


def test_conversation_shows_progress_and_caches(fake):
    client = fake(scripted(message("ok")))

    conversation = llm.start_conversation("s", [], model="claude-opus-5-5", show_progress=True)
    conversation.add_user("Hi")
    assert conversation.next().text == "ok"

    request = client.calls[0][1]
    assert request["thinking"] == {"type": "adaptive", "display": "updates"}
    assert claude.PROGRESS_BETA in request["betas"]
    assert request["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"GEMINI_API_KEY": "g"}, "gemini"),
        ({"GOOGLE_API_KEY": "g"}, "gemini"),
        ({"ANTHROPIC_API_KEY": "a"}, "claude"),
        ({"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g"}, "claude"),
        ({}, "claude"),
    ],
)
def test_provider_is_picked_from_the_keys_that_are_set(monkeypatch, env, expected):
    monkeypatch.setattr(llm, "PROVIDER", None)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert llm.provider_name() == expected


def test_explicit_provider_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setattr(llm, "PROVIDER", "gemini")
    assert llm.provider_name() == "gemini"
    monkeypatch.setattr(llm, "PROVIDER", "openai")
    with pytest.raises(ValueError):
        llm.provider_name()
