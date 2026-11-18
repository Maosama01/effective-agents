import pytest

from effective_agents import llm, trace
from effective_agents.providers import claude, gemini
from fakes import FakeClient, FakeGemini


def _isolate(monkeypatch, provider):
    """Pin the provider and clear model overrides; restored after each test."""
    monkeypatch.setattr(trace, "enabled", False)
    monkeypatch.setattr(trace, "full_notes", False)
    monkeypatch.setattr(trace, "max_chars", 800)
    monkeypatch.setattr(llm, "PROVIDER", provider)
    monkeypatch.setattr(llm, "MODEL", None)
    monkeypatch.setattr(llm, "SMALL_MODEL", None)


@pytest.fixture
def fake(monkeypatch):
    """Use a fake Claude client: fake(handler) -> FakeClient."""
    _isolate(monkeypatch, "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    def install(handler):
        client = FakeClient(handler)
        monkeypatch.setattr(claude, "_client", client)
        return client

    return install


@pytest.fixture
def fake_gemini(monkeypatch):
    """Use a fake Gemini client: fake_gemini(handler) -> FakeGemini."""
    _isolate(monkeypatch, "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")

    def install(handler):
        client = FakeGemini(handler)
        monkeypatch.setattr(gemini, "_client", client)
        return client

    return install
