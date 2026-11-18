from effective_agents import trace
from effective_agents.cli import main
from effective_agents.llm import ToolCall, Turn

GEMINI_NOTE = """Here's my take on the process, thinking through it as if I were working on the task:

**Finding the Customer's Order for Headphones**

Okay, great, I've got the customer ID, `C-1001`. Now we can get down to **business** and find the orders."""


def test_headline_prefers_a_bold_title_line():
    assert trace.headline(GEMINI_NOTE) == "Finding the Customer's Order for Headphones"


def test_headline_falls_back_to_the_first_line():
    assert trace.headline("\n  Checking the refund window next.\nMore detail.") == "Checking the refund window next."


def test_headline_cuts_long_lines():
    line = trace.headline("word " * 100, width=40)
    assert len(line) == 40 and line.endswith("…")


def test_turn_shows_headlines_by_default_and_full_notes_when_verbose(monkeypatch, capsys):
    monkeypatch.setattr(trace, "enabled", True)
    monkeypatch.setattr(trace, "full_notes", False)
    turn = Turn(notes=[GEMINI_NOTE], tool_calls=[ToolCall("1", "list_orders", {"customer_id": "C-1001"})])

    trace.turn(turn)
    short = capsys.readouterr().err
    assert "[progress] Finding the Customer's Order for Headphones\n" in short
    assert "Okay, great" not in short
    assert "list_orders" in short

    monkeypatch.setattr(trace, "full_notes", True)
    trace.turn(turn)
    assert "Okay, great" in capsys.readouterr().err


def test_verbose_flag_turns_on_full_notes(fake, monkeypatch):
    from test_demos import generic_handler

    fake(generic_handler)
    main(["evaluate", "--quiet", "--verbose"])
    assert trace.full_notes is True and trace.max_chars == 0
