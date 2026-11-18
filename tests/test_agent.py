import pytest

from effective_agents import Agent, AugmentedLLM, StepLimitReached, Tool, ToolError
from effective_agents.providers import claude
from fakes import message, scripted, tool_use


def adder(calls=None):
    def add(a: int, b: int) -> int:
        if calls is not None:
            calls.append((a, b))
        return a + b

    return Tool(
        "add",
        "Add two integers.",
        {"a": {"type": "integer"}, "b": {"type": "integer"}},
        add,
    )


def test_tool_loop_runs_tools_and_returns_final_answer(fake):
    client = fake(scripted(message("Let me add.", tool_use("add", {"a": 2, "b": 3}, id="t1")), message("It is 5.")))

    assistant = AugmentedLLM("You add numbers.", [adder()])
    assert assistant.run("What is 2 + 3?") == "It is 5."

    roles = [m["role"] for m in assistant.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert assistant.messages[2]["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "5"}]

    tool_spec = client.calls[0][1]["tools"][0]
    assert tool_spec["strict"] is True
    assert tool_spec["input_schema"]["additionalProperties"] is False
    # The same tool definitions are sent on every request.
    assert client.calls[0][1]["tools"] == client.calls[1][1]["tools"]


def test_all_tool_results_of_one_turn_go_back_in_one_message(fake):
    fake(
        scripted(
            message(tool_use("add", {"a": 1, "b": 1}, id="x"), tool_use("add", {"a": 2, "b": 2}, id="y")),
            message("2 and 4"),
        )
    )
    assistant = AugmentedLLM("s", [adder()])
    assistant.run("two sums")

    results = assistant.messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["x", "y"]
    assert [r["content"] for r in results] == ["2", "4"]


def test_conversation_history_is_kept_between_runs(fake):
    client = fake(scripted(message("Hi Ada."), message("Your name is Ada.")))
    assistant = AugmentedLLM("s")

    assistant.run("I'm Ada.")
    assistant.run("What's my name?")

    assert len(client.calls[1][1]["messages"]) == 3


def test_declined_approval_skips_the_tool(fake):
    calls = []
    risky = adder(calls)
    risky.needs_approval = True
    fake(scripted(message(tool_use("add", {"a": 1, "b": 1})), message("OK, I won't.")))

    assistant = AugmentedLLM("s", [risky], approve=lambda tool, args: False)
    assistant.run("add")

    result = assistant.messages[2]["content"][0]
    assert calls == []
    assert result["is_error"] is True and "declined" in result["content"]


def test_tool_errors_are_reported_back_to_the_model(fake):
    def broken(path: str) -> str:
        raise ToolError("Paths must be absolute.")

    def crashing(path: str) -> str:
        raise KeyError("boom")

    tools = [
        Tool("broken", "d", {"path": {"type": "string"}}, broken),
        Tool("crashing", "d", {"path": {"type": "string"}}, crashing),
    ]
    fake(
        scripted(
            message(tool_use("broken", {"path": "x"}), tool_use("crashing", {"path": "x"}), tool_use("missing", {})),
            message("Recovered."),
        )
    )

    assistant = AugmentedLLM("s", tools)
    assert assistant.run("go") == "Recovered."

    results = assistant.messages[2]["content"]
    assert all(r["is_error"] for r in results)
    assert results[0]["content"] == "Paths must be absolute."
    assert results[1]["content"].startswith("KeyError")
    assert "Unknown tool" in results[2]["content"]


def test_step_limit_stops_a_runaway_loop(fake):
    fake(lambda kind, kwargs: message(tool_use("add", {"a": 1, "b": 1})))

    with pytest.raises(StepLimitReached):
        AugmentedLLM("s", [adder()], max_steps=3).run("loop forever")

    agent = Agent("s", [adder()], max_steps=3, approve=None)
    assert agent.run_task("loop forever").startswith("[stopped]")


def test_agent_adds_guidelines_and_shows_progress(fake):
    client = fake(scripted(message("Done.")))

    agent = Agent("You are a test agent.", [adder()], model="claude-opus-5-5")
    assert agent.run_task("do it") == "Done."

    request = client.calls[0][1]
    assert request["system"].startswith("You are a test agent.")
    assert "state your plan" in request["system"]
    assert request["thinking"]["display"] == "updates"
    assert claude.PROGRESS_BETA in request["betas"]
