"""The Gemini provider: requests, structured output, and the tool loop."""

import pytest
from google.genai import types
from pydantic import BaseModel

from effective_agents import AugmentedLLM, Tool, llm
from effective_agents.providers import gemini
from fakes import gemini_call, gemini_response, gemini_thought, scripted


def adder():
    return Tool("add", "Add two integers.", {"a": {"type": "integer"}, "b": {"type": "integer"}}, lambda a, b: a + b)


def test_complete_maps_effort_and_system(fake_gemini):
    client = fake_gemini(scripted(gemini_response(gemini_thought("hmm"), "Bonjour")))

    assert llm.call_llm("Say hello", system="Speak French", effort="low") == "Bonjour"

    request = client.calls[0][1]
    assert request["model"] == gemini.DEFAULT_MODEL
    assert request["contents"] == "Say hello"
    assert request["config"].system_instruction == "Speak French"
    assert request["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_small_model_and_top_effort_levels(fake_gemini):
    client = fake_gemini(scripted(gemini_response("ok")))

    llm.call_llm("Hi", model=llm.SMALL, effort="max")

    request = client.calls[0][1]
    assert request["model"] == gemini.SMALL_MODEL
    assert request["config"].thinking_config.thinking_level == types.ThinkingLevel.HIGH


def test_structured_output_is_validated(fake_gemini):
    class Answer(BaseModel):
        value: int

    client = fake_gemini(scripted(gemini_response('{"value": 42}')))

    assert llm.call_structured("What is 6 * 7?", Answer).value == 42
    config = client.calls[0][1]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer


def test_tool_loop_keeps_thought_signatures_and_returns_results(fake_gemini):
    client = fake_gemini(
        scripted(
            gemini_response(gemini_thought("I should add."), gemini_call("add", {"a": 2, "b": 3}, signature=b"S1")),
            gemini_response("It is 5."),
        )
    )

    assistant = AugmentedLLM("You add numbers.", [adder()])
    assert assistant.run("What is 2 + 3?") == "It is 5."

    config = client.calls[0][1]["config"]
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "add"
    assert declaration.parameters_json_schema["required"] == ["a", "b"]
    assert config.automatic_function_calling.disable is True

    second = client.calls[1][1]["contents"]
    model_turn, results = second[1], second[2]
    assert model_turn.role == "model"
    assert model_turn.parts[1].thought_signature == b"S1"  # sent back unchanged
    response = results.parts[0].function_response
    assert (response.name, response.response) == ("add", {"result": "5"})


def test_tool_errors_and_gemini_call_ids_are_passed_back(fake_gemini):
    client = fake_gemini(
        scripted(
            gemini_response(gemini_call("missing", {}, id="fc-7")),
            gemini_response("Sorry."),
        )
    )

    AugmentedLLM("s", [adder()]).run("go")

    response = client.calls[1][1]["contents"][2].parts[0].function_response
    assert response.id == "fc-7"
    assert "Unknown tool" in response.response["error"]


def test_tool_without_arguments_has_no_parameters():
    tool = Tool("recall", "Read memory.", {}, lambda: "nothing")
    assert gemini.declaration(tool).parameters_json_schema is None


def test_blocked_answers_raise_refusal(fake_gemini):
    fake_gemini(scripted(gemini_response("partial", finish="SAFETY"), gemini_response(block_reason="PROHIBITED_CONTENT")))
    with pytest.raises(llm.RefusalError, match="SAFETY"):
        llm.call_llm("bad")
    with pytest.raises(llm.RefusalError, match="PROHIBITED_CONTENT"):
        llm.call_llm("worse")


def test_parallel_first_calls_share_one_client(monkeypatch):
    """Regression: two threads creating clients at once closed one mid-request."""
    import threading
    import time

    created = []

    def slow_client(**kwargs):
        time.sleep(0.05)  # widen the race window
        created.append(object())
        return created[-1]

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(gemini, "_client", None)
    monkeypatch.setattr(gemini.genai, "Client", slow_client)

    seen = []
    threads = [threading.Thread(target=lambda: seen.append(gemini.get_client())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert all(client is created[0] for client in seen)


def test_progress_requests_thought_summaries(fake_gemini):
    client = fake_gemini(scripted(gemini_response(gemini_thought("Planning."), "Done.")))

    conversation = llm.start_conversation("s", [], show_progress=True)
    conversation.add_user("Hi")
    turn = conversation.next()

    assert turn.notes == ["Planning."] and turn.text == "Done."
    assert client.calls[0][1]["config"].thinking_config.include_thoughts is True
