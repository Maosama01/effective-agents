import time

import pytest

from effective_agents.workflows import (
    Evaluation,
    GateFailed,
    Plan,
    Route,
    Step,
    Subtask,
    Vote,
    chain,
    optimize,
    orchestrate,
    route,
    run_parallel,
    vote,
)
from fakes import message, parsed, prompt_of, scripted


def test_chain_feeds_each_output_into_the_next_step(fake):
    client = fake(scripted(message("one"), message("two"), message("three")))

    result = chain("start", ["A", Step("B"), Step("C")])

    assert result == "three"
    prompts = [prompt_of(kwargs) for _, kwargs in client.calls]
    assert "start" in prompts[0]
    assert prompts[1].startswith("B") and "one" in prompts[1]
    assert "two" in prompts[2]


def test_chain_stops_when_a_gate_fails(fake):
    client = fake(scripted(message("too short"), message("never used")))

    with pytest.raises(GateFailed) as failure:
        chain("start", [Step("A", gate=lambda out: "needs bullets", name="points"), "B"])

    assert failure.value.step == 1
    assert failure.value.output == "too short"
    assert len(client.calls) == 1


def test_route_classifies_then_uses_the_routes_prompt_and_model(fake):
    routes = {
        "billing": Route("Money questions", system="BILLING PROMPT"),
        "general": Route("Easy questions", system="GENERAL PROMPT", model="claude-haiku-5-5", effort="low"),
    }
    client = fake(
        scripted(
            lambda kw: parsed(kw["output_format"](route="general", reason="simple question")),
            message("We do have dark mode."),
        )
    )

    result = route("Do you have dark mode?", routes)

    assert (result.route, result.output) == ("general", "We do have dark mode.")
    _, classify_request = client.calls[0]
    assert "Money questions" in prompt_of(classify_request)
    _, answer_request = client.calls[1]
    assert answer_request["system"] == "GENERAL PROMPT"
    assert answer_request["model"] == "claude-haiku-5-5"


def test_run_parallel_runs_concurrently_and_keeps_order():
    def job(n):
        def run():
            time.sleep(0.2)
            return n
        return run

    start = time.perf_counter()
    assert run_parallel([job(n) for n in range(5)]) == [0, 1, 2, 3, 4]
    assert time.perf_counter() - start < 0.8  # 5 x 0.2s sequentially would be 1s


@pytest.mark.parametrize("threshold, flagged", [(1, True), (2, True), (3, False)])
def test_vote_flags_at_threshold(fake, threshold, flagged):
    def handler(kind, kwargs):
        return parsed(Vote(flagged="FLAGGER" in prompt_of(kwargs), reason="r"))

    fake(handler)
    result = vote("some code", ["FLAGGER one", "FLAGGER two", "calm reviewer"], threshold=threshold)

    assert result.count == 2
    assert result.flagged is flagged


def test_orchestrator_plans_runs_workers_and_synthesizes(fake):
    plan = Plan(
        approach="Split by audience.",
        subtasks=[Subtask(title="Costs", instructions="c"), Subtask(title="Risks", instructions="r")],
    )

    def handler(kind, kwargs):
        if kind == "parse":
            return parsed(plan)
        prompt = prompt_of(kwargs)
        if "worker_result" in prompt:
            assert "COSTS TEXT" in prompt and "RISKS TEXT" in prompt
            return message("FINAL DOC")
        return message("COSTS TEXT" if 'title="Costs"' in prompt else "RISKS TEXT")

    fake(handler)
    result = orchestrate("Write a guide")

    assert result.worker_outputs == ["COSTS TEXT", "RISKS TEXT"]
    assert result.final == "FINAL DOC"


def test_optimizer_revises_until_the_evaluator_passes(fake):
    client = fake(
        scripted(
            message("draft 1"),
            parsed(Evaluation(verdict="NEEDS_IMPROVEMENT", feedback="Fix the idiom.")),
            message("draft 2"),
            parsed(Evaluation(verdict="PASS", feedback="Good.")),
        )
    )

    result = optimize("Translate it", "- natural")

    assert result.passed and result.output == "draft 2"
    assert len(result.rounds) == 2
    second_generation = prompt_of(client.calls[2][1])
    assert "draft 1" in second_generation and "Fix the idiom." in second_generation


def test_optimizer_stops_at_max_rounds(fake):
    fail = parsed(Evaluation(verdict="NEEDS_IMPROVEMENT", feedback="no"))
    fake(scripted(message("a"), fail, message("b"), fail))

    result = optimize("t", "c", max_rounds=2)

    assert not result.passed and result.output == "b"
