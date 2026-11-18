"""The demo tools work for real, and every demo runs end to end through the CLI."""

import json
from pathlib import Path

import pytest

from effective_agents.cli import build_parser, main
from effective_agents.demos import DEMOS, coding_agent, hr_assistant, marketing_chain, support_agent
from effective_agents.providers import claude, gemini
from effective_agents.tools import ToolError
from fakes import fake_instance, gemini_response, message, parsed

FIXED_CART = coding_agent.SEED_PROJECT["cart.py"].replace(
    'sum(item["price"] for item in items)', 'sum(item["price"] * item["qty"] for item in items)'
).replace("amount * percent / 100", "amount * (1 - percent / 100)").replace(
    "amount > FREE_SHIPPING_THRESHOLD", "amount >= FREE_SHIPPING_THRESHOLD"
)


def tools_by_name(tools):
    return {tool.name: tool.function for tool in tools}


# --- coding agent -----------------------------------------------------------

def test_coding_tools_fix_the_seed_project(tmp_path):
    workspace = tmp_path.resolve()
    for name, content in coding_agent.SEED_PROJECT.items():
        (workspace / name).write_text(content)
    tools = tools_by_name(coding_agent.make_coding_tools(workspace))
    cart = str(workspace / "cart.py")

    assert str(workspace / "test_cart.py") in tools["list_files"](str(workspace))
    assert "TESTS FAILED" in tools["run_tests"]()  # the seed project really is broken

    assert "def subtotal" in tools["read_file"](cart)
    tools["write_file"](cart, FIXED_CART)
    assert tools["run_tests"]().startswith("ALL TESTS PASSED")  # and it is fixable


def test_coding_tools_reject_relative_and_outside_paths(tmp_path):
    tools = tools_by_name(coding_agent.make_coding_tools(tmp_path))

    with pytest.raises(ToolError, match="absolute"):
        tools["read_file"]("cart.py")
    with pytest.raises(ToolError, match="outside the workspace"):
        tools["read_file"]("/etc/hosts")
    with pytest.raises(ToolError, match="outside the workspace"):
        tools["write_file"](str(tmp_path / ".." / "escape.py"), "x")
    with pytest.raises(ToolError, match="does not exist"):
        tools["read_file"](str(tmp_path / "nope.py"))


# --- support agent ----------------------------------------------------------

def test_support_refund_rules():
    store = support_agent.make_store()
    tools = tools_by_name(support_agent.make_support_tools(store))

    jane = json.loads(tools["find_customer"]("JANE@example.com"))
    assert jane["customer_id"] == "C-1001"
    assert "A-5001" in tools["list_orders"]("C-1001")
    assert "Returns and refunds" in tools["search_help_center"]("refund policy")

    with pytest.raises(ToolError, match="only delivered"):
        tools["issue_refund"]("A-5002", 12.99, "not arrived")
    with pytest.raises(ToolError, match="outside the 30-day"):
        tools["issue_refund"]("A-5003", 39.00, "changed mind")
    with pytest.raises(ToolError, match="at most the order total"):
        tools["issue_refund"]("A-5001", 500, "broken")

    assert "issued" in tools["issue_refund"]("A-5001", 89.99, "stopped working")
    with pytest.raises(ToolError, match="already been refunded"):
        tools["issue_refund"]("A-5001", 89.99, "again")
    with pytest.raises(ToolError, match="No customer"):
        tools["find_customer"]("nobody@example.com")


def test_refund_tool_requires_approval():
    tools = {t.name: t for t in support_agent.make_support_tools(support_agent.make_store())}
    assert tools["issue_refund"].needs_approval
    assert not tools["list_orders"].needs_approval


# --- augmented LLM ----------------------------------------------------------

def test_calculator_is_exact_and_safe():
    assert hr_assistant.calculate("(15 + 4) * 7.5") == "142.5"
    with pytest.raises(ToolError):
        hr_assistant.calculate("__import__('os').system('echo hi')")
    with pytest.raises(ToolError):
        hr_assistant.calculate("1 / 0")


def test_handbook_search_and_memory_file(tmp_path):
    assert "## Vacation" in hr_assistant.search_handbook("vacation days")

    path = tmp_path / "memory.json"
    hr_assistant.MemoryStore(str(path)).remember("Prefers bullet points.")
    assert "Prefers bullet points." in hr_assistant.MemoryStore(str(path)).recall()


def test_remember_ignores_facts_it_already_has(tmp_path):
    path = tmp_path / "memory.json"
    memory = hr_assistant.MemoryStore(str(path))

    assert memory.remember("Prefers short bullet-point answers.").startswith("Saved")
    assert memory.remember("  prefers SHORT bullet point answers  ").startswith("Already saved")
    assert memory.facts == ["Prefers short bullet-point answers."]
    assert json.loads(path.read_text()) == ["Prefers short bullet-point answers."]


def test_duplicates_from_older_memory_files_are_dropped_on_load(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(["Likes tea.", "Likes tea.", "Works in Berlin."]))

    assert hr_assistant.MemoryStore(str(path)).facts == ["Likes tea.", "Works in Berlin."]


# --- routing ------------------------------------------------------------------

def test_every_route_forbids_invented_facts():
    from effective_agents.demos import support_router

    for name, route in support_router.ROUTES.items():
        assert route.system.endswith(support_router.FACTS_ONLY), name


# --- chaining gates ----------------------------------------------------------

def test_marketing_gates():
    assert marketing_chain.has_three_bullets("- a\n- b\n- c") is None
    assert marketing_chain.has_three_bullets("- a\n- b") is not None
    assert marketing_chain.short_enough("word " * 50) is None
    assert marketing_chain.short_enough("word " * 200) is not None


# --- CLI smoke tests --------------------------------------------------------

BULLETS = "- point one\n- point two\n- point three"


def generic_handler(kind, kwargs):
    """Answers any Claude request well enough for every demo to run to completion."""
    if kind == "parse":
        return parsed(fake_instance(kwargs["output_format"]))
    return message(BULLETS)


def generic_gemini_handler(kind, kwargs):
    """The same for Gemini: JSON when a schema is requested, bullets otherwise."""
    schema = kwargs["config"].response_schema
    return gemini_response(fake_instance(schema).model_dump_json() if schema else BULLETS)


HANDLERS = {"fake": generic_handler, "fake_gemini": generic_gemini_handler}


@pytest.mark.parametrize("provider", ["fake", "fake_gemini"])
@pytest.mark.parametrize(
    "argv",
    [
        ["augmented"],
        ["chain", "--language", "German"],
        ["route", "--ticket", "Does it have dark mode?"],
        ["parallel"],
        ["orchestrate"],
        ["evaluate"],
        ["support-agent", "--message", "hello", "--yes"],
        ["coding-agent"],
    ],
)
def test_every_demo_runs_through_the_cli(request, capsys, provider, argv):
    request.getfixturevalue(provider)(HANDLERS[provider])
    assert main(argv + ["--quiet"]) == 0
    assert "──" in capsys.readouterr().out


def test_every_demo_is_registered_with_the_parser():
    parser = build_parser()
    for name in DEMOS:
        assert parser.parse_args([name]).demo == name


@pytest.mark.parametrize("provider, module, variable", [("claude", claude, "ANTHROPIC_API_KEY"), ("gemini", gemini, "GEMINI_API_KEY")])
def test_cli_explains_missing_credentials(monkeypatch, capsys, tmp_path, provider, module, variable):
    from effective_agents import llm

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either
    monkeypatch.setattr(llm, "PROVIDER", llm.PROVIDER)  # restored after the test
    monkeypatch.setattr(module, "_client", None)

    assert main(["chain", "--quiet", "--provider", provider]) == 1
    assert variable in capsys.readouterr().err


def test_cli_without_a_demo_prints_help(capsys):
    assert main([]) == 0
    assert "coding-agent" in capsys.readouterr().out


def test_cli_model_flag_overrides_default(fake):
    client = fake(generic_handler)
    main(["evaluate", "--quiet", "--model", "claude-sonnet-5-5"])
    assert client.calls[0][1]["model"] == "claude-sonnet-5-5"


def test_cli_provider_flag_switches_to_gemini(fake, monkeypatch):
    from fakes import FakeGemini

    fake(generic_handler)  # Claude is the default here...
    gemini_client = FakeGemini(generic_gemini_handler)
    monkeypatch.setattr(gemini, "_client", gemini_client)

    assert main(["evaluate", "--quiet", "--provider", "gemini"]) == 0  # ...but the flag wins
    assert gemini_client.calls[0][1]["model"] == gemini.DEFAULT_MODEL
