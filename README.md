# Building Effective Agents

Every pattern from Anthropic's article [Building effective agents](https://www.anthropic.com/research/building-effective-agents), implemented directly on model APIs with no agent framework, plus a runnable demo for each. It runs on **Claude** (Anthropic API) or **Gemini** (Google Gen AI API).

The article's main advice is to start with simple, composable patterns and to use the LLM API directly, so you can see every prompt and response. That's what this project does. The workflows and agents never touch an SDK: they call three functions in [`effective_agents/llm.py`](effective_agents/llm.py), and a small provider module ([`claude.py`](effective_agents/providers/claude.py) or [`gemini.py`](effective_agents/providers/gemini.py)) turns those into API requests.

| Article section | Code | Demo |
|---|---|---|
| Building block: the augmented LLM | [`augmented.py`](effective_agents/augmented.py) | `augmented`: HR assistant with handbook search, a calculator and memory |
| Workflow: prompt chaining | [`workflows/chaining.py`](effective_agents/workflows/chaining.py) | `chain`: product notes → selling points → marketing copy → translation, with gates |
| Workflow: routing | [`workflows/routing.py`](effective_agents/workflows/routing.py) | `route`: support tickets go to specialist prompts; easy ones go to a smaller model |
| Workflow: parallelization | [`workflows/parallelization.py`](effective_agents/workflows/parallelization.py) | `parallel`: guardrail screening next to the answer (sectioning) and three code reviewers (voting) |
| Workflow: orchestrator-workers | [`workflows/orchestrator_workers.py`](effective_agents/workflows/orchestrator_workers.py) | `orchestrate`: an orchestrator plans a guide's sections, workers write them, a final call merges them |
| Workflow: evaluator-optimizer | [`workflows/evaluator_optimizer.py`](effective_agents/workflows/evaluator_optimizer.py) | `evaluate`: literary translation revised until a critic passes it |
| Agents | [`agent.py`](effective_agents/agent.py) | see the two below |
| Appendix 1A: customer support | [`demos/support_agent.py`](effective_agents/demos/support_agent.py) | `support-agent`: looks up orders and issues refunds, with a human approving each refund |
| Appendix 1B: coding agents | [`demos/coding_agent.py`](effective_agents/demos/coding_agent.py) | `coding-agent`: fixes a buggy project, using the test results as ground truth |
| Appendix 2: prompt engineering your tools | [`tools.py`](effective_agents/tools.py) and the demo tools | absolute paths, whole-file writes, strict schemas, errors the model can act on |

## Quick start

You need Python 3.10 or newer and an API key for at least one provider:

- Claude: a key from the [Claude Console](https://console.anthropic.com/)
- Gemini: a key from [Google AI Studio](https://aistudio.google.com/apikey)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then set one key (or both):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

Then run any demo:

```bash
effective-agents coding-agent
```

If only a Gemini key is set, Gemini is used; otherwise Claude is. To choose explicitly, pass `--provider gemini` or `--provider claude`, or set `EFFECTIVE_AGENTS_PROVIDER`. `python -m effective_agents <demo>` works too. Run `effective-agents` on its own to list the demos, and `effective-agents <demo> --help` to see a demo's options.

The step-by-step trace (plans, tool calls, results, gate checks, votes) goes to stderr and final results go to stdout. The first trace line shows which provider and model are in use. The model's progress notes are shortened to one-line headlines; add `--verbose` to see them in full, along with untruncated tool output, or `--quiet` to hide the trace.

## The demos

**`augmented`**: the building block. One model call that can search a handbook (retrieval), use a calculator (a tool), and save or recall facts about the user (memory). The model decides when to use each one. Run it twice with `--memory-file memory.json` and it remembers your preferences between runs. Use `--chat` for a conversation.

**`chain`**: three fixed steps. Code checks the output between steps: there must be at least 3 selling points, and the copy must be 90 words or fewer. If a check fails, the chain stops instead of passing bad output along. Try `--language Japanese`.

**`route`**: a classifier picks billing, technical, account or general for each ticket, using structured outputs so the code can act on the choice. Each route has its own system prompt, and general questions go to the provider's small model at low effort. Pass your own tickets with `--ticket "..."`.

**`parallel`**: part 1 answers each customer message while a separate call screens it at the same time, and the answer is withheld if the screen blocks the message. Part 2 has three security reviewers with different focuses check the same code in parallel. `--threshold` sets how many votes it takes to flag the code.

**`orchestrate`**: the orchestrator decides which sections the guide needs (they change with the task), workers write the sections in parallel, and a synthesizer merges them. Try `--task "..."` with your own writing or research task.

**`evaluate`**: a translator drafts, an evaluator checks the draft against explicit criteria and returns feedback, and the translator revises. This repeats until the evaluator passes it or `--max-rounds` is reached.

**`support-agent`**: a chat with a support agent for a made-up shop. The data is fake and kept in memory. Every refund stops for your approval in the terminal, which is the article's human checkpoint. Try *"Hi, I'm jane@example.com, my headphones stopped working. Can I get my money back?"*. You can also send one message with `--message "..."`, and add `--yes` to approve refunds automatically.

**`coding-agent`**: copies a small project with three bugs into a temporary directory and asks the agent to make its tests pass. Watch it run the tests, read the code, rewrite the file and run the tests again. Point it at your own project with `--workspace PATH --task "..."`. Outside the temporary sandbox, every file write asks for your approval unless you pass `--yes`.

## Using the patterns in your own code

```python
from effective_agents import SMALL, Agent, Tool, call_llm
from effective_agents.workflows import Route, Step, chain, route, optimize, orchestrate, vote

# Prompt chaining with a programmatic gate
copy = chain(
    "notes about my product...",
    [
        Step("List 3 selling points as '- ' bullets.", gate=lambda out: None if out.count("- ") >= 3 else "need 3 bullets"),
        Step("Write 50 words of marketing copy from these points."),
        Step("Translate into French."),
    ],
)

# Routing (SMALL means the provider's smaller, cheaper model)
result = route(ticket, {
    "billing": Route("Charges and refunds", system="You are a billing specialist..."),
    "general": Route("Simple questions", system="Answer briefly.", model=SMALL),
})

# An agent: an LLM using tools in a loop
def get_weather(city: str) -> str:
    return f"18°C and cloudy in {city}"

weather = Tool(
    "get_weather",
    "Current weather for a city. Use the city's English name, e.g. 'Paris'.",
    {"city": {"type": "string", "description": "City name"}},
    get_weather,
)
agent = Agent("You are a travel assistant.", [weather])
print(agent.run_task("Should I pack an umbrella for Paris?"))
```

All of this runs unchanged on either provider.

## How the article's principles show up in the code

- **Simplicity.** There's no framework. Each workflow is one short function built from `call_llm` (text in, text out) and `call_structured` (text in, a validated Pydantic object out). The agent loop is about 25 lines in [`augmented.py`](effective_agents/augmented.py). It's written out by hand instead of using an SDK's built-in tool runner, so you can see each step.
- **Transparency.** The trace shows every step. Agents state their plan before acting, and the model's progress notes between tool calls are shown as they arrive.
- **Agent-computer interface.** Tool descriptions read like docstrings, with examples. The coding agent's tools only accept absolute paths, which is the article's SWE-bench lesson; a relative path gets an error that shows the correct form. Edits rewrite the whole file instead of asking the model to write a diff. Tool errors go back to the model as results it can learn from, instead of crashing the loop.
- **Ground truth and guardrails.** Agents check tool results at each step, stop after `max_steps`, and pause for human approval before risky actions (`needs_approval=True`).

## Providers

Both providers implement the same small interface (see [`providers/__init__.py`](effective_agents/providers/__init__.py)). Adding another means writing one more module like these.

| | Claude | Gemini |
|---|---|---|
| SDK | `anthropic` | `google-genai` |
| Default model | Claude Opus 5.5 (`claude-opus-5-5`) | Gemini 3.8 Flash (`gemini-3.8-flash`) |
| Small model (routing demo) | Claude Haiku 5.5 (`claude-haiku-5-5`) | Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) |
| Effort | sent as `output_config.effort` | mapped to `thinking_level`; `xhigh` and `max` become `HIGH` |
| Structured outputs | `messages.parse` with the Pydantic model | `response_schema` with the Pydantic model |
| Tool schemas | `strict: true`, so inputs always match | JSON Schema function declarations; the loop runs tools itself |
| Progress notes | `thinking.display: "updates"` | thought summaries (`include_thoughts`) |
| History | assistant turns stored exactly as returned, thinking blocks included | model turns stored exactly as returned, so thought signatures go back unchanged |
| Reliability | refusal fallback: if a safety check wrongly declines a request, the API retries it on a fallback model; prompt caching in tool loops | 2-minute request timeout, with retries and backoff on timeouts, 429s and 5xx errors |

On either provider, a request that is declined or blocked raises `RefusalError` instead of returning an empty answer.

**Gemini free tier.** Free-tier keys get a small daily request allowance per model; at the time of writing, Gemini 3.8 Flash allowed 20 requests a day. One coding-agent run uses about 8, so the demos use it up quickly. Allowances are counted per model, so you can switch to a model with its own allowance, such as Flash Lite:

```bash
export EFFECTIVE_AGENTS_MODEL=gemini-3.5-flash-lite
```

When a limit is hit, the error message quotes Google's explanation, including when the allowance resets.

## Configuration

| Setting | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | for Claude (or sign in with `ant auth login`) |
| `GEMINI_API_KEY` | | for Gemini (`GOOGLE_API_KEY` also works) |
| `EFFECTIVE_AGENTS_PROVIDER` / `--provider` | Gemini if only a Gemini key is set, else Claude | `claude` or `gemini` |
| `EFFECTIVE_AGENTS_MODEL` / `--model` | the provider's default model | |
| `EFFECTIVE_AGENTS_SMALL_MODEL` | the provider's small model | used by the routing demo's "general" route |
| `EFFECTIVE_AGENTS_EFFORT` / `--effort` | `medium` | `low`, `medium`, `high`, `xhigh`, `max` |

## Tests

```bash
pip install -e .
python -m pytest
```

The tests replace both API clients with fakes that return real SDK response objects (see [`tests/fakes.py`](tests/fakes.py)), so they run offline in about a second and cost nothing. They cover each pattern's control flow, the agent loop (tool errors, approvals, step limits), what each provider sends and how it reads replies (including Gemini's thought signatures), the demo tools (the coding agent's tools really fix the seed project, and the refund rules hold), and an end-to-end run of every demo through the CLI on both providers.

## Layout

```
effective_agents/
  llm.py            the facade: call_llm, call_structured, start_conversation
  providers/        claude.py and gemini.py: the only code that calls an API
  tools.py          Tool definition and execution
  augmented.py      AugmentedLLM: the building block and its tool loop
  agent.py          Agent: loop + checkpoints + stopping condition
  trace.py          console tracing
  workflows/        chaining, routing, parallelization, orchestrator_workers, evaluator_optimizer
  demos/            one runnable demo per pattern
  cli.py            the effective-agents command
tests/
```
