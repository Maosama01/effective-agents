"""Building-block demo: an augmented LLM with retrieval, tools and memory.

An HR assistant for a fictional company that can:
* retrieve: search the employee handbook (simple keyword search here; in
  production this would be a vector or full-text search);
* use a tool: a calculator for exact arithmetic;
* remember: save facts about the user to a memory store, and recall them later.
  Pass --memory-file to keep memories between runs.

The model decides on its own when to search, calculate or remember.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from pathlib import Path

from .. import trace
from ..augmented import AugmentedLLM, chat
from ..tools import Tool, ToolError

SUMMARY = "Augmented LLM: HR assistant with retrieval, a calculator tool, and memory."

HANDBOOK = {
    "Vacation": "Full-time employees get 15 vacation days a year, plus 1 extra day for each full year of "
                "service, up to a maximum of 25 days. Up to 5 unused days carry over to the next year.",
    "Remote work": "Employees may work remotely up to 3 days a week. Core hours, when everyone should be "
                   "reachable, are 10:00 to 15:00 local time.",
    "Equipment": "Laptops are replaced every 3 years. New employees get a one-time $500 home-office stipend.",
    "Expenses": "Submit expenses within 30 days. While travelling, meals are covered up to $60 a day.",
    "Parental leave": "Parents get 16 weeks of fully paid leave, which can be taken within the child's first year.",
    "Working hours": "A standard workday is 7.5 hours, and a standard week is 37.5 hours.",
}

DEFAULT_QUESTION = (
    "I've been at Northwind for 4 years. How many vacation days do I get this year, and how many "
    "working hours is that? Also, please remember that I prefer short bullet-point answers."
)

SYSTEM = """You are the HR assistant for Northwind Labs.

- Answer from the employee handbook: search it rather than relying on assumptions, and name the sections you used.
- Every number in your answer must come from a search result or the calculator. If a figure you need, such as the length of a workday, is not in your search results yet, search for it before answering. Never assume a typical value.
- Use the calculator for any arithmetic.
- At the start of a conversation, recall saved memories. Apply every preference you recall to your answer, even a one-line answer: if the user prefers bullet points, answer in bullet points.
- When the user tells you a lasting fact or preference about themselves, save it with remember."""


class MemoryStore:
    """Long-term memory: a list of facts, optionally saved to a JSON file.

    Duplicates are handled by the tool, not left to the model (the article's
    "poka-yoke" advice): saving a fact that is already stored changes nothing,
    whatever the model does.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path).expanduser() if path else None
        self.facts: list[str] = []
        if self.path and self.path.exists():
            for fact in json.loads(self.path.read_text(encoding="utf-8")):
                if not self._has(fact):  # also drops duplicates saved by older versions
                    self.facts.append(fact)

    @staticmethod
    def _key(fact: str) -> str:
        """Compare facts ignoring case, spacing and punctuation."""
        return re.sub(r"[\W_]+", " ", fact.lower()).strip()

    def _has(self, fact: str) -> bool:
        return self._key(fact) in {self._key(known) for known in self.facts}

    def remember(self, fact: str) -> str:
        fact = fact.strip()
        if self._has(fact):
            return f"Already saved, nothing changed: {fact}"
        self.facts.append(fact)
        if self.path:
            self.path.write_text(json.dumps(self.facts, indent=2), encoding="utf-8")
        return f"Saved. {len(self.facts)} memories stored."

    def recall(self) -> str:
        if not self.facts:
            return "No memories saved yet."
        return "\n".join(f"- {fact}" for fact in self.facts)


def search_handbook(query: str) -> str:
    words = set(re.findall(r"[a-z]+", query.lower()))
    scored = []
    for title, body in HANDBOOK.items():
        score = len(words & set(re.findall(r"[a-z]+", f"{title} {body}".lower())))
        if score:
            scored.append((score, title, body))
    if not scored:
        return "No matching sections. Try other keywords."
    scored.sort(reverse=True)
    return "\n\n".join(f"## {title}\n{body}" for _, title, body in scored[:3])


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculate(expression: str) -> str:
    """Safely evaluate arithmetic (no names, calls or attributes)."""

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ToolError("Only numbers and + - * / // % ** and parentheses are allowed, e.g. '(15 + 4) * 7.5'.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError(f"{expression!r} is not a valid arithmetic expression.")
    try:
        value = evaluate(tree.body)
    except ZeroDivisionError:
        raise ToolError("Division by zero.")
    return str(round(value, 10))


def build_assistant(memory: MemoryStore, effort: str | None = None) -> AugmentedLLM:
    tools = [
        Tool(
            "search_handbook",
            "Search the Northwind Labs employee handbook. Returns up to 3 matching sections with their titles. "
            "Use plain keywords, e.g. 'vacation days' or 'remote work'.",
            {"query": {"type": "string", "description": "Keywords to search for."}},
            search_handbook,
        ),
        Tool(
            "calculate",
            "Evaluate an arithmetic expression exactly, e.g. '(15 + 4) * 7.5'. Supports + - * / // % ** and parentheses.",
            {"expression": {"type": "string", "description": "The expression to evaluate."}},
            calculate,
        ),
        Tool(
            "remember",
            "Save one lasting fact or preference about the user to long-term memory, as a short sentence, "
            "e.g. 'Prefers short bullet-point answers.' Don't save one-off questions. Saving a fact that "
            "is already stored does nothing.",
            {"fact": {"type": "string", "description": "The fact to remember."}},
            memory.remember,
        ),
        Tool(
            "recall",
            "Return everything saved in long-term memory about the user.",
            {},
            memory.recall,
        ),
    ]
    return AugmentedLLM(SYSTEM, tools, effort=effort)


def add_arguments(parser) -> None:
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="The question to ask.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive chat instead.")
    parser.add_argument("--memory-file", help="JSON file to keep memories in between runs, e.g. memory.json")


def main(args) -> None:
    memory = MemoryStore(args.memory_file)
    assistant = build_assistant(memory, args.effort)
    if args.chat:
        chat(assistant, "Northwind Labs HR assistant.")
    else:
        trace.log("step", "question", args.question)
        trace.show("answer", assistant.run(args.question))
    trace.show("memory", memory.recall())
