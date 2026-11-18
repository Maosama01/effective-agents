"""Agent demo: a coding agent (the article's Appendix 1B).

The agent gets a small project with failing tests and four tools: list files,
read a file, write a file, and run the tests. The test results are the ground
truth it checks after every change, which is what makes coding such a good
fit for agents.

The tools follow Appendix 2's advice on agent-computer interfaces:
* paths must be absolute (the article's SWE-bench lesson): relative paths are
  rejected with an error that shows the correct form;
* edits rewrite the whole file instead of asking for a diff, a format that is
  much easier for a model to produce correctly;
* every error message says what to do next.

By default the project is copied into a fresh temporary directory, so nothing
on your machine is touched. Use --workspace to point it at another directory
(writes then need your approval unless you pass --yes).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import trace
from ..agent import Agent, ask_human, auto_approve
from ..tools import Tool, ToolError

SUMMARY = "Coding agent fixes a buggy project until its tests pass (Appendix 1B)."

SEED_PROJECT = {
    "README.md": "# cart\n\nA tiny shopping-cart pricing module. Run the tests with `python -m pytest`.\n",
    "cart.py": '''"""A tiny shopping-cart pricing module."""

FREE_SHIPPING_THRESHOLD = 50.00
SHIPPING_FEE = 4.99


def subtotal(items):
    """Sum of price * qty for every item in the cart."""
    return sum(item["price"] for item in items)


def apply_discount(amount, percent):
    """Return `amount` after taking `percent` percent off."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return amount * percent / 100


def shipping(amount):
    """Orders of FREE_SHIPPING_THRESHOLD or more ship free."""
    return 0.0 if amount > FREE_SHIPPING_THRESHOLD else SHIPPING_FEE


def total(items, discount_percent=0):
    """Discounted subtotal plus shipping, rounded to cents."""
    discounted = apply_discount(subtotal(items), discount_percent)
    return round(discounted + shipping(discounted), 2)
''',
    "test_cart.py": '''import pytest

from cart import apply_discount, shipping, subtotal, total

ITEMS = [
    {"name": "pen", "price": 1.50, "qty": 4},
    {"name": "notebook", "price": 3.25, "qty": 2},
]


def test_subtotal_counts_quantity():
    assert subtotal(ITEMS) == pytest.approx(12.50)


def test_subtotal_of_empty_cart_is_zero():
    assert subtotal([]) == 0


def test_discount_takes_percent_off():
    assert apply_discount(200, 25) == pytest.approx(150)


def test_zero_discount_keeps_amount():
    assert apply_discount(80, 0) == pytest.approx(80)


def test_discount_rejects_bad_percent():
    with pytest.raises(ValueError):
        apply_discount(10, 120)


def test_free_shipping_at_threshold():
    assert shipping(50.00) == 0.0


def test_shipping_fee_below_threshold():
    assert shipping(49.99) == 4.99


def test_small_order_pays_shipping():
    assert total(ITEMS) == pytest.approx(17.49)


def test_discount_can_unlock_free_shipping():
    chair = [{"name": "chair", "price": 62.50, "qty": 1}]
    assert total(chair, discount_percent=20) == pytest.approx(50.00)
''',
}

SYSTEM = """You are a careful software engineer working on the Python project in {workspace}.

Every file path you pass to a tool must be an absolute path inside {workspace}.
Run the tests before changing anything, so you know what is failing, and run them again after your changes to confirm the fix. Keep changes minimal, and do not edit the tests unless a test is clearly wrong."""

DEFAULT_TASK = "The test suite is failing. Find the bugs in the code and fix them so that all the tests pass."

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}


def make_coding_tools(workspace: Path) -> list[Tool]:
    workspace = workspace.resolve()
    example = workspace / "example.py"

    def resolve(path: str) -> Path:
        if not os.path.isabs(path):
            raise ToolError(
                f"Paths must be absolute. You gave {path!r}; use the form {example}."
            )
        resolved = Path(path).resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ToolError(f"{path} is outside the workspace. Only paths inside {workspace} are allowed.")
        return resolved

    def list_files(directory: str) -> str:
        root = resolve(directory)
        if not root.is_dir():
            raise ToolError(f"{directory} is not a directory.")
        files = []
        for current, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
            files += [str(Path(current) / name) for name in sorted(names)]
        if not files:
            return "(no files)"
        shown = files[:300]
        more = f"\n… and {len(files) - 300} more" if len(files) > 300 else ""
        return "\n".join(shown) + more

    def read_file(path: str) -> str:
        file = resolve(path)
        if not file.is_file():
            raise ToolError(f"{path} does not exist. Use list_files to see what is there.")
        text = file.read_text(encoding="utf-8", errors="replace")
        if len(text) > 100_000:
            raise ToolError(f"{path} is {len(text):,} characters, too large to read in full.")
        return text

    def write_file(path: str, content: str) -> str:
        file = resolve(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.splitlines())} lines to {file}."

    def run_tests() -> str:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise ToolError("The tests took longer than 120 seconds and were stopped.")
        output = (proc.stdout + proc.stderr).strip()
        if len(output) > 6000:
            output = "…(earlier output cut)…\n" + output[-6000:]
        status = "ALL TESTS PASSED" if proc.returncode == 0 else f"TESTS FAILED (exit code {proc.returncode})"
        return f"{status}\n\n{output}"

    return [
        Tool(
            "list_files",
            "List every file under a directory, recursively, as absolute paths. "
            "Hidden folders and caches are skipped. Use it first to see the project layout.",
            {"directory": {"type": "string", "description": f"Absolute path of a directory, e.g. {workspace}"}},
            list_files,
        ),
        Tool(
            "read_file",
            "Return the full text of one file. Read a file before you change it.",
            {"path": {"type": "string", "description": f"Absolute path of the file, e.g. {example}"}},
            read_file,
        ),
        Tool(
            "write_file",
            "Replace the entire contents of a file (or create it) with `content`. "
            "Always send the complete new file, not a diff or a fragment: anything you leave out is deleted. "
            "Read the file first so you keep the parts that should not change.",
            {
                "path": {"type": "string", "description": f"Absolute path of the file, e.g. {example}"},
                "content": {"type": "string", "description": "The complete new contents of the file."},
            },
            write_file,
            needs_approval=True,
        ),
        Tool(
            "run_tests",
            "Run the project's test suite with pytest and return the result and output. "
            "Use it before making changes and again after every change to check your work.",
            {},
            run_tests,
        ),
    ]


def create_workspace(path: str | None) -> tuple[Path, bool]:
    """Return (workspace, is_sandbox). A sandbox is a temp copy of the seed project."""
    if path:
        workspace = Path(path).expanduser().resolve()
        if not workspace.is_dir():
            raise SystemExit(f"--workspace {path} is not a directory")
        return workspace, False
    workspace = Path(tempfile.mkdtemp(prefix="coding-agent-")).resolve()
    for name, content in SEED_PROJECT.items():
        (workspace / name).write_text(content, encoding="utf-8")
    return workspace, True


def add_arguments(parser) -> None:
    parser.add_argument("--task", default=DEFAULT_TASK, help="What the agent should do.")
    parser.add_argument("--workspace", help="Work in this directory instead of a temp copy of the demo project.")
    parser.add_argument("--yes", action="store_true", help="Allow file writes without asking.")
    parser.add_argument("--max-steps", type=int, default=30, help="Stop after this many model calls.")


def main(args) -> None:
    workspace, sandbox = create_workspace(args.workspace)
    trace.log("info", "workspace", f"{workspace}{' (temporary copy of the demo project)' if sandbox else ''}")
    # In the throwaway sandbox, writes are safe to allow; elsewhere, ask first.
    approve = auto_approve if (sandbox or args.yes) else ask_human
    agent = Agent(
        SYSTEM.format(workspace=workspace),
        make_coding_tools(workspace),
        effort=args.effort,
        max_steps=args.max_steps,
        approve=approve,
    )
    trace.show("agent summary", agent.run_task(args.task))
    print(f"Workspace: {workspace}")
