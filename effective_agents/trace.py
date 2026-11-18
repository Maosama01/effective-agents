"""Console tracing, so you can watch what an agentic system is doing.

The article's second principle is transparency: show the planning steps.
Everything here prints to stderr, which keeps stdout for final answers.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

_COLORS = {
    "step": "\033[1;36m",    # bold cyan
    "plan": "\033[35m",      # magenta
    "tool": "\033[33m",      # yellow
    "result": "\033[2m",     # dim
    "error": "\033[31m",     # red
    "ok": "\033[32m",        # green
    "info": "\033[34m",      # blue
}
_RESET = "\033[0m"

enabled = True
max_chars = 800       # longer trace entries are cut; 0 shows everything
full_notes = False    # False: show each progress note as a one-line headline


def _use_color() -> bool:
    return sys.stderr.isatty() and "NO_COLOR" not in os.environ


def _short(text: str) -> str:
    text = text.strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + f" … [{len(text) - max_chars} more chars]"
    return text


def log(kind: str, label: str, text: str = "") -> None:
    if not enabled:
        return
    color = _COLORS.get(kind, "") if _use_color() else ""
    reset = _RESET if color else ""
    body = f" {_short(text)}" if text else ""
    print(f"{color}[{label}]{reset}{body}", file=sys.stderr, flush=True)


def headline(note: str, width: int = 120) -> str:
    """One line that sums up a progress note.

    Gemini's thought summaries are paragraphs under a bold title line, often
    after a filler sentence ("Here's my take..."), so a title line wins;
    otherwise (e.g. Claude's short notes) the first line is used.
    """
    title = re.search(r"^\s*\*\*(.+?)\*\*\s*$", note, re.MULTILINE)
    line = title.group(1) if title else next((l for l in note.splitlines() if l.strip()), "")
    line = line.strip()
    return line if len(line) <= width else line[: width - 1] + "…"


def turn(t: Any) -> None:
    """Show the interesting parts of one model turn (an llm.Turn)."""
    for note in t.notes:
        log("plan", "progress", note if full_notes else headline(note))
    if t.tool_calls and t.text:
        log("plan", "says", t.text)
    for call in t.tool_calls:
        log("tool", f"tool → {call.name}", json.dumps(call.input, ensure_ascii=False))


def tool_result(result: Any) -> None:
    """Show one tool result (an llm.ToolResult)."""
    log("error" if result.is_error else "result", f"{result.call.name} ←", result.content)


def show(title: str, body: str) -> None:
    """Print a final result to stdout."""
    bar = "─" * max(8, 60 - len(title))
    print(f"\n── {title} {bar}\n{body.strip()}\n", flush=True)
