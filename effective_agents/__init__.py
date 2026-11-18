"""Building effective agents: the patterns from Anthropic's article, implemented
directly on model APIs (Claude or Gemini) with no framework.

    Building block   AugmentedLLM                       (augmented.py)
    Workflows        chain, route, sectioning / vote,   (workflows/)
                     orchestrate, optimize
    Agents           Agent                              (agent.py)
    Providers        claude, gemini                     (providers/)
"""

from .agent import Agent, ask_human, auto_approve
from .augmented import AugmentedLLM, StepLimitReached, chat
from .llm import SMALL, MissingCredentials, RefusalError, call_llm, call_structured, start_conversation
from .tools import Tool, ToolError

__all__ = [
    "Agent",
    "AugmentedLLM",
    "MissingCredentials",
    "RefusalError",
    "SMALL",
    "StepLimitReached",
    "Tool",
    "ToolError",
    "ask_human",
    "auto_approve",
    "call_llm",
    "call_structured",
    "chat",
    "start_conversation",
]
