"""Workflow: routing.

Classify the input, then hand it to a specialized follow-up: its own prompt,
and optionally its own model. Each route's prompt can be tuned without
hurting the others, and easy inputs can go to a smaller, cheaper model.

Use it when inputs fall into distinct categories that are better handled
separately, and the classification itself can be done reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, create_model

from .. import trace
from ..llm import call_llm, call_structured, resolve_model


@dataclass
class Route:
    description: str          # what belongs on this route (shown to the classifier)
    system: str               # specialized system prompt for the follow-up call
    model: str | None = None  # e.g. llm.SMALL for the provider's cheaper model
    effort: str | None = None


@dataclass
class RoutingResult:
    route: str
    reason: str
    output: str


def classify(input_text: str, routes: dict[str, Route]) -> tuple[str, str]:
    """Pick a route name with structured outputs, so code can act on it."""
    names = tuple(routes)
    Decision: type[BaseModel] = create_model(
        "Decision",
        route=(Literal[names], Field(description="The best route for this input.")),
        reason=(str, Field(description="One short sentence on why.")),
    )
    options = "\n".join(f"- {name}: {route.description}" for name, route in routes.items())
    decision = call_structured(
        f"Choose the route that best fits the input.\n\nRoutes:\n{options}\n\n"
        f"<input>\n{input_text}\n</input>",
        Decision,
        effort="low",  # classification is a simple task
        max_tokens=4000,
    )
    return decision.route, decision.reason  # type: ignore[attr-defined]


def route(input_text: str, routes: dict[str, Route]) -> RoutingResult:
    name, reason = classify(input_text, routes)
    chosen = routes[name]
    trace.log("step", f"route → {name}", f"{reason} (model: {resolve_model(chosen.model)})")
    output = call_llm(input_text, system=chosen.system, model=chosen.model, effort=chosen.effort)
    return RoutingResult(name, reason, output)
