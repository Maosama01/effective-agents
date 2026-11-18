"""Routing demo: sort support tickets and answer each with a specialist prompt.

Both of the article's routing examples in one: each ticket type gets its own
specialized prompt, and simple general questions go to the provider's smaller,
cheaper model (Claude Haiku or Gemini Flash Lite; set
EFFECTIVE_AGENTS_SMALL_MODEL to change it).
"""

from __future__ import annotations

from .. import trace
from ..llm import SMALL
from ..workflows import Route, route

SUMMARY = "Routing: classify support tickets and send each to a specialist prompt or a cheaper model."

# Without this, the model fills gaps about the (fictional) company with
# plausible inventions: support phone numbers, status pages, refund timelines.
FACTS_ONLY = (
    "\n\nYou know only what this prompt and the ticket tell you about Lumen. Never invent phone numbers, "
    "email addresses, links, prices, timeframes, features or policies. When the customer needs one of "
    "those, say that the Lumen team will confirm it in their reply."
)

ROUTES = {
    "billing": Route(
        description="Charges, invoices, refunds, plans and payment methods.",
        system="You are a billing specialist for Lumen, a project-management app. Be precise about amounts "
               "and dates, explain charges plainly, and say exactly what the customer should do next."
               + FACTS_ONLY,
    ),
    "technical": Route(
        description="Bugs, error messages, crashes, performance, setup and integrations.",
        system="You are a senior support engineer for Lumen, a project-management app. Ask for any missing "
               "details, then give numbered troubleshooting steps, most likely fix first." + FACTS_ONLY,
    ),
    "account": Route(
        description="Signing in, passwords, two-factor authentication, profile and security settings, "
                    "deleting an account.",
        system="You are an account-security specialist for Lumen. Never ask for passwords or codes. Explain "
               "the safe self-service path first, then how to reach a human if that fails." + FACTS_ONLY,
    ),
    "general": Route(
        description="Simple general questions about Lumen's features, pricing or company.",
        system="You are a friendly assistant for Lumen, a project-management app. Answer in two or three "
               "sentences." + FACTS_ONLY,
        model=SMALL,
        effort="low",
    ),
}

SAMPLE_TICKETS = [
    "I was charged twice for my Pro subscription this month. Can you fix it?",
    "Since this morning the Gantt view shows 'Error 502' whenever I open a project with more than 200 tasks.",
    "I lost my phone and can't get past the two-factor screen anymore.",
    "Does Lumen have a dark mode?",
]


def add_arguments(parser) -> None:
    parser.add_argument("--ticket", action="append", help="A ticket to route (repeatable). Defaults to 4 samples.")


def main(args) -> None:
    for number, ticket in enumerate(args.ticket or SAMPLE_TICKETS, 1):
        trace.log("step", f"ticket {number}", ticket)
        result = route(ticket, ROUTES)
        trace.show(f"ticket {number} → {result.route}", f"> {ticket}\n\n{result.output}")
