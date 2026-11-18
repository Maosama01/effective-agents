"""Agent demo: a customer support agent (the article's Appendix 1A).

Support is a natural fit for agents: it is a conversation that also needs
actions. This agent can look up customers and orders, search the help center,
and issue refunds. Refunds are a human checkpoint: the agent proposes one and
a person at the terminal approves or declines it.

All data is fake and lives in memory, so it is safe to experiment.
Try: "Hi, I'm jane@example.com. My headphones stopped working, can I get my money back?"
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

from .. import trace
from ..agent import Agent, ask_human, auto_approve
from ..augmented import chat
from ..tools import Tool, ToolError

SUMMARY = "Customer support agent with order lookup and human-approved refunds (Appendix 1A)."

RETURN_WINDOW_DAYS = 30


def _days_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def make_store() -> dict:
    """A fresh copy of the fake database."""
    return {
        "customers": {
            "C-1001": {"name": "Jane Doe", "email": "jane@example.com", "plan": "Pro", "customer_since": "2023-04-12"},
            "C-1002": {"name": "Sam Lee", "email": "sam@example.com", "plan": "Free", "customer_since": "2025-01-30"},
        },
        "orders": {
            "A-5001": {"customer_id": "C-1001", "item": "Wireless headphones", "amount": 89.99,
                       "status": "delivered", "delivered_on": _days_ago(16), "refunded": False},
            "A-5002": {"customer_id": "C-1001", "item": "USB-C cable (2-pack)", "amount": 12.99,
                       "status": "shipped", "delivered_on": None, "refunded": False},
            "A-5003": {"customer_id": "C-1002", "item": "Laptop stand", "amount": 39.00,
                       "status": "delivered", "delivered_on": _days_ago(59), "refunded": False},
        },
    }


HELP_CENTER = [
    ("Returns and refunds",
     f"Delivered items can be returned for a full refund within {RETURN_WINDOW_DAYS} days of delivery. "
     "Refunds go to the original payment method and arrive in 5-7 business days. Orders that have not "
     "been delivered yet cannot be refunded; customers can refuse the delivery instead."),
    ("Shipping times",
     "Standard shipping takes 3-5 business days and express takes 1-2. A tracking link is emailed when "
     "an order ships."),
    ("Warranty",
     "Electronics have a one-year warranty against defects. Within the return window a refund is usually "
     "simplest; after it, we repair or replace the item under warranty."),
    ("Resetting your password",
     "Use 'Forgot password' on the sign-in page. The reset link expires after one hour. Support staff can "
     "never see or set a customer's password."),
]


def make_support_tools(store: dict) -> list[Tool]:
    customers, orders = store["customers"], store["orders"]

    def find_customer(email: str) -> str:
        for customer_id, customer in customers.items():
            if customer["email"].lower() == email.strip().lower():
                return json.dumps({"customer_id": customer_id, **customer})
        raise ToolError(f"No customer has the email {email!r}. Ask the customer to double-check it.")

    def list_orders(customer_id: str) -> str:
        if customer_id not in customers:
            raise ToolError(f"Unknown customer_id {customer_id!r}. Get it from find_customer first.")
        found = [{"order_id": oid, **o} for oid, o in orders.items() if o["customer_id"] == customer_id]
        return json.dumps(found) if found else "This customer has no orders."

    def search_help_center(query: str) -> str:
        words = set(re.findall(r"[a-z]+", query.lower()))
        scored = sorted(
            HELP_CENTER,
            key=lambda article: -len(words & set(re.findall(r"[a-z]+", " ".join(article).lower()))),
        )
        return "\n\n".join(f"## {title}\n{body}" for title, body in scored[:2])

    def issue_refund(order_id: str, amount: float, reason: str) -> str:
        order = orders.get(order_id)
        if order is None:
            raise ToolError(f"Unknown order_id {order_id!r}. Use list_orders to find the right one.")
        if order["refunded"]:
            raise ToolError(f"Order {order_id} has already been refunded.")
        if order["status"] != "delivered":
            raise ToolError(f"Order {order_id} is '{order['status']}', and only delivered orders can be refunded.")
        age = (date.today() - date.fromisoformat(order["delivered_on"])).days
        if age > RETURN_WINDOW_DAYS:
            raise ToolError(
                f"Order {order_id} was delivered {age} days ago, outside the {RETURN_WINDOW_DAYS}-day refund "
                "window. Offer a warranty repair or replacement instead."
            )
        if not 0 < amount <= order["amount"]:
            raise ToolError(f"Amount must be more than 0 and at most the order total of {order['amount']:.2f}.")
        order["refunded"] = True
        return f"Refund of ${amount:.2f} for order {order_id} issued (ref R-{order_id[2:]}). Reason: {reason}"

    return [
        Tool(
            "find_customer",
            "Look up a customer by email address. Returns their customer_id, name and plan. "
            "Use this first: the other tools need the customer_id.",
            {"email": {"type": "string", "description": "The customer's email, e.g. jane@example.com"}},
            find_customer,
        ),
        Tool(
            "list_orders",
            "List all orders for a customer, with order_id, item, amount, status and delivery date.",
            {"customer_id": {"type": "string", "description": "From find_customer, e.g. C-1001"}},
            list_orders,
        ),
        Tool(
            "search_help_center",
            "Search the help-center articles (refund policy, shipping, warranty, passwords). "
            "Check policy here before promising the customer anything.",
            {"query": {"type": "string", "description": "A few keywords, e.g. 'refund window'"}},
            search_help_center,
        ),
        Tool(
            "issue_refund",
            "Refund a delivered order to the customer's original payment method. A human must approve each "
            "refund. Only use it after confirming the order belongs to the customer and is eligible under "
            "the refund policy.",
            {
                "order_id": {"type": "string", "description": "e.g. A-5001"},
                "amount": {"type": "number", "description": "Amount in US dollars, at most the order total."},
                "reason": {"type": "string", "description": "One-line reason for the refund."},
            },
            issue_refund,
            needs_approval=True,
        ),
    ]


SYSTEM = f"""You are the support assistant for Gadgetly, an online electronics shop. Today's date is {date.today().isoformat()}.

Help customers with orders, refunds, shipping and account questions. Identify the customer by email before discussing any order, and only discuss that customer's own orders. Follow the help-center policy; don't promise anything it doesn't allow. Keep replies short, warm and plain."""

GUIDELINES = """

How to work:
- Use the tools to check facts instead of guessing.
- When the customer asks for something the policy allows, such as a refund, call the tool in this same reply instead of only saying you will. A staff member approves every refund, so you don't need to ask the customer to confirm first.
- Only tell the customer an action is done after its tool reports success. If it fails or is declined, say so plainly.
- If a tool returns an error, explain the situation to the customer and offer the best alternative.
- If you can't help, say so and suggest what the customer can do next."""


def build_agent(args, store: dict | None = None) -> Agent:
    return Agent(
        SYSTEM,
        make_support_tools(store or make_store()),
        effort=args.effort,
        max_steps=15,
        approve=auto_approve if args.yes else ask_human,
        guidelines=GUIDELINES,
    )


def add_arguments(parser) -> None:
    parser.add_argument("--message", help="Send one message instead of starting a chat.")
    parser.add_argument("--yes", action="store_true", help="Approve refunds automatically.")


def main(args) -> None:
    agent = build_agent(args)
    if args.message:
        trace.show("assistant", agent.run_task(args.message))
        return
    chat(
        agent,
        "Gadgetly support. Fake customers: jane@example.com (orders A-5001, A-5002) "
        "and sam@example.com (order A-5003).",
    )
