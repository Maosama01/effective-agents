"""Runnable demos, one per pattern, in the order the article introduces them."""

from . import (
    coding_agent,
    guardrails_review,
    guide_writer,
    hr_assistant,
    literary_translation,
    marketing_chain,
    support_agent,
    support_router,
)

DEMOS = {
    "augmented": hr_assistant,
    "chain": marketing_chain,
    "route": support_router,
    "parallel": guardrails_review,
    "orchestrate": guide_writer,
    "evaluate": literary_translation,
    "support-agent": support_agent,
    "coding-agent": coding_agent,
}
