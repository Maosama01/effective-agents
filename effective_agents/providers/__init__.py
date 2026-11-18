"""Model providers. Each module exposes the same interface (see ``effective_agents.llm``):

    DEFAULT_MODEL, SMALL_MODEL
    complete(prompt, *, system, model, effort, max_tokens) -> str
    structured(prompt, output, *, system, model, effort, max_tokens) -> output instance
    Conversation(system, tools, *, model, effort, max_tokens, show_progress)
    friendly_error(exc) -> str | None
"""
