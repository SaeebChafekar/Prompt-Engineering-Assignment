"""Prompts for the codebase Q&A bot.

The system prompt is short and rule-based on purpose: small local models follow a few
clear rules better than a long essay. The context goes in the user message, each chunk
under a header that carries the exact citation label the model must copy.
"""

from __future__ import annotations

from .retrieve import Retrieval

NOT_FOUND_TEXT = "I couldn't find this in the indexed code."
EFFORT_LEVELS = ("low", "medium", "high")

_COMMON_RULES = f"""Rules:
1. Use ONLY the provided context. If it does not contain the answer, reply exactly: "{NOT_FOUND_TEXT}" and stop. Never guess and never use outside knowledge about this repository.
2. Cite every claim with its source in square brackets, copied exactly from the context headers, for example [app/auth/routes.py:20-45]. Put the citation right after the claim.
3. "Relationship facts" come from a code graph. Facts marked (likely) or (guess) may be wrong: say "probably" when you rely on them."""

SYSTEM_PROMPT = f"""You answer questions about ONE software repository, using only the context provided in the user message.

{_COMMON_RULES}
4. Be concise: give the direct answer first, then the key details. Name the files and functions involved. Do not paste long code.
5. When asked where something is handled, list the relevant files and functions, most important first."""

HIGH_EFFORT_RAG_PROMPT = f"""You are an expert software engineer and codebase guide answering questions about ONE software repository using the provided context.

{_COMMON_RULES}
4. Provide a thorough, comprehensive, and in-depth explanation:
   - Direct answer: Start with a clear executive summary answering the question directly.
   - Deep code walkthrough: Walk through the relevant functions, classes, and logic step-by-step.
   - Architecture & Call flow: Explain how components interact, citing callers, callees, decorators, and imports from the graph and code.
   - Code illustrations: Quote and explain relevant snippets from the context (with line references) to demonstrate exactly how the implementation works.
   - Edge cases & nuances: Explain error handling, data flow, parameters, and design patterns observed in the code.
5. When asked where something is handled or how it works, thoroughly describe the role of each relevant file and function, tracing the execution flow from entry point through backend processing."""

LOW_EFFORT_RAG_PROMPT = f"""You answer questions about ONE software repository, using only the context provided in the user message.

{_COMMON_RULES}
4. Be very concise: give a direct 1-3 sentence answer naming the files and functions. Do not paste long code.
5. When asked where something is handled, list the relevant files and functions, most important first."""


def get_rag_system_prompt(effort: str = "medium") -> str:
    level = (effort or "medium").strip().lower()
    if level == "high":
        return HIGH_EFFORT_RAG_PROMPT
    if level == "low":
        return LOW_EFFORT_RAG_PROMPT
    return SYSTEM_PROMPT

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question so that it can be understood on its own. "
    "Use the conversation to replace pronouns and vague references (it, that function, the same file) "
    "with the concrete names they refer to. Output ONLY the rewritten question, nothing else. "
    "If the question is already standalone, output it unchanged."
)


def format_context(retrieval: Retrieval) -> str:
    """Render the retrieved chunks and graph facts into the text shown to the model."""
    blocks: list[str] = []
    if retrieval.facts:
        blocks.append("Relationship facts (from the code graph):\n" + "\n".join(f"- {f}" for f in retrieval.facts))
    if retrieval.items:
        chunks = []
        for item in retrieval.items:
            c = item.chunk
            header = f"### [{c.citation}] ({c.kind} {c.name}) - {item.reason}"
            chunks.append(f"{header}\n```\n{c.text}\n```")
        blocks.append("Context:\n\n" + "\n\n".join(chunks))
    else:
        blocks.append("Context: (nothing relevant was found)")
    return "\n\n".join(blocks)


def build_user_message(repo_name: str, question: str, retrieval: Retrieval) -> str:
    return f"Repository: {repo_name}\n\n{format_context(retrieval)}\n\nQuestion: {question}"


def build_rewrite_message(history: list[tuple[str, str]], question: str) -> str:
    lines = []
    for asked, answered in history[-3:]:
        lines.append(f"User: {asked}")
        lines.append(f"Assistant: {answered[:300]}")
    lines.append(f"Follow-up question: {question}")
    return "Conversation so far:\n" + "\n".join(lines) + "\n\nRewritten standalone question:"
