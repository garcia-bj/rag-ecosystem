"""Prompt builder — per-query-type templates with token budget management."""

from __future__ import annotations

import logging
import re
from typing import List, Literal

logger = logging.getLogger(__name__)

QueryType = Literal["factual", "analytical", "generative"]

# Token budgets (context) per tier model
_TOKEN_BUDGETS: dict[int | str, int] = {
    1: 6_000,    # claude-haiku — keep context small for speed
    2: 8_000,    # gpt-4o-mini
    3: 14_000,   # gpt-4o
    "pii": 4_000,  # ollama/llama3 — conservative
}

_TEMPLATES: dict[str, str] = {
    "factual": (
        "You are a precise assistant. Answer the question based ONLY on the "
        "provided context. If the answer is not in the context, say exactly: "
        "\"I don't have enough information to answer this.\"\n\n"
        "{tenant_header}"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    "analytical": (
        "You are an analytical assistant. Synthesize the provided context to "
        "answer the question. Reference specific sources when drawing conclusions.\n\n"
        "{tenant_header}"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Analysis:"
    ),
    "generative": (
        "You are a helpful assistant. Using the provided context as your primary "
        "source, answer the question thoughtfully. You may use your knowledge to "
        "clarify, but stay grounded in the context.\n\n"
        "{tenant_header}"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Response:"
    ),
}

# Keywords that shift query type
_ANALYTICAL_RE = re.compile(
    r"\b(analyz|analys|compar|evaluat|assess|explain why|contrast|review|"
    r"discuss|examine|investigat|synthesiz|interpret)\w*\b",
    re.IGNORECASE,
)
_GENERATIVE_RE = re.compile(
    r"\b(write|creat|generat|draft|compose|suggest|design|develop|build|"
    r"produc|summariz)\w*\b",
    re.IGNORECASE,
)


def classify_query_type(query: str) -> QueryType:
    """Classify query as factual / analytical / generative."""
    if _GENERATIVE_RE.search(query):
        return "generative"
    if _ANALYTICAL_RE.search(query):
        return "analytical"
    return "factual"


class PromptBuilder:
    """Build a prompt from retrieved chunks and query context.

    Usage::

        builder = PromptBuilder()
        prompt, token_est = builder.build(
            query="What is GPT-4?",
            chunks=ranked_chunks,
            tier=1,
            tenant_name="Acme Corp",
        )
    """

    def build(
        self,
        query: str,
        chunks: list,
        tier: int | str,
        tenant_name: str = "",
        query_type: QueryType | None = None,
    ) -> tuple[str, int]:
        """Build the final prompt string.

        Returns (prompt_text, estimated_token_count).
        """
        if query_type is None:
            query_type = classify_query_type(query)

        budget = _TOKEN_BUDGETS.get(tier, 6_000)
        context = self._build_context(chunks, budget)

        tenant_header = (
            f"[Tenant: {tenant_name}]\n\n" if tenant_name else ""
        )

        template = _TEMPLATES[query_type]
        prompt = template.format(
            query=query,
            context=context,
            tenant_header=tenant_header,
        )

        token_estimate = len(prompt) // 4
        return prompt, token_estimate

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_context(self, chunks: list, token_budget: int) -> str:
        """Concatenate chunk texts respecting the token budget."""
        parts: list[str] = []
        used = 0
        for i, chunk in enumerate(chunks, 1):
            text = getattr(chunk, "text", str(chunk))
            source = getattr(chunk, "source_id", "unknown")
            entry = f"[{i}] (source: {source})\n{text}"
            tokens = len(entry) // 4
            if used + tokens > token_budget:
                break
            parts.append(entry)
            used += tokens

        return "\n\n---\n\n".join(parts) if parts else "(No context available)"
