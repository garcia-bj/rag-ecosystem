"""LLM Router — tier classification, circuit breaker, fallback chain, Langfuse logging."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List

from llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from llm.pii_detector import PIIDetector
from llm.prompt_builder import PromptBuilder, classify_query_type

logger = logging.getLogger(__name__)

# ── Tier configuration ────────────────────────────────────────────────────────

# Token thresholds for tier classification
_TIER1_MAX_TOKENS = 30
_TIER2_MAX_TOKENS = 100

# litellm model identifiers
_TIER_MODELS: dict[int | str, str] = {
    1:     "claude-haiku-4-5-20251001",  # cheap, fast
    2:     "gpt-4o-mini",                # balanced
    3:     "gpt-4o",                     # powerful
    "pii": "claude-haiku-4-5-20251001",   # fallback when Ollama unavailable
}

# Estimated cost per query (USD) — rough average assuming ~500 tokens
_TIER_COSTS: dict[int | str, float] = {
    1:     0.001,
    2:     0.006,
    3:     0.03,
    "pii": 0.0,
}

# Fallback order on failure (downgrade)
_FALLBACK: dict[int | str, list[int | str]] = {
    3:     [2, 1],
    2:     [1],
    1:     [],
    "pii": [1],
}

# Keywords indicating complex reasoning (always Tier 3)
_COMPLEX_RE = __import__("re").compile(
    r"\b(reason|infer|deduc|prove|contradict|hypothes|implicat|causal"
    r"|step[- ]by[- ]step|chain[- ]of[- ]thought)\w*\b",
    __import__("re").IGNORECASE,
)


@dataclass
class LLMResponse:
    text:         str
    model:        str
    tier:         int | str
    cost_usd:     float
    prompt_tokens: int = 0
    cached:       bool = False
    pii_detected: bool = False
    metadata:     dict = field(default_factory=dict)


class LLMRouter:
    """Route LLM requests to the cheapest sufficient model.

    Classification:
    - PII detected        → ollama/llama3 (offline)
    - query < 30 tokens   → Tier 1 (claude-haiku)
    - 30-100 tokens       → Tier 2 (gpt-4o-mini)
    - > 100 tokens or     → Tier 3 (gpt-4o)
      complex reasoning

    Each provider has its own circuit breaker.  On failure, the router
    downgrades to the next available tier.
    """

    def __init__(self) -> None:
        self._pii       = PIIDetector()
        self._builder   = PromptBuilder()
        self._breakers: dict[int | str, CircuitBreaker] = {
            1:     CircuitBreaker("anthropic"),
            2:     CircuitBreaker("openai-mini"),
            3:     CircuitBreaker("openai"),
            "pii": CircuitBreaker("ollama"),
        }

    # ── Public ────────────────────────────────────────────────────────────────

    async def complete(
        self,
        query: str,
        chunks: list,
        tenant_name: str = "",
        tenant_id: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Route query + chunks to the appropriate LLM tier.

        Builds prompt internally, handles fallback, logs to Langfuse.
        """
        pii = self._pii.contains_pii(query)
        tier = "pii" if pii else self._classify_tier(query)

        query_type = classify_query_type(query)
        prompt, token_est = self._builder.build(
            query=query,
            chunks=chunks,
            tier=tier,
            tenant_name=tenant_name,
            query_type=query_type,
        )

        t_llm = time.perf_counter()
        response = await self._call_with_fallback(tier, prompt, max_tokens)
        llm_latency_ms = (time.perf_counter() - t_llm) * 1000
        response.pii_detected = pii
        response.metadata["latency_ms"] = round(llm_latency_ms, 1)

        from core.telemetry import trace_llm_call
        trace_llm_call(
            tenant_id=tenant_id,
            tier=tier,
            model=response.model,
            tokens_in=response.prompt_tokens,
            tokens_out=0,
            cost_usd=response.cost_usd,
            latency_ms=llm_latency_ms,
        )
        return response

    def classify_tier(self, query: str) -> int | str:
        """Public wrapper — exposed for testing."""
        return "pii" if self._pii.contains_pii(query) else self._classify_tier(query)

    # ── Internal — classification ─────────────────────────────────────────────

    def _classify_tier(self, query: str) -> int | str:
        tokens = len(query.split())
        if tokens < _TIER1_MAX_TOKENS:
            return 1
        if _COMPLEX_RE.search(query) or tokens > _TIER2_MAX_TOKENS:
            return 3
        return 2

    # ── Internal — LLM calls ──────────────────────────────────────────────────

    async def _call_with_fallback(
        self, tier: int | str, prompt: str, max_tokens: int
    ) -> LLMResponse:
        """Try tier; on CircuitBreakerOpen/exception → try fallbacks in order."""
        tiers_to_try = [tier] + _FALLBACK.get(tier, [])
        last_exc: Exception = RuntimeError("No tiers available")

        for t in tiers_to_try:
            try:
                return await self._breakers[t].call(
                    self._call_litellm(t, prompt, max_tokens)
                )
            except CircuitBreakerOpen as exc:
                logger.info("Tier %s circuit open, trying fallback. %s", t, exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("Tier %s failed: %s", t, exc)
                last_exc = exc

        raise last_exc

    async def _call_litellm(
        self, tier: int | str, prompt: str, max_tokens: int
    ) -> LLMResponse:
        import litellm

        model = _TIER_MODELS[tier]
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage or {}
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        cost = litellm.completion_cost(completion_response=response) or _TIER_COSTS[tier]

        return LLMResponse(
            text=text,
            model=model,
            tier=tier,
            cost_usd=cost,
            prompt_tokens=total_tokens,
        )

