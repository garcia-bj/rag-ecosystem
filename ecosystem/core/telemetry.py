"""Langfuse telemetry — centralised tracing for queries, ingest, and LLM calls."""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Lazy singleton: None = uninitialised, False = disabled (no keys / import error)
_langfuse_client: Any = None


def _get_langfuse() -> Any | None:
    """Return the shared Langfuse client, or None if not configured."""
    global _langfuse_client
    if _langfuse_client is None:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sk = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not pk or not sk or "CHANGE_ME" in pk:
            _langfuse_client = False
            return None
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(public_key=pk, secret_key=sk)
            logger.info("Langfuse telemetry enabled")
        except Exception as exc:
            logger.debug("Langfuse init failed (non-critical): %s", exc)
            _langfuse_client = False
    return _langfuse_client if _langfuse_client is not False else None


# ── Public trace helpers ───────────────────────────────────────────────────────

def trace_query(
    tenant_id: str,
    query: str,
    response: str,
    cost_usd: float,
    latency_ms: float,
    cached: bool = False,
) -> None:
    """Emit a full query trace (retrieval + LLM) to Langfuse.

    Fails silently — never raises.
    """
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        trace = lf.trace(
            name="rag_query",
            user_id=tenant_id,
            input=query[:500],
            output=response[:500],
            metadata={
                "tenant_id":  tenant_id,
                "cost_usd":   round(cost_usd, 6),
                "latency_ms": round(latency_ms, 1),
                "cached":     cached,
            },
        )
        trace.span(
            name="query_summary",
            metadata={"cost_usd": cost_usd, "latency_ms": latency_ms},
        )
    except Exception as exc:
        logger.debug("trace_query failed (non-critical): %s", exc)


def trace_ingest(
    tenant_id: str,
    doc_id: str,
    chunk_count: int,
    modality: str,
    cost_usd: float = 0.0,
) -> None:
    """Emit a document-ingest trace."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name="rag_ingest",
            user_id=tenant_id,
            input=doc_id,
            metadata={
                "tenant_id":   tenant_id,
                "doc_id":      doc_id,
                "chunk_count": chunk_count,
                "modality":    modality,
                "cost_usd":    round(cost_usd, 6),
            },
        )
    except Exception as exc:
        logger.debug("trace_ingest failed (non-critical): %s", exc)


def trace_llm_call(
    tenant_id: str,
    tier: int | str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: float = 0.0,
    retry: bool = False,
) -> None:
    """Emit a single LLM generation span."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.generation(
            name="llm_call",
            model=model,
            usage={"input": tokens_in, "output": tokens_out, "total_cost": cost_usd},
            metadata={
                "tenant_id":  tenant_id,
                "tier":       str(tier),
                "latency_ms": round(latency_ms, 1),
                "retry":      retry,
            },
        )
    except Exception as exc:
        logger.debug("trace_llm_call failed (non-critical): %s", exc)


def trace_retrieval(
    tenant_id: str,
    query: str,
    cache_hit: bool,
    chunk_count: int,
    max_rrf_score: float,
    latency_ms: float,
    expanded: bool = False,
    reranked: bool = False,
    compressed: bool = False,
) -> None:
    """Emit a retrieval pipeline trace span."""
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.span(
            name="retrieval_pipeline",
            input=query[:200],
            metadata={
                "tenant_id":     tenant_id,
                "cache_hit":     cache_hit,
                "chunk_count":   chunk_count,
                "max_rrf_score": round(max_rrf_score, 6),
                "latency_ms":    round(latency_ms, 1),
                "expanded":      expanded,
                "reranked":      reranked,
                "compressed":    compressed,
            },
        )
    except Exception as exc:
        logger.debug("trace_retrieval failed (non-critical): %s", exc)


# ── Decorator ─────────────────────────────────────────────────────────────────

def langfuse_trace(fn: Callable) -> Callable:
    """Wrap an async function in a Langfuse span.

    Records function name, latency, and any exception that propagates.

    Usage::

        @langfuse_trace
        async def my_pipeline(arg1, arg2):
            ...
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        exc_val: Exception | None = None
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            exc_val = exc
            raise
        finally:
            lf = _get_langfuse()
            if lf is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                try:
                    lf.span(
                        name=fn.__name__,
                        metadata={
                            "latency_ms": round(latency_ms, 1),
                            "error":      str(exc_val) if exc_val else None,
                        },
                    )
                except Exception:
                    pass

    return wrapper
