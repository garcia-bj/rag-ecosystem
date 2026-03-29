"""MCP tool: get_tenant_stats — metrics for admin/editor roles."""

from __future__ import annotations

import logging
import os
from datetime import date

logger = logging.getLogger(__name__)


async def get_tenant_stats(tenant_id: str, role: str) -> dict:
    """Return usage statistics for the given tenant.

    Access:
        Only admin and editor roles may call this tool.

    Returns:
        {
          "chunks_by_modality": {...},
          "cache_hit_rate_24h": float,
          "llm_cost_today_usd": float,
          "total_documents": int,
        }
    """
    if role not in ("admin", "editor"):
        raise PermissionError("Only admin and editor roles can access tenant stats")

    import asyncio
    chunks, cache_rate, llm_cost, total_docs = await asyncio.gather(
        _count_chunks_by_modality(tenant_id),
        _cache_hit_rate(tenant_id),
        _llm_cost_today(tenant_id),
        _count_documents(tenant_id),
        return_exceptions=True,
    )

    return {
        "chunks_by_modality": chunks if isinstance(chunks, dict) else {},
        "cache_hit_rate_24h":  cache_rate if isinstance(cache_rate, float) else 0.0,
        "llm_cost_today_usd":  llm_cost if isinstance(llm_cost, float) else 0.0,
        "total_documents":     total_docs if isinstance(total_docs, int) else 0,
    }


async def _count_chunks_by_modality(tenant_id: str) -> dict:
    """Count chunks per modality from Elasticsearch."""
    from core.tenant import get_tenant_index
    from elasticsearch import AsyncElasticsearch

    es = AsyncElasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
        request_timeout=10,
    )
    try:
        result = await es.search(
            index=get_tenant_index(tenant_id),
            size=0,
            aggs={"by_modality": {"terms": {"field": "modality"}}},
        )
        buckets = result["aggregations"]["by_modality"]["buckets"]
        return {b["key"]: b["doc_count"] for b in buckets}
    except Exception:
        return {}
    finally:
        await es.close()


async def _cache_hit_rate(tenant_id: str) -> float:
    """Compute cache hit rate from Redis counters over last 24h."""
    import redis.asyncio as aioredis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    today = date.today().isoformat()
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        hits   = int(await r.get(f"stats:{tenant_id}:cache_hits:{today}")   or 0)
        misses = int(await r.get(f"stats:{tenant_id}:cache_misses:{today}") or 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
    except Exception:
        return 0.0
    finally:
        await r.aclose()


async def _llm_cost_today(_tenant_id: str) -> float:
    """Return today's LLM cost from Redis accumulator (written by LLMRouter)."""
    import redis.asyncio as aioredis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    today = date.today().isoformat()
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        raw = await r.get(f"stats:{_tenant_id}:llm_cost:{today}")
        return float(raw) if raw else 0.0
    except Exception:
        return 0.0
    finally:
        await r.aclose()


async def _count_documents(tenant_id: str) -> int:
    """Count documents from PostgreSQL."""
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            row = await conn.fetchrow(
                "SELECT count(*) AS n FROM documents WHERE tenant_id = $1",
                tenant_id,
            )
            return int(row["n"]) if row else 0
        finally:
            await conn.close()
    except Exception:
        return 0
