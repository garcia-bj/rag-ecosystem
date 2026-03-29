"""Per-tenant rate limiter backed by Redis."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ── Plan limits ───────────────────────────────────────────────────────────────

_PLANS: dict[str, dict[str, int]] = {
    "free":       {"queries_day": 100,   "docs_month": 10},
    "starter":    {"queries_day": 500,   "docs_month": 50},
    "pro":        {"queries_day": 1_000, "docs_month": 100},
    "enterprise": {"queries_day": 999_999, "docs_month": 999_999},
}

_DEFAULT_PLAN = "free"


class TenantRateLimiter:
    """Redis-backed per-tenant rate limiter.

    Keys:
      ``rate:{tenant_id}:queries:{YYYY-MM-DD}``  → daily query counter (TTL: 48h)
      ``rate:{tenant_id}:docs:{YYYY-MM}``        → monthly doc counter  (TTL: 62d)
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis = None  # lazy

    async def check_query(self, tenant_id: str, plan: str) -> None:
        """Raise 429 if tenant has exceeded daily query limit."""
        limit = _PLANS.get(plan, _PLANS[_DEFAULT_PLAN])["queries_day"]
        key   = f"rate:{tenant_id}:queries:{date.today().isoformat()}"
        count = await self._increment(key, ttl=172_800)  # 48h
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily query limit ({limit}) reached for plan '{plan}'.",
                headers={"Retry-After": str(_seconds_until_midnight())},
            )

    async def check_doc_ingest(self, tenant_id: str, plan: str) -> None:
        """Raise 429 if tenant has exceeded monthly document limit."""
        limit = _PLANS.get(plan, _PLANS[_DEFAULT_PLAN])["docs_month"]
        month = date.today().strftime("%Y-%m")
        key   = f"rate:{tenant_id}:docs:{month}"
        count = await self._increment(key, ttl=5_356_800)  # ~62 days
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Monthly document limit ({limit}) reached for plan '{plan}'.",
            )

    async def current_query_count(self, tenant_id: str) -> int:
        key = f"rate:{tenant_id}:queries:{date.today().isoformat()}"
        r   = await self._get_redis()
        raw = await r.get(key)
        return int(raw) if raw else 0

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _increment(self, key: str, ttl: int) -> int:
        r = await self._get_redis()
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, ttl)
        return count

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._url, encoding="utf-8", decode_responses=True
            )
        return self._redis


def _seconds_until_midnight() -> int:
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    next_midnight = midnight + timedelta(days=1)
    return int((next_midnight - now).total_seconds())


# Singleton shared across requests
_limiter: TenantRateLimiter | None = None

def get_rate_limiter() -> TenantRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = TenantRateLimiter()
    return _limiter
