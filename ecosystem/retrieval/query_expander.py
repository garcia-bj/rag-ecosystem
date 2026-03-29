"""Query expander — HyDE + multi-query variants via Claude Haiku."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Token threshold to trigger HyDE (approximate, 4 chars ≈ 1 token)
_HYDE_MIN_TOKENS = 20

_HAIKU_MODEL = "claude-haiku-4-5-20251001"


class QueryExpander:
    """Expand a query with HyDE and multi-query variants.

    - HyDE: only when query > _HYDE_MIN_TOKENS tokens.
    - Multi-query: always generates 3 rephrased variants.
    - Both run in parallel via asyncio.gather.
    - Results are cached in Redis (TTL 1h).
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._client: object | None = None  # anthropic.AsyncAnthropic — lazy
        self._redis: object | None = None   # aioredis.Redis — lazy

    # ── Public ────────────────────────────────────────────────────────────────

    async def expand(self, query: str) -> "ExpandedQuery":
        """Return HyDE doc + multi-query variants for the given query.

        Results are cached in Redis under ``qexpand:<sha256>`` with TTL 1h.
        """
        import hashlib
        cache_key = f"qexpand:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = await self._cache_get(cache_key)
        if cached is not None:
            logger.debug("QueryExpander cache HIT query=%r", query[:60])
            return ExpandedQuery(**cached)

        token_estimate = len(query) / 4
        use_hyde = token_estimate >= _HYDE_MIN_TOKENS

        tasks = [self._multi_query(query)]
        if use_hyde:
            tasks.append(self._hyde(query))

        if use_hyde:
            variants_list, hyde_doc = await asyncio.gather(*tasks)
        else:
            (variants_list,) = await asyncio.gather(*tasks)
            hyde_doc = None

        result = ExpandedQuery(
            original=query,
            hyde_doc=hyde_doc,
            variants=variants_list,
        )

        await self._cache_set(cache_key, result.to_dict(), ttl=3600)
        return result

    async def close(self) -> None:
        if self._redis is not None:
            import redis.asyncio as aioredis
            await self._redis.aclose()  # type: ignore[union-attr]

    # ── Internal — LLM calls ─────────────────────────────────────────────────

    async def _hyde(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = (
            "Write a short, factual paragraph (3-5 sentences) that directly "
            f"answers the following question. Do not mention the question itself.\n\n"
            f"Question: {query}"
        )
        return await self._complete(prompt, max_tokens=256)

    async def _multi_query(self, query: str) -> List[str]:
        """Generate 3 rephrasings of the query."""
        prompt = (
            "Rephrase the following query in 3 different ways to improve "
            "retrieval coverage. Return ONLY a JSON array of 3 strings, "
            "no explanation.\n\n"
            f"Query: {query}"
        )
        raw = await self._complete(prompt, max_tokens=200)
        try:
            variants = json.loads(raw)
            if isinstance(variants, list):
                return [str(v) for v in variants[:3]]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: return original query as single variant
        return [query]

    async def _complete(self, prompt: str, max_tokens: int = 256) -> str:
        client = await self._get_client()
        if client is None:
            return ""
        try:
            msg = await client.messages.create(  # type: ignore[union-attr]
                model=_HAIKU_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            logger.warning("QueryExpander LLM error: %s", exc)
            return ""

    # ── Internal — helpers ───────────────────────────────────────────────────

    async def _get_client(self) -> object | None:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key or "CHANGE_ME" in api_key:
                logger.warning("ANTHROPIC_API_KEY not configured — query expansion disabled")
                return None
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
        return self._redis

    async def _cache_get(self, key: str) -> dict | None:
        try:
            r = await self._get_redis()
            raw = await r.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    async def _cache_set(self, key: str, value: dict, ttl: int) -> None:
        try:
            r = await self._get_redis()
            await r.set(key, json.dumps(value), ex=ttl)
        except Exception:
            pass


class ExpandedQuery:
    """Container for expansion results."""

    def __init__(
        self,
        original: str,
        variants: List[str],
        hyde_doc: str | None = None,
    ) -> None:
        self.original  = original
        self.hyde_doc  = hyde_doc
        self.variants  = variants

    def all_texts(self) -> List[str]:
        """All texts to embed for retrieval (original + variants + hyde)."""
        texts = [self.original] + self.variants
        if self.hyde_doc:
            texts.append(self.hyde_doc)
        return texts

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "hyde_doc": self.hyde_doc,
            "variants": self.variants,
        }
