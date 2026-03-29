"""Semantic cache — Redis backend, cosine similarity threshold, Gemini embeddings.

Keys are tenant-scoped: ``cache:{tenant_id}:emb:{hash}`` /
``cache:{tenant_id}:resp:{hash}``.  When tenant_id is empty the legacy
``semcache:`` prefix is used so existing tests continue to work.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from cache.normalizer import normalize_query
from cache.ttl_classifier import classify_query_ttl

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.87


def _prefix_emb(tenant_id: str) -> str:
    return f"cache:{tenant_id}:emb:" if tenant_id else "semcache:emb:"


def _prefix_resp(tenant_id: str) -> str:
    return f"cache:{tenant_id}:resp:" if tenant_id else "semcache:resp:"


class SemanticCache:
    """Semantic similarity cache backed by Redis.

    On ``get``:
      1. Normalize and embed the incoming query.
      2. Scan all stored embeddings *for this tenant*.
      3. Return the cached response if cosine similarity ≥ threshold.

    On ``set``:
      Store embedding + response with the given TTL under a tenant-scoped key.

    Usage::

        cache = SemanticCache(tenant_id="550e8400-...")
        hit = await cache.get("What is GPT-4?")
        if hit is None:
            result = await expensive_pipeline(query)
            ttl = classify_query_ttl(query)
            await cache.set(query, result, ttl=ttl)
    """

    def __init__(
        self,
        tenant_id: str = "",
        redis_url: str | None = None,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis: aioredis.Redis = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=False,
        )
        self._tenant_id = tenant_id
        self._threshold = threshold
        self._embedder: Any = None  # lazy init

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(self, query: str) -> Any | None:
        """Return cached response for query, or None on miss."""
        norm = normalize_query(query)
        query_vec = await self._embed(norm)

        prefix_emb = _prefix_emb(self._tenant_id)
        keys = await self._redis.keys(f"{prefix_emb}*")
        if not keys:
            return None

        best_score = -1.0
        best_key   = None

        for key in keys:
            raw = await self._redis.get(key)
            if raw is None:
                continue
            stored_vec = np.array(json.loads(raw), dtype=np.float32)
            score = _cosine(query_vec, stored_vec)
            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= self._threshold and best_key is not None:
            cache_id = best_key.decode().removeprefix(prefix_emb)
            resp_raw = await self._redis.get(f"{_prefix_resp(self._tenant_id)}{cache_id}")
            if resp_raw is not None:
                logger.debug(
                    "SemanticCache HIT  score=%.4f tenant=%s query=%r",
                    best_score, self._tenant_id[:8] or "system", query[:60],
                )
                return json.loads(resp_raw)

        logger.debug(
            "SemanticCache MISS score=%.4f tenant=%s query=%r",
            best_score, self._tenant_id[:8] or "system", query[:60],
        )
        return None

    async def set(self, query: str, response: Any, ttl: int | None = None) -> None:
        """Store query embedding + response in Redis with optional TTL."""
        norm = normalize_query(query)
        if ttl is None:
            ttl = classify_query_ttl(query)

        query_vec = await self._embed(norm)

        import hashlib
        cache_id = hashlib.sha256(norm.encode()).hexdigest()

        emb_key  = f"{_prefix_emb(self._tenant_id)}{cache_id}"
        resp_key = f"{_prefix_resp(self._tenant_id)}{cache_id}"

        pipe = self._redis.pipeline()
        pipe.set(emb_key,  json.dumps(query_vec.tolist()),    ex=ttl)
        pipe.set(resp_key, json.dumps(response, default=str), ex=ttl)
        await pipe.execute()

        logger.debug(
            "SemanticCache SET  ttl=%ds cache_id=%s tenant=%s query=%r",
            ttl, cache_id[:8], self._tenant_id[:8] or "system", query[:60],
        )

    async def close(self) -> None:
        await self._redis.aclose()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            from embeddings.gemini_embedder import GeminiEmbedder
            self._embedder = GeminiEmbedder()
        return await self._embedder.embed_one(text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (returns -1..1)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
