"""Gemini Embedding client — gemini-embedding-exp-03-07, dim=768, batch async."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from typing import List

import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# gemini-embedding-exp-03-07 requires explicit allowlist access.
# gemini-embedding-2-preview is the production equivalent (MRL, dim=768).
MODEL = "models/gemini-embedding-2-preview"
OUTPUT_DIM = 768
BATCH_SIZE = 256          # max texts per API call
CACHE_MAXSIZE = 1000
# Gemini free-tier / paid rate limit guard — max parallel requests
_MAX_CONCURRENCY = 4


# ── LRU cache ─────────────────────────────────────────────────────────────────

class _LRUCache:
    """Thread-safe LRU cache keyed by text hash."""

    def __init__(self, maxsize: int = CACHE_MAXSIZE) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        k = self._key(text)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        k = self._key(text)
        self._cache[k] = embedding
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# ── Embedder ───────────────────────────────────────────────────────────────────

class GeminiEmbedder:
    """Async Gemini embedding client with batching, retry, and LRU cache.

    Usage::

        embedder = GeminiEmbedder()
        vectors = await embedder.embed_batch(["text1", "text2", ...])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        output_dimensionality: int = OUTPUT_DIM,
        batch_size: int = BATCH_SIZE,
        cache_maxsize: int = CACHE_MAXSIZE,
    ) -> None:
        from google import genai
        from google.genai import types as genai_types

        self._api_key = api_key or os.environ["GOOGLE_API_KEY"]
        self._model = model
        self._dim = output_dimensionality
        self._batch_size = batch_size
        self._cache = _LRUCache(cache_maxsize)
        self._sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        self._client = genai.Client(api_key=self._api_key)
        self._types = genai_types

    # ── Public ────────────────────────────────────────────────────────────────

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts.  Returns embeddings in the same order.

        Hits the LRU cache first; uncached texts are batched and sent to Gemini.
        """
        if not texts:
            return []

        # Split into cached / uncached
        result: List[np.ndarray | None] = [None] * len(texts)
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                result[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            embeddings = await self._embed_uncached(uncached_texts)
            for idx, embedding in zip(uncached_indices, embeddings):
                self._cache.put(texts[idx], embedding)
                result[idx] = embedding

        logger.debug(
            "embed_batch: %d total, %d cached, %d fetched",
            len(texts),
            len(texts) - len(uncached_indices),
            len(uncached_indices),
        )
        return result  # type: ignore[return-value]

    async def embed_one(self, text: str) -> np.ndarray:
        results = await self.embed_batch([text])
        return results[0]

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _embed_uncached(self, texts: List[str]) -> List[np.ndarray]:
        """Split into batches ≤ BATCH_SIZE and fetch concurrently."""
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        batch_results = await asyncio.gather(
            *[self._fetch_batch(b) for b in batches]
        )
        return [emb for batch in batch_results for emb in batch]

    async def _fetch_batch(self, texts: List[str]) -> List[np.ndarray]:
        async with self._sem:
            return await self._fetch_with_retry(texts)

    async def _fetch_with_retry(self, texts: List[str]) -> List[np.ndarray]:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        ):
            with attempt:
                return await self._call_api(texts)
        raise RuntimeError("unreachable")  # tenacity reraises

    async def _call_api(self, texts: List[str]) -> List[np.ndarray]:
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
            config=self._types.EmbedContentConfig(
                output_dimensionality=self._dim,
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        return [np.array(e.values, dtype=np.float32) for e in response.embeddings]

    # ── Cost estimation ────────────────────────────────────────────────────────

    @staticmethod
    def estimate_cost_usd(texts: List[str]) -> float:
        """Rough cost estimate at $0.025 / 1M tokens (4 chars ≈ 1 token)."""
        total_chars = sum(len(t) for t in texts)
        tokens = total_chars / 4
        return tokens / 1_000_000 * 0.025
