"""Conditional reranker — local cross-encoder or Cohere, selectable via env var."""

from __future__ import annotations

import logging
import os
from typing import List

from retrieval.hybrid_retriever import RankedChunk

logger = logging.getLogger(__name__)

# Only rerank when best RRF score is below this (low confidence in ranking)
_RERANK_THRESHOLD = 0.72

# Cross-encoder model for local backend
_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cohere model
_COHERE_MODEL = "rerank-v3.5"


class ConditionalReranker:
    """Rerank chunks only when the top RRF score < threshold.

    Backend is selected via ``RERANKER_BACKEND`` env var:
    - ``local``  (default) → sentence-transformers cross-encoder (no cost)
    - ``cohere``           → Cohere Rerank 3.5 (requires COHERE_API_KEY)

    Both backends expose the same ``rerank(query, chunks)`` interface.
    """

    def __init__(self, backend: str | None = None) -> None:
        self._backend = (backend or os.getenv("RERANKER_BACKEND", "local")).lower()
        self._model: object | None = None   # lazy loaded

    # ── Public ────────────────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        chunks: List[RankedChunk],
        top_k: int | None = None,
    ) -> List[RankedChunk]:
        """Rerank chunks if confidence is low; otherwise return as-is.

        Args:
            query:  The user's query string.
            chunks: RRF-fused results (sorted by rrf_score descending).
            top_k:  Cap the returned list (None = return all).

        Returns:
            Re-sorted list of RankedChunk.
        """
        if not chunks:
            return []

        max_score = chunks[0].rrf_score
        if max_score >= _RERANK_THRESHOLD:
            logger.debug(
                "Reranker skipped — top RRF=%.4f ≥ threshold=%.2f",
                max_score, _RERANK_THRESHOLD,
            )
            return chunks[:top_k] if top_k else chunks

        logger.debug(
            "Reranker active — top RRF=%.4f < threshold=%.2f backend=%s",
            max_score, _RERANK_THRESHOLD, self._backend,
        )

        try:
            if self._backend == "cohere":
                reranked = await self._rerank_cohere(query, chunks)
            else:
                reranked = await self._rerank_local(query, chunks)
        except Exception as exc:
            logger.warning("Reranker error (%s): %s — returning RRF order", self._backend, exc)
            return chunks[:top_k] if top_k else chunks

        return reranked[:top_k] if top_k else reranked

    # ── Local cross-encoder ───────────────────────────────────────────────────

    async def _rerank_local(self, query: str, chunks: List[RankedChunk]) -> List[RankedChunk]:
        import asyncio
        from sentence_transformers import CrossEncoder

        if self._model is None:
            logger.info("Loading local cross-encoder: %s", _LOCAL_MODEL)
            # Load in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, lambda: CrossEncoder(_LOCAL_MODEL)
            )

        pairs = [(query, c.text) for c in chunks]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, lambda: self._model.predict(pairs)  # type: ignore[union-attr]
        )

        for chunk, score in zip(chunks, scores):
            chunk.rrf_score = float(score)

        return sorted(chunks, key=lambda c: c.rrf_score, reverse=True)

    # ── Cohere reranker ───────────────────────────────────────────────────────

    async def _rerank_cohere(self, query: str, chunks: List[RankedChunk]) -> List[RankedChunk]:
        import cohere

        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key or "CHANGE_ME" in api_key:
            raise EnvironmentError("COHERE_API_KEY not configured")

        client = cohere.AsyncClientV2(api_key=api_key)
        documents = [c.text for c in chunks]

        response = await client.rerank(
            model=_COHERE_MODEL,
            query=query,
            documents=documents,
            return_documents=False,
        )

        scored: List[RankedChunk] = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.rrf_score = result.relevance_score
            scored.append(chunk)

        return sorted(scored, key=lambda c: c.rrf_score, reverse=True)
