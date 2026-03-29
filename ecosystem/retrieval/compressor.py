"""Context compressor — extract relevant fragment from each chunk via Claude Haiku."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List

from retrieval.hybrid_retriever import RankedChunk

logger = logging.getLogger(__name__)

# Only compress when total context exceeds this many tokens (approx: chars / 4)
_COMPRESS_THRESHOLD_TOKENS = 2000

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Max concurrent Haiku calls to avoid rate limits
_MAX_CONCURRENCY = 6


class ContextCompressor:
    """Extract the most relevant fragment of each chunk relative to the query.

    Compression is skipped when total tokens ≤ threshold to avoid unnecessary
    API cost on short contexts.

    Usage::

        compressor = ContextCompressor()
        compressed = await compressor.compress(query, chunks)
    """

    def __init__(self) -> None:
        self._client: object | None = None   # anthropic.AsyncAnthropic — lazy
        self._sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    # ── Public ────────────────────────────────────────────────────────────────

    async def compress(
        self,
        query: str,
        chunks: List[RankedChunk],
    ) -> List[RankedChunk]:
        """Return chunks with text replaced by the most relevant fragment.

        If total token estimate ≤ threshold or API is unavailable, returns
        chunks unchanged.
        """
        if not chunks:
            return []

        total_tokens = sum(len(c.text) for c in chunks) // 4
        if total_tokens <= _COMPRESS_THRESHOLD_TOKENS:
            logger.debug(
                "Compressor skipped — estimated %d tokens ≤ threshold %d",
                total_tokens, _COMPRESS_THRESHOLD_TOKENS,
            )
            return chunks

        client = await self._get_client()
        if client is None:
            logger.debug("Compressor skipped — ANTHROPIC_API_KEY not configured")
            return chunks

        logger.debug(
            "Compressor active — %d chunks, ~%d tokens", len(chunks), total_tokens
        )

        compressed = await asyncio.gather(
            *[self._compress_one(query, chunk, client) for chunk in chunks],
            return_exceptions=True,
        )

        result: List[RankedChunk] = []
        for original, maybe_chunk in zip(chunks, compressed):
            if isinstance(maybe_chunk, Exception):
                logger.debug("Compression failed for chunk %s: %s", original.chunk_id, maybe_chunk)
                result.append(original)
            else:
                result.append(maybe_chunk)

        original_tokens = total_tokens
        compressed_tokens = sum(len(c.text) for c in result) // 4
        logger.info(
            "Compressor: %d → %d tokens (~%.0f%% reduction)",
            original_tokens, compressed_tokens,
            100 * (1 - compressed_tokens / max(original_tokens, 1)),
        )

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _compress_one(
        self,
        query: str,
        chunk: RankedChunk,
        client: object,
    ) -> RankedChunk:
        async with self._sem:
            prompt = (
                f"Extract the most relevant 1-3 sentences from the passage below "
                f"that directly answer or support the query. "
                f"Return ONLY the extracted sentences, no explanation.\n\n"
                f"Query: {query}\n\n"
                f"Passage:\n{chunk.text}"
            )
            try:
                msg = await client.messages.create(  # type: ignore[union-attr]
                    model=_HAIKU_MODEL,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                compressed_text = msg.content[0].text.strip()
            except Exception as exc:
                raise exc

        # Return a copy with compressed text; preserve all other fields
        import copy
        result = copy.copy(chunk)
        result.text = compressed_text
        return result

    async def _get_client(self) -> object | None:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key or "CHANGE_ME" in api_key:
                return None
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client
