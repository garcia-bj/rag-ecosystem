"""Retrieval pipeline — orchestrates cache → expand → retrieve → rerank → compress."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from retrieval.hybrid_retriever import HybridRetriever, RankedChunk
from retrieval.reranker import ConditionalReranker
from retrieval.compressor import ContextCompressor
from retrieval.query_expander import QueryExpander
from cache.semantic_cache import SemanticCache
from cache.ttl_classifier import classify_query_ttl

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Output of the full retrieval pipeline."""
    query:         str
    chunks:        List[RankedChunk]
    cache_hit:     bool      = False
    expanded:      bool      = False
    reranked:      bool      = False
    compressed:    bool      = False
    latency_ms:    float     = 0.0
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query":      self.query,
            "cache_hit":  self.cache_hit,
            "expanded":   self.expanded,
            "reranked":   self.reranked,
            "compressed": self.compressed,
            "latency_ms": round(self.latency_ms, 1),
            "chunks": [
                {
                    "chunk_id":    c.chunk_id,
                    "source_id":   c.source_id,
                    "chunk_index": c.chunk_index,
                    "text":        c.text,
                    "modality":    c.modality,
                    "rrf_score":   round(c.rrf_score, 6),
                    "sources":     c.sources,
                    "metadata":    c.metadata,
                }
                for c in self.chunks
            ],
        }


class RetrievalPipeline:
    """Full retrieval pipeline.

    Steps:
    1. Semantic cache check (return immediately on hit).
    2. Query expansion: HyDE + multi-query variants.
    3. Hybrid retrieval: Qdrant (dense) + ES (sparse) + Neo4j (graph).
    4. RRF fusion.
    5. Conditional reranking (only when max RRF < 0.72).
    6. Context compression (only when total tokens > 2000).
    7. Cache set with TTL based on query type.

    Usage::

        pipeline = RetrievalPipeline()
        result = await pipeline.run("What is GPT-4?", top_k=5)
        await pipeline.close()
    """

    def __init__(
        self,
        qdrant_collection: str = "rag_ecosystem",
        es_index: str = "rag_chunks",
        use_cache: bool = True,
        use_expansion: bool = True,
        use_reranker: bool = True,
        use_compressor: bool = True,
    ) -> None:
        self._use_cache      = use_cache
        self._use_expansion  = use_expansion
        self._use_reranker   = use_reranker
        self._use_compressor = use_compressor

        self._cache      = SemanticCache()           if use_cache      else None
        self._expander   = QueryExpander()           if use_expansion  else None
        self._retriever  = HybridRetriever(
            qdrant_collection=qdrant_collection,
            es_index=es_index,
        )
        self._reranker   = ConditionalReranker()     if use_reranker   else None
        self._compressor = ContextCompressor()       if use_compressor else None
        self._embedder: Any = None   # GeminiEmbedder — lazy

    # ── Public ────────────────────────────────────────────────────────────────

    async def run(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Execute the full retrieval pipeline for a query."""
        t0 = time.perf_counter()

        # 1. Cache check
        if self._cache is not None:
            cached = await self._cache.get(query)
            if cached is not None:
                chunks = _dict_list_to_ranked_chunks(cached.get("chunks", []))
                return RetrievalResult(
                    query=query,
                    chunks=chunks[:top_k],
                    cache_hit=True,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # 2. Query expansion
        expanded = False
        texts_to_embed = [query]
        if self._expander is not None:
            exp = await self._expander.expand(query)
            texts_to_embed = exp.all_texts()
            expanded = len(texts_to_embed) > 1
            logger.debug("Query expanded to %d texts", len(texts_to_embed))

        # 3. Embed all query texts and average the vectors
        embedder = await self._get_embedder()
        embeddings = await embedder.embed_batch(texts_to_embed)
        query_vec  = np.mean(np.stack(embeddings), axis=0).astype(np.float32)

        # 4. Hybrid retrieval + RRF fusion
        chunks = await self._retriever.retrieve(query_vec, query, top_k=top_k * 5)

        # 5. Conditional reranking
        reranked = False
        if self._reranker is not None and chunks:
            before_top = chunks[0].rrf_score if chunks else 0.0
            chunks = await self._reranker.rerank(query, chunks, top_k=top_k * 2)
            after_top = chunks[0].rrf_score if chunks else 0.0
            reranked = after_top != before_top

        chunks = chunks[:top_k]

        # 6. Context compression
        compressed = False
        if self._compressor is not None and chunks:
            total_tokens = sum(len(c.text) for c in chunks) // 4
            chunks = await self._compressor.compress(query, chunks)
            compressed_tokens = sum(len(c.text) for c in chunks) // 4
            compressed = compressed_tokens < total_tokens

        # 7. Cache the result
        if self._cache is not None:
            result_dict = RetrievalResult(
                query=query, chunks=chunks,
                expanded=expanded, reranked=reranked, compressed=compressed,
            ).to_dict()
            ttl = classify_query_ttl(query)
            await self._cache.set(query, result_dict, ttl=ttl)

        total_latency_ms = (time.perf_counter() - t0) * 1000
        max_rrf = chunks[0].rrf_score if chunks else 0.0

        try:
            from core.telemetry import trace_retrieval
            trace_retrieval(
                tenant_id=getattr(self, "_tenant_id", ""),
                query=query,
                cache_hit=False,
                chunk_count=len(chunks),
                max_rrf_score=max_rrf,
                latency_ms=total_latency_ms,
                expanded=expanded,
                reranked=reranked,
                compressed=compressed,
            )
        except Exception:
            pass

        return RetrievalResult(
            query=query,
            chunks=chunks,
            cache_hit=False,
            expanded=expanded,
            reranked=reranked,
            compressed=compressed,
            latency_ms=total_latency_ms,
        )

    async def close(self) -> None:
        import asyncio
        coros = [self._retriever.close()]
        if self._cache is not None:
            coros.append(self._cache.close())
        if self._expander is not None:
            coros.append(self._expander.close())
        await asyncio.gather(*coros, return_exceptions=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_embedder(self):
        if self._embedder is None:
            from embeddings.gemini_embedder import GeminiEmbedder
            self._embedder = GeminiEmbedder()
        return self._embedder


def _dict_list_to_ranked_chunks(data: list) -> List[RankedChunk]:
    chunks = []
    for d in data:
        chunks.append(RankedChunk(
            chunk_id=d.get("chunk_id", ""),
            source_id=d.get("source_id", ""),
            chunk_index=d.get("chunk_index", 0),
            text=d.get("text", ""),
            modality=d.get("modality", "text"),
            rrf_score=d.get("rrf_score", 0.0),
            metadata=d.get("metadata", {}),
            sources=d.get("sources", []),
        ))
    return chunks
