"""Celery task — consumes 'embed' queue: embed → index Qdrant + ES + Neo4j."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ingestion.parsers.base import ParsedChunk
from ingestion.workers.celery_app import app

logger = logging.getLogger(__name__)

# ── Lazy singletons (one per worker process) ──────────────────────────────────

_embedder = None
_qdrant   = None
_es       = None
_graph    = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from embeddings.gemini_embedder import GeminiEmbedder
        _embedder = GeminiEmbedder()
    return _embedder


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from embeddings.indexers.qdrant_indexer import QdrantIndexer
        _qdrant = QdrantIndexer()
    return _qdrant


def _get_es():
    global _es
    if _es is None:
        from embeddings.indexers.es_indexer import ESIndexer
        _es = ESIndexer()
    return _es


def _get_graph():
    global _graph
    if _graph is None:
        from graph.knowledge_graph import KnowledgeGraph
        _graph = KnowledgeGraph()
    return _graph


# ── Task ──────────────────────────────────────────────────────────────────────

@app.task(
    name="embeddings.workers.index_task.embed_chunks",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    queue="embed",
)
def embed_chunks(self, chunks: list[dict[str, Any]], tenant_id: str = "") -> dict[str, Any]:
    """Embed and index a batch of ParsedChunks.

    Pipeline (all parallel at the indexing stage):
        1. Parse chunk dicts back to ParsedChunk objects
        2. Embed texts via Gemini
        3. asyncio.gather → Qdrant upsert + ES bulk_index + Neo4j ingest

    Returns summary dict with counts and estimated cost.
    """
    if not chunks:
        return {"status": "empty", "chunk_count": 0}

    # Reset singletons so each task gets fresh async clients (avoids closed event loop)
    global _embedder, _qdrant, _es, _graph
    _embedder = _qdrant = _es = _graph = None

    try:
        return asyncio.run(_pipeline(chunks, tenant_id=tenant_id))
    except Exception as exc:
        logger.exception("embed_chunks failed for %d chunks", len(chunks))
        raise self.retry(exc=exc)


# ── Async pipeline ─────────────────────────────────────────────────────────────

async def _pipeline(raw_chunks: list[dict], tenant_id: str = "") -> dict[str, Any]:
    # 1. Deserialise
    parsed = [ParsedChunk.model_validate(c) for c in raw_chunks]
    texts = [c.text for c in parsed]

    # 2. Cost estimate (logged before API call)
    from embeddings.gemini_embedder import GeminiEmbedder
    cost_usd = GeminiEmbedder.estimate_cost_usd(texts)
    logger.info(
        "embed_chunks: batch=%d  est_cost=USD %.6f",
        len(texts),
        cost_usd,
    )

    # 3. Embed
    embedder = _get_embedder()
    embeddings = await embedder.embed_batch(texts)
    logger.info(
        "embed_chunks: got %d embeddings  cache_size=%d",
        len(embeddings),
        embedder.cache_size,
    )

    # 4. Index in parallel: Qdrant + ES + Neo4j (using tenant-specific collection/index)
    from embeddings.indexers.qdrant_indexer import QdrantIndexer
    from embeddings.indexers.es_indexer import ESIndexer
    qdrant = QdrantIndexer(tenant_id=tenant_id)
    es     = ESIndexer(tenant_id=tenant_id)
    qdrant_count, es_count = await asyncio.gather(
        qdrant.upsert(parsed, embeddings),
        es.bulk_index(parsed),
    )
    # Neo4j is graph-only (no embeddings), run after so it doesn't block embedding
    await _get_graph().ingest_chunks(parsed)

    result = {
        "status": "ok",
        "chunk_count": len(parsed),
        "qdrant_upserted": qdrant_count,
        "es_indexed": es_count,
        "neo4j_ingested": len(parsed),
        "estimated_cost_usd": round(cost_usd, 8),
        "source_ids": list({c.source_id for c in parsed}),
    }
    logger.info("embed_chunks done: %s", result)
    return result
