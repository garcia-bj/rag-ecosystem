"""MCP tool: search_knowledge — retrieves chunks for a query, scoped to tenant."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


async def search_knowledge(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    modalities: List[str] | None = None,
) -> dict:
    """Retrieve top-k chunks relevant to *query* for the given tenant.

    Args:
        query:      Natural language query.
        tenant_id:  Caller's tenant UUID — enforces data isolation.
        top_k:      Maximum number of chunks to return.
        modalities: Filter by modality (text, image, audio, structured, web).

    Returns:
        {"chunks": [...], "total": int, "cached": bool}
    """
    from core.tenant import get_tenant_collection, get_tenant_index
    from retrieval.pipeline import RetrievalPipeline

    collection = get_tenant_collection(tenant_id)
    index      = get_tenant_index(tenant_id)

    pipeline = RetrievalPipeline(
        qdrant_collection=collection,
        es_index=index,
        use_expansion=False,   # keep MCP calls fast
        use_reranker=True,
        use_compressor=False,
    )
    try:
        result = await pipeline.run(query, top_k=top_k)
    finally:
        await pipeline.close()

    chunks_out = []
    for c in result.chunks:
        if modalities and c.modality not in modalities:
            continue
        chunks_out.append({
            "text":        c.text,
            "source_id":   c.source_id,
            "chunk_index": c.chunk_index,
            "modality":    c.modality,
            "score":       round(c.rrf_score, 6),
            "sources":     c.sources,
        })

    return {
        "chunks": chunks_out,
        "total":  len(chunks_out),
        "cached": result.cache_hit,
    }
