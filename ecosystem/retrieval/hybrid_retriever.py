"""Hybrid retriever — dense (Qdrant) + sparse (ES) + graph (Neo4j), fused with RRF."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# RRF constant (Cormack et al., 2009)
_RRF_K = 60


@dataclass
class RankedChunk:
    """A retrieved chunk with its fusion score and origin metadata."""
    chunk_id:    str
    source_id:   str
    chunk_index: int
    text:        str
    modality:    str
    rrf_score:   float
    metadata:    Dict[str, Any] = field(default_factory=dict)
    # Which retrievers returned this chunk
    sources:     List[str] = field(default_factory=list)


class HybridRetriever:
    """Retrieve chunks from Qdrant (dense) + ES (sparse) + Neo4j (graph),
    then fuse results with Reciprocal Rank Fusion.

    Usage::

        retriever = HybridRetriever()
        chunks = await retriever.retrieve(query_embedding, query_text, top_k=10)
        await retriever.close()
    """

    def __init__(
        self,
        qdrant_collection: str = "rag_ecosystem",
        es_index: str = "rag_chunks",
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        graph_top_k: int = 20,
    ) -> None:
        self._qdrant_collection = qdrant_collection
        self._es_index = es_index
        self._dense_top_k  = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._graph_top_k  = graph_top_k

        # Lazy clients
        self._qdrant: Any | None = None
        self._es:     Any | None = None
        self._neo4j:  Any | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
    ) -> List[RankedChunk]:
        """Run all three retrievers in parallel and return top_k fused results."""
        dense_task  = self._dense_search(query_embedding)
        sparse_task = self._sparse_search(query_text)
        graph_task  = self._graph_search(query_text)

        dense_hits, sparse_hits, graph_hits = await asyncio.gather(
            dense_task, sparse_task, graph_task
        )

        fused = _rrf_fuse(
            [dense_hits, sparse_hits, graph_hits],
            ["qdrant", "elasticsearch", "neo4j"],
            k=_RRF_K,
        )

        return fused[:top_k]

    async def close(self) -> None:
        coros = []
        if self._qdrant is not None:
            coros.append(self._qdrant.close())
        if self._es is not None:
            coros.append(self._es.close())
        if self._neo4j is not None:
            coros.append(self._neo4j.close())
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    # ── Dense — Qdrant ────────────────────────────────────────────────────────

    async def _dense_search(self, embedding: np.ndarray) -> List[RankedChunk]:
        try:
            client = await self._get_qdrant()
            response = await client.query_points(
                collection_name=self._qdrant_collection,
                query=embedding.tolist(),
                limit=self._dense_top_k,
            )
            return [
                RankedChunk(
                    chunk_id=str(p.id),
                    source_id=p.payload.get("source_id", ""),
                    chunk_index=p.payload.get("chunk_index", 0),
                    text=p.payload.get("text", ""),
                    modality=p.payload.get("modality", "text"),
                    rrf_score=0.0,
                    metadata=p.payload.get("metadata", {}),
                    sources=["qdrant"],
                )
                for p in response.points
            ]
        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return []

    # ── Sparse — Elasticsearch ────────────────────────────────────────────────

    async def _sparse_search(self, query_text: str) -> List[RankedChunk]:
        try:
            es = await self._get_es()
            result = await es.search(
                index=self._es_index,
                query={"match": {"text": {"query": query_text}}},
                size=self._sparse_top_k,
            )
            hits = result["hits"]["hits"]
            return [
                RankedChunk(
                    chunk_id=h["_id"],
                    source_id=h["_source"].get("source_id", ""),
                    chunk_index=h["_source"].get("chunk_index", 0),
                    text=h["_source"].get("text", ""),
                    modality=h["_source"].get("modality", "text"),
                    rrf_score=0.0,
                    metadata=h["_source"].get("metadata", {}),
                    sources=["elasticsearch"],
                )
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Sparse search failed: %s", exc)
            return []

    # ── Graph — Neo4j ─────────────────────────────────────────────────────────

    async def _graph_search(self, query_text: str) -> List[RankedChunk]:
        """Find chunks related to entities mentioned in the query."""
        try:
            kg = await self._get_neo4j()
            driver = await kg._get_driver()

            # Extract potential entity names from query (capitalized words as heuristic)
            entity_names = [
                w for w in query_text.split()
                if w and w[0].isupper() and len(w) > 1
            ]
            if not entity_names:
                return []

            async with driver.session() as s:
                result = await s.run(
                    """
                    MATCH (e:Entity)-[:MENTIONS]-(c:Chunk)
                    WHERE e.name IN $names
                    RETURN DISTINCT
                        c.chunk_id   AS chunk_id,
                        c.source_id  AS source_id,
                        c.chunk_index AS chunk_index,
                        c.text_preview AS text,
                        c.modality   AS modality
                    LIMIT $limit
                    """,
                    names=entity_names,
                    limit=self._graph_top_k,
                )
                records = await result.data()

            return [
                RankedChunk(
                    chunk_id=r.get("chunk_id", ""),
                    source_id=r.get("source_id", ""),
                    chunk_index=r.get("chunk_index", 0),
                    text=r.get("text", ""),
                    modality=r.get("modality", "text"),
                    rrf_score=0.0,
                    sources=["neo4j"],
                )
                for r in records
            ]
        except Exception as exc:
            logger.warning("Graph search failed: %s", exc)
            return []

    # ── Lazy client initialisation ────────────────────────────────────────────

    async def _get_qdrant(self):
        if self._qdrant is None:
            from embeddings.indexers.qdrant_indexer import _make_client
            url     = os.environ["QDRANT_URL"]
            api_key = os.environ.get("QDRANT_API_KEY")
            self._qdrant = _make_client(url, api_key)
        return self._qdrant

    async def _get_es(self):
        if self._es is None:
            from elasticsearch import AsyncElasticsearch
            host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
            self._es = AsyncElasticsearch(hosts=[host], request_timeout=30)
        return self._es

    async def _get_neo4j(self):
        if self._neo4j is None:
            from graph.knowledge_graph import KnowledgeGraph
            self._neo4j = KnowledgeGraph()
        return self._neo4j


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    result_lists: List[List[RankedChunk]],
    source_names: List[str],
    k: int = 60,
) -> List[RankedChunk]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    score(d) = Σ  1 / (k + rank_i(d))
    """
    # chunk_id → accumulated RRF score + merged RankedChunk
    scores: Dict[str, float] = {}
    chunks: Dict[str, RankedChunk] = {}

    for result_list, source in zip(result_lists, source_names):
        for rank, chunk in enumerate(result_list, start=1):
            cid = chunk.chunk_id or f"{chunk.source_id}::{chunk.chunk_index}"
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

            if cid not in chunks:
                chunks[cid] = chunk
            else:
                # Merge source provenance
                if source not in chunks[cid].sources:
                    chunks[cid].sources.append(source)

    # Assign final scores and sort
    for cid, score in scores.items():
        chunks[cid].rrf_score = score
        # Ensure canonical chunk_id
        if not chunks[cid].chunk_id:
            chunks[cid].chunk_id = cid

    return sorted(chunks.values(), key=lambda c: c.rrf_score, reverse=True)
