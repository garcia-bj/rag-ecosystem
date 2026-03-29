"""Elasticsearch BM25 indexer — bulk index chunks, tenant-aware."""

from __future__ import annotations

import logging
import os
from typing import List

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

INDEX = "rag_chunks"

_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "rag_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "text":        {"type": "text", "analyzer": "rag_text"},
            "chunk_id":    {"type": "keyword"},
            "tenant_id":   {"type": "keyword"},
            "source_id":   {"type": "keyword"},
            "modality":    {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "uploaded_by": {"type": "keyword"},
            "metadata":    {"type": "object", "dynamic": True},
        }
    },
}


class ESIndexer:
    """Bulk-index ParsedChunks into Elasticsearch for BM25 keyword search.

    When ``tenant_id`` is provided the index name is derived from it via
    ``get_tenant_index()``.  Pass ``index`` explicitly (and leave ``tenant_id``
    empty) for system-level / test usage.

    Elasticsearch is expected at localhost:9200 (local Docker service).
    """

    def __init__(
        self,
        tenant_id: str = "",
        host: str | None = None,
        index: str = INDEX,
    ) -> None:
        es_host = host or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self._es = AsyncElasticsearch(
            hosts=[es_host],
            request_timeout=30,
        )
        self._tenant_id = tenant_id

        # Resolve index: tenant-scoped name takes priority
        if tenant_id:
            from core.tenant import get_tenant_index
            self.index = get_tenant_index(tenant_id)
        else:
            self.index = index

        self._index_ready = False

    # ── Public ────────────────────────────────────────────────────────────────

    async def bulk_index(self, chunks: List[ParsedChunk]) -> int:
        """Bulk-index chunks.  Returns number of documents indexed."""
        if not chunks:
            return 0

        await self._ensure_index()

        actions = [self._chunk_action(chunk) for chunk in chunks]
        success, errors = await async_bulk(
            self._es,
            actions,
            raise_on_error=False,
            chunk_size=500,
        )

        if errors:
            logger.warning("ES bulk_index: %d errors", len(errors))
            for err in errors[:5]:
                logger.debug("ES error: %s", err)

        logger.info(
            "ESIndexer: indexed %d docs → index=%s tenant=%s (%d errors)",
            success,
            self.index,
            self._tenant_id[:8] if self._tenant_id else "system",
            len(errors),
        )
        return success

    async def close(self) -> None:
        await self._es.close()

    # ── Index management ──────────────────────────────────────────────────────

    async def _ensure_index(self) -> None:
        if self._index_ready:
            return
        exists = await self._es.indices.exists(index=self.index)
        if not exists:
            logger.info("Creating ES index '%s'", self.index)
            await self._es.indices.create(index=self.index, **_MAPPING)
        self._index_ready = True

    # ── Document serialisation ────────────────────────────────────────────────

    def _chunk_action(self, chunk: ParsedChunk) -> dict:
        effective_tenant = chunk.tenant_id or self._tenant_id
        chunk_id = (
            f"{effective_tenant}::{chunk.source_id}::{chunk.chunk_index}"
            if effective_tenant
            else f"{chunk.source_id}::{chunk.chunk_index}"
        )
        meta: dict = {}
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool, list, type(None))):
                meta[k] = v
            else:
                meta[k] = str(v)

        return {
            "_index": self.index,
            "_id": chunk_id,
            "_source": {
                "text":        chunk.text,
                "chunk_id":    chunk_id,
                "tenant_id":   effective_tenant,
                "source_id":   chunk.source_id,
                "modality":    chunk.modality,
                "chunk_index": chunk.chunk_index,
                "uploaded_by": chunk.uploaded_by,
                "metadata":    meta,
            },
        }
