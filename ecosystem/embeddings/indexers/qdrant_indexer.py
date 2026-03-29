"""Qdrant indexer — upsert chunks with 768-dim Cosine vectors, tenant-aware."""

from __future__ import annotations

import logging
import os
import uuid
from typing import List
from urllib.parse import urlparse

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

COLLECTION = "rag_ecosystem"
VECTOR_SIZE = 768


def _point_id(tenant_id: str, source_id: str, chunk_index: int) -> str:
    """Stable UUID from tenant_id + source_id + chunk_index."""
    key = f"{tenant_id}::{source_id}::{chunk_index}" if tenant_id else f"{source_id}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _make_client(url: str, api_key: str | None) -> AsyncQdrantClient:
    """Build AsyncQdrantClient that works with HTTPS traefik URLs.

    qdrant-client 1.x needs explicit host/port/https when behind a TLS proxy;
    the url= shorthand does not always resolve SSL correctly on Windows.
    """
    parsed = urlparse(url)
    https = parsed.scheme == "https"
    port = parsed.port or (443 if https else 80)
    return AsyncQdrantClient(
        host=parsed.hostname,
        port=port,
        api_key=api_key,
        https=https,
        prefer_grpc=False,
        check_compatibility=False,
    )


class QdrantIndexer:
    """Upsert ParsedChunks with pre-computed embeddings into Qdrant.

    When ``tenant_id`` is provided the collection name is derived from it via
    ``get_tenant_collection()``.  Pass ``collection`` explicitly (and leave
    ``tenant_id`` empty) for system-level / test usage.

    Reads QDRANT_URL and QDRANT_API_KEY exclusively from environment variables —
    never falls back to localhost.
    """

    def __init__(
        self,
        tenant_id: str = "",
        collection: str = COLLECTION,
        vector_size: int = VECTOR_SIZE,
    ) -> None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")

        if not url:
            raise EnvironmentError(
                "QDRANT_URL is not set. "
                "This indexer requires an external Qdrant instance — "
                "do NOT use localhost."
            )

        self._client = _make_client(url, api_key or None)
        self._tenant_id = tenant_id

        # Resolve collection: tenant-scoped name takes priority
        if tenant_id:
            from core.tenant import get_tenant_collection
            self.collection = get_tenant_collection(tenant_id)
        else:
            self.collection = collection

        self.vector_size = vector_size
        self._collection_ready = False

    # ── Public ────────────────────────────────────────────────────────────────

    async def upsert(
        self,
        chunks: List[ParsedChunk],
        embeddings: List[np.ndarray],
    ) -> int:
        """Upsert chunks with their embeddings.  Returns number of points upserted."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length"
            )
        if not chunks:
            return 0

        await self._ensure_collection()

        points = [
            PointStruct(
                id=_point_id(chunk.tenant_id or self._tenant_id, chunk.source_id, chunk.chunk_index),
                vector=embedding.tolist(),
                payload=self._chunk_payload(chunk, self._tenant_id),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        await self._client.upsert(collection_name=self.collection, points=points)
        logger.info(
            "QdrantIndexer: upserted %d points → collection=%s tenant=%s",
            len(points),
            self.collection,
            self._tenant_id[:8] if self._tenant_id else "system",
        )
        return len(points)

    async def close(self) -> None:
        await self._client.close()

    # ── Collection management ─────────────────────────────────────────────────

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            await self._client.get_collection(self.collection)
            logger.debug("Collection '%s' already exists", self.collection)
        except Exception:
            logger.info("Creating collection '%s'", self.collection)
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for fast filtering
            for field, schema in (
                ("tenant_id",   PayloadSchemaType.KEYWORD),
                ("source_id",   PayloadSchemaType.KEYWORD),
                ("modality",    PayloadSchemaType.KEYWORD),
                ("chunk_index", PayloadSchemaType.INTEGER),
            ):
                await self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )

        self._collection_ready = True

    # ── Payload serialisation ─────────────────────────────────────────────────

    @staticmethod
    def _chunk_payload(chunk: ParsedChunk, tenant_id: str = "") -> dict:
        """Flatten chunk to a JSON-safe Qdrant payload."""
        meta: dict = {}
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool, list, type(None))):
                meta[k] = v
            else:
                meta[k] = str(v)

        effective_tenant = chunk.tenant_id or tenant_id

        return {
            "text":        chunk.text[:2000],
            "tenant_id":   effective_tenant,
            "source_id":   chunk.source_id,
            "chunk_index": chunk.chunk_index,
            "modality":    chunk.modality,
            "uploaded_by": chunk.uploaded_by,
            "metadata":    meta,
        }
