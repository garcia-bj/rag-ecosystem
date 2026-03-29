"""Neo4j knowledge graph — Chunk, Document, Entity nodes and relationships.

All nodes carry a ``tenant_id`` property.  Every query filters by tenant_id to
guarantee cross-tenant isolation.
"""

from __future__ import annotations

import logging
import os
from typing import List

from neo4j import AsyncGraphDatabase, AsyncDriver

from ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

_NER_FIELDS = {
    "ner_person":  "PERSON",
    "ner_org":     "ORG",
    "ner_gpe":     "GPE",
    "ner_loc":     "LOC",
    "ner_date":    "DATE",
    "ner_money":   "MONEY",
    "ner_product": "PRODUCT",
    "ner_event":   "EVENT",
    "ner_law":     "LAW",
}


def _chunk_id(tenant_id: str, source_id: str, chunk_index: int) -> str:
    """Build a globally-unique chunk ID that includes the tenant prefix."""
    if tenant_id:
        return f"{tenant_id}::{source_id}::{chunk_index}"
    return f"{source_id}::{chunk_index}"


class KnowledgeGraph:
    """Build a property graph from ingested chunks.

    Schema::

        (:Document {source_id, tenant_id, modality})
          -[:CONTAINS]->
        (:Chunk {chunk_id, source_id, tenant_id, chunk_index, modality, text_preview})
          -[:MENTIONS {entity_type}]->
        (:Entity {name, entity_type, tenant_id})

        (:Chunk)-[:RELATED_TO {score: 1.0}]->(:Chunk)   # consecutive, same source+tenant
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri  = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user = user     or os.getenv("NEO4J_USER",     "neo4j")
        self._pass = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver: AsyncDriver | None = None

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._pass),
            )
            await self._ensure_constraints()
        return self._driver

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ── Public ────────────────────────────────────────────────────────────────

    async def ingest_chunks(self, chunks: List[ParsedChunk]) -> None:
        """Merge Documents, Chunks, Entities, and relationships into the graph."""
        if not chunks:
            return
        driver = await self._get_driver()
        async with driver.session() as session:
            for chunk in chunks:
                await session.execute_write(self._write_chunk, chunk)

        logger.info(
            "KnowledgeGraph: ingested %d chunks from %d sources",
            len(chunks),
            len({c.source_id for c in chunks}),
        )

    # ── Schema ────────────────────────────────────────────────────────────────

    async def _ensure_constraints(self) -> None:
        driver = await self._get_driver()
        async with driver.session() as session:
            # Drop legacy single-property constraints that conflict with composite keys
            legacy_drops = [
                "DROP CONSTRAINT document_source_id_unique IF EXISTS",
                "DROP CONSTRAINT entity_name_type_unique IF EXISTS",
            ]
            for cypher in legacy_drops:
                try:
                    await session.run(cypher)
                except Exception:
                    pass

            # Also drop by constraint name pattern used by Neo4j auto-naming
            try:
                result = await session.run("SHOW CONSTRAINTS")
                rows = await result.data()
                for row in rows:
                    name = row.get("name", "")
                    props = row.get("properties", [])
                    label = row.get("labelsOrTypes", [])
                    # Old Document constraint: single property source_id
                    if "Document" in label and props == ["source_id"]:
                        await session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                    # Old Entity constraint: two properties without tenant_id
                    if "Entity" in label and set(props) == {"name", "entity_type"}:
                        await session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            except Exception as exc:
                logger.debug("Constraint cleanup (non-critical): %s", exc)

            constraints = [
                # chunk_id is globally unique (includes tenant prefix when set)
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                # Composite: (source_id, tenant_id) per Document
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) "
                "REQUIRE (d.source_id, d.tenant_id) IS UNIQUE",
                # Composite: (name, entity_type, tenant_id) per Entity
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) "
                "REQUIRE (e.name, e.entity_type, e.tenant_id) IS UNIQUE",
            ]
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id, c.chunk_id)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.tenant_id)",
            ]
            for cypher in constraints + indexes:
                try:
                    await session.run(cypher)
                except Exception as exc:
                    logger.debug("Constraint/index (may already exist): %s", exc)

    # ── Write transaction ─────────────────────────────────────────────────────

    @staticmethod
    async def _write_chunk(tx, chunk: ParsedChunk) -> None:
        tenant_id = chunk.tenant_id or ""
        cid = _chunk_id(tenant_id, chunk.source_id, chunk.chunk_index)

        # 1. Merge Document (scoped to tenant)
        await tx.run(
            """
            MERGE (d:Document {source_id: $source_id, tenant_id: $tenant_id})
            ON CREATE SET d.modality = $modality, d.created_at = timestamp()
            """,
            source_id=chunk.source_id,
            tenant_id=tenant_id,
            modality=chunk.modality,
        )

        # 2. Merge Chunk + CONTAINS (scoped to tenant)
        await tx.run(
            """
            MATCH (d:Document {source_id: $source_id, tenant_id: $tenant_id})
            MERGE (c:Chunk {chunk_id: $chunk_id})
            ON CREATE SET
                c.source_id    = $source_id,
                c.tenant_id    = $tenant_id,
                c.chunk_index  = $chunk_index,
                c.modality     = $modality,
                c.text_preview = $text_preview,
                c.created_at   = timestamp()
            MERGE (d)-[:CONTAINS]->(c)
            """,
            source_id=chunk.source_id,
            tenant_id=tenant_id,
            chunk_id=cid,
            chunk_index=chunk.chunk_index,
            modality=chunk.modality,
            text_preview=chunk.text[:200],
        )

        # 3. Merge Entities + MENTIONS (tenant-scoped)
        for meta_key, entity_type in _NER_FIELDS.items():
            entities: list[str] = chunk.metadata.get(meta_key, [])
            for name in entities[:10]:
                if not name or not name.strip():
                    continue
                await tx.run(
                    """
                    MERGE (e:Entity {name: $name, entity_type: $entity_type, tenant_id: $tenant_id})
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS {entity_type: $entity_type}]->(e)
                    """,
                    name=name.strip(),
                    entity_type=entity_type,
                    tenant_id=tenant_id,
                    chunk_id=cid,
                )

        # 4. RELATED_TO between consecutive chunks (same source + tenant)
        if chunk.chunk_index > 0:
            prev_cid = _chunk_id(tenant_id, chunk.source_id, chunk.chunk_index - 1)
            await tx.run(
                """
                MATCH (prev:Chunk {chunk_id: $prev_id})
                WHERE prev.tenant_id = $tenant_id
                MATCH (curr:Chunk {chunk_id: $curr_id})
                WHERE curr.tenant_id = $tenant_id
                MERGE (prev)-[:RELATED_TO {score: 1.0}]->(curr)
                """,
                prev_id=prev_cid,
                curr_id=cid,
                tenant_id=tenant_id,
            )
