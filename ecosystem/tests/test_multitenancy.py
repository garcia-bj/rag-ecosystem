"""Multi-tenancy isolation tests.

Verifies that tenant_A data is never visible to tenant_B across all stores:
- Qdrant (separate collections)
- Elasticsearch (separate indices)
- Neo4j (tenant_id property + WHERE filtering)
- PostgreSQL RLS (row-level security)
- Redis semantic cache (tenant-prefixed keys)
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parsers.base import ParsedChunk

# ── Two synthetic tenants ─────────────────────────────────────────────────────

_TENANT_A = "aaaaaaaa-0000-4000-8000-000000000001"
_TENANT_B = "bbbbbbbb-0000-4000-8000-000000000002"

# Chunks: same source_id to prove uniqueness is enforced per-tenant
_CHUNK_A = ParsedChunk(
    text="TenantA secret document about project Alpha.",
    metadata={"ner_org": ["AlphaCorp"]},
    modality="text",
    source_id="shared_doc_001",
    chunk_index=0,
    tenant_id=_TENANT_A,
)
_CHUNK_B = ParsedChunk(
    text="TenantB confidential document about project Beta.",
    metadata={"ner_org": ["BetaCorp"]},
    modality="text",
    source_id="shared_doc_001",   # same source_id — proves cross-tenant isolation
    chunk_index=0,
    tenant_id=_TENANT_B,
)

_EMB_A = np.random.rand(768).astype(np.float32)
_EMB_B = np.random.rand(768).astype(np.float32)


def _env_ok(*keys: str) -> bool:
    for k in keys:
        v = os.getenv(k, "")
        if not v or "CHANGE_ME" in v:
            return False
    return True


# ── core/tenant.py unit tests (no I/O) ───────────────────────────────────────

class TestTenantHelpers:
    def test_get_tenant_collection(self):
        from core.tenant import get_tenant_collection
        coll = get_tenant_collection(_TENANT_A)
        assert coll == "rag_aaaaaaaa"

    def test_get_tenant_index(self):
        from core.tenant import get_tenant_index
        idx = get_tenant_index(_TENANT_B)
        assert idx == "rag_chunks_bbbbbbbb"

    def test_collections_differ_per_tenant(self):
        from core.tenant import get_tenant_collection
        assert get_tenant_collection(_TENANT_A) != get_tenant_collection(_TENANT_B)

    def test_validate_tenant_id_valid(self):
        from core.tenant import validate_tenant_id
        assert validate_tenant_id(_TENANT_A) is True

    def test_validate_tenant_id_invalid(self):
        from core.tenant import validate_tenant_id
        assert validate_tenant_id("not-a-uuid") is False

    def test_context_var_set_get(self):
        from core.tenant import TenantContext, set_tenant_context, get_current_tenant
        ctx = TenantContext(tenant_id=_TENANT_A, user_id="user_001", role="admin")
        set_tenant_context(ctx)
        retrieved = get_current_tenant()
        assert retrieved.tenant_id == _TENANT_A
        assert retrieved.role == "admin"

    def test_context_var_raises_without_set(self):
        from contextvars import copy_context
        from core.tenant import get_current_tenant

        # Run in a fresh context where no tenant is set
        def _check():
            import contextvars
            ctx = contextvars.copy_context()
            # Clear the var inside a new context
            from core.tenant import _current_tenant
            _current_tenant.set(None)
            with pytest.raises(RuntimeError):
                get_current_tenant()
        _check()

    def test_parsed_chunk_has_tenant_fields(self):
        chunk = ParsedChunk(
            text="hello",
            modality="text",
            source_id="doc1",
            chunk_index=0,
        )
        assert hasattr(chunk, "tenant_id")
        assert hasattr(chunk, "uploaded_by")
        assert chunk.tenant_id == ""
        assert chunk.uploaded_by == ""

    def test_parsed_chunk_with_tenant_id(self):
        chunk = ParsedChunk(
            text="hello",
            modality="text",
            source_id="doc1",
            chunk_index=0,
            tenant_id=_TENANT_A,
            uploaded_by="user_abc",
        )
        assert chunk.tenant_id == _TENANT_A
        assert chunk.uploaded_by == "user_abc"


# ── Qdrant isolation ──────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("QDRANT_URL", "QDRANT_API_KEY"),
    reason="QDRANT_URL / QDRANT_API_KEY not configured",
)
class TestQdrantTenantIsolation:
    def _make(self, tenant_id: str):
        from embeddings.indexers.qdrant_indexer import QdrantIndexer
        return QdrantIndexer(tenant_id=tenant_id)

    def teardown_method(self):
        async def _clean():
            from core.tenant import get_tenant_collection
            from embeddings.indexers.qdrant_indexer import _make_client
            import os
            client = _make_client(os.environ["QDRANT_URL"], os.environ.get("QDRANT_API_KEY"))
            for tid in [_TENANT_A, _TENANT_B]:
                try:
                    await client.delete_collection(get_tenant_collection(tid))
                except Exception:
                    pass
            await client.close()
        asyncio.run(_clean())

    def test_separate_collections(self):
        from core.tenant import get_tenant_collection
        assert get_tenant_collection(_TENANT_A) != get_tenant_collection(_TENANT_B)

    def test_tenant_a_doc_not_visible_to_tenant_b(self):
        async def _run():
            q_a = self._make(_TENANT_A)
            q_b = self._make(_TENANT_B)
            try:
                await q_a.upsert([_CHUNK_A], [_EMB_A])

                # Tenant B's collection doesn't exist yet; search returns empty
                try:
                    response = await q_b._client.query_points(
                        collection_name=q_b.collection,
                        query=_EMB_A.tolist(),
                        limit=5,
                    )
                    points = response.points
                except Exception:
                    points = []

                return points
            finally:
                await q_a.close()
                await q_b.close()

        points = asyncio.run(_run())
        assert len(points) == 0

    def test_tenant_a_doc_visible_to_tenant_a(self):
        async def _run():
            q_a = self._make(_TENANT_A)
            try:
                await q_a.upsert([_CHUNK_A], [_EMB_A])
                response = await q_a._client.query_points(
                    collection_name=q_a.collection,
                    query=_EMB_A.tolist(),
                    limit=5,
                )
                return response.points
            finally:
                await q_a.close()

        points = asyncio.run(_run())
        assert len(points) >= 1
        assert points[0].payload["tenant_id"] == _TENANT_A

    def test_both_tenants_indexed_independently(self):
        async def _run():
            q_a = self._make(_TENANT_A)
            q_b = self._make(_TENANT_B)
            try:
                await q_a.upsert([_CHUNK_A], [_EMB_A])
                await q_b.upsert([_CHUNK_B], [_EMB_B])

                resp_a = await q_a._client.query_points(
                    collection_name=q_a.collection, query=_EMB_A.tolist(), limit=5
                )
                resp_b = await q_b._client.query_points(
                    collection_name=q_b.collection, query=_EMB_B.tolist(), limit=5
                )
                return resp_a.points, resp_b.points
            finally:
                await q_a.close()
                await q_b.close()

        pts_a, pts_b = asyncio.run(_run())
        assert all(p.payload["tenant_id"] == _TENANT_A for p in pts_a)
        assert all(p.payload["tenant_id"] == _TENANT_B for p in pts_b)


# ── Elasticsearch isolation ───────────────────────────────────────────────────

class TestElasticsearchTenantIsolation:
    def _make(self, tenant_id: str):
        from embeddings.indexers.es_indexer import ESIndexer
        return ESIndexer(tenant_id=tenant_id)

    def setup_method(self):
        asyncio.run(self._drop())

    def teardown_method(self):
        asyncio.run(self._drop())

    async def _drop(self):
        from core.tenant import get_tenant_index
        from elasticsearch import AsyncElasticsearch
        es = AsyncElasticsearch(
            hosts=[os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
            request_timeout=10,
        )
        try:
            for tid in [_TENANT_A, _TENANT_B]:
                await es.indices.delete(
                    index=get_tenant_index(tid), ignore_unavailable=True
                )
        except Exception:
            pass
        finally:
            await es.close()

    def _skip_if_down(self):
        async def _ping():
            from elasticsearch import AsyncElasticsearch
            es = AsyncElasticsearch(
                hosts=[os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
                request_timeout=5,
            )
            try:
                return await es.ping()
            except Exception:
                return False
            finally:
                await es.close()
        if not asyncio.run(_ping()):
            pytest.skip("Elasticsearch not reachable")

    def test_separate_indices(self):
        from core.tenant import get_tenant_index
        assert get_tenant_index(_TENANT_A) != get_tenant_index(_TENANT_B)

    def test_tenant_a_doc_not_visible_to_tenant_b(self):
        self._skip_if_down()

        async def _run():
            es_a = self._make(_TENANT_A)
            es_b = self._make(_TENANT_B)
            try:
                await es_a.bulk_index([_CHUNK_A])
                await es_a._es.indices.refresh(index=es_a.index)

                # tenant_b's index doesn't exist → search returns empty
                try:
                    from elasticsearch import NotFoundError
                    result = await es_b._es.search(
                        index=es_b.index,
                        query={"match": {"text": "Alpha"}},
                    )
                    hits = result["hits"]["hits"]
                except Exception:
                    hits = []
                return hits
            finally:
                await es_a.close()
                await es_b.close()

        hits = asyncio.run(_run())
        assert len(hits) == 0

    def test_tenant_a_doc_visible_to_tenant_a(self):
        self._skip_if_down()

        async def _run():
            es_a = self._make(_TENANT_A)
            try:
                await es_a.bulk_index([_CHUNK_A])
                await es_a._es.indices.refresh(index=es_a.index)
                result = await es_a._es.search(
                    index=es_a.index,
                    query={"match": {"text": "Alpha"}},
                )
                return result["hits"]["hits"]
            finally:
                await es_a.close()

        hits = asyncio.run(_run())
        assert len(hits) >= 1
        assert hits[0]["_source"]["tenant_id"] == _TENANT_A

    def test_chunk_id_includes_tenant(self):
        self._skip_if_down()

        async def _run():
            es_a = self._make(_TENANT_A)
            try:
                await es_a.bulk_index([_CHUNK_A])
                await es_a._es.indices.refresh(index=es_a.index)
                result = await es_a._es.search(
                    index=es_a.index,
                    query={"match_all": {}},
                )
                return result["hits"]["hits"][0]["_id"]
            finally:
                await es_a.close()

        doc_id = asyncio.run(_run())
        assert _TENANT_A in doc_id


# ── Neo4j isolation ───────────────────────────────────────────────────────────

class TestNeo4jTenantIsolation:
    def setup_method(self):
        asyncio.run(self._clean())

    def teardown_method(self):
        asyncio.run(self._clean())

    async def _clean(self):
        from graph.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        try:
            driver = await kg._get_driver()
            async with driver.session() as s:
                await s.run(
                    "MATCH (n) WHERE n.tenant_id IN $tids DETACH DELETE n",
                    tids=[_TENANT_A, _TENANT_B],
                )
        except Exception:
            pass
        finally:
            await kg.close()

    def _skip_if_down(self):
        async def _ping():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg._get_driver()
                return True
            except Exception:
                return False
            finally:
                await kg.close()
        if not asyncio.run(_ping()):
            pytest.skip("Neo4j not reachable")

    def test_tenant_a_chunks_not_visible_to_b(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks([_CHUNK_A])
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (c:Chunk) WHERE c.tenant_id = $tid RETURN count(c) AS n",
                        tid=_TENANT_B,
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        count = asyncio.run(_run())
        assert count == 0

    def test_tenant_a_chunks_visible_to_a(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks([_CHUNK_A])
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (c:Chunk) WHERE c.tenant_id = $tid RETURN count(c) AS n",
                        tid=_TENANT_A,
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        count = asyncio.run(_run())
        assert count == 1

    def test_same_source_id_different_tenants(self):
        """Both tenants can index the same source_id without constraint violation."""
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                # Both chunks share source_id="shared_doc_001"
                await kg.ingest_chunks([_CHUNK_A, _CHUNK_B])
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (d:Document {source_id: 'shared_doc_001'}) "
                        "RETURN count(d) AS n"
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        # Two Document nodes: one per tenant
        count = asyncio.run(_run())
        assert count == 2

    def test_entities_are_tenant_scoped(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks([_CHUNK_A, _CHUNK_B])
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (e:Entity) WHERE e.tenant_id = $tid RETURN e.name AS name",
                        tid=_TENANT_A,
                    )
                    records = await result.data()
                    return [r["name"] for r in records]
            finally:
                await kg.close()

        names = asyncio.run(_run())
        assert "AlphaCorp" in names
        assert "BetaCorp" not in names


# ── PostgreSQL RLS isolation ──────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("POSTGRES_HOST", "POSTGRES_PASSWORD"),
    reason="PostgreSQL not configured",
)
class TestPostgreSQLRLS:
    def setup_method(self):
        asyncio.run(self._bootstrap())

    def teardown_method(self):
        asyncio.run(self._cleanup())

    async def _bootstrap(self):
        from core.database import bootstrap_schema
        await bootstrap_schema()

    async def _cleanup(self):
        import asyncpg
        from core.database import _dsn
        conn = await asyncpg.connect(_dsn())
        try:
            # Delete test tenants (cascades to users, documents, api_keys)
            await conn.execute(
                "DELETE FROM tenants WHERE slug IN ('tenant-a-test', 'tenant-b-test')"
            )
        finally:
            await conn.close()

    def test_create_tenant_inserts_row(self):
        async def _run():
            from core.database import create_tenant, get_tenant_by_id
            tenant = await create_tenant("Tenant A Test", "tenant-a-test")
            fetched = await get_tenant_by_id(tenant.id)
            return fetched
        tenant = asyncio.run(_run())
        assert tenant is not None
        assert tenant.slug == "tenant-a-test"

    def _app_conn_kwargs(self) -> dict:
        """Connection kwargs for the non-superuser rag_app role (respects RLS)."""
        return dict(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user="rag_app",
            password="rag_app_rls_2024",
            database=os.getenv("POSTGRES_DB", "rag_metadata"),
        )

    def test_rls_tenant_cannot_see_other_tenant(self):
        """rag_app (non-superuser) cannot see another tenant's rows via RLS."""
        async def _run():
            import asyncpg
            from core.database import create_tenant

            t_a = await create_tenant("TenantA RLS", "tenant-a-test")
            t_b = await create_tenant("TenantB RLS", "tenant-b-test")

            # Connect as the application role (not superuser — RLS applies)
            conn = await asyncpg.connect(**self._app_conn_kwargs())
            try:
                async with conn.transaction():
                    await conn.execute(
                        f"SET LOCAL \"app.tenant_id\" = '{t_a.id}'"
                    )
                    rows = await conn.fetch(
                        "SELECT id FROM tenants WHERE id = $1", t_b.id
                    )
                return len(rows)
            finally:
                await conn.close()

        visible_count = asyncio.run(_run())
        assert visible_count == 0

    def test_rls_tenant_can_see_own_row(self):
        async def _run():
            import asyncpg
            from core.database import create_tenant

            t_a = await create_tenant("TenantA Own", "tenant-a-test")

            conn = await asyncpg.connect(**self._app_conn_kwargs())
            try:
                async with conn.transaction():
                    await conn.execute(
                        f"SET LOCAL \"app.tenant_id\" = '{t_a.id}'"
                    )
                    rows = await conn.fetch(
                        "SELECT id FROM tenants WHERE id = $1", t_a.id
                    )
                return len(rows)
            finally:
                await conn.close()

        assert asyncio.run(_run()) == 1


# ── Redis cache isolation ─────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("REDIS_URL", "GOOGLE_API_KEY"),
    reason="REDIS_URL / GOOGLE_API_KEY not configured",
)
class TestRedisCacheIsolation:
    def teardown_method(self):
        async def _clean():
            import redis.asyncio as aioredis
            r = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=False)
            for tid in [_TENANT_A, _TENANT_B]:
                keys = await r.keys(f"cache:{tid}:*")
                if keys:
                    await r.delete(*keys)
            await r.aclose()
        asyncio.run(_clean())

    def test_cache_keys_are_tenant_scoped(self):
        from cache.semantic_cache import _prefix_emb
        assert _prefix_emb(_TENANT_A) != _prefix_emb(_TENANT_B)
        assert _TENANT_A in _prefix_emb(_TENANT_A)
        assert _TENANT_B in _prefix_emb(_TENANT_B)

    def test_tenant_a_cache_not_visible_to_tenant_b(self):
        async def _run():
            from cache.semantic_cache import SemanticCache
            cache_a = SemanticCache(tenant_id=_TENANT_A)
            cache_b = SemanticCache(tenant_id=_TENANT_B)
            try:
                await cache_a.set("What is AlphaCorp?", {"answer": "secret_A"}, ttl=60)
                result_b = await cache_b.get("What is AlphaCorp?")
                return result_b
            finally:
                await cache_a.close()
                await cache_b.close()

        result = asyncio.run(_run())
        assert result is None

    def test_tenant_a_cache_visible_to_tenant_a(self):
        async def _run():
            from cache.semantic_cache import SemanticCache
            cache_a = SemanticCache(tenant_id=_TENANT_A)
            try:
                await cache_a.set("What is AlphaCorp?", {"answer": "secret_A"}, ttl=60)
                return await cache_a.get("What is AlphaCorp?")
            finally:
                await cache_a.close()

        result = asyncio.run(_run())
        assert result is not None
        assert result["answer"] == "secret_A"
