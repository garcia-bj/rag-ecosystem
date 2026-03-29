"""Integration tests — index 3 chunks, search, verify Qdrant + ES + Neo4j.

Each test creates a fresh async client inside a single asyncio.run() to avoid
aiohttp / Neo4j driver event-loop affinity issues.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parsers.base import ParsedChunk

# ── Shared fixture chunks ──────────────────────────────────────────────────────

_CHUNKS = [
    ParsedChunk(
        text=(
            "OpenAI released GPT-4 in March 2023. "
            "It outperforms previous models on many benchmarks."
        ),
        metadata={"ner_org": ["OpenAI"], "ner_date": ["March 2023"], "title": "GPT-4"},
        modality="text",
        source_id="test_doc_001",
        chunk_index=0,
    ),
    ParsedChunk(
        text=(
            "Google DeepMind published Gemini in December 2023. "
            "The model supports multimodal inputs."
        ),
        metadata={"ner_org": ["Google", "DeepMind"], "ner_date": ["December 2023"]},
        modality="text",
        source_id="test_doc_001",
        chunk_index=1,
    ),
    ParsedChunk(
        text="employee_id,name,salary\n1,Alice,90000\n2,Bob,85000",
        metadata={"columns": ["employee_id", "name", "salary"]},
        modality="structured",
        source_id="test_csv_001",
        chunk_index=0,
    ),
]

_EMBEDDINGS = [np.random.rand(768).astype(np.float32) for _ in _CHUNKS]


def _env_ok(*keys: str) -> bool:
    for k in keys:
        v = os.getenv(k, "")
        if not v or "CHANGE_ME" in v:
            return False
    return True


# ── GeminiEmbedder unit tests (no API calls) ───────────────────────────────────

class TestGeminiEmbedderUnit:
    def test_lru_cache_stores_and_retrieves(self):
        from embeddings.gemini_embedder import _LRUCache
        cache = _LRUCache(maxsize=3)
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.put("hello", vec)
        result = cache.get("hello")
        assert result is not None
        assert np.allclose(result, vec)

    def test_lru_cache_evicts_oldest(self):
        from embeddings.gemini_embedder import _LRUCache
        cache = _LRUCache(maxsize=2)
        cache.put("a", np.zeros(3, dtype=np.float32))
        cache.put("b", np.ones(3, dtype=np.float32))
        cache.put("c", np.full(3, 2.0, dtype=np.float32))
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_lru_cache_miss_returns_none(self):
        from embeddings.gemini_embedder import _LRUCache
        assert _LRUCache().get("nonexistent") is None

    def test_cost_estimate_positive(self):
        from embeddings.gemini_embedder import GeminiEmbedder
        assert GeminiEmbedder.estimate_cost_usd(["hello world"] * 100) > 0

    def test_cost_estimate_scales(self):
        from embeddings.gemini_embedder import GeminiEmbedder
        short = GeminiEmbedder.estimate_cost_usd(["hi"])
        long_ = GeminiEmbedder.estimate_cost_usd(["hi " * 1000])
        assert long_ > short


# ── ESIndexer ──────────────────────────────────────────────────────────────────

_ES_TEST_INDEX = "rag_test_chunks"


def _make_es():
    from embeddings.indexers.es_indexer import ESIndexer
    return ESIndexer(index=_ES_TEST_INDEX)


async def _es_ping() -> bool:
    es = _make_es()
    try:
        return await es._es.ping()
    except Exception:
        return False
    finally:
        await es.close()


async def _es_drop():
    es = _make_es()
    try:
        await es._es.indices.delete(index=_ES_TEST_INDEX, ignore_unavailable=True)
    except Exception:
        pass
    finally:
        await es.close()


class TestESIndexer:
    def setup_method(self):
        asyncio.run(_es_drop())

    def teardown_method(self):
        asyncio.run(_es_drop())

    def _skip_if_down(self):
        if not asyncio.run(_es_ping()):
            pytest.skip("Elasticsearch not reachable at localhost:9200")

    def test_bulk_index_returns_count(self):
        self._skip_if_down()

        async def _run():
            es = _make_es()
            try:
                return await es.bulk_index(_CHUNKS)
            finally:
                await es.close()

        count = asyncio.run(_run())
        assert count == len(_CHUNKS)

    def test_index_created(self):
        self._skip_if_down()

        async def _run():
            es = _make_es()
            try:
                await es.bulk_index(_CHUNKS)
                return await es._es.indices.exists(index=_ES_TEST_INDEX)
            finally:
                await es.close()

        assert asyncio.run(_run())

    def test_search_after_index(self):
        self._skip_if_down()

        async def _run():
            es = _make_es()
            try:
                await es.bulk_index(_CHUNKS)
                await es._es.indices.refresh(index=_ES_TEST_INDEX)
                result = await es._es.search(
                    index=_ES_TEST_INDEX,
                    query={"match": {"text": "OpenAI"}},
                )
                return result["hits"]["hits"]
            finally:
                await es.close()

        hits = asyncio.run(_run())
        assert len(hits) >= 1
        assert "OpenAI" in hits[0]["_source"]["text"]

    def test_chunk_id_field(self):
        self._skip_if_down()

        async def _run():
            es = _make_es()
            try:
                await es.bulk_index(_CHUNKS)
                await es._es.indices.refresh(index=_ES_TEST_INDEX)
                chunk = _CHUNKS[0]
                doc_id = f"{chunk.source_id}::{chunk.chunk_index}"
                doc = await es._es.get(index=_ES_TEST_INDEX, id=doc_id)
                return doc["_source"]["source_id"]
            finally:
                await es.close()

        assert asyncio.run(_run()) == _CHUNKS[0].source_id

    def test_bulk_empty_is_noop(self):
        self._skip_if_down()

        async def _run():
            es = _make_es()
            try:
                return await es.bulk_index([])
            finally:
                await es.close()

        assert asyncio.run(_run()) == 0


# ── KnowledgeGraph ─────────────────────────────────────────────────────────────

_TEST_SOURCE_IDS = ["test_doc_001", "test_csv_001"]


async def _neo4j_ping() -> bool:
    from graph.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    try:
        await kg._get_driver()
        return True
    except Exception:
        return False
    finally:
        await kg.close()


async def _neo4j_clean():
    from graph.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    try:
        driver = await kg._get_driver()
        async with driver.session() as s:
            await s.run(
                "MATCH (n) WHERE n.source_id IN $ids DETACH DELETE n",
                ids=_TEST_SOURCE_IDS,
            )
    except Exception:
        pass
    finally:
        await kg.close()


class TestKnowledgeGraph:
    def setup_method(self):
        asyncio.run(_neo4j_clean())

    def teardown_method(self):
        asyncio.run(_neo4j_clean())

    def _skip_if_down(self):
        if not asyncio.run(_neo4j_ping()):
            pytest.skip("Neo4j not reachable at bolt://localhost:7687")

    def test_ingest_creates_documents(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks(_CHUNKS)
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (d:Document) WHERE d.source_id IN $ids RETURN count(d) AS n",
                        ids=_TEST_SOURCE_IDS,
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        assert asyncio.run(_run()) == 2

    def test_ingest_creates_chunks(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks(_CHUNKS)
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (c:Chunk) WHERE c.source_id IN $ids RETURN count(c) AS n",
                        ids=_TEST_SOURCE_IDS,
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        assert asyncio.run(_run()) == len(_CHUNKS)

    def test_ingest_creates_entities(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks(_CHUNKS)
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (e:Entity) WHERE e.name IN ['OpenAI','Google','DeepMind'] "
                        "RETURN count(e) AS n"
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        assert asyncio.run(_run()) >= 3

    def test_ingest_idempotent(self):
        self._skip_if_down()

        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks(_CHUNKS)
                await kg.ingest_chunks(_CHUNKS)  # second call — same driver, same loop
                driver = await kg._get_driver()
                async with driver.session() as s:
                    result = await s.run(
                        "MATCH (c:Chunk) WHERE c.source_id IN $ids RETURN count(c) AS n",
                        ids=_TEST_SOURCE_IDS,
                    )
                    record = await result.single()
                    return record["n"]
            finally:
                await kg.close()

        assert asyncio.run(_run()) == len(_CHUNKS)

    def test_empty_ingest_is_noop(self):
        async def _run():
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            try:
                await kg.ingest_chunks([])
            finally:
                await kg.close()

        asyncio.run(_run())  # must not raise


# ── QdrantIndexer ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("QDRANT_URL", "QDRANT_API_KEY"),
    reason="QDRANT_URL / QDRANT_API_KEY not configured",
)
class TestQdrantIndexer:
    _COLL = "rag_test"

    def _make(self):
        from embeddings.indexers.qdrant_indexer import QdrantIndexer
        return QdrantIndexer(collection=self._COLL)

    def teardown_method(self):
        async def _drop():
            q = self._make()
            try:
                await q._client.delete_collection(self._COLL)
            except Exception:
                pass
            await q.close()
        asyncio.run(_drop())

    def test_upsert_returns_correct_count(self):
        async def _run():
            q = self._make()
            try:
                return await q.upsert(_CHUNKS, _EMBEDDINGS)
            finally:
                await q.close()
        assert asyncio.run(_run()) == len(_CHUNKS)

    def test_upsert_creates_collection(self):
        async def _run():
            q = self._make()
            try:
                await q.upsert(_CHUNKS, _EMBEDDINGS)
                colls = await q._client.get_collections()
                return [c.name for c in colls.collections]
            finally:
                await q.close()
        assert self._COLL in asyncio.run(_run())

    def test_search_returns_results(self):
        async def _run():
            q = self._make()
            try:
                await q.upsert(_CHUNKS, _EMBEDDINGS)
                response = await q._client.query_points(
                    collection_name=self._COLL,
                    query=_EMBEDDINGS[0].tolist(),
                    limit=3,
                )
                return response.points
            finally:
                await q.close()
        results = asyncio.run(_run())
        assert len(results) >= 1
        assert results[0].payload["source_id"] in {"test_doc_001", "test_csv_001"}

    def test_upsert_idempotent(self):
        async def _run():
            q = self._make()
            try:
                await q.upsert(_CHUNKS, _EMBEDDINGS)
                await q.upsert(_CHUNKS, _EMBEDDINGS)
                count = await q._client.count(self._COLL)
                return count.count
            finally:
                await q.close()
        assert asyncio.run(_run()) == len(_CHUNKS)

    def test_mismatched_lengths_raise(self):
        async def _run():
            q = self._make()
            try:
                await q.upsert(_CHUNKS, _EMBEDDINGS[:1])
            finally:
                await q.close()
        with pytest.raises(ValueError):
            asyncio.run(_run())


# ── Full pipeline (real Gemini API + Qdrant) ───────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"),
    reason="GOOGLE_API_KEY / QDRANT_URL not configured",
)
class TestFullPipeline:
    _COLL = "rag_integration_test"

    def teardown_method(self):
        async def _drop():
            from embeddings.indexers.qdrant_indexer import QdrantIndexer
            q = QdrantIndexer(collection=self._COLL)
            try:
                await q._client.delete_collection(self._COLL)
            except Exception:
                pass
            await q.close()
        asyncio.run(_drop())

    def test_embed_and_search(self):
        async def _run():
            from embeddings.gemini_embedder import GeminiEmbedder
            from embeddings.indexers.qdrant_indexer import QdrantIndexer
            embedder = GeminiEmbedder()
            indexer  = QdrantIndexer(collection=self._COLL)
            try:
                chunks = _CHUNKS[:2]
                embeddings = await embedder.embed_batch([c.text for c in chunks])

                assert len(embeddings) == 2
                assert all(e.shape == (768,) for e in embeddings)
                assert all(e.dtype == np.float32 for e in embeddings)

                await indexer.upsert(chunks, embeddings)

                query_emb = await embedder.embed_one("GPT-4 language model")
                response = await indexer._client.query_points(
                    collection_name=self._COLL,
                    query=query_emb.tolist(),
                    limit=2,
                )
                return response.points
            finally:
                await indexer.close()

        results = asyncio.run(_run())
        assert len(results) >= 1
        assert results[0].payload["source_id"] == "test_doc_001"
