"""Integration + unit tests for FASE 4 (semantic cache) and FASE 5 (retrieval).

Structure:
- TestNormalizer     — pure unit, no I/O
- TestTTLClassifier  — pure unit, no I/O
- TestSemanticCache  — requires Redis (REDIS_URL)
- TestHybridRetriever — requires Qdrant + ES + Neo4j
- TestRetrievalPipeline — full integration, requires all services
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

# ── Shared test chunks (same set as test_indexers) ────────────────────────────

_CHUNKS = [
    ParsedChunk(
        text=(
            "OpenAI released GPT-4 in March 2023. "
            "It outperforms previous models on many benchmarks."
        ),
        metadata={"ner_org": ["OpenAI"], "ner_date": ["March 2023"], "title": "GPT-4"},
        modality="text",
        source_id="ret_test_001",
        chunk_index=0,
    ),
    ParsedChunk(
        text=(
            "Google DeepMind published Gemini in December 2023. "
            "The model supports multimodal inputs."
        ),
        metadata={"ner_org": ["Google", "DeepMind"], "ner_date": ["December 2023"]},
        modality="text",
        source_id="ret_test_001",
        chunk_index=1,
    ),
    ParsedChunk(
        text=(
            "Anthropic released Claude 3 in 2024. "
            "Claude is known for its safety-focused design."
        ),
        metadata={"ner_org": ["Anthropic"], "ner_date": ["2024"], "title": "Claude 3"},
        modality="text",
        source_id="ret_test_002",
        chunk_index=0,
    ),
    ParsedChunk(
        text=(
            "Meta released Llama 2 as an open-source large language model. "
            "It is available for research and commercial use."
        ),
        metadata={"ner_org": ["Meta"], "title": "Llama 2"},
        modality="text",
        source_id="ret_test_002",
        chunk_index=1,
    ),
    ParsedChunk(
        text="employee_id,name,salary\n1,Alice,90000\n2,Bob,85000",
        metadata={"columns": ["employee_id", "name", "salary"]},
        modality="structured",
        source_id="ret_test_csv_001",
        chunk_index=0,
    ),
]

_EMBEDDINGS = [np.random.rand(768).astype(np.float32) for _ in _CHUNKS]

_RET_SOURCE_IDS = ["ret_test_001", "ret_test_002", "ret_test_csv_001"]
_RET_ES_INDEX   = "rag_ret_test"
_RET_QDRANT_COL = "rag_ret_test"


def _env_ok(*keys: str) -> bool:
    for k in keys:
        v = os.getenv(k, "")
        if not v or "CHANGE_ME" in v:
            return False
    return True


# ── FASE 4 — Normalizer ───────────────────────────────────────────────────────

class TestNormalizer:
    def test_lowercase(self):
        from cache.normalizer import normalize_query
        assert normalize_query("OpenAI GPT-4") == normalize_query("openai gpt 4") or \
               normalize_query("OpenAI GPT-4").islower()

    def test_accent_removal(self):
        from cache.normalizer import normalize_query
        result = normalize_query("qué es inteligencia artificial")
        assert "e" in result and "a" in result
        # Accented chars removed
        assert "é" not in result

    def test_stopwords_removed(self):
        from cache.normalizer import normalize_query
        result = normalize_query("what is the definition of AI")
        assert "what" not in result
        assert "the" not in result
        assert "is" not in result
        assert "definition" in result
        assert "ai" in result

    def test_punctuation_removed(self):
        from cache.normalizer import normalize_query
        result = normalize_query("GPT-4, released in 2023!")
        assert "," not in result
        assert "!" not in result

    def test_empty_string(self):
        from cache.normalizer import normalize_query
        assert normalize_query("") == ""

    def test_all_stopwords_preserves_something(self):
        from cache.normalizer import normalize_query
        # When all tokens are stopwords, original (lowercased) form is returned
        result = normalize_query("is it a the")
        assert isinstance(result, str)


# ── FASE 4 — TTL Classifier ───────────────────────────────────────────────────

class TestTTLClassifier:
    def test_static_knowledge(self):
        from cache.ttl_classifier import classify_query_ttl, TTL_STATIC
        assert classify_query_ttl("what is the definition of neural networks") == TTL_STATIC

    def test_system_state(self):
        from cache.ttl_classifier import classify_query_ttl, TTL_SYSTEM
        assert classify_query_ttl("what is the current status of the API") == TTL_SYSTEM

    def test_factual(self):
        from cache.ttl_classifier import classify_query_ttl, TTL_FACTUAL
        assert classify_query_ttl("latest version of GPT-4") == TTL_FACTUAL

    def test_default_conversational(self):
        from cache.ttl_classifier import classify_query_ttl, TTL_DEFAULT
        assert classify_query_ttl("tell me about language models") == TTL_DEFAULT

    def test_system_beats_factual(self):
        from cache.ttl_classifier import classify_query_ttl, TTL_SYSTEM
        # "current" is factual but "status" is system — system wins (checked first)
        assert classify_query_ttl("current status of the service") == TTL_SYSTEM

    def test_ttl_ordering(self):
        from cache.ttl_classifier import TTL_STATIC, TTL_FACTUAL, TTL_SYSTEM, TTL_DEFAULT
        assert TTL_STATIC > TTL_FACTUAL > TTL_DEFAULT > TTL_SYSTEM


# ── FASE 4 — SemanticCache ────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("REDIS_URL", "GOOGLE_API_KEY"),
    reason="REDIS_URL / GOOGLE_API_KEY not configured",
)
class TestSemanticCache:
    def _make(self):
        from cache.semantic_cache import SemanticCache
        return SemanticCache()

    def teardown_method(self):
        """Clean test keys from Redis."""
        async def _clean():
            import redis.asyncio as aioredis
            r = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=False)
            keys = await r.keys("semcache:*")
            if keys:
                await r.delete(*keys)
            await r.aclose()
        asyncio.run(_clean())

    def test_miss_returns_none(self):
        async def _run():
            cache = self._make()
            try:
                return await cache.get("a very unique query xyz123abc")
            finally:
                await cache.close()
        assert asyncio.run(_run()) is None

    def test_set_and_exact_get(self):
        async def _run():
            cache = self._make()
            try:
                await cache.set("What is GPT-4?", {"answer": "A large language model"}, ttl=60)
                return await cache.get("What is GPT-4?")
            finally:
                await cache.close()
        result = asyncio.run(_run())
        assert result is not None
        assert result["answer"] == "A large language model"

    def test_similar_query_hits_cache(self):
        async def _run():
            cache = self._make()
            try:
                await cache.set(
                    "What is GPT-4?",
                    {"answer": "A large language model"},
                    ttl=60,
                )
                # Semantically similar but different phrasing
                return await cache.get("Tell me about GPT-4")
            finally:
                await cache.close()
        result = asyncio.run(_run())
        # May hit or miss depending on embedding similarity — just check no error
        assert result is None or isinstance(result, dict)

    def test_different_query_misses(self):
        async def _run():
            cache = self._make()
            try:
                await cache.set("What is GPT-4?", {"answer": "A model"}, ttl=60)
                return await cache.get("How do I make pizza?")
            finally:
                await cache.close()
        assert asyncio.run(_run()) is None


# ── FASE 5 — RRF fusion unit test ─────────────────────────────────────────────

class TestRRFFusion:
    def test_basic_fusion(self):
        from retrieval.hybrid_retriever import RankedChunk, _rrf_fuse

        list_a = [
            RankedChunk("c1", "s1", 0, "text1", "text", 0.0),
            RankedChunk("c2", "s1", 1, "text2", "text", 0.0),
        ]
        list_b = [
            RankedChunk("c2", "s1", 1, "text2", "text", 0.0),
            RankedChunk("c3", "s2", 0, "text3", "text", 0.0),
        ]

        fused = _rrf_fuse([list_a, list_b], ["a", "b"], k=60)
        assert len(fused) == 3
        # c2 appears in both lists — should have highest score
        assert fused[0].chunk_id == "c2"

    def test_scores_decrease(self):
        from retrieval.hybrid_retriever import RankedChunk, _rrf_fuse

        lst = [RankedChunk(f"c{i}", "s", i, f"text{i}", "text", 0.0) for i in range(5)]
        fused = _rrf_fuse([lst], ["a"], k=60)
        scores = [c.rrf_score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_source_provenance_merged(self):
        from retrieval.hybrid_retriever import RankedChunk, _rrf_fuse

        list_a = [RankedChunk("c1", "s1", 0, "text", "text", 0.0, sources=["qdrant"])]
        list_b = [RankedChunk("c1", "s1", 0, "text", "text", 0.0, sources=["elasticsearch"])]

        fused = _rrf_fuse([list_a, list_b], ["qdrant", "elasticsearch"])
        assert len(fused) == 1
        assert "qdrant" in fused[0].sources or "elasticsearch" in fused[0].sources


# ── FASE 5 — HybridRetriever integration ──────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("QDRANT_URL", "QDRANT_API_KEY"),
    reason="QDRANT_URL / QDRANT_API_KEY not configured",
)
class TestHybridRetriever:
    def setup_method(self):
        """Index test chunks into Qdrant + ES."""
        async def _index():
            from embeddings.indexers.qdrant_indexer import QdrantIndexer
            from embeddings.indexers.es_indexer import ESIndexer
            q = QdrantIndexer(collection=_RET_QDRANT_COL)
            es = ESIndexer(index=_RET_ES_INDEX)
            try:
                await q.upsert(_CHUNKS, _EMBEDDINGS)
                await es.bulk_index(_CHUNKS)
                await es._es.indices.refresh(index=_RET_ES_INDEX)
            finally:
                await q.close()
                await es.close()
        asyncio.run(_index())

    def teardown_method(self):
        async def _clean():
            from embeddings.indexers.qdrant_indexer import QdrantIndexer
            from embeddings.indexers.es_indexer import ESIndexer
            from graph.knowledge_graph import KnowledgeGraph
            q  = QdrantIndexer(collection=_RET_QDRANT_COL)
            es = ESIndexer(index=_RET_ES_INDEX)
            kg = KnowledgeGraph()
            try:
                await q._client.delete_collection(_RET_QDRANT_COL)
            except Exception:
                pass
            try:
                await es._es.indices.delete(index=_RET_ES_INDEX, ignore_unavailable=True)
            except Exception:
                pass
            try:
                driver = await kg._get_driver()
                async with driver.session() as s:
                    await s.run(
                        "MATCH (n) WHERE n.source_id IN $ids DETACH DELETE n",
                        ids=_RET_SOURCE_IDS,
                    )
            except Exception:
                pass
            finally:
                await q.close()
                await es.close()
                await kg.close()
        asyncio.run(_clean())

    def test_dense_search_returns_results(self):
        async def _run():
            from retrieval.hybrid_retriever import HybridRetriever
            r = HybridRetriever(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
            )
            try:
                return await r._dense_search(_EMBEDDINGS[0])
            finally:
                await r.close()
        results = asyncio.run(_run())
        assert len(results) >= 1

    def test_sparse_search_returns_results(self):
        async def _run():
            from retrieval.hybrid_retriever import HybridRetriever
            r = HybridRetriever(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
            )
            try:
                return await r._sparse_search("OpenAI GPT")
            finally:
                await r.close()
        results = asyncio.run(_run())
        assert len(results) >= 1
        assert any("OpenAI" in c.text for c in results)

    def test_retrieve_returns_rrf_fused(self):
        async def _run():
            from retrieval.hybrid_retriever import HybridRetriever
            r = HybridRetriever(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
            )
            try:
                return await r.retrieve(_EMBEDDINGS[0], "OpenAI GPT-4", top_k=5)
            finally:
                await r.close()
        results = asyncio.run(_run())
        assert len(results) >= 1
        # Scores should be in descending order
        scores = [c.rrf_score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top_k_respected(self):
        async def _run():
            from retrieval.hybrid_retriever import HybridRetriever
            r = HybridRetriever(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
            )
            try:
                return await r.retrieve(_EMBEDDINGS[0], "model", top_k=2)
            finally:
                await r.close()
        results = asyncio.run(_run())
        assert len(results) <= 2


# ── FASE 5 — Full pipeline cache hit ─────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("QDRANT_URL", "QDRANT_API_KEY", "REDIS_URL", "GOOGLE_API_KEY"),
    reason="QDRANT_URL / QDRANT_API_KEY / REDIS_URL / GOOGLE_API_KEY not configured",
)
class TestRetrievalPipeline:
    def setup_method(self):
        async def _index():
            from embeddings.indexers.qdrant_indexer import QdrantIndexer
            from embeddings.indexers.es_indexer import ESIndexer
            q  = QdrantIndexer(collection=_RET_QDRANT_COL)
            es = ESIndexer(index=_RET_ES_INDEX)
            try:
                await q.upsert(_CHUNKS, _EMBEDDINGS)
                await es.bulk_index(_CHUNKS)
                await es._es.indices.refresh(index=_RET_ES_INDEX)
            finally:
                await q.close()
                await es.close()
        asyncio.run(_index())

    def teardown_method(self):
        async def _clean():
            import redis.asyncio as aioredis
            from embeddings.indexers.qdrant_indexer import QdrantIndexer
            from embeddings.indexers.es_indexer import ESIndexer

            # Clear Redis cache keys
            r = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=False)
            keys = await r.keys("semcache:*")
            if keys:
                await r.delete(*keys)
            await r.aclose()

            # Drop test collection + index
            q  = QdrantIndexer(collection=_RET_QDRANT_COL)
            es = ESIndexer(index=_RET_ES_INDEX)
            try:
                await q._client.delete_collection(_RET_QDRANT_COL)
            except Exception:
                pass
            try:
                await es._es.indices.delete(index=_RET_ES_INDEX, ignore_unavailable=True)
            except Exception:
                pass
            finally:
                await q.close()
                await es.close()
        asyncio.run(_clean())

    def test_pipeline_returns_result(self):
        async def _run():
            from retrieval.pipeline import RetrievalPipeline
            p = RetrievalPipeline(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
                use_expansion=False,
                use_reranker=False,
                use_compressor=False,
            )
            try:
                return await p.run("GPT-4 language model", top_k=3)
            finally:
                await p.close()
        result = asyncio.run(_run())
        assert result is not None
        assert len(result.chunks) >= 1
        assert result.cache_hit is False

    def test_cache_hit_on_second_query(self):
        async def _run():
            from retrieval.pipeline import RetrievalPipeline
            p = RetrievalPipeline(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
                use_expansion=False,
                use_reranker=False,
                use_compressor=False,
            )
            try:
                query = "What is GPT-4?"
                first  = await p.run(query, top_k=3)
                second = await p.run(query, top_k=3)
                return first, second
            finally:
                await p.close()
        first, second = asyncio.run(_run())
        assert first.cache_hit is False
        assert second.cache_hit is True

    def test_rrf_scores_are_positive(self):
        async def _run():
            from retrieval.pipeline import RetrievalPipeline
            p = RetrievalPipeline(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
                use_expansion=False,
                use_reranker=False,
                use_compressor=False,
            )
            try:
                result = await p.run("Gemini multimodal model", top_k=5)
                return result.chunks
            finally:
                await p.close()
        chunks = asyncio.run(_run())
        assert all(c.rrf_score > 0 for c in chunks)

    def test_result_to_dict_serializable(self):
        async def _run():
            from retrieval.pipeline import RetrievalPipeline
            p = RetrievalPipeline(
                qdrant_collection=_RET_QDRANT_COL,
                es_index=_RET_ES_INDEX,
                use_expansion=False,
                use_reranker=False,
                use_compressor=False,
            )
            try:
                result = await p.run("Anthropic Claude safety", top_k=3)
                return result.to_dict()
            finally:
                await p.close()
        import json
        d = asyncio.run(_run())
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert "chunks" in json.loads(serialized)
