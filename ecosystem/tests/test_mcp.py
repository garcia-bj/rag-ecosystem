"""MCP Server tool + resource tests (FASE 7).

Tests tool logic directly without starting the full MCP server.
External services (Qdrant, ES) are skipped when not configured.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_TENANT_A = "aaaaaaaa-0000-4000-8000-000000000001"


def _env_ok(*keys: str) -> bool:
    for k in keys:
        v = os.getenv(k, "")
        if not v or "CHANGE_ME" in v:
            return False
    return True


# ── Resource tests (no I/O) ───────────────────────────────────────────────────

class TestMCPResources:
    def test_chunks_schema_is_valid_json(self):
        from mcp_server.resources.schema import CHUNKS_SCHEMA
        dumped = json.dumps(CHUNKS_SCHEMA)
        parsed = json.loads(dumped)
        assert parsed["title"] == "ParsedChunk"

    def test_chunks_schema_has_required_fields(self):
        from mcp_server.resources.schema import CHUNKS_SCHEMA
        required = set(CHUNKS_SCHEMA.get("required", []))
        assert "text" in required
        assert "modality" in required
        assert "source_id" in required
        assert "chunk_index" in required

    def test_retrieval_config_returns_dict(self):
        from mcp_server.resources.schema import get_retrieval_config
        config = get_retrieval_config()
        assert "pipeline" in config
        assert "retrieval" in config
        assert "rrf" in config

    def test_retrieval_config_rrf_k(self):
        from mcp_server.resources.schema import get_retrieval_config
        assert get_retrieval_config()["rrf"]["k"] == 60

    def test_retrieval_config_is_json_serializable(self):
        from mcp_server.resources.schema import get_retrieval_config
        dumped = json.dumps(get_retrieval_config())
        assert len(dumped) > 0


# ── MCP server tool registration ──────────────────────────────────────────────

class TestMCPToolRegistration:
    def test_server_app_created(self):
        from mcp_server.server import app
        assert app is not None
        assert app.name == "rag-ecosystem"

    def test_list_tools_returns_all_tools(self):
        async def _run():
            from mcp_server.server import list_tools
            tools = await list_tools()
            return [t.name for t in tools]
        names = asyncio.run(_run())
        assert "search_knowledge"  in names
        assert "ingest_document"   in names
        assert "get_ingest_status" in names
        assert "get_tenant_stats"  in names

    def test_list_resources_returns_resources(self):
        async def _run():
            from mcp_server.server import list_resources
            return await list_resources()
        resources = asyncio.run(_run())
        uris = [str(r.uri) for r in resources]
        assert any("schema" in u for u in uris)
        assert any("config" in u for u in uris)

    def test_read_resource_chunks_schema(self):
        async def _run():
            import mcp.types as types
            from mcp_server.server import read_resource
            return await read_resource(types.AnyUrl("schema://chunks"))
        result = asyncio.run(_run())
        parsed = json.loads(result)
        assert parsed["title"] == "ParsedChunk"

    def test_read_resource_retrieval_config(self):
        async def _run():
            import mcp.types as types
            from mcp_server.server import read_resource
            return await read_resource(types.AnyUrl("config://retrieval"))
        result = asyncio.run(_run())
        parsed = json.loads(result)
        assert "pipeline" in parsed

    def test_read_resource_unknown_raises(self):
        async def _run():
            import mcp.types as types
            from mcp_server.server import read_resource
            return await read_resource(types.AnyUrl("unknown://xyz"))
        with pytest.raises(ValueError, match="Unknown resource URI"):
            asyncio.run(_run())


# ── stats tool — permission check (no I/O) ───────────────────────────────────

class TestStatsTool:
    def test_viewer_role_raises_permission_error(self):
        async def _run():
            from mcp_server.tools.stats import get_tenant_stats
            return await get_tenant_stats(_TENANT_A, role="viewer")
        with pytest.raises(PermissionError):
            asyncio.run(_run())

    def test_admin_role_allowed(self):
        async def _run():
            from mcp_server.tools.stats import get_tenant_stats
            # Mock all sub-functions to avoid real DB/Redis calls
            with (
                patch("mcp_server.tools.stats._count_chunks_by_modality", return_value={"text": 5}),
                patch("mcp_server.tools.stats._cache_hit_rate",  return_value=0.75),
                patch("mcp_server.tools.stats._llm_cost_today",  return_value=0.05),
                patch("mcp_server.tools.stats._count_documents",  return_value=3),
            ):
                return await get_tenant_stats(_TENANT_A, role="admin")
        result = asyncio.run(_run())
        assert result["chunks_by_modality"] == {"text": 5}
        assert result["cache_hit_rate_24h"] == 0.75
        assert result["total_documents"] == 3

    def test_editor_role_allowed(self):
        async def _run():
            from mcp_server.tools.stats import get_tenant_stats
            with (
                patch("mcp_server.tools.stats._count_chunks_by_modality", return_value={}),
                patch("mcp_server.tools.stats._cache_hit_rate",  return_value=0.0),
                patch("mcp_server.tools.stats._llm_cost_today",  return_value=0.0),
                patch("mcp_server.tools.stats._count_documents",  return_value=0),
            ):
                return await get_tenant_stats(_TENANT_A, role="editor")
        result = asyncio.run(_run())
        assert isinstance(result, dict)


# ── search tool — tenant isolation ───────────────────────────────────────────

class TestSearchToolIsolation:
    def test_search_uses_tenant_collection(self):
        """search_knowledge must route to the correct tenant collection."""
        from core.tenant import get_tenant_collection, get_tenant_index
        from retrieval.pipeline import RetrievalResult
        from retrieval.hybrid_retriever import RankedChunk

        calls = []

        async def _run():
            from mcp_server.tools.search import search_knowledge

            chunk = RankedChunk("c1", "doc1", 0, "tenant A content", "text", 0.9)
            mock_result = RetrievalResult(query="test query", chunks=[chunk], cache_hit=False)
            mock_pipeline = MagicMock()
            mock_pipeline.run   = AsyncMock(return_value=mock_result)
            mock_pipeline.close = AsyncMock()

            # Patch at the source module (import is inside the function)
            with patch("retrieval.pipeline.RetrievalPipeline", return_value=mock_pipeline) as mock_cls:
                result = await search_knowledge(
                    query="test query",
                    tenant_id=_TENANT_A,
                    top_k=3,
                )
                call_kwargs = mock_cls.call_args.kwargs
                calls.append(call_kwargs)
                return result

        result = asyncio.run(_run())
        assert result["total"] == 1
        from core.tenant import get_tenant_collection, get_tenant_index
        assert calls[0]["qdrant_collection"] == get_tenant_collection(_TENANT_A)
        assert calls[0]["es_index"]          == get_tenant_index(_TENANT_A)

    def test_search_modality_filter(self):
        async def _run():
            from mcp_server.tools.search import search_knowledge
            from retrieval.pipeline import RetrievalResult
            from retrieval.hybrid_retriever import RankedChunk

            chunks = [
                RankedChunk("c1", "doc1", 0, "text content",  "text",  0.9),
                RankedChunk("c2", "doc2", 0, "image content", "image", 0.8),
                RankedChunk("c3", "doc3", 0, "audio content", "audio", 0.7),
            ]
            mock_pipeline = MagicMock()
            mock_pipeline.run   = AsyncMock(
                return_value=RetrievalResult(query="q", chunks=chunks, cache_hit=False)
            )
            mock_pipeline.close = AsyncMock()

            with patch("retrieval.pipeline.RetrievalPipeline", return_value=mock_pipeline):
                return await search_knowledge(
                    query="test",
                    tenant_id=_TENANT_A,
                    top_k=5,
                    modalities=["text"],
                )

        result = asyncio.run(_run())
        assert result["total"] == 1
        assert result["chunks"][0]["modality"] == "text"


# ── ingest tool ───────────────────────────────────────────────────────────────

class TestIngestTool:
    def test_ingest_document_returns_job_id(self):
        async def _run():
            from mcp_server.tools.ingest import ingest_document
            import ingestion.workers.celery_app as celery_module

            mock_celery = MagicMock()
            mock_celery.send_task = MagicMock()

            # Patch the attribute on the source module (import is inside the function)
            with patch.object(celery_module, "app", mock_celery):
                return await ingest_document(
                    file_path="/tmp/test_doc.pdf",
                    tenant_id=_TENANT_A,
                    user_id="user_001",
                )

        result = asyncio.run(_run())
        assert "job_id" in result
        assert result["status"] == "queued"
        assert result["file_path"] == "/tmp/test_doc.pdf"
        import re
        assert re.match(r"[0-9a-f\-]{36}", result["job_id"])

    def test_get_ingest_status_returns_state(self):
        async def _run():
            from mcp_server.tools.ingest import get_ingest_status
            import ingestion.workers.celery_app as celery_module

            mock_result = MagicMock()
            mock_result.state = "PENDING"
            mock_result.result = None

            mock_celery = MagicMock()
            mock_celery.AsyncResult = MagicMock(return_value=mock_result)

            with patch.object(celery_module, "app", mock_celery):
                return await get_ingest_status("test-job-id")

        result = asyncio.run(_run())
        assert result["job_id"] == "test-job-id"
        assert result["status"] == "pending"

    def test_call_tool_dispatch_search(self):
        """call_tool routes to search_knowledge correctly."""
        async def _run():
            from mcp_server.server import call_tool

            mock_result = {"chunks": [], "total": 0, "cached": False}

            with (
                patch("mcp_server.server._get_context", return_value={
                    "tenant_id": _TENANT_A, "user_id": "u1", "role": "admin"
                }),
                patch("mcp_server.tools.search.search_knowledge",
                      AsyncMock(return_value=mock_result)),
            ):
                return await call_tool("search_knowledge", {"query": "hello"})

        result = asyncio.run(_run())
        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert "chunks" in parsed

    def test_call_tool_unknown_name(self):
        async def _run():
            from mcp_server.server import call_tool
            with patch("mcp_server.server._get_context", return_value={
                "tenant_id": _TENANT_A, "user_id": "u1", "role": "viewer"
            }):
                return await call_tool("nonexistent_tool", {})
        result = asyncio.run(_run())
        assert "Unknown tool" in result[0].text
