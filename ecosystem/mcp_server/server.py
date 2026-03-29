"""MCP Server — stdio (Claude Desktop) + SSE (web integrations)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from mcp_server.resources.schema import CHUNKS_SCHEMA, get_retrieval_config
from mcp_server.tools.search import search_knowledge
from mcp_server.tools.ingest import ingest_document, get_ingest_status
from mcp_server.tools.stats import get_tenant_stats

logger = logging.getLogger(__name__)

app = Server("rag-ecosystem")

# ── Tool registry ─────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_knowledge",
            description=(
                "Search the tenant's knowledge base using hybrid retrieval "
                "(semantic + keyword + graph). Returns ranked chunks."
            ),
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query":      {"type": "string"},
                    "top_k":     {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    "modalities": {
                        "type": "array",
                        "items": {"type": "string",
                                  "enum": ["text", "image", "audio", "structured", "web"]},
                        "default": ["text", "image", "audio", "structured", "web"],
                    },
                },
            },
        ),
        types.Tool(
            name="ingest_document",
            description=(
                "Enqueue a document for ingestion into the knowledge base. "
                "Returns a job_id for status tracking."
            ),
            inputSchema={
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {"type": "string"},
                    "metadata":  {"type": "object", "default": {}},
                },
            },
        ),
        types.Tool(
            name="get_ingest_status",
            description="Check the status of a previously submitted ingest job.",
            inputSchema={
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="get_tenant_stats",
            description=(
                "Return usage statistics for the current tenant. "
                "Requires admin or editor role."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    args = arguments or {}
    ctx  = _get_context()

    try:
        if name == "search_knowledge":
            result = await search_knowledge(
                query=args["query"],
                tenant_id=ctx["tenant_id"],
                top_k=args.get("top_k", 5),
                modalities=args.get("modalities"),
            )

        elif name == "ingest_document":
            result = await ingest_document(
                file_path=args["file_path"],
                tenant_id=ctx["tenant_id"],
                user_id=ctx["user_id"],
                metadata=args.get("metadata", {}),
            )

        elif name == "get_ingest_status":
            result = await get_ingest_status(job_id=args["job_id"])

        elif name == "get_tenant_stats":
            result = await get_tenant_stats(
                tenant_id=ctx["tenant_id"],
                role=ctx.get("role", "viewer"),
            )

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except PermissionError as exc:
        return [types.TextContent(type="text", text=f"Permission denied: {exc}")]
    except Exception as exc:
        logger.exception("Tool %s error", name)
        return [types.TextContent(type="text", text=f"Error: {exc}")]

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ── Resource registry ─────────────────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri=types.AnyUrl("schema://chunks"),
            name="ParsedChunk Schema",
            description="JSON Schema for the ParsedChunk data model",
            mimeType="application/json",
        ),
        types.Resource(
            uri=types.AnyUrl("config://retrieval"),
            name="Retrieval Configuration",
            description="Current retrieval pipeline parameters",
            mimeType="application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: types.AnyUrl) -> str:
    uri_str = str(uri)
    if uri_str == "schema://chunks":
        return json.dumps(CHUNKS_SCHEMA, indent=2)
    if uri_str == "config://retrieval":
        return json.dumps(get_retrieval_config(), indent=2)
    raise ValueError(f"Unknown resource URI: {uri_str}")


# ── Context helpers ───────────────────────────────────────────────────────────

def _get_context() -> dict:
    """Return tenant context from environment (stdio) or SSE request."""
    return {
        "tenant_id": os.getenv("TENANT_ID", ""),
        "user_id":   os.getenv("USER_ID",   ""),
        "role":      os.getenv("TENANT_ROLE", "viewer"),
    }


# ── Entry points ──────────────────────────────────────────────────────────────

async def run_stdio() -> None:
    """Run the MCP server over stdio (for Claude Desktop)."""
    async with stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="rag-ecosystem",
            server_version="1.0.0",
            capabilities=app.get_capabilities(
                notification_options=None,
                experimental_capabilities={},
            ),
        )
        await app.run(read_stream, write_stream, init_options)


def run_sse(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Run the MCP server over SSE (for web integrations)."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from mcp.server.sse import SseServerTransport

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        _inject_context_from_request(request)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            init_options = InitializationOptions(
                server_name="rag-ecosystem",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            )
            await app.run(streams[0], streams[1], init_options)

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ]
    )
    uvicorn.run(starlette_app, host=host, port=port)


def _inject_context_from_request(request) -> None:
    try:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return
        token = auth[7:]
        from jose import jwt
        secret = os.getenv("JWT_SECRET_KEY", "")
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        os.environ["TENANT_ID"]   = payload.get("tenant_id", "")
        os.environ["USER_ID"]     = payload.get("sub", "")
        os.environ["TENANT_ROLE"] = payload.get("role", "viewer")
    except Exception:
        pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_stdio())
