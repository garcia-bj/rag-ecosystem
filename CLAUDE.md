# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

All application code lives under `ecosystem/`. Tests, scripts, and Docker configs are inside that directory. The working directory for most commands is `ecosystem/`.

## Running Tests

Tests live in `ecosystem/tests/`. Run from the `ecosystem/` directory:

```bash
# All tests
cd ecosystem && python -m pytest tests/ -v

# Single test file
cd ecosystem && python -m pytest tests/test_router.py -v

# Single test by name
cd ecosystem && python -m pytest tests/test_router.py::test_function_name -v
```

`conftest.py` adds `ecosystem/` to `sys.path` automatically, so no install is needed. It also loads `ecosystem/.env` for integration tests that need real service credentials (Qdrant, Neo4j, etc.).

## Running the API

```bash
cd ecosystem && PYTHONPATH=. python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or directly: `cd ecosystem && python api/main.py`

## Local Services (Docker)

External services (PostgreSQL, Qdrant, Redis, RustFS) run on Dokploy at `147.93.1.182`. Local services (Elasticsearch, Neo4j, RabbitMQ, ClickHouse, Langfuse, Traefik) run via Docker Compose.

```bash
# Dev (16GB RAM) — applies resource-limited overrides
cd ecosystem && ./scripts/start-local.sh

# Production
cd ecosystem && ./scripts/start-prod.sh

# Wait for services to be healthy
cd ecosystem && ./scripts/wait_for_services.sh

# Check connectivity to all services
cd ecosystem && python scripts/verify_connections.py
```

Copy `ecosystem/.env.example` to `ecosystem/.env` and fill in credentials before starting.

## Architecture Overview

### Multi-tenancy

Every resource (Qdrant collection, Elasticsearch index, PostgreSQL rows) is namespaced by `tenant_id` (first 8 hex chars of the UUID). `core/tenant.py` provides `TenantContext` (user_id, tenant_id, role). PostgreSQL uses RLS via the `rag_app` role enforced in `core/database.py`.

### Request Flow

```
POST /query
  → api/middleware/auth.py       (JWT decode → TenantContext)
  → api/middleware/rate_limit.py (plan-based Redis rate limiting)
  → retrieval/pipeline.py        (orchestrates everything below)
      → cache/semantic_cache.py  (Redis — skip retrieval on hit)
      → retrieval/query_expander.py
      → retrieval/hybrid_retriever.py
          → embeddings/qdrant_indexer.py   (semantic)
          → embeddings/es_indexer.py        (BM25 keyword)
          → graph/knowledge_graph.py        (Neo4j)
      → retrieval/reranker.py
      → retrieval/compressor.py
  → llm/router.py                (Haiku → GPT-4o-mini → GPT-4o tiering)
  → llm/hallucination_guard.py
  → llm/pii_detector.py
  → core/telemetry.py + Langfuse (observability)
```

### Ingestion Flow

```
POST /ingest/file or /ingest/url
  → ingestion/parsers/           (PDF/DOCX, image OCR, audio STT, CSV/JSON, web)
  → ingestion/chunkers/          (document chunking)
  → embeddings/gemini_embedder.py
  → embeddings/indexers/         (Qdrant + Elasticsearch)
  → graph/knowledge_graph.py     (Neo4j entity extraction)
  (Celery workers via RabbitMQ for async jobs)
```

### LLM Routing

`llm/router.py` classifies queries by complexity:
- **Tier 1** — Claude Haiku 4.5 (simple lookups, <30 tokens context)
- **Tier 2** — GPT-4o Mini (factual QA, balanced)
- **Tier 3** — GPT-4o (complex reasoning)

`llm/circuit_breaker.py` handles fallback chaining on provider failures.

### MCP Server

`mcp_server/server.py` exposes tools (`search_knowledge`, `ingest_document`, `get_tenant_stats`) over both stdio (Claude Desktop) and SSE (web). Run with:

```bash
cd ecosystem && PYTHONPATH=. python mcp_server/server.py
```

### Observability

Langfuse (port 3000) is the observability UI backed by ClickHouse. `core/telemetry.py` wraps trace creation. `evaluation/evaluator.py` runs RAGAS metrics; `evaluation/quality_monitor.py` scores answer quality continuously.

```bash
cd ecosystem && python scripts/run_evaluation.py
cd ecosystem && python scripts/benchmark.py
```

## Key Conventions

- **PYTHONPATH** must include `ecosystem/` when running any module directly (conftest handles this for pytest automatically).
- **Tenant prefixing**: collection/index names use `tenant_id[:8]`; never access raw store names without the prefix.
- **No install required**: the project uses direct path injection, not an installable package.
- **`.env` location**: `ecosystem/.env` — loaded by conftest and by application code via `python-dotenv`.
