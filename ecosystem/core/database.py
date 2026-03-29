"""Async PostgreSQL connection + tenant schema bootstrap."""

from __future__ import annotations

import logging
import os
from typing import Any

import asyncpg

from core.tenant import Tenant, get_tenant_collection, get_tenant_index

logger = logging.getLogger(__name__)

# ── Connection ─────────────────────────────────────────────────────────────────


def _dsn() -> str:
    """Build a PostgreSQL DSN from env vars."""
    explicit = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5432")
    user     = os.getenv("POSTGRES_USER", "rag")
    password = os.getenv("POSTGRES_PASSWORD", "")
    db       = os.getenv("POSTGRES_DB",       "rag_metadata")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def get_connection() -> asyncpg.Connection:
    """Return a single asyncpg connection (caller must close)."""
    return await asyncpg.connect(_dsn())


async def get_pool(min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """Return a connection pool (caller must close)."""
    return await asyncpg.create_pool(_dsn(), min_size=min_size, max_size=max_size)


# ── Bootstrap — run once at startup ──────────────────────────────────────────

_BOOTSTRAP_SQL = r"""
-- Application role: non-superuser, respects RLS (created once, idempotent)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
        CREATE ROLE rag_app LOGIN PASSWORD 'rag_app_rls_2024' NOSUPERUSER NOCREATEDB NOCREATEROLE;
    END IF;
END $$;

"""

_BOOTSTRAP_SQL += """
-- ── Tenants ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tenants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    slug        TEXT NOT NULL UNIQUE,
    plan        TEXT NOT NULL DEFAULT 'free',
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Users ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email           TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',
    hashed_password TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, email)
);

-- ── Documents ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    modality    TEXT NOT NULL DEFAULT 'text',
    status      TEXT NOT NULL DEFAULT 'pending',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    storage_key TEXT,
    created_by  UUID REFERENCES users(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Add storage_key to existing tables (idempotent migration)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_key TEXT;

-- ── API Keys ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash    TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes for FK lookups ────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_users_tenant       ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_tenant   ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant    ON api_keys(tenant_id);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- FORCE ensures RLS applies even to the table owner (no bypass)
ALTER TABLE tenants   ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenants   FORCE ROW LEVEL SECURITY;
ALTER TABLE users     ENABLE ROW LEVEL SECURITY;
ALTER TABLE users     FORCE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE api_keys  ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys  FORCE ROW LEVEL SECURITY;

-- Policy: application sets current_setting('app.tenant_id') per request
DO $$
BEGIN
    -- tenants: own row only
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'tenants' AND policyname = 'tenant_isolation'
    ) THEN
        CREATE POLICY tenant_isolation ON tenants
            USING (id::text = current_setting('app.tenant_id', TRUE));
    END IF;

    -- users: own tenant only
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'users' AND policyname = 'tenant_isolation'
    ) THEN
        CREATE POLICY tenant_isolation ON users
            USING (tenant_id::text = current_setting('app.tenant_id', TRUE));
    END IF;

    -- documents: own tenant only
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'documents' AND policyname = 'tenant_isolation'
    ) THEN
        CREATE POLICY tenant_isolation ON documents
            USING (tenant_id::text = current_setting('app.tenant_id', TRUE));
    END IF;

    -- api_keys: own tenant only
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'api_keys' AND policyname = 'tenant_isolation'
    ) THEN
        CREATE POLICY tenant_isolation ON api_keys
            USING (tenant_id::text = current_setting('app.tenant_id', TRUE));
    END IF;
END $$;

-- Grant table access to the application role (superuser manages schema separately)
GRANT SELECT, INSERT, UPDATE, DELETE ON tenants   TO rag_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON users     TO rag_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO rag_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON api_keys  TO rag_app;
"""


async def bootstrap_schema() -> None:
    """Create tables and RLS policies if they don't exist.

    Safe to call on every startup (all statements use IF NOT EXISTS).
    """
    conn = await get_connection()
    try:
        await conn.execute(_BOOTSTRAP_SQL)
        logger.info("PostgreSQL schema bootstrapped")
    finally:
        await conn.close()


# ── Tenant CRUD ───────────────────────────────────────────────────────────────

async def create_tenant(name: str, slug: str, plan: str = "free") -> Tenant:
    """Insert a new tenant row and provision its vector/search stores."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO tenants (name, slug, plan)
            VALUES ($1, $2, $3)
            RETURNING id, name, slug, plan, is_active, created_at
            """,
            name, slug, plan,
        )
    finally:
        await conn.close()

    tenant = Tenant(
        id=str(row["id"]),
        name=row["name"],
        slug=row["slug"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=row["created_at"],
    )

    # Provision vector + search stores for this tenant
    await _provision_tenant_stores(tenant.id)
    return tenant


async def get_tenant_by_id(tenant_id: str) -> Tenant | None:
    """Fetch a tenant by UUID (bypasses RLS — internal use only)."""
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT id, name, slug, plan, is_active, created_at "
            "FROM tenants WHERE id = $1",
            tenant_id,
        )
    finally:
        await conn.close()

    if row is None:
        return None
    return Tenant(
        id=str(row["id"]),
        name=row["name"],
        slug=row["slug"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=row["created_at"],
    )


async def delete_tenant(tenant_id: str) -> None:
    """Delete a tenant and cascade (internal / test cleanup use only)."""
    conn = await get_connection()
    try:
        await conn.execute("DELETE FROM tenants WHERE id = $1", tenant_id)
    finally:
        await conn.close()


# ── Store provisioning ────────────────────────────────────────────────────────

async def _provision_tenant_stores(tenant_id: str) -> None:
    """Create Qdrant collection, ES index, and Neo4j indexes for a new tenant."""
    import asyncio
    await asyncio.gather(
        _provision_qdrant(tenant_id),
        _provision_es(tenant_id),
        _provision_neo4j(tenant_id),
        return_exceptions=True,
    )


async def _provision_qdrant(tenant_id: str) -> None:
    from embeddings.indexers.qdrant_indexer import QdrantIndexer
    q = QdrantIndexer(tenant_id=tenant_id)
    try:
        await q._ensure_collection()
        logger.info("Qdrant collection provisioned for tenant %s", tenant_id[:8])
    except Exception as exc:
        logger.warning("Qdrant provision failed for %s: %s", tenant_id[:8], exc)
    finally:
        await q.close()


async def _provision_es(tenant_id: str) -> None:
    from embeddings.indexers.es_indexer import ESIndexer
    es = ESIndexer(tenant_id=tenant_id)
    try:
        await es._ensure_index()
        logger.info("ES index provisioned for tenant %s", tenant_id[:8])
    except Exception as exc:
        logger.warning("ES provision failed for %s: %s", tenant_id[:8], exc)
    finally:
        await es.close()


async def _provision_neo4j(tenant_id: str) -> None:
    from graph.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    try:
        await kg._ensure_constraints()
        logger.info("Neo4j indexes provisioned for tenant %s", tenant_id[:8])
    except Exception as exc:
        logger.warning("Neo4j provision failed for %s: %s", tenant_id[:8], exc)
    finally:
        await kg.close()
