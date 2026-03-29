"""FastAPI application — RAG Ecosystem API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth import (
    TOKEN_TTL_HOURS, create_access_token, hash_password, verify_password,
)
from api.routers import admin as admin_router
from api.routers import ingest as ingest_router
from api.routers import query as query_router
from api.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse

logger = logging.getLogger(__name__)


# ── Auth endpoints ─────────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    """Authenticate with email + password; return JWT."""
    from core.database import get_connection
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "SELECT id, tenant_id, role, hashed_password FROM users WHERE email = $1",
            body.email,
        )
    finally:
        await conn.close()

    if not row or not verify_password(body.password, row["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        user_id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        role=row["role"],
    )
    return LoginResponse(access_token=token, expires_in=TOKEN_TTL_HOURS * 3600)


@auth_router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(body: RegisterRequest):
    """Create a new tenant + admin user; return JWT."""
    from core.database import create_tenant, get_connection

    tenant = await create_tenant(body.tenant_name, body.tenant_slug, body.plan)
    conn   = await get_connection()
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO users (tenant_id, email, role, hashed_password)
            VALUES ($1, $2, 'admin', $3)
            RETURNING id
            """,
            tenant.id, body.email, hash_password(body.password),
        )
        user_id = str(row["id"])
    finally:
        await conn.close()

    token = create_access_token(user_id=user_id, tenant_id=tenant.id, role="admin")
    return RegisterResponse(tenant_id=tenant.id, user_id=user_id, access_token=token)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("RAG API starting…")

    # Bootstrap PostgreSQL schema (tables + RLS)
    try:
        from core.database import bootstrap_schema, get_connection
        await bootstrap_schema()
        conn = await get_connection()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id  UUID NOT NULL,
                    user_id    TEXT NOT NULL,
                    query      TEXT NOT NULL,
                    answer     TEXT NOT NULL,
                    cost_usd   NUMERIC(10,6) NOT NULL DEFAULT 0,
                    cached     BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_qh_tenant
                    ON query_history(tenant_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS evaluations (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id   UUID NOT NULL,
                    sample_size INT NOT NULL DEFAULT 0,
                    scores      JSONB NOT NULL DEFAULT '{}',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_eval_tenant
                    ON evaluations(tenant_id, created_at DESC);
            """)
        finally:
            await conn.close()
        logger.info("PostgreSQL schema ready")
    except Exception as exc:
        logger.warning("DB bootstrap (non-fatal): %s", exc)

    yield

    try:
        from api.middleware.rate_limit import _limiter
        if _limiter:
            await _limiter.close()
    except Exception:
        pass
    logger.info("RAG API shut down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    _app = FastAPI(
        title="RAG Ecosystem API",
        version="1.0.0",
        description="Multi-tenant multimodal RAG with hybrid retrieval and LLM routing",
        lifespan=lifespan,
    )

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @_app.exception_handler(Exception)
    async def _global_exc(request: Request, exc: Exception):
        logger.exception("Unhandled: %s %s", request.method, request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    _app.include_router(auth_router)
    _app.include_router(query_router.router)
    _app.include_router(ingest_router.router)
    _app.include_router(admin_router.router)

    @_app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    return _app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
