"""Admin router — stats, users, API keys (admin role only)."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import get_current_user, require_admin
from api.schemas import (
    ApiKeyCreate, ApiKeyResponse, CostsResponse, CostEntry,
    StatsResponse, UserCreate, UserResponse,
)
from core.tenant import TenantContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(ctx: TenantContext = Depends(get_current_user)):
    """Tenant usage statistics (admin + editor)."""
    if ctx.role not in ("admin", "editor"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    from mcp_server.tools.stats import get_tenant_stats
    result = await get_tenant_stats(ctx.tenant_id, ctx.role)
    return StatsResponse(**result)


@router.get("/users", response_model=list[UserResponse])
async def list_users(ctx: TenantContext = Depends(require_admin)):
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            rows = await conn.fetch(
                "SELECT id, email, role, created_at FROM users WHERE tenant_id = $1 ORDER BY created_at",
                ctx.tenant_id,
            )
            return [UserResponse(id=str(r["id"]), email=r["email"], role=r["role"], created_at=r["created_at"]) for r in rows]
        finally:
            await conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(body: UserCreate, ctx: TenantContext = Depends(require_admin)):
    from api.middleware.auth import hash_password
    from core.database import get_connection
    try:
        conn = await get_connection()
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO users (tenant_id, email, role, hashed_password)
                VALUES ($1, $2, $3, $4)
                RETURNING id, email, role, created_at
                """,
                ctx.tenant_id, body.email, body.role, hash_password(body.password),
            )
            return UserResponse(id=str(row["id"]), email=row["email"], role=row["role"], created_at=row["created_at"])
        finally:
            await conn.close()
    except Exception as exc:
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail="User already exists")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, ctx: TenantContext = Depends(require_admin)):
    from core.database import get_connection
    conn = await get_connection()
    try:
        result = await conn.execute(
            "DELETE FROM users WHERE id = $1 AND tenant_id = $2", user_id, ctx.tenant_id
        )
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="User not found")
    finally:
        await conn.close()


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(body: ApiKeyCreate, ctx: TenantContext = Depends(require_admin)):
    from api.middleware.auth import generate_api_key
    from core.database import get_connection
    raw, key_hash = generate_api_key()
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            "INSERT INTO api_keys (tenant_id, key_hash, name) VALUES ($1, $2, $3) RETURNING id, name, created_at",
            ctx.tenant_id, key_hash, body.name,
        )
        return ApiKeyResponse(id=str(row["id"]), name=row["name"], key=raw, created_at=row["created_at"])
    finally:
        await conn.close()


@router.get("/costs", response_model=CostsResponse)
async def get_costs(ctx: TenantContext = Depends(require_admin), days: int = 30):
    """Return daily LLM cost breakdown for the last N days."""
    import redis.asyncio as aioredis
    import os
    r = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
    try:
        entries = []
        total = 0.0
        for i in range(days):
            d = (date.today() - timedelta(days=i)).isoformat()
            cost_raw    = await r.get(f"stats:{ctx.tenant_id}:llm_cost:{d}")
            queries_raw = await r.get(f"rate:{ctx.tenant_id}:queries:{d}")
            cost    = float(cost_raw)    if cost_raw    else 0.0
            queries = int(queries_raw)   if queries_raw else 0
            entries.append(CostEntry(date=d, cost_usd=cost, queries=queries))
            total += cost
        return CostsResponse(entries=entries, total_usd=round(total, 6))
    finally:
        await r.aclose()
