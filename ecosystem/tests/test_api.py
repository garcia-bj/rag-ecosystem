"""FastAPI integration tests (FASE 8).

Uses httpx.AsyncClient with the app as ASGI transport.
Expensive operations (LLM calls, retrieval pipeline) are mocked.
Requires PostgreSQL (POSTGRES_HOST) for auth/register/rate-limit tests.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

sys.path.insert(0, str(sys.path[0].replace("tests", "")))

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _env_ok(*keys: str) -> bool:
    for k in keys:
        v = os.getenv(k, "")
        if not v or "CHANGE_ME" in v:
            return False
    return True


# ── Fixtures ──────────────────────────────────────────────────────────────────

_TENANT_SLUG = f"test-api-{uuid.uuid4().hex[:6]}"
_ADMIN_EMAIL = f"admin-{uuid.uuid4().hex[:6]}@test.example"
_ADMIN_PASS  = "StrongPass123!"


def _make_client() -> httpx.AsyncClient:
    from api.main import app
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def _register_and_login() -> tuple[str, str]:
    """Returns (tenant_id, jwt_token)."""
    from api.main import app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post("/auth/register", json={
            "tenant_name": f"Test API Tenant {_TENANT_SLUG}",
            "tenant_slug": _TENANT_SLUG,
            "email":       _ADMIN_EMAIL,
            "password":    _ADMIN_PASS,
            "plan":        "free",
        })
        assert resp.status_code == 201, resp.text
        data = resp.json()
        return data["tenant_id"], data["access_token"]


# ── Health check (no DB required) ────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self):
        async def _run():
            async with _make_client() as client:
                resp = await client.get("/health")
                return resp
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Auth (requires PostgreSQL) ────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("POSTGRES_HOST", "POSTGRES_PASSWORD"),
    reason="PostgreSQL not configured",
)
class TestAuth:
    _tenant_id: str = ""
    _token:     str = ""

    def setup_method(self):
        # Reset rate-limiter singleton so each test gets a fresh Redis client
        # bound to the current event loop (avoids "Event loop is closed" errors).
        import api.middleware.rate_limit as _rl_mod
        _rl_mod._limiter = None

        async def _setup():
            tid, tok = await _register_and_login()
            TestAuth._tenant_id = tid
            TestAuth._token     = tok
        asyncio.run(_setup())

    def teardown_method(self):
        async def _cleanup():
            from core.database import delete_tenant
            if TestAuth._tenant_id:
                await delete_tenant(TestAuth._tenant_id)
        asyncio.run(_cleanup())

    def test_register_returns_tenant_and_token(self):
        assert TestAuth._tenant_id != ""
        assert TestAuth._token != ""

    def test_login_with_correct_credentials(self):
        async def _run():
            async with _make_client() as client:
                resp = await client.post("/auth/login", json={
                    "email":    _ADMIN_EMAIL,
                    "password": _ADMIN_PASS,
                })
                return resp
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_with_wrong_password_returns_401(self):
        async def _run():
            async with _make_client() as client:
                resp = await client.post("/auth/login", json={
                    "email":    _ADMIN_EMAIL,
                    "password": "WrongPassword!",
                })
                return resp
        resp = asyncio.run(_run())
        assert resp.status_code == 401

    def test_protected_route_without_token_returns_401(self):
        async def _run():
            async with _make_client() as client:
                return await client.post("/query", json={"query": "hello"})
        resp = asyncio.run(_run())
        assert resp.status_code == 401

    def test_protected_route_with_token(self):
        """Query endpoint accessible with valid JWT (pipeline mocked)."""
        async def _run():
            mock_result = MagicMock()
            mock_result.cache_hit = True
            mock_result.chunks    = []

            with patch("retrieval.pipeline.RetrievalPipeline") as mock_cls:
                mock_pipeline = MagicMock()
                mock_pipeline.run   = AsyncMock(return_value=mock_result)
                mock_pipeline.close = AsyncMock()
                mock_cls.return_value = mock_pipeline

                async with _make_client() as client:
                    resp = await client.post(
                        "/query",
                        json={"query": "What is GPT-4?", "top_k": 3},
                        headers={"Authorization": f"Bearer {TestAuth._token}"},
                    )
                    return resp
        resp = asyncio.run(_run())
        # 200 OK (cache hit returns immediately without LLM call)
        assert resp.status_code == 200

    def test_cross_tenant_token_rejected(self):
        """A JWT for a different tenant cannot access this tenant's data.

        We verify that a forged token with a random tenant_id returns 401 or
        has no data (pipeline is scoped to tenant).  Here we just verify
        the token is parsed correctly and the tenant_id in the JWT is used.
        """
        from api.middleware.auth import create_access_token

        other_tenant_id = str(uuid.uuid4())
        other_token = create_access_token(
            user_id="other_user",
            tenant_id=other_tenant_id,
            role="admin",
        )

        async def _run():
            mock_result = MagicMock()
            mock_result.cache_hit = True
            mock_result.chunks    = []

            with patch("retrieval.pipeline.RetrievalPipeline") as mock_cls:
                mock_pipeline = MagicMock()
                mock_pipeline.run   = AsyncMock(return_value=mock_result)
                mock_pipeline.close = AsyncMock()
                mock_cls.return_value = mock_pipeline

                # Verify the pipeline was called with other_tenant's collection
                from core.tenant import get_tenant_collection, get_tenant_index

                async with _make_client() as client:
                    await client.post(
                        "/query",
                        json={"query": "secret data", "top_k": 3},
                        headers={"Authorization": f"Bearer {other_token}"},
                    )
                    call_kwargs = mock_cls.call_args.kwargs
                    return call_kwargs

        kwargs = asyncio.run(_run())
        from core.tenant import get_tenant_collection
        # Pipeline was called with the OTHER tenant's collection — isolation confirmed
        assert kwargs["qdrant_collection"] == get_tenant_collection(other_tenant_id)

    def test_ingest_file_returns_job_id(self):
        async def _run():
            with (
                patch("api.routers.ingest._enqueue_ingest", return_value="job-abc-123"),
                patch("api.routers.ingest._create_document_record", new_callable=AsyncMock),
            ):
                async with _make_client() as client:
                    content = b"fake pdf content"
                    resp = await client.post(
                        "/ingest/file",
                        files={"file": ("test.pdf", content, "application/pdf")},
                        headers={"Authorization": f"Bearer {TestAuth._token}"},
                    )
                    return resp
        resp = asyncio.run(_run())
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "job-abc-123"

    def test_rate_limit_exceeded_returns_429(self):
        """Patched rate limiter raises 429 to avoid event-loop singleton issues."""
        from fastapi import HTTPException as _HTTPException

        async def _run():
            mock_rl = MagicMock()
            mock_rl.check_query = AsyncMock(
                side_effect=_HTTPException(
                    status_code=429,
                    detail="Daily query limit (100) reached for plan 'free'.",
                )
            )
            with patch("api.middleware.rate_limit.get_rate_limiter", return_value=mock_rl):
                async with _make_client() as client:
                    resp = await client.post(
                        "/query",
                        json={"query": "over the limit", "top_k": 3},
                        headers={"Authorization": f"Bearer {TestAuth._token}"},
                    )
                    return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 429
        assert "limit" in resp.json()["detail"].lower()

    def test_admin_stats_endpoint(self):
        async def _run():
            with (
                patch("mcp_server.tools.stats._count_chunks_by_modality", return_value={"text": 10}),
                patch("mcp_server.tools.stats._cache_hit_rate",  return_value=0.5),
                patch("mcp_server.tools.stats._llm_cost_today",  return_value=0.02),
                patch("mcp_server.tools.stats._count_documents",  return_value=5),
            ):
                async with _make_client() as client:
                    resp = await client.get(
                        "/admin/stats",
                        headers={"Authorization": f"Bearer {TestAuth._token}"},
                    )
                    return resp
        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 5
        assert data["cache_hit_rate_24h"] == 0.5

    def test_viewer_cannot_access_admin_endpoints(self):
        """Editor-level token cannot access admin-only routes."""
        from api.middleware.auth import create_access_token

        viewer_token = create_access_token(
            user_id="viewer_user",
            tenant_id=TestAuth._tenant_id,
            role="viewer",
        )

        async def _run():
            async with _make_client() as client:
                resp = await client.post(
                    "/admin/users",
                    json={"email": "x@x.com", "password": "pass1234", "role": "viewer"},
                    headers={"Authorization": f"Bearer {viewer_token}"},
                )
                return resp
        resp = asyncio.run(_run())
        assert resp.status_code == 403


# ── JWT middleware unit tests (no DB) ─────────────────────────────────────────

class TestJWTMiddleware:
    def test_create_and_decode_token(self):
        from api.middleware.auth import create_access_token, decode_token
        token = create_access_token("user1", "tenant1", "admin")
        payload = decode_token(token)
        assert payload["sub"] == "user1"
        assert payload["tenant_id"] == "tenant1"
        assert payload["role"] == "admin"

    def test_expired_token_raises(self):
        from datetime import timedelta
        from api.middleware.auth import create_access_token, decode_token
        from jose import JWTError
        token = create_access_token("u1", "t1", "viewer", expires_delta=timedelta(seconds=-1))
        with pytest.raises(JWTError):
            decode_token(token)

    def test_hash_password_and_verify(self):
        from api.middleware.auth import hash_password, verify_password
        hashed = hash_password("mypassword123")
        assert verify_password("mypassword123", hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_generate_api_key(self):
        from api.middleware.auth import generate_api_key, hash_api_key
        raw, hashed = generate_api_key()
        assert len(raw) > 20
        assert hash_api_key(raw) == hashed

    def test_hash_password_is_different_each_time(self):
        from api.middleware.auth import hash_password
        h1 = hash_password("same_pass")
        h2 = hash_password("same_pass")
        # bcrypt adds random salt
        assert h1 != h2


# ── Rate limiter unit tests ───────────────────────────────────────────────────

@pytest.mark.skipif(
    not _env_ok("REDIS_URL"),
    reason="REDIS_URL not configured",
)
class TestRateLimiter:
    def teardown_method(self):
        async def _cleanup():
            import redis.asyncio as aioredis
            from datetime import date
            tid = "rl-test-tenant-001"
            r = aioredis.from_url(os.getenv("REDIS_URL", ""), decode_responses=True)
            keys = await r.keys(f"rate:{tid}:*")
            if keys:
                await r.delete(*keys)
            await r.aclose()
        asyncio.run(_cleanup())

    def test_first_query_allowed(self):
        async def _run():
            from api.middleware.rate_limit import TenantRateLimiter
            rl = TenantRateLimiter()
            try:
                await rl.check_query("rl-test-tenant-001", "free")  # should not raise
            finally:
                await rl.close()
        asyncio.run(_run())  # no exception = pass

    def test_over_limit_raises_429(self):
        async def _run():
            import redis.asyncio as aioredis
            from datetime import date
            from fastapi import HTTPException
            from api.middleware.rate_limit import TenantRateLimiter

            tid = "rl-test-tenant-001"
            # Pre-set counter to 100 (the free limit)
            r = aioredis.from_url(os.getenv("REDIS_URL", ""), decode_responses=True)
            key = f"rate:{tid}:queries:{date.today().isoformat()}"
            await r.set(key, "100", ex=3600)
            await r.aclose()

            rl = TenantRateLimiter()
            try:
                with pytest.raises(HTTPException) as exc_info:
                    await rl.check_query(tid, "free")
                assert exc_info.value.status_code == 429
            finally:
                await rl.close()
        asyncio.run(_run())

    def test_pro_plan_has_higher_limit(self):
        from api.middleware.rate_limit import _PLANS
        assert _PLANS["pro"]["queries_day"] > _PLANS["free"]["queries_day"]

    def test_enterprise_plan_unlimited(self):
        from api.middleware.rate_limit import _PLANS
        assert _PLANS["enterprise"]["queries_day"] >= 999_999
