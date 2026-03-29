"""JWT + API-key authentication middleware and dependency."""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt_lib

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt

from core.tenant import TenantContext, set_tenant_context

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM  = "HS256"
TOKEN_TTL_HOURS = 24

_bearer         = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Password helpers (direct bcrypt — avoids passlib 4.x compat issues) ───────

def hash_password(password: str) -> str:
    return _bcrypt_lib.hashpw(password.encode()[:72], _bcrypt_lib.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt_lib.checkpw(plain.encode()[:72], hashed.encode())
    except Exception:
        return False

# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    tenant_id: str,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=TOKEN_TTL_HOURS)
    )
    payload = {
        "sub":       user_id,
        "tenant_id": tenant_id,
        "role":      role,
        "exp":       expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

# ── API-key helpers ───────────────────────────────────────────────────────────

def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()

def generate_api_key() -> tuple[str, str]:
    """Return (raw_key, key_hash).  Store only the hash; send raw to user."""
    raw  = secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)

# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def get_current_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    api_key: Optional[str] = Security(_api_key_header),
) -> TenantContext:
    """Resolve JWT or API-key → TenantContext.  Raises 401 on failure."""
    # ── JWT path ──────────────────────────────────────────────────────────────
    if bearer is not None:
        try:
            payload  = decode_token(bearer.credentials)
            user_id  = payload["sub"]
            tenant_id = payload["tenant_id"]
            role     = payload.get("role", "viewer")
        except (JWTError, KeyError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        ctx = TenantContext(tenant_id=tenant_id, user_id=user_id, role=role)
        set_tenant_context(ctx)
        return ctx

    # ── API-key path ──────────────────────────────────────────────────────────
    if api_key:
        ctx = await _resolve_api_key(api_key)
        if ctx:
            set_tenant_context(ctx)
            return ctx

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_admin(ctx: TenantContext = Depends(get_current_user)) -> TenantContext:
    if ctx.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return ctx


async def require_editor_or_admin(ctx: TenantContext = Depends(get_current_user)) -> TenantContext:
    if ctx.role not in ("admin", "editor"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Editor role required")
    return ctx


# ── DB resolution of API key ──────────────────────────────────────────────────

async def _resolve_api_key(raw_key: str) -> TenantContext | None:
    try:
        from core.database import get_connection
        conn = await get_connection()
        key_hash = hash_api_key(raw_key)
        try:
            row = await conn.fetchrow(
                "SELECT tenant_id FROM api_keys WHERE key_hash = $1",
                key_hash,
            )
        finally:
            await conn.close()

        if row is None:
            return None
        # API-key callers always get viewer role (can be extended per-key)
        return TenantContext(
            tenant_id=str(row["tenant_id"]),
            user_id="api_key",
            role="viewer",
        )
    except Exception:
        return None
