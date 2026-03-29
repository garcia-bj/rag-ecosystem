"""Tenant model, context variable, and store-naming helpers."""

from __future__ import annotations

import re
from contextvars import ContextVar
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ── Constants ─────────────────────────────────────────────────────────────────

Plan = Literal["free", "starter", "pro", "enterprise"]
Role = Literal["admin", "editor", "viewer"]

# Max chars from tenant_id UUID used in store names (first 8 hex chars)
_TENANT_PREFIX_LEN = 8

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# ── Data models ───────────────────────────────────────────────────────────────

class Tenant(BaseModel):
    id:         str
    name:       str
    slug:       str
    plan:       Plan = "free"
    is_active:  bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TenantContext(BaseModel):
    """Carries the authenticated tenant + user info for a single request."""
    tenant_id: str
    user_id:   str
    role:      Role = "viewer"

    model_config = {"frozen": True}


# ── Context variable (one per async task / request) ───────────────────────────

_current_tenant: ContextVar[TenantContext | None] = ContextVar(
    "_current_tenant", default=None
)


def set_tenant_context(ctx: TenantContext) -> None:
    """Set the tenant context for the current async task."""
    _current_tenant.set(ctx)


def get_current_tenant() -> TenantContext:
    """Return the TenantContext for the running request.

    Raises RuntimeError if no context has been set (e.g., outside a request).
    """
    ctx = _current_tenant.get()
    if ctx is None:
        raise RuntimeError(
            "No tenant context set. Call set_tenant_context() before accessing "
            "tenant-scoped resources."
        )
    return ctx


def get_current_tenant_or_none() -> TenantContext | None:
    return _current_tenant.get()


# ── Store-naming helpers ──────────────────────────────────────────────────────

def get_tenant_collection(tenant_id: str) -> str:
    """Qdrant collection name for the given tenant.

    Example: ``rag_550e8400``
    """
    prefix = tenant_id.replace("-", "")[:_TENANT_PREFIX_LEN]
    return f"rag_{prefix}"


def get_tenant_index(tenant_id: str) -> str:
    """Elasticsearch index name for the given tenant.

    Example: ``rag_chunks_550e8400``
    """
    prefix = tenant_id.replace("-", "")[:_TENANT_PREFIX_LEN]
    return f"rag_chunks_{prefix}"


def validate_tenant_id(tenant_id: str) -> bool:
    """Return True if tenant_id is a valid UUID v4 string."""
    return bool(_UUID4_RE.match(tenant_id))
