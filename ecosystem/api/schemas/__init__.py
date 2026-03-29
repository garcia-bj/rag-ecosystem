"""Pydantic schemas for all API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class RegisterRequest(BaseModel):
    tenant_name: str = Field(min_length=2, max_length=80)
    tenant_slug: str = Field(min_length=2, max_length=40, pattern=r"^[a-z0-9\-]+$")
    email: str
    password: str = Field(min_length=8)
    plan: str = "free"

class RegisterResponse(BaseModel):
    tenant_id:    str
    user_id:      str
    access_token: str
    token_type:   str = "bearer"


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class ChunkOut(BaseModel):
    text:        str
    source_id:   str
    chunk_index: int
    modality:    str
    score:       float
    sources:     List[str]

class QueryResponse(BaseModel):
    answer:     str
    sources:    List[ChunkOut]
    cost_usd:   float
    cached:     bool
    model:      str
    latency_ms: float

class QueryHistoryItem(BaseModel):
    query:      str
    answer:     str
    cost_usd:   float
    cached:     bool
    created_at: datetime


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id:  str
    status:  str
    source:  str

class IngestStatusResponse(BaseModel):
    job_id:  str
    status:  str
    result:  Optional[Any] = None

class DocumentOut(BaseModel):
    id:          str
    filename:    str
    modality:    str
    status:      str
    chunk_count: int
    created_at:  datetime

class DocumentList(BaseModel):
    documents: List[DocumentOut]
    total:     int


# ── Admin ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email:    str
    role:     str = "viewer"
    password: str = Field(min_length=8)

class UserResponse(BaseModel):
    id:         str
    email:      str
    role:       str
    created_at: datetime

class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)

class ApiKeyResponse(BaseModel):
    id:         str
    name:       str
    key:        str   # shown only on creation
    created_at: datetime

class StatsResponse(BaseModel):
    chunks_by_modality: Dict[str, int]
    cache_hit_rate_24h: float
    llm_cost_today_usd: float
    total_documents:    int

class CostEntry(BaseModel):
    date:     str
    cost_usd: float
    queries:  int

class CostsResponse(BaseModel):
    entries:   List[CostEntry]
    total_usd: float
