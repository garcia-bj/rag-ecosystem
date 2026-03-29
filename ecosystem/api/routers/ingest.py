"""Ingest router — file upload, URL ingest, status, list, delete."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.middleware.auth import get_current_user, require_editor_or_admin
from api.middleware.rate_limit import get_rate_limiter
from api.schemas import DocumentList, DocumentOut, IngestResponse, IngestStatusResponse
from core.tenant import TenantContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/rag_uploads"))
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_file(
    file: UploadFile = File(...),
    ctx: TenantContext = Depends(require_editor_or_admin),
):
    """Upload a file and enqueue it for ingestion."""
    plan = await _get_tenant_plan(ctx.tenant_id)
    await get_rate_limiter().check_doc_ingest(ctx.tenant_id, plan)

    # Validate file size
    content = await file.read()
    if len(content) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds 50 MB limit (got {len(content) // (1024*1024)} MB)",
        )

    # Save to local temp dir (Celery worker reads from here)
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{Path(file.filename or 'upload').name}"
    dest = _UPLOAD_DIR / safe_name
    dest.write_bytes(content)

    # Upload to RustFS: {tenant[:8]}/{user[:8]}/{safe_name}
    storage_key = await _upload_to_storage(content, ctx.tenant_id, ctx.user_id, safe_name)

    # Enqueue Celery task
    job_id = _enqueue_ingest(str(dest), file.filename or safe_name, ctx)

    # Record in documents table
    await _create_document_record(
        ctx.tenant_id, ctx.user_id, file.filename or safe_name, job_id, storage_key
    )

    return IngestResponse(job_id=job_id, status="queued", source=file.filename or safe_name)


@router.post("/url", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_url(
    url: str,
    ctx: TenantContext = Depends(require_editor_or_admin),
):
    """Enqueue a public URL for ingestion."""
    plan = await _get_tenant_plan(ctx.tenant_id)
    await get_rate_limiter().check_doc_ingest(ctx.tenant_id, plan)

    job_id = _enqueue_ingest(url, url, ctx)
    await _create_document_record(ctx.tenant_id, ctx.user_id, url, job_id)
    return IngestResponse(job_id=job_id, status="queued", source=url)


@router.get("/status/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(
    job_id: str,
    _: TenantContext = Depends(get_current_user),
):
    """Return the current status of an ingest job."""
    try:
        from ingestion.workers.celery_app import app as celery_app
        res = celery_app.AsyncResult(job_id)
        state  = res.state.lower()
        result = None
        if state == "success":
            result = res.result
        elif state == "failure":
            result = str(res.result)
        return IngestStatusResponse(job_id=job_id, status=state, result=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    ctx: TenantContext = Depends(get_current_user),
    offset: int = 0,
    limit: int = 50,
):
    """Return all documents for this tenant."""
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT id, filename, modality, status, chunk_count, created_at
                FROM documents
                WHERE tenant_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                ctx.tenant_id, limit, offset,
            )
            total = await conn.fetchval(
                "SELECT count(*) FROM documents WHERE tenant_id = $1", ctx.tenant_id
            )
            docs = [
                DocumentOut(
                    id=str(r["id"]),
                    filename=r["filename"],
                    modality=r["modality"],
                    status=r["status"],
                    chunk_count=r["chunk_count"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]
            return DocumentList(documents=docs, total=total or 0)
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("list_documents error: %s", exc)
        return DocumentList(documents=[], total=0)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    ctx: TenantContext = Depends(require_editor_or_admin),
):
    """Delete a document and its chunks from all stores."""
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            row = await conn.fetchrow(
                "SELECT filename FROM documents WHERE id = $1 AND tenant_id = $2",
                doc_id, ctx.tenant_id,
            )
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")
            await conn.execute(
                "DELETE FROM documents WHERE id = $1", doc_id
            )
        finally:
            await conn.close()

        # Remove from vector stores (best-effort)
        await _delete_from_stores(doc_id, ctx.tenant_id)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enqueue_ingest(source: str, source_id: str, ctx: TenantContext) -> str:
    from ingestion.workers.celery_app import app as celery_app
    job_id = str(uuid.uuid4())
    celery_app.send_task(
        "ingestion.workers.ingest_task.ingest_document",
        kwargs={
            "source":    source,
            "source_id": source_id,
            "tenant_id": ctx.tenant_id,
            "user_id":   ctx.user_id,
            "metadata":  {},
        },
        task_id=job_id,
        queue="ingest",
    )
    return job_id


async def _upload_to_storage(
    content: bytes, tenant_id: str, user_id: str, safe_name: str
) -> str | None:
    """Upload file bytes to RustFS. Returns storage_key or None if S3 not configured."""
    from core.storage import is_configured, get_storage, make_storage_key
    if not is_configured():
        logger.debug("S3 storage not configured — skipping RustFS upload")
        return None
    try:
        storage = get_storage()
        key = make_storage_key(tenant_id, user_id, safe_name)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, storage.upload_bytes, content, key)
        return key
    except Exception as exc:
        logger.warning("RustFS upload failed (non-fatal): %s", exc)
        return None


async def _create_document_record(
    tenant_id: str, user_id: str, filename: str, job_id: str,
    storage_key: str | None = None,
) -> None:
    try:
        from core.database import get_connection
        from ingestion.workers.ingest_task import detect_modality
        conn = await get_connection()
        try:
            await conn.execute(
                """
                INSERT INTO documents (id, tenant_id, filename, modality, status, storage_key, created_by)
                VALUES ($1, $2, $3, $4, 'queued', $5, $6)
                ON CONFLICT DO NOTHING
                """,
                job_id, tenant_id, filename,
                detect_modality(filename), storage_key, user_id,
            )
        finally:
            await conn.close()
    except Exception as exc:
        logger.debug("create_document_record failed: %s", exc)


async def _delete_from_stores(doc_id: str, tenant_id: str) -> None:
    import asyncio
    async def _del_es():
        try:
            from core.tenant import get_tenant_index
            from elasticsearch import AsyncElasticsearch
            es = AsyncElasticsearch(
                hosts=[os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")]
            )
            await es.delete_by_query(
                index=get_tenant_index(tenant_id),
                query={"term": {"source_id": doc_id}},
            )
            await es.close()
        except Exception:
            pass

    async def _del_neo4j():
        try:
            from graph.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            d  = await kg._get_driver()
            async with d.session() as s:
                await s.run(
                    "MATCH (n) WHERE n.source_id = $sid AND n.tenant_id = $tid DETACH DELETE n",
                    sid=doc_id, tid=tenant_id,
                )
            await kg.close()
        except Exception:
            pass

    await asyncio.gather(_del_es(), _del_neo4j(), return_exceptions=True)


async def _get_tenant_plan(tenant_id: str) -> str:
    try:
        from core.database import get_tenant_by_id
        t = await get_tenant_by_id(tenant_id)
        return t.plan if t else "free"
    except Exception:
        return "free"
