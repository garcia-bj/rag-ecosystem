"""MCP tools: ingest_document and get_ingest_status."""

from __future__ import annotations

import logging
import os
import uuid

logger = logging.getLogger(__name__)


async def ingest_document(
    file_path: str,
    tenant_id: str,
    user_id: str,
    metadata: dict | None = None,
) -> dict:
    """Enqueue a document for ingestion via Celery.

    Args:
        file_path:  Absolute path to the file (accessible by the worker).
        tenant_id:  Caller's tenant UUID.
        user_id:    ID of the user submitting the document.
        metadata:   Optional extra metadata (title, description, …).

    Returns:
        {"job_id": str, "status": "queued", "file_path": str}
    """
    from ingestion.workers.celery_app import app as celery_app

    job_id = str(uuid.uuid4())
    celery_app.send_task(
        "ingestion.workers.ingest_task.ingest_document",
        kwargs={
            "source":     file_path,
            "source_id":  os.path.basename(file_path),
            "tenant_id":  tenant_id,
            "user_id":    user_id,
            "metadata":   metadata or {},
        },
        task_id=job_id,
        queue="ingest",
    )
    logger.info("MCP ingest queued job_id=%s tenant=%s", job_id, tenant_id[:8])
    return {"job_id": job_id, "status": "queued", "file_path": file_path}


async def get_ingest_status(job_id: str) -> dict:
    """Return the status of a previously submitted ingest job.

    Returns:
        {"job_id": str, "status": str, "result": any}
    """
    from ingestion.workers.celery_app import app as celery_app
    from celery.result import AsyncResult

    res: AsyncResult = celery_app.AsyncResult(job_id)
    state = res.state.lower()
    result = None
    if state == "success":
        result = res.result
    elif state == "failure":
        result = str(res.result)

    return {"job_id": job_id, "status": state, "result": result}
