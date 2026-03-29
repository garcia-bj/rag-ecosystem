"""Main ingestion task — detect modality → parse → chunk → extract metadata → queue."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from celery import Task

from ingestion.workers.celery_app import app
from ingestion.parsers.base import ParsedChunk
from ingestion.chunkers.semantic import SemanticChunker, ChunkConfig
from ingestion.chunkers.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

# ── Modality detection ─────────────────────────────────────────────────────────

_TEXT_EXT     = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt", ".rst"}
_IMAGE_EXT    = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_AUDIO_EXT    = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
_STRUCT_EXT   = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet"}


def detect_modality(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return "web"
    ext = Path(source).suffix.lower()
    if ext in _TEXT_EXT:
        return "text"
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _AUDIO_EXT:
        return "audio"
    if ext in _STRUCT_EXT:
        return "structured"
    return "text"  # safe default


# ── Lazy singletons (loaded once per worker process) ─────────────────────────

_chunker: SemanticChunker | None = None
_extractor: MetadataExtractor | None = None


def _get_chunker() -> SemanticChunker:
    global _chunker
    if _chunker is None:
        _chunker = SemanticChunker(
            ChunkConfig(
                min_tokens=int(os.getenv("CHUNK_MIN_TOKENS", "256")),
                max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "1500")),
                overlap_ratio=float(os.getenv("CHUNK_OVERLAP_RATIO", "0.20")),
            )
        )
    return _chunker


def _get_extractor() -> MetadataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = MetadataExtractor(
            use_llm=os.getenv("METADATA_LLM_ENABLED", "true").lower() == "true"
        )
    return _extractor


# ── Task ───────────────────────────────────────────────────────────────────────

class IngestTask(Task):
    """Custom Task base for typed error handling and retries."""
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "ingest_document FAILED task_id=%s source=%s: %s",
            task_id,
            kwargs.get("source", args[0] if args else "?"),
            exc,
        )


@app.task(
    bind=True,
    base=IngestTask,
    name="ingestion.workers.ingest_task.ingest_document",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def ingest_document(
    self: IngestTask,
    source: str,
    source_id: str | None = None,
    modality_hint: str | None = None,
    chunk_config: dict[str, Any] | None = None,
    extract_metadata: bool = True,
    tenant_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a document through the full pipeline.

    Args:
        source: File path or URL.
        source_id: Stable ID for deduplication; defaults to source.
        modality_hint: Override auto-detection ("text"|"image"|"audio"|"structured"|"web").
        chunk_config: Optional overrides for ChunkConfig fields.
        extract_metadata: Whether to run NER + LLM metadata extraction.

    Returns:
        Summary dict with source_id, chunk_count, modality.
    """
    modality = modality_hint or detect_modality(source)
    sid = source_id or source

    logger.info("ingest_document: source=%s modality=%s", source, modality)

    try:
        chunks = asyncio.run(_run_pipeline(
            source=source,
            source_id=sid,
            modality=modality,
            chunk_config=chunk_config,
            extract_metadata=extract_metadata,
        ))
    except Exception as exc:
        logger.exception("Pipeline error for %s", source)
        raise self.retry(exc=exc)

    # Stamp tenant_id and user_id on every chunk
    for chunk in chunks:
        chunk.tenant_id   = tenant_id or ""
        chunk.uploaded_by = user_id   or ""

    # Dispatch chunks to the embed queue
    _dispatch_to_embed(chunks, tenant_id=tenant_id or "")

    result = {
        "source_id": sid,
        "modality": modality,
        "chunk_count": len(chunks),
        "chunks": [c.model_dump() for c in chunks],
    }
    logger.info(
        "ingest_document done: source=%s chunks=%d", source, len(chunks)
    )

    # Update document status in PostgreSQL
    try:
        asyncio.run(_update_document_status(
            job_id=self.request.id,
            status="completed",
            chunk_count=len(chunks),
        ))
    except Exception as exc:
        logger.warning("Failed to update document status: %s", exc)

    # Clean up temp file (already persisted to RustFS by the API layer)
    _cleanup_temp_file(source)

    return result


# ── Pipeline (async) ───────────────────────────────────────────────────────────

async def _run_pipeline(
    source: str,
    source_id: str,
    modality: str,
    chunk_config: dict[str, Any] | None,
    extract_metadata: bool,
) -> list[ParsedChunk]:
    # 1. Parse
    parser = _build_parser(modality)
    raw_chunks = await parser.parse(source, source_id=source_id)

    if not raw_chunks:
        logger.warning("No chunks parsed from %s", source)
        return []

    # 2. Semantic chunking
    chunker = _get_chunker()
    if chunk_config:
        cfg = ChunkConfig(**chunk_config)
        chunker = SemanticChunker(cfg)
    chunked = chunker.chunk(raw_chunks)

    # 3. Metadata extraction
    if extract_metadata:
        extractor = _get_extractor()
        chunked = await extractor.extract_batch(chunked)

    return chunked


def _build_parser(modality: str):
    if modality == "text":
        from ingestion.parsers.document import DocumentParser
        return DocumentParser()
    if modality == "image":
        from ingestion.parsers.image import ImageParser
        return ImageParser()
    if modality == "audio":
        from ingestion.parsers.audio import AudioParser
        return AudioParser()
    if modality == "structured":
        from ingestion.parsers.structured import StructuredParser
        return StructuredParser()
    if modality == "web":
        from ingestion.parsers.web import WebParser
        return WebParser()
    raise ValueError(f"Unknown modality: {modality!r}")


# ── Dispatch ───────────────────────────────────────────────────────────────────

def _dispatch_to_embed(chunks: list[ParsedChunk], tenant_id: str = "") -> None:
    """Publish processed chunks to the embed queue for vector embedding."""
    if not chunks:
        return
    try:
        from ingestion.workers.celery_app import app as _app
        _app.send_task(
            "embeddings.workers.index_task.embed_chunks",
            kwargs={"chunks": [c.model_dump() for c in chunks], "tenant_id": tenant_id},
            queue="embed",
            routing_key="embed",
        )
        logger.debug("Dispatched %d chunks to embed queue", len(chunks))
    except Exception as exc:
        # Non-fatal: embedding is a separate stage
        logger.warning("Could not dispatch to embed queue: %s", exc)


async def _update_document_status(job_id: str, status: str, chunk_count: int) -> None:
    """Update the documents table with final ingest status."""
    from core.database import get_connection
    conn = await get_connection()
    try:
        await conn.execute(
            "UPDATE documents SET status=$1, chunk_count=$2 WHERE id=$3",
            status, chunk_count, job_id,
        )
    finally:
        await conn.close()


def _cleanup_temp_file(source: str) -> None:
    """Delete the local temp file after successful ingestion.

    Only deletes files under the configured UPLOAD_DIR to avoid accidental
    deletion of user-provided paths. URLs are ignored.
    """
    if source.startswith(("http://", "https://")):
        return
    try:
        upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp/rag_uploads")).resolve()
        source_path = Path(source).resolve()
        if str(source_path).startswith(str(upload_dir)):
            source_path.unlink(missing_ok=True)
            logger.debug("Cleaned up temp file: %s", source_path)
    except Exception as exc:
        logger.debug("Temp file cleanup skipped: %s", exc)
