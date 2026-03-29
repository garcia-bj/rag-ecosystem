"""S3-compatible object storage client — RustFS / MinIO / AWS S3.

Documents are stored under:  {tenant_id[:8]}/{user_id[:8]}/{uuid_filename}

Uses a separate bucket (S3_RAG_BUCKET) from the Langfuse bucket (S3_BUCKET)
so observability data and RAG documents don't mix.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_S3_ENDPOINT   = os.getenv("S3_ENDPOINT", "")
_S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
_S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
_S3_REGION     = os.getenv("S3_REGION", "us-east-1")
_RAG_BUCKET    = os.getenv("S3_RAG_BUCKET", "rag-documents")


def is_configured() -> bool:
    """Return True if S3 credentials are set (not placeholder values)."""
    return bool(
        _S3_ENDPOINT
        and _S3_ACCESS_KEY
        and "CHANGE_ME" not in _S3_ACCESS_KEY
    )


def make_storage_key(tenant_id: str, user_id: str, filename: str) -> str:
    """Build a namespaced S3 key: <tenant[:8]>/<user[:8]>/<filename>"""
    tenant_prefix = (tenant_id or "unknown")[:8]
    user_prefix   = (user_id   or "unknown")[:8]
    return f"{tenant_prefix}/{user_prefix}/{filename}"


class StorageClient:
    """Thin sync wrapper around boto3 for RustFS uploads/downloads.

    All public methods are synchronous — call them from a thread executor
    if you need to use them inside async code:

        loop = asyncio.get_event_loop()
        key  = await loop.run_in_executor(None, storage.upload_bytes, content, key)
    """

    def __init__(self) -> None:
        self._client = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "s3",
                endpoint_url=_S3_ENDPOINT or None,
                aws_access_key_id=_S3_ACCESS_KEY or None,
                aws_secret_access_key=_S3_SECRET_KEY or None,
                region_name=_S3_REGION,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=_RAG_BUCKET)
        except Exception:
            try:
                self._client.create_bucket(Bucket=_RAG_BUCKET)
                logger.info("Created S3 bucket: %s", _RAG_BUCKET)
            except Exception as exc:
                logger.warning("Could not create bucket %s: %s", _RAG_BUCKET, exc)

    # ── Public ────────────────────────────────────────────────────────────────

    def upload_bytes(self, content: bytes, storage_key: str) -> str:
        """Upload raw bytes to S3 and return the storage key."""
        import io
        client = self._get_client()
        client.upload_fileobj(io.BytesIO(content), _RAG_BUCKET, storage_key)
        logger.info("Uploaded %d bytes → s3://%s/%s", len(content), _RAG_BUCKET, storage_key)
        return storage_key

    def upload_file(self, local_path: str, storage_key: str) -> str:
        """Upload a local file to S3 and return the storage key."""
        client = self._get_client()
        client.upload_file(local_path, _RAG_BUCKET, storage_key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, _RAG_BUCKET, storage_key)
        return storage_key

    def download_file(self, storage_key: str, local_path: str) -> None:
        """Download a file from S3 to a local path."""
        client = self._get_client()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(_RAG_BUCKET, storage_key, local_path)
        logger.info("Downloaded s3://%s/%s → %s", _RAG_BUCKET, storage_key, local_path)

    def delete_file(self, storage_key: str) -> None:
        """Delete a file from S3 (best-effort)."""
        try:
            client = self._get_client()
            client.delete_object(Bucket=_RAG_BUCKET, Key=storage_key)
            logger.info("Deleted s3://%s/%s", _RAG_BUCKET, storage_key)
        except Exception as exc:
            logger.warning("S3 delete failed for %s: %s", storage_key, exc)

    def presigned_url(self, storage_key: str, expires: int = 3600) -> str:
        """Generate a presigned GET URL (default 1 hour expiry)."""
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": _RAG_BUCKET, "Key": storage_key},
            ExpiresIn=expires,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_storage: StorageClient | None = None


def get_storage() -> StorageClient:
    global _storage
    if _storage is None:
        _storage = StorageClient()
    return _storage
