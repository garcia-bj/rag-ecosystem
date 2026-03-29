"""Document parser — PDF, DOCX, HTML, Markdown via Unstructured.io.

For scanned PDFs (image-based, no embedded text), Mistral OCR is used as a
fallback when OCR_BACKEND=mistral and MISTRAL_API_KEY is configured.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path

from .base import BaseParser, ParsedChunk

logger = logging.getLogger(__name__)

# Unstructured is imported lazily inside _parse_sync to avoid
# loading libmagic at collection time (causes segfault on some Windows setups).
_UNSTRUCTURED_OK: bool | None = None  # None = not yet checked

# Minimum characters per page to consider text extraction successful.
# Below this threshold the PDF is likely scanned — trigger Mistral OCR fallback.
_MIN_CHARS_PER_PAGE = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))

_MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")


def _check_unstructured() -> bool:
    global _UNSTRUCTURED_OK
    if _UNSTRUCTURED_OK is None:
        try:
            import unstructured.partition.auto  # noqa: F401
            _UNSTRUCTURED_OK = True
        except Exception:
            _UNSTRUCTURED_OK = False
    return _UNSTRUCTURED_OK

_SUPPORTED = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt", ".rst"}


class DocumentParser(BaseParser):
    """Parse text-heavy documents into page/section-aware chunks.

    Each returned chunk maps to a logical element (paragraph, heading,
    table, list item) extracted by Unstructured.

    For scanned/image-based PDFs that yield little text, Mistral OCR is
    used as a fallback when OCR_BACKEND=mistral and MISTRAL_API_KEY is set.
    """

    def __init__(
        self,
        strategy: str = "fast",  # "fast" | "hi_res" | "ocr_only"
        languages: list[str] | None = None,
    ) -> None:
        if not _check_unstructured():
            raise ImportError(
                "unstructured is required: pip install 'unstructured[pdf,docx,html]'"
            )
        self.strategy = strategy
        self.languages = languages or ["eng"]

    async def parse(
        self,
        source: str | Path | bytes,
        source_id: str | None = None,
    ) -> list[ParsedChunk]:
        sid = self._make_source_id(source, source_id)
        return await asyncio.to_thread(self._parse_sync, source, sid)

    # ------------------------------------------------------------------ internals

    def _parse_sync(
        self, source: str | Path | bytes, source_id: str
    ) -> list[ParsedChunk]:
        kwargs: dict = {
            "strategy": self.strategy,
            "languages": self.languages,
            "include_metadata": True,
        }

        from unstructured.partition.auto import partition as _partition

        raw_bytes: bytes | None = None

        if isinstance(source, bytes):
            import tempfile
            raw_bytes = source
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(source)
                tmp_path = f.name
            try:
                elements = _partition(filename=tmp_path, **kwargs)
            finally:
                os.unlink(tmp_path)
        else:
            elements = _partition(filename=str(source), **kwargs)

        chunks: list[ParsedChunk] = []
        for idx, el in enumerate(elements):
            text = str(el).strip()
            if not text:
                continue

            meta = self._element_metadata(el)
            meta["element_type"] = type(el).__name__
            meta["source_id"] = source_id

            chunks.append(
                ParsedChunk(
                    text=text,
                    metadata=meta,
                    modality="text",
                    source_id=source_id,
                    chunk_index=idx,
                )
            )

        logger.info("DocumentParser: %d elements from %s", len(chunks), source_id)

        # If PDF yielded very little text it's likely a scanned document.
        # Attempt Mistral OCR as fallback.
        is_pdf = (
            not isinstance(source, bytes)
            and str(source).lower().endswith(".pdf")
        )
        if is_pdf and self._is_scanned(chunks, source):
            logger.info(
                "DocumentParser: sparse text detected — attempting Mistral OCR for %s",
                source_id,
            )
            ocr_chunks = self._mistral_ocr_pdf(source, source_id)
            if ocr_chunks:
                logger.info(
                    "Mistral OCR: replaced %d Unstructured chunks with %d OCR chunks",
                    len(chunks),
                    len(ocr_chunks),
                )
                return ocr_chunks

        return chunks

    @staticmethod
    def _is_scanned(chunks: list[ParsedChunk], source: str | Path | bytes) -> bool:
        """Heuristic: very low text yield for a PDF suggests a scanned document."""
        if isinstance(source, bytes):
            return False
        total_chars = sum(len(c.text) for c in chunks)
        # Rough estimate: assume at least 1 page, flag if < threshold chars total
        is_sparse = total_chars < _MIN_CHARS_PER_PAGE
        return is_sparse

    def _mistral_ocr_pdf(
        self, source: str | Path, source_id: str
    ) -> list[ParsedChunk]:
        """Run Mistral OCR on a PDF file and return ParsedChunks (one per page)."""
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key or "CHANGE_ME" in api_key:
            return []
        ocr_backend = os.getenv("OCR_BACKEND", "").lower()
        if ocr_backend not in ("mistral", ""):
            return []

        try:
            from mistralai import Mistral
        except ImportError:
            logger.warning("mistralai not installed: pip install mistralai")
            return []

        try:
            with open(str(source), "rb") as f:
                pdf_bytes = f.read()

            b64 = base64.b64encode(pdf_bytes).decode()
            client = Mistral(api_key=api_key)

            response = client.ocr.process(
                model=_MISTRAL_OCR_MODEL,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{b64}",
                },
                include_image_base64=False,
            )

            chunks: list[ParsedChunk] = []
            for page_idx, page in enumerate(response.pages):
                text = (page.markdown or "").strip()
                if not text:
                    continue
                chunks.append(
                    ParsedChunk(
                        text=text,
                        metadata={
                            "source_id": source_id,
                            "page_number": page_idx + 1,
                            "ocr_backend": "mistral",
                            "element_type": "OCRPage",
                        },
                        modality="text",
                        source_id=source_id,
                        chunk_index=page_idx,
                    )
                )

            logger.info(
                "Mistral OCR PDF: %d pages extracted from %s", len(chunks), source_id
            )
            return chunks

        except Exception as exc:
            logger.warning("Mistral OCR PDF failed: %s", exc)
            return []

    @staticmethod
    def _element_metadata(el: "Element") -> dict:
        meta: dict = {}
        raw = getattr(el, "metadata", None)
        if raw is None:
            return meta
        # Unstructured metadata is a dataclass-like object
        for attr in (
            "page_number",
            "filename",
            "file_directory",
            "filetype",
            "section",
            "url",
            "text_as_html",
        ):
            val = getattr(raw, attr, None)
            if val is not None:
                meta[attr] = val
        return meta
