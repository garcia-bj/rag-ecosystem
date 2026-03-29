"""Image parser — Mistral OCR (primary) or Gemini Vision (fallback) + EXIF metadata.

Backend selection via OCR_BACKEND env var:
  - "mistral" (default if MISTRAL_API_KEY set) — extracts actual text via Mistral OCR API
  - "gemini"                                   — generates a visual description via Gemini Vision
  - "none"                                     — EXIF metadata only
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path

from .base import BaseParser, ParsedChunk

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    _GENAI_OK = True
    _GENAI_V2 = True
except ImportError:
    try:
        import google.generativeai as genai  # type: ignore[no-redef]
        _GENAI_OK = True
        _GENAI_V2 = False
    except ImportError:
        _GENAI_OK = False
        _GENAI_V2 = False

_SUPPORTED = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Select OCR backend: "mistral" | "gemini" | "none"
# Auto-selects mistral if MISTRAL_API_KEY is present, otherwise gemini.
def _default_backend() -> str:
    key = os.getenv("MISTRAL_API_KEY", "")
    if key and "CHANGE_ME" not in key:
        return "mistral"
    return "gemini"

_OCR_BACKEND = os.getenv("OCR_BACKEND", _default_backend()).lower()

_CAPTION_PROMPT = (
    "Describe this image in detail for a document retrieval system. "
    "Include: main subject, setting, any visible text, colors, and relevant objects. "
    "Be factual and concise (2-4 sentences)."
)

_MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")


class ImageParser(BaseParser):
    """Extract text from images via Mistral OCR or Gemini Vision + EXIF metadata.

    OCR priority:
      1. Mistral OCR   — when OCR_BACKEND=mistral (returns actual text content)
      2. Gemini Vision — when OCR_BACKEND=gemini  (returns visual description)
      3. EXIF only     — when neither API key is configured

    Mistral OCR is preferred for images containing text (invoices, screenshots,
    scanned forms). Gemini Vision is better for natural-scene photos.
    """

    def __init__(
        self,
        backend: str | None = None,
        gemini_model: str = "gemini-1.5-flash",
        caption_prompt: str = _CAPTION_PROMPT,
    ) -> None:
        if not _PIL_OK:
            raise ImportError("Pillow is required: pip install Pillow")
        self._backend = (backend or _OCR_BACKEND).lower()
        self._gemini_model = gemini_model
        self._caption_prompt = caption_prompt
        self._gemini_client = None

    # ── Public ────────────────────────────────────────────────────────────────

    async def parse(
        self,
        source: str | Path | bytes,
        source_id: str | None = None,
    ) -> list[ParsedChunk]:
        sid = self._make_source_id(source, source_id)
        return await asyncio.to_thread(self._parse_sync, source, sid)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse_sync(
        self, source: str | Path | bytes, source_id: str
    ) -> list[ParsedChunk]:
        if isinstance(source, bytes):
            import io
            img = Image.open(io.BytesIO(source))
            raw_bytes = source
        else:
            img = Image.open(str(source))
            with open(str(source), "rb") as f:
                raw_bytes = f.read()

        fmt = (img.format or "JPEG").upper()
        exif = self._extract_exif(img)

        # Extract text / description
        if self._backend == "mistral":
            ocr_text = self._mistral_ocr(raw_bytes, fmt)
            if not ocr_text:
                # Fallback to Gemini if Mistral returns nothing
                logger.debug("Mistral OCR empty — falling back to Gemini Vision")
                ocr_text = self._gemini_caption(img, raw_bytes, fmt)
        elif self._backend == "gemini":
            ocr_text = self._gemini_caption(img, raw_bytes, fmt)
        else:
            ocr_text = ""

        text_parts = []
        if ocr_text:
            text_parts.append(ocr_text)
        if exif:
            exif_summary = ", ".join(f"{k}: {v}" for k, v in list(exif.items())[:8])
            text_parts.append(f"[EXIF: {exif_summary}]")

        text = "\n\n".join(text_parts) if text_parts else "[image: no text extracted]"

        meta: dict = {
            "source_id": source_id,
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "ocr_backend": self._backend,
            **exif,
        }

        return [
            ParsedChunk(
                text=text,
                metadata=meta,
                modality="image",
                source_id=source_id,
                chunk_index=0,
            )
        ]

    # ── Mistral OCR ───────────────────────────────────────────────────────────

    def _mistral_ocr(self, raw_bytes: bytes, fmt: str) -> str:
        """Extract text from image using Mistral OCR API."""
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key or "CHANGE_ME" in api_key:
            logger.debug("MISTRAL_API_KEY not configured — skipping Mistral OCR")
            return ""
        try:
            from mistralai import Mistral
        except ImportError:
            logger.warning("mistralai not installed: pip install mistralai")
            return ""

        try:
            client = Mistral(api_key=api_key)
            mime = _fmt_to_mime(fmt)
            b64 = base64.b64encode(raw_bytes).decode()

            response = client.ocr.process(
                model=_MISTRAL_OCR_MODEL,
                document={
                    "type": "image_url",
                    "image_url": f"data:{mime};base64,{b64}",
                },
                include_image_base64=False,
            )

            pages = [p.markdown for p in response.pages if p.markdown]
            text = "\n\n".join(pages).strip()
            logger.info(
                "Mistral OCR: extracted %d chars from image (%s)", len(text), fmt
            )
            return text

        except Exception as exc:
            logger.warning("Mistral OCR failed: %s", exc)
            return ""

    # ── Gemini Vision ─────────────────────────────────────────────────────────

    def _get_gemini_client(self):
        if not _GENAI_OK:
            return None
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key or api_key.startswith("CHANGE_ME"):
            return None
        if self._gemini_client is None:
            if _GENAI_V2:
                self._gemini_client = genai_new.Client(api_key=api_key)
            else:
                genai.configure(api_key=api_key)  # type: ignore[name-defined]
                self._gemini_client = genai.GenerativeModel(  # type: ignore[name-defined]
                    self._gemini_model
                )
        return self._gemini_client

    def _gemini_caption(self, img: "Image.Image", raw_bytes: bytes, fmt: str) -> str:
        """Generate a visual description using Gemini Vision."""
        client = self._get_gemini_client()
        if client is None:
            return ""
        try:
            import io
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            img_bytes = buf.getvalue()
            mime = _fmt_to_mime(fmt)

            if _GENAI_V2:
                response = client.models.generate_content(
                    model=self._gemini_model,
                    contents=[
                        self._caption_prompt,
                        genai_types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    ],
                )
                return response.text.strip()
            else:
                image_part = {
                    "mime_type": mime,
                    "data": base64.b64encode(img_bytes).decode(),
                }
                response = client.generate_content(
                    [self._caption_prompt, image_part]
                )
                return response.text.strip()
        except Exception as exc:
            logger.warning("Gemini Vision failed: %s", exc)
            return ""

    # ── EXIF ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_exif(img: "Image.Image") -> dict:
        meta: dict = {}
        try:
            raw = img._getexif()  # type: ignore[attr-defined]
            if not raw:
                return meta
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, str(tag_id))
                if tag == "GPSInfo" and isinstance(value, dict):
                    gps: dict = {}
                    for gps_tag_id, gps_val in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                        gps[gps_tag] = gps_val
                    meta["gps"] = gps
                elif isinstance(value, (str, int, float, tuple)):
                    meta[tag] = value
        except Exception:
            pass
        return meta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_to_mime(fmt: str) -> str:
    """Convert PIL format string to MIME type."""
    mapping = {
        "JPEG": "image/jpeg",
        "JPG":  "image/jpeg",
        "PNG":  "image/png",
        "GIF":  "image/gif",
        "WEBP": "image/webp",
        "TIFF": "image/tiff",
        "TIF":  "image/tiff",
        "BMP":  "image/bmp",
    }
    return mapping.get(fmt.upper(), f"image/{fmt.lower()}")
