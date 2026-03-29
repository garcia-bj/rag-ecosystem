"""Base classes and shared data model for all parsers."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["text", "image", "audio", "structured", "web"]


class ParsedChunk(BaseModel):
    """Single unit of parsed content, ready for embedding."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    modality: Modality
    source_id: str
    chunk_index: int
    # Multi-tenancy fields — empty string means "no tenant / system context"
    tenant_id:   str = ""
    uploaded_by: str = ""

    model_config = {"extra": "allow"}


class BaseParser(ABC):
    """Abstract base for all modality parsers."""

    @abstractmethod
    async def parse(
        self,
        source: str | Path | bytes,
        source_id: str | None = None,
    ) -> list[ParsedChunk]:
        """Parse source into chunks.

        Args:
            source: File path, URL string, or raw bytes.
            source_id: Stable identifier for deduplication. Auto-generated if None.

        Returns:
            Ordered list of ParsedChunk objects.
        """
        ...

    # ------------------------------------------------------------------ helpers

    def _make_source_id(
        self, source: str | Path | bytes, provided: str | None
    ) -> str:
        if provided:
            return provided
        if isinstance(source, (str, Path)):
            return str(source)
        return str(uuid.uuid4())
