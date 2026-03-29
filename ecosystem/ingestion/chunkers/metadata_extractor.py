"""Metadata extractor — spaCy NER + Claude LLM structured extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

try:
    import spacy
    _SPACY_OK = True
except ImportError:
    _SPACY_OK = False

try:
    from anthropic import AsyncAnthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

_SPACY_MODEL = "en_core_web_sm"
_LLM_EXTRACT_PROMPT = """\
Extract structured metadata from the following text.
Return ONLY a valid JSON object with these keys (omit keys with no clear value):
- title: string — document or section title if identifiable
- summary: string — 1–2 sentence summary
- topics: list[string] — 3–5 key topics or themes
- language: string — ISO 639-1 language code (e.g. "en", "es")
- doc_type: string — article | report | email | code | conversation | other
- sentiment: string — positive | negative | neutral
- date: string — any date mentioned (ISO format if possible)

Text:
{text}"""


class MetadataExtractor:
    """Two-stage metadata extraction:

    1. **spaCy NER** — fast, local, no API cost.
       Extracts: PERSON, ORG, GPE, LOC, DATE, MONEY, PRODUCT, EVENT, LAW.

    2. **Claude LLM** — for richer structured fields (title, summary, topics, …).
       Optional: only runs when ANTHROPIC_API_KEY is set and not placeholder.
    """

    def __init__(
        self,
        spacy_model: str = _SPACY_MODEL,
        use_llm: bool = True,
        llm_model: str = "claude-haiku-4-5-20251001",  # cheapest, fast enough
    ) -> None:
        self.use_llm = use_llm and _ANTHROPIC_OK
        self.llm_model = llm_model
        self._nlp = None
        self._llm: "AsyncAnthropic | None" = None

        if _SPACY_OK:
            try:
                self._nlp = spacy.load(spacy_model)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found. "
                    "Run: python -m spacy download %s",
                    spacy_model,
                    spacy_model,
                )

    def _get_llm(self) -> "AsyncAnthropic | None":
        if not self.use_llm:
            return None
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key or api_key.startswith("sk-ant-CHANGE"):
            return None
        if self._llm is None:
            self._llm = AsyncAnthropic(api_key=api_key)
        return self._llm

    async def extract(self, chunk: ParsedChunk) -> ParsedChunk:
        """Return a new ParsedChunk with enriched metadata."""
        meta = dict(chunk.metadata)

        # 1. spaCy NER (sync → thread)
        ner_meta = await asyncio.to_thread(self._run_ner, chunk.text)
        meta.update(ner_meta)

        # 2. LLM extraction (async)
        llm_client = self._get_llm()
        if llm_client and chunk.modality in ("text", "web", "structured"):
            llm_meta = await self._run_llm(chunk.text, llm_client)
            # LLM fields take precedence over defaults but not over already-set values
            for k, v in llm_meta.items():
                if k not in meta or not meta[k]:
                    meta[k] = v

        return chunk.model_copy(update={"metadata": meta})

    async def extract_batch(self, chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        tasks = [self.extract(c) for c in chunks]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------ spaCy

    def _run_ner(self, text: str) -> dict[str, Any]:
        if self._nlp is None:
            return {}
        # Truncate to avoid slow processing of very long texts
        doc = self._nlp(text[:10_000])
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            val = ent.text.strip()
            if val and val not in entities[label]:
                entities[label].append(val)

        meta: dict[str, Any] = {}
        for label in ("PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "PRODUCT", "EVENT", "LAW"):
            if label in entities:
                meta[f"ner_{label.lower()}"] = entities[label][:20]

        return meta

    # ------------------------------------------------------------------ LLM

    async def _run_llm(
        self, text: str, client: "AsyncAnthropic"
    ) -> dict[str, Any]:
        # Truncate to avoid large token bills
        truncated = text[:4000]
        prompt = _LLM_EXTRACT_PROMPT.format(text=truncated)
        try:
            response = await client.messages.create(
                model=self.llm_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            logger.warning("LLM metadata extraction failed: %s", exc)
            return {}
