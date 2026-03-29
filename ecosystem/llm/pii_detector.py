"""PII detector — spaCy NER + regex patterns.  PII forces Ollama offline routing."""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# spaCy entity labels considered personally identifiable
# NOTE: ORG removed — brand/company names (iPhone, Apple, Tesla) are not PII
_PII_NER_LABELS = frozenset({"PERSON"})

# Regex patterns for PII not reliably caught by NER
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d{1,3}[\s.\-]?)?"
    r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"
    r"(?!\d)"
)


class PIIDetector:
    """Detect PII entities in text using spaCy NER + regex patterns.

    Detected types:
    - PERSON  — personal names (via spaCy)
    - ORG     — organizations (via spaCy)
    - EMAIL   — email addresses (via regex)
    - PHONE   — phone numbers (via regex)

    When PII is found the LLMRouter routes to Ollama (offline) to prevent
    data leakage to external LLM providers.
    """

    def __init__(self) -> None:
        self._nlp = None  # lazy — spaCy model loaded on first use

    # ── Public ────────────────────────────────────────────────────────────────

    def contains_pii(self, text: str) -> bool:
        """Return True if *text* contains any PII entity."""
        if _EMAIL_RE.search(text) or _PHONE_RE.search(text):
            return True
        return self._spacy_contains(text)

    def get_entities(self, text: str) -> List[dict]:
        """Return all detected PII entities as [{text, type}, ...]."""
        entities: List[dict] = []

        for m in _EMAIL_RE.finditer(text):
            entities.append({"text": m.group(), "type": "EMAIL"})
        for m in _PHONE_RE.finditer(text):
            entities.append({"text": m.group(), "type": "PHONE"})

        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in _PII_NER_LABELS:
                    entities.append({"text": ent.text, "type": ent.label_})

        return entities

    # ── Internal ──────────────────────────────────────────────────────────────

    def _spacy_contains(self, text: str) -> bool:
        nlp = self._get_nlp()
        if nlp is None:
            return False
        doc = nlp(text)
        return any(ent.label_ in _PII_NER_LABELS for ent in doc.ents)

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as exc:
                logger.warning("spaCy model unavailable — NER PII detection disabled: %s", exc)
                self._nlp = False  # sentinel: don't retry
        return self._nlp if self._nlp is not False else None
