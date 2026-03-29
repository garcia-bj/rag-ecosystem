"""Query normalizer — lowercase, strip accents, remove noise for cache key matching."""

from __future__ import annotations

import re
import unicodedata

# Stopwords that carry no semantic weight for cache matching
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "that", "this", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they",
    "them", "their", "what", "which", "who", "and", "but", "or", "nor",
    "so", "yet", "both", "either", "neither", "not", "no", "nor", "only",
    "own", "same", "than", "too", "very", "just", "because", "as", "until",
    "while", "although", "though", "unless", "since", "if", "when", "where",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "en",
    "con", "por", "para", "que", "del", "al", "se", "su", "sus", "le",
    "les", "me", "nos", "os", "lo", "les", "hay", "es", "son", "fue",
    "era", "ser", "estar", "tener", "hacer", "y", "o", "pero", "si",
    "como", "cuando", "donde", "porque", "aunque", "sino",
})

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_query(q: str) -> str:
    """Normalize a query string for semantic cache key matching.

    Steps:
    1. Lowercase
    2. Strip accents (NFD decomposition → drop combining chars)
    3. Remove punctuation
    4. Collapse whitespace
    5. Remove irrelevant stopwords
    6. Strip leading/trailing whitespace
    """
    # 1. Lowercase
    text = q.lower().strip()

    # 2. Strip accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # 3. Remove punctuation
    text = _PUNCT_RE.sub(" ", text)

    # 4. Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    # 5. Remove stopwords (preserve single-word queries)
    tokens = text.split()
    filtered = [t for t in tokens if t not in _STOPWORDS]
    if filtered:
        text = " ".join(filtered)
    # else: all tokens were stopwords — keep original collapsed form

    return text
