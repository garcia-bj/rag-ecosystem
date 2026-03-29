"""TTL classifier — assigns cache TTL (seconds) based on query type."""

from __future__ import annotations

import re

# ── TTL constants ─────────────────────────────────────────────────────────────

TTL_STATIC    = 7 * 24 * 3600   # 7 days  — static/encyclopaedic knowledge
TTL_FACTUAL   = 24 * 3600       # 24 hours — factual but may change
TTL_SYSTEM    = 5 * 60          # 5 min   — live system state / metrics
TTL_DEFAULT   = 1 * 3600        # 1 hour  — conversational / default

# ── Keyword patterns (lowercase, applied to normalized query) ─────────────────

_STATIC_PATTERNS = re.compile(
    r"\b("
    r"definition|define|what is|history|origin|explain|concept|theory|"
    r"formula|algorithm|protocol|standard|specification|rfc|iso|"
    r"biography|born|founded|invented|discovered|equation|law of|"
    r"programming language|framework|library|tutorial|how does .* work"
    r")\b",
    re.IGNORECASE,
)

_FACTUAL_PATTERNS = re.compile(
    r"\b("
    r"latest|current|recent|release|version|update|changelog|"
    r"price|cost|rate|revenue|market cap|ceo|cto|cfo|president|"
    r"population|gdp|index|ranking|statistics|report|"
    r"today|this year|this month|last year|last month|2024|2025|2026"
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_PATTERNS = re.compile(
    r"\b("
    r"status|uptime|health|ping|latency|throughput|error rate|"
    r"cpu|memory|disk|load|queue|backlog|lag|metric|alert|incident|"
    r"running|active|deployed|live|production|monitoring|"
    r"right now|at this moment|currently down|is.*down|are.*down"
    r")\b",
    re.IGNORECASE,
)


def classify_query_ttl(q: str) -> int:
    """Return cache TTL in seconds for the given query.

    Categories (checked in priority order):
    - system_state   →   5 min  (live metrics, health checks)
    - static_knowledge → 7 days (definitions, history, theory)
    - factual          → 24 h   (versioned facts, prices, stats)
    - default          →  1 h   (conversational / unclassified)
    """
    if _SYSTEM_PATTERNS.search(q):
        return TTL_SYSTEM

    if _STATIC_PATTERNS.search(q):
        return TTL_STATIC

    if _FACTUAL_PATTERNS.search(q):
        return TTL_FACTUAL

    return TTL_DEFAULT
