"""MCP resources: schema://chunks and config://retrieval."""

from __future__ import annotations

import json
import os

# ── schema://chunks ───────────────────────────────────────────────────────────

CHUNKS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title":   "ParsedChunk",
    "description": "Single unit of parsed + embedded content in the RAG ecosystem",
    "type": "object",
    "required": ["text", "modality", "source_id", "chunk_index"],
    "properties": {
        "text":        {"type": "string", "description": "Chunk text content"},
        "modality":    {
            "type": "string",
            "enum": ["text", "image", "audio", "structured", "web"],
            "description": "Content modality",
        },
        "source_id":   {"type": "string", "description": "Source document identifier"},
        "chunk_index": {"type": "integer", "description": "Position within source document"},
        "tenant_id":   {"type": "string", "description": "Owning tenant UUID"},
        "uploaded_by": {"type": "string", "description": "User UUID who uploaded the document"},
        "metadata": {
            "type": "object",
            "description": "NER entities, columns, and other extracted metadata",
            "properties": {
                "ner_person":  {"type": "array", "items": {"type": "string"}},
                "ner_org":     {"type": "array", "items": {"type": "string"}},
                "ner_date":    {"type": "array", "items": {"type": "string"}},
                "title":       {"type": "string"},
                "summary":     {"type": "string"},
                "topics":      {"type": "array", "items": {"type": "string"}},
                "language":    {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
}

# ── config://retrieval ────────────────────────────────────────────────────────

def get_retrieval_config() -> dict:
    """Return current retrieval pipeline configuration."""
    return {
        "pipeline": {
            "stages": [
                "semantic_cache",
                "query_expansion",
                "hybrid_retrieval",
                "rrf_fusion",
                "conditional_reranking",
                "context_compression",
                "cache_set",
            ],
        },
        "semantic_cache": {
            "similarity_threshold": 0.87,
            "backend": "redis",
        },
        "query_expansion": {
            "hyde_min_tokens": 20,
            "multi_query_variants": 3,
            "model": "claude-haiku-4-5-20251001",
            "cache_ttl_seconds": 3600,
        },
        "retrieval": {
            "qdrant_top_k":  int(os.getenv("QDRANT_TOP_K",  "50")),
            "es_top_k":      int(os.getenv("ES_TOP_K",      "50")),
            "graph_top_k":   int(os.getenv("GRAPH_TOP_K",   "20")),
        },
        "rrf": {
            "k": 60,
        },
        "reranker": {
            "backend":             os.getenv("RERANKER_BACKEND", "local"),
            "activation_threshold": 0.72,
            "local_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cohere_model": "rerank-v3.5",
        },
        "compressor": {
            "threshold_tokens": 2000,
            "model": "claude-haiku-4-5-20251001",
        },
    }
