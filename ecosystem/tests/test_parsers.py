"""Tests for parsers, chunkers, and the ingest task."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.parsers.base import ParsedChunk
from ingestion.chunkers.semantic import SemanticChunker, ChunkConfig

# ── Fixtures ───────────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


def _chunk(
    text: str,
    modality: str = "text",
    source_id: str = "test",
    idx: int = 0,
) -> ParsedChunk:
    return ParsedChunk(
        text=text,
        metadata={},
        modality=modality,
        source_id=source_id,
        chunk_index=idx,
    )


# ── ParsedChunk model ──────────────────────────────────────────────────────────

class TestParsedChunk:
    def test_required_fields(self):
        c = ParsedChunk(
            text="Hello world",
            modality="text",
            source_id="src_001",
            chunk_index=0,
        )
        assert c.text == "Hello world"
        assert c.modality == "text"
        assert c.metadata == {}

    def test_metadata_stored(self):
        c = ParsedChunk(
            text="test",
            modality="structured",
            source_id="s",
            chunk_index=0,
            metadata={"columns": ["a", "b"], "rows": 10},
        )
        assert c.metadata["columns"] == ["a", "b"]

    def test_invalid_modality(self):
        with pytest.raises(Exception):
            ParsedChunk(
                text="x",
                modality="unknown_modality",  # type: ignore
                source_id="s",
                chunk_index=0,
            )

    def test_model_copy_update(self):
        c = _chunk("original")
        c2 = c.model_copy(update={"text": "updated"})
        assert c2.text == "updated"
        assert c.text == "original"


# ── StructuredParser ───────────────────────────────────────────────────────────

class TestStructuredParser:
    def setup_method(self):
        from ingestion.parsers.structured import StructuredParser
        self.parser = StructuredParser(rows_per_chunk=5)

    def test_csv_file(self):
        chunks = asyncio.run(
            self.parser.parse(FIXTURES / "sample.csv", source_id="csv_test")
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert c.modality == "structured"
            assert c.source_id == "csv_test"
            assert c.text  # non-empty

    def test_csv_has_schema_metadata(self):
        chunks = asyncio.run(
            self.parser.parse(FIXTURES / "sample.csv", source_id="csv_schema")
        )
        assert "schema" in chunks[0].metadata
        assert "columns" in chunks[0].metadata

    def test_csv_chunked_correctly(self):
        # 10 data rows, rows_per_chunk=5 → 2 chunks
        chunks = asyncio.run(
            self.parser.parse(FIXTURES / "sample.csv", source_id="csv_chunks")
        )
        assert len(chunks) == 2

    def test_json_object(self):
        chunks = asyncio.run(
            self.parser.parse(FIXTURES / "sample.json", source_id="json_obj")
        )
        assert len(chunks) == 1
        assert chunks[0].modality == "structured"
        data = json.loads(chunks[0].text)
        assert data["dataset"] == "RAG Evaluation Benchmark"

    def test_jsonl_file(self):
        chunks = asyncio.run(
            self.parser.parse(FIXTURES / "sample.jsonl", source_id="jsonl_test")
        )
        assert len(chunks) >= 1
        assert all(c.modality == "structured" for c in chunks)

    def test_bytes_input_csv(self):
        csv_bytes = (FIXTURES / "sample.csv").read_bytes()
        chunks = asyncio.run(
            self.parser.parse(csv_bytes, source_id="bytes_csv")
        )
        assert len(chunks) >= 1

    def test_bytes_input_json(self):
        json_bytes = (FIXTURES / "sample.json").read_bytes()
        chunks = asyncio.run(
            self.parser.parse(json_bytes, source_id="bytes_json")
        )
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        chunks = asyncio.run(
            self.parser.parse(FIXTURES / "sample.csv")
        )
        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(indices)


# ── SemanticChunker ────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(
            ChunkConfig(min_tokens=10, max_tokens=50, overlap_ratio=0.2)
        )

    def test_short_text_unchanged(self):
        c = _chunk("Short text.")
        result = self.chunker.chunk([c])
        assert len(result) == 1
        assert result[0].text == "Short text."

    def test_long_text_split(self):
        # ~200 words → should exceed max_tokens=50
        long_text = " ".join(["This is a sentence that has many words."] * 20)
        c = _chunk(long_text)
        result = self.chunker.chunk([c])
        assert len(result) > 1

    def test_overlap_creates_context(self):
        long_text = " ".join([f"Sentence number {i} has some content here." for i in range(30)])
        c = _chunk(long_text)
        result = self.chunker.chunk([c])
        # Sub-chunks beyond the first should have parent metadata
        for r in result[1:]:
            assert "parent_chunk_index" in r.metadata

    def test_image_chunk_passthrough(self):
        c = ParsedChunk(
            text="image caption",
            modality="image",
            source_id="img",
            chunk_index=0,
        )
        result = self.chunker.chunk([c])
        assert len(result) == 1
        assert result[0].text == "image caption"

    def test_audio_chunk_passthrough(self):
        c = ParsedChunk(
            text="[SPEAKER_00] Hello there.",
            modality="audio",
            source_id="aud",
            chunk_index=0,
        )
        result = self.chunker.chunk([c])
        assert len(result) == 1

    def test_multiple_chunks_processed(self):
        chunks = [_chunk(f"Text block {i}. More words here.") for i in range(5)]
        result = self.chunker.chunk(chunks)
        assert len(result) >= 5

    def test_empty_input(self):
        result = self.chunker.chunk([])
        assert result == []

    def test_source_id_preserved(self):
        c = _chunk("Some text for testing.", source_id="preserve_me")
        result = self.chunker.chunk([c])
        assert all(r.source_id == "preserve_me" for r in result)

    def test_txt_file_chunking(self):
        """Integration test: read sample.txt and chunk it."""
        text = (FIXTURES / "sample.txt").read_text(encoding="utf-8")
        c = _chunk(text, source_id="sample_txt")
        chunker = SemanticChunker(ChunkConfig(min_tokens=50, max_tokens=200, overlap_ratio=0.2))
        result = chunker.chunk([c])
        assert len(result) >= 2
        for r in result:
            assert r.text
            assert r.modality == "text"


# ── MetadataExtractor ──────────────────────────────────────────────────────────

class TestMetadataExtractor:
    def setup_method(self):
        from ingestion.chunkers.metadata_extractor import MetadataExtractor
        # Disable LLM for unit tests (no API key in CI)
        self.extractor = MetadataExtractor(use_llm=False)

    def test_extract_returns_enriched_chunk(self):
        c = _chunk("Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.")
        result = asyncio.run(self.extractor.extract(c))
        assert isinstance(result, ParsedChunk)
        assert result.text == c.text  # text unchanged

    def test_ner_extracts_orgs(self):
        c = _chunk("Google and Microsoft announced a partnership in New York.")
        result = asyncio.run(self.extractor.extract(c))
        # spaCy may or may not be installed; just check metadata key exists if it is
        if hasattr(self.extractor, "_nlp") and self.extractor._nlp is not None:
            assert "ner_org" in result.metadata or "ner_gpe" in result.metadata

    def test_extract_batch(self):
        chunks = [
            _chunk("OpenAI was founded in San Francisco in 2015.", idx=i)
            for i in range(3)
        ]
        results = asyncio.run(self.extractor.extract_batch(chunks))
        assert len(results) == 3

    def test_metadata_not_overwritten(self):
        c = ParsedChunk(
            text="test text",
            modality="text",
            source_id="s",
            chunk_index=0,
            metadata={"existing_key": "existing_value"},
        )
        result = asyncio.run(self.extractor.extract(c))
        assert result.metadata.get("existing_key") == "existing_value"


# ── Modality detection ─────────────────────────────────────────────────────────

class TestModalityDetection:
    def setup_method(self):
        from ingestion.workers.ingest_task import detect_modality
        self.detect = detect_modality

    @pytest.mark.parametrize("path,expected", [
        ("document.pdf", "text"),
        ("report.docx", "text"),
        ("page.html", "text"),
        ("README.md", "text"),
        ("photo.jpg", "image"),
        ("scan.png", "image"),
        ("recording.mp3", "audio"),
        ("meeting.wav", "audio"),
        ("data.csv", "structured"),
        ("records.json", "structured"),
        ("table.parquet", "structured"),
        ("https://example.com/page", "web"),
        ("http://docs.site.org/intro", "web"),
    ])
    def test_extension_routing(self, path: str, expected: str):
        assert self.detect(path) == expected

    def test_unknown_defaults_to_text(self):
        assert self.detect("file.xyz") == "text"


# ── Integration: full text pipeline (no LLM, no heavy deps) ───────────────────

class TestTextPipeline:
    """End-to-end test using only stdlib + pydantic (no unstructured)."""

    def test_structured_pipeline(self):
        """CSV → StructuredParser → SemanticChunker → check output."""
        from ingestion.parsers.structured import StructuredParser

        parser = StructuredParser(rows_per_chunk=3)
        chunker = SemanticChunker(ChunkConfig(min_tokens=5, max_tokens=500))

        raw = asyncio.run(parser.parse(FIXTURES / "sample.csv", source_id="pipe_csv"))
        chunked = chunker.chunk(raw)

        assert len(chunked) >= 1
        for c in chunked:
            assert isinstance(c, ParsedChunk)
            assert c.source_id == "pipe_csv"
            assert c.text

    def test_json_pipeline(self):
        from ingestion.parsers.structured import StructuredParser

        parser = StructuredParser()
        raw = asyncio.run(parser.parse(FIXTURES / "sample.json", source_id="pipe_json"))
        chunker = SemanticChunker()
        chunked = chunker.chunk(raw)

        assert len(chunked) >= 1
        assert chunked[0].modality == "structured"
