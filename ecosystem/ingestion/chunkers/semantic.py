"""Semantic chunker — sentence-aware splitting with token budget and 20% overlap."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Generator

from ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        # Fallback: 1 token ≈ 4 chars
        return max(1, len(text) // 4)


@dataclass
class ChunkConfig:
    min_tokens: int = 256
    max_tokens: int = 1500
    overlap_ratio: float = 0.20  # 20 % of max_tokens

    @property
    def overlap_tokens(self) -> int:
        return int(self.max_tokens * self.overlap_ratio)


# Sentence boundary: end with . ! ? followed by whitespace or end-of-string.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Split ParsedChunk.text into token-bounded sub-chunks with overlap.

    For non-text modalities (image, audio) the chunk is returned as-is,
    since they are already semantically atomic.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.cfg = config or ChunkConfig()

    def chunk(self, parsed: list[ParsedChunk]) -> list[ParsedChunk]:
        """Re-chunk a list of ParsedChunk objects respecting token limits."""
        result: list[ParsedChunk] = []
        for pc in parsed:
            if pc.modality in ("image", "audio"):
                result.append(pc)
                continue
            sub = list(self._split_chunk(pc))
            result.extend(sub)
        return result

    # ------------------------------------------------------------------ core

    def _split_chunk(self, pc: ParsedChunk) -> Generator[ParsedChunk, None, None]:
        tokens = _count_tokens(pc.text)

        if tokens <= self.cfg.max_tokens:
            yield pc
            return

        sentences = _SENTENCE_END.split(pc.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        windows = list(self._build_windows(sentences))
        base_idx = pc.chunk_index * 1000  # keep original chunk ordering

        for sub_i, window_text in enumerate(windows):
            yield ParsedChunk(
                text=window_text,
                metadata={
                    **pc.metadata,
                    "parent_chunk_index": pc.chunk_index,
                    "token_count": _count_tokens(window_text),
                    "sub_chunk_index": sub_i,
                    "total_sub_chunks": len(windows),
                },
                modality=pc.modality,
                source_id=pc.source_id,
                chunk_index=base_idx + sub_i,
            )

    def _build_windows(self, sentences: list[str]) -> Generator[str, None, None]:
        """Greedily fill windows, then rewind by overlap_tokens."""
        buf: list[str] = []
        buf_tokens = 0
        overlap_buf: list[str] = []

        for sent in sentences:
            sent_tokens = _count_tokens(sent)

            # Single sentence exceeds max — hard-split it
            if sent_tokens > self.cfg.max_tokens:
                if buf:
                    yield " ".join(buf)
                    overlap_buf = self._tail_sentences(buf, self.cfg.overlap_tokens)
                    buf, buf_tokens = list(overlap_buf), sum(_count_tokens(s) for s in overlap_buf)
                for piece in self._hard_split(sent):
                    yield piece
                continue

            if buf_tokens + sent_tokens > self.cfg.max_tokens:
                if buf:
                    text = " ".join(buf)
                    if _count_tokens(text) >= self.cfg.min_tokens:
                        yield text
                    overlap_buf = self._tail_sentences(buf, self.cfg.overlap_tokens)
                    buf = list(overlap_buf)
                    buf_tokens = sum(_count_tokens(s) for s in buf)

            buf.append(sent)
            buf_tokens += sent_tokens

        if buf:
            text = " ".join(buf)
            if _count_tokens(text) >= self.cfg.min_tokens:
                yield text
            elif text:  # below min_tokens but non-empty: yield anyway
                yield text

    @staticmethod
    def _tail_sentences(sentences: list[str], target_tokens: int) -> list[str]:
        """Return suffix of sentences totalling ≤ target_tokens."""
        tail: list[str] = []
        acc = 0
        for sent in reversed(sentences):
            t = _count_tokens(sent)
            if acc + t > target_tokens:
                break
            tail.insert(0, sent)
            acc += t
        return tail

    def _hard_split(self, text: str) -> Generator[str, None, None]:
        """Split an oversized sentence by word boundaries."""
        words = text.split()
        buf: list[str] = []
        buf_tokens = 0
        for word in words:
            w_tok = _count_tokens(word)
            if buf_tokens + w_tok > self.cfg.max_tokens:
                if buf:
                    yield " ".join(buf)
                buf = [word]
                buf_tokens = w_tok
            else:
                buf.append(word)
                buf_tokens += w_tok
        if buf:
            yield " ".join(buf)
