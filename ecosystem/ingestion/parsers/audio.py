"""Audio parser — transcription with faster-whisper + optional speaker diarization."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from .base import BaseParser, ParsedChunk

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False

try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    _PYANNOTE_OK = True
except ImportError:
    _PYANNOTE_OK = False

_SUPPORTED = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}

# Segment shorter than this (seconds) are merged with the next
_MIN_SEGMENT_SECONDS = 1.5


class AudioParser(BaseParser):
    """Transcribe audio files and optionally diarize speakers.

    Each logical segment (or merged group) becomes one ParsedChunk.
    With diarization, segments are labelled SPEAKER_00, SPEAKER_01, etc.
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        diarize: bool = False,
    ) -> None:
        if not _WHISPER_OK:
            raise ImportError(
                "faster-whisper is required: pip install faster-whisper"
            )
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        self.device = device
        self.compute_type = compute_type
        self.diarize = diarize and _PYANNOTE_OK
        if diarize and not _PYANNOTE_OK:
            logger.warning("pyannote.audio not installed — diarization disabled")

        self._whisper: "WhisperModel | None" = None
        self._diarizer: "DiarizationPipeline | None" = None

    async def parse(
        self,
        source: str | Path | bytes,
        source_id: str | None = None,
    ) -> list[ParsedChunk]:
        sid = self._make_source_id(source, source_id)
        return await asyncio.to_thread(self._parse_sync, source, sid)

    # ------------------------------------------------------------------ internals

    def _get_whisper(self) -> "WhisperModel":
        if self._whisper is None:
            logger.info("Loading Whisper model: %s", self.model_size)
            self._whisper = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._whisper

    def _get_diarizer(self) -> "DiarizationPipeline | None":
        if not self.diarize:
            return None
        if self._diarizer is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning("HUGGINGFACE_TOKEN not set — diarization disabled")
                self.diarize = False
                return None
            logger.info("Loading pyannote diarization pipeline")
            self._diarizer = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        return self._diarizer

    def _parse_sync(
        self, source: str | Path | bytes, source_id: str
    ) -> list[ParsedChunk]:
        tmp_path: str | None = None

        try:
            if isinstance(source, bytes):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as f:
                    f.write(source)
                    tmp_path = f.name
                audio_path = tmp_path
            else:
                audio_path = str(source)

            model = self._get_whisper()
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True,
            )
            raw_segments = list(segments)  # consume generator

            # Optional diarization
            speaker_map: dict[tuple[float, float], str] = {}
            diarizer = self._get_diarizer()
            if diarizer is not None:
                try:
                    diarization = diarizer(audio_path)
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        speaker_map[(round(turn.start, 2), round(turn.end, 2))] = speaker
                except Exception as exc:
                    logger.warning("Diarization failed: %s", exc)

            chunks = self._build_chunks(raw_segments, speaker_map, info, source_id)

        finally:
            if tmp_path:
                import os as _os
                _os.unlink(tmp_path)

        logger.info(
            "AudioParser: %d chunks, lang=%s, dur=%.1fs from %s",
            len(chunks),
            info.language,
            info.duration,
            source_id,
        )
        return chunks

    def _build_chunks(
        self,
        segments: list,
        speaker_map: dict[tuple[float, float], str],
        info,
        source_id: str,
    ) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        idx = 0

        # Merge very short segments
        merged: list[dict] = []
        for seg in segments:
            if (
                merged
                and (seg.end - seg.start) < _MIN_SEGMENT_SECONDS
                and merged[-1].get("speaker") == self._lookup_speaker(speaker_map, seg)
            ):
                merged[-1]["text"] += " " + seg.text.strip()
                merged[-1]["end"] = seg.end
            else:
                merged.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "speaker": self._lookup_speaker(speaker_map, seg),
                    }
                )

        for seg in merged:
            text = seg["text"]
            if not text:
                continue
            speaker = seg["speaker"]
            if speaker:
                text = f"[{speaker}] {text}"

            chunks.append(
                ParsedChunk(
                    text=text,
                    metadata={
                        "source_id": source_id,
                        "start_seconds": round(seg["start"], 2),
                        "end_seconds": round(seg["end"], 2),
                        "language": info.language,
                        "language_probability": round(info.language_probability, 3),
                        "speaker": speaker,
                        "duration_seconds": round(info.duration, 1),
                    },
                    modality="audio",
                    source_id=source_id,
                    chunk_index=idx,
                )
            )
            idx += 1

        return chunks

    @staticmethod
    def _lookup_speaker(
        speaker_map: dict[tuple[float, float], str], seg
    ) -> str:
        """Find the dominant speaker for a segment via overlap heuristic."""
        if not speaker_map:
            return ""
        for (start, end), speaker in speaker_map.items():
            if start <= seg.start and end >= seg.end:
                return speaker
        return ""
