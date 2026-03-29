"""Structured data parser — CSV, JSON, JSONL, Excel via Pandas/Polars."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .base import BaseParser, ParsedChunk

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

try:
    import polars as pl
    _POLARS_OK = True
except ImportError:
    _POLARS_OK = False

_SUPPORTED = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet"}

# Max rows per chunk for tabular data
_ROWS_PER_CHUNK = 50


class StructuredParser(BaseParser):
    """Parse tabular and semi-structured files into text-serialised chunks.

    Strategy:
    - CSV / Excel / Parquet → chunked by row groups, serialised as Markdown table.
    - JSON (object)        → formatted string.
    - JSON (array)         → chunked by items.
    - JSONL / NDJSON       → one chunk per line-batch.
    """

    def __init__(self, rows_per_chunk: int = _ROWS_PER_CHUNK) -> None:
        if not (_PANDAS_OK or _POLARS_OK):
            raise ImportError(
                "pandas or polars required: pip install pandas polars"
            )
        self.rows_per_chunk = rows_per_chunk

    async def parse(
        self,
        source: str | Path | bytes,
        source_id: str | None = None,
    ) -> list[ParsedChunk]:
        sid = self._make_source_id(source, source_id)
        return await asyncio.to_thread(self._parse_sync, source, sid)

    # ------------------------------------------------------------------ internals

    def _parse_sync(
        self, source: str | Path | bytes, source_id: str
    ) -> list[ParsedChunk]:
        if isinstance(source, bytes):
            return self._parse_bytes(source, source_id)

        path = Path(str(source))
        ext = path.suffix.lower()

        if ext in (".json", ".jsonl", ".ndjson"):
            return self._parse_json_file(path, source_id)
        if ext in (".csv", ".tsv"):
            return self._parse_tabular(path, source_id, sep="\t" if ext == ".tsv" else ",")
        if ext in (".xlsx", ".xls"):
            return self._parse_tabular(path, source_id)
        if ext == ".parquet":
            return self._parse_tabular(path, source_id)

        # Fallback: try JSON then CSV
        try:
            return self._parse_json_file(path, source_id)
        except Exception:
            return self._parse_tabular(path, source_id)

    def _parse_bytes(self, data: bytes, source_id: str) -> list[ParsedChunk]:
        """Try JSON then CSV for raw bytes."""
        text = data.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text)
            return self._json_to_chunks(obj, source_id)
        except json.JSONDecodeError:
            import io
            if _PANDAS_OK:
                df = pd.read_csv(io.StringIO(text))
                return self._df_to_chunks(df, source_id)
            raise

    def _parse_json_file(self, path: Path, source_id: str) -> list[ParsedChunk]:
        ext = path.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            return self._parse_jsonl(path, source_id)

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return self._json_to_chunks(obj, source_id)

    def _parse_jsonl(self, path: Path, source_id: str) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        batch: list[Any] = []
        idx = 0

        def flush():
            nonlocal idx
            if not batch:
                return
            text = json.dumps(batch, ensure_ascii=False, indent=2)
            chunks.append(
                ParsedChunk(
                    text=text,
                    metadata={"source_id": source_id, "batch_size": len(batch)},
                    modality="structured",
                    source_id=source_id,
                    chunk_index=idx,
                )
            )
            idx += 1
            batch.clear()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(batch) >= self.rows_per_chunk:
                    flush()
        flush()
        return chunks

    def _json_to_chunks(self, obj: Any, source_id: str) -> list[ParsedChunk]:
        if isinstance(obj, list):
            chunks: list[ParsedChunk] = []
            for i in range(0, len(obj), self.rows_per_chunk):
                batch = obj[i : i + self.rows_per_chunk]
                text = json.dumps(batch, ensure_ascii=False, indent=2)
                chunks.append(
                    ParsedChunk(
                        text=text,
                        metadata={
                            "source_id": source_id,
                            "total_items": len(obj),
                            "batch_start": i,
                        },
                        modality="structured",
                        source_id=source_id,
                        chunk_index=i // self.rows_per_chunk,
                    )
                )
            return chunks
        # Plain object — single chunk
        return [
            ParsedChunk(
                text=json.dumps(obj, ensure_ascii=False, indent=2),
                metadata={"source_id": source_id, "keys": list(obj.keys()) if isinstance(obj, dict) else []},
                modality="structured",
                source_id=source_id,
                chunk_index=0,
            )
        ]

    def _parse_tabular(
        self, path: Path, source_id: str, sep: str = ","
    ) -> list[ParsedChunk]:
        if _POLARS_OK and path.suffix.lower() in (".csv", ".tsv", ".parquet"):
            try:
                if path.suffix.lower() == ".parquet":
                    df_pl = pl.read_parquet(path)
                else:
                    df_pl = pl.read_csv(path, separator=sep, infer_schema_length=1000)
                return self._polars_to_chunks(df_pl, source_id)
            except Exception:
                pass  # fallback to pandas

        if _PANDAS_OK:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, sep=sep)
            return self._df_to_chunks(df, source_id)

        raise ImportError("pandas or polars required")

    def _df_to_chunks(self, df: "pd.DataFrame", source_id: str) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

        for i in range(0, len(df), self.rows_per_chunk):
            batch = df.iloc[i : i + self.rows_per_chunk]
            text = batch.to_markdown(index=False)
            chunks.append(
                ParsedChunk(
                    text=text,
                    metadata={
                        "source_id": source_id,
                        "schema": schema,
                        "total_rows": len(df),
                        "row_start": i,
                        "row_end": min(i + self.rows_per_chunk, len(df)),
                        "columns": list(df.columns),
                    },
                    modality="structured",
                    source_id=source_id,
                    chunk_index=i // self.rows_per_chunk,
                )
            )
        return chunks

    def _polars_to_chunks(self, df: "pl.DataFrame", source_id: str) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        schema = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}

        for i in range(0, len(df), self.rows_per_chunk):
            batch = df.slice(i, self.rows_per_chunk)
            text = batch.to_pandas().to_markdown(index=False)
            chunks.append(
                ParsedChunk(
                    text=text,
                    metadata={
                        "source_id": source_id,
                        "schema": schema,
                        "total_rows": len(df),
                        "row_start": i,
                        "row_end": min(i + self.rows_per_chunk, len(df)),
                        "columns": df.columns,
                    },
                    modality="structured",
                    source_id=source_id,
                    chunk_index=i // self.rows_per_chunk,
                )
            )
        return chunks
