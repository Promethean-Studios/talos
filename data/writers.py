"""Sharded writers: write processed records out as on-disk shards + manifest.

:class:`ShardedWriter` buffers records and flushes them to numbered shard files
as soon as a size limit is crossed, then writes a JSON ``manifest.json`` that
records each shard's file name, record count and per-``source`` distribution.

Atomicity: each shard is written to a ``*.tmp`` temp file in the same
directory and ``os.replace``-renamed to its final name only after a successful
close of the file handle, so an interrupted run never leaves a half-written
shard under its real name.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from data._logging import get_logger
from data.types import Record

log = get_logger("writer")


class ShardedWriter:
    """Write records to sharded JSONL (and optionally Parquet) + manifest.

    Args:
        output_dir: directory to write shards + ``manifest.json`` into.
        shard_size: max *records* per shard if ``size_by=="records"`` (default)
            or max *bytes* if ``size_by=="bytes"``.
        format: ``"jsonl"`` (default) or ``"parquet"`` (requires pyarrow).
        prefix: files are named ``{prefix}-{index:04d}.jsonl``.
        write_manifest: whether to (re)write ``manifest.json`` on close.
    """

    def __init__(
        self,
        output_dir: str,
        shard_size: int = 100_000,
        size_by: str = "records",
        format: str = "jsonl",
        prefix: str = "shard",
        write_manifest: bool = True,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if size_by not in ("records", "bytes"):
            raise ValueError("size_by must be 'records' or 'bytes'")
        if format not in ("jsonl", "parquet"):
            raise ValueError("format must be 'jsonl' or 'parquet'")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.size_by = size_by
        self.format = format
        self.prefix = prefix
        self.write_manifest = write_manifest
        os.makedirs(output_dir, exist_ok=True)
        self._index = 0
        self._current_records: List[Record] = []
        self._current_bytes = 0
        self._manifest: List[Dict[str, Any]] = []
        self._closed = False

    # -- public API ---------------------------------------------------------
    def write(self, record: Record) -> None:
        """Append one record to the current shard buffer."""
        if self._closed:
            raise RuntimeError("writer already closed")
        self._current_records.append(record)
        if self.format == "jsonl":
            self._current_bytes += len(
                json.dumps(record, ensure_ascii=False).encode("utf-8")
            ) + 1  # newline
        else:
            self._current_bytes += 0
        limit = (
            len(self._current_records)
            if self.size_by == "records"
            else self._current_bytes
        )
        if limit >= self.shard_size:
            self._flush()

    def close(self) -> List[Dict[str, Any]]:
        """Flush any pending shard, write the manifest, and return it."""
        if self._closed:
            return self._manifest
        self._flush()
        self._closed = True
        if self.write_manifest:
            self._write_manifest()
        log.info(
            "wrote %d shards (%d records) to %s",
            len(self._manifest),
            sum(s["count"] for s in self._manifest),
            self.output_dir,
        )
        return self._manifest

    def __enter__(self) -> "ShardedWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internals ----------------------------------------------------------
    def _flush(self) -> None:
        if not self._current_records:
            return
        shard_idx = self._index
        self._index += 1
        if self.format == "jsonl":
            final = os.path.join(
                self.output_dir, f"{self.prefix}-{shard_idx:04d}.jsonl"
            )
            tmp = final + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for rec in self._current_records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, final)  # atomic rename
        else:  # parquet
            self._write_parquet(shard_idx, self._current_records)
        sources: Dict[str, int] = {}
        for rec in self._current_records:
            src = rec.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        self._manifest.append(
            {
                "shard": f"{self.prefix}-{shard_idx:04d}.{self.format}",
                "index": shard_idx,
                "count": len(self._current_records),
                "sources": sources,
            }
        )
        self._current_records = []
        self._current_bytes = 0

    def _write_parquet(self, shard_idx: int, records: List[Record]) -> None:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "parquet output requires 'pip install pyarrow'"
            ) from exc
        if not records:
            return
        fields = {k: [] for k in records[0]}
        for rec in records:
            for k in fields:
                fields[k].append(rec.get(k))
        table = pa.table({k: pa.array(v) for k, v in fields.items()})
        final = os.path.join(self.output_dir, f"{self.prefix}-{shard_idx:04d}.parquet")
        tmp = final + ".tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, final)

    def _write_manifest(self) -> None:
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        tmp = manifest_path + ".tmp"
        payload = {
            "format": self.format,
            "size_by": self.size_by,
            "shard_size": self.shard_size,
            "num_shards": len(self._manifest),
            "total_records": sum(s["count"] for s in self._manifest),
            "shards": self._manifest,
        }
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, manifest_path)
