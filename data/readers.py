"""Readers: turn raw files / external corpora into streams of :data:`Record`.

Every reader subclasses :class:`DatasetReader` which exposes two core methods:

* ``__iter__`` — stream records one at a time (never materialise the whole set).
* ``num_records()`` — optional; may return ``None`` if not cheaply knowable.

Adding a new format is just: subclass, implement ``_iter_records()`` yielding
``dict{"text": ..., **metadata}``, and register it in ``reader_from_config``.
No processor or writer code needs to change.

Supported now:
* :class:`JSONLReader` — newline-delimited JSON objects (stdlib only).
* :class:`TextReader` — plain UTF-8 text files, one or more; each file becomes
  one record (or, optionally, each blank-line-separated paragraph).
* :class:`ParquetReader` — Apache Parquet via optional ``pyarrow``.
* :class:`HuggingFaceReader` — HF ``datasets`` streaming via optional ``datasets``.

Parquet/HF backends are optional and guarded with ``importorskip``-style lazy
imports so the core runs on stdlib + numpy alone.
"""
from __future__ import annotations

import gzip
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from data._logging import get_logger
from data.types import Record, validate_record

log = get_logger("reader")


class DatasetReader(ABC):
    """Abstract streaming source of :data:`Record` documents."""

    source: str = "unknown"

    def __iter__(self) -> Iterator[Record]:
        count = 0
        for record in self._iter_records():
            validate_record(record)
            record.setdefault("source", self.source)
            count += 1
            yield record
        log.info("reader %s yielded %d records", type(self).__name__, count)

    @abstractmethod
    def _iter_records(self) -> Iterator[Record]:
        """Yield validated ``dict`` records with a ``text`` field."""

    def num_records(self) -> Optional[int]:
        """Best-effort length; ``None`` when not cheaply known."""
        return None

    def close(self) -> None:
        """Release any held resources. Default no-op."""
        return None


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------
class JSONLReader(DatasetReader):
    """Stream newline-delimited JSON documents from one or more files.

    Each line is a JSON object; the ``text`` field is required unless
    ``text_field`` is remapped. Supports plain and ``.gz`` files.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        source: str = "jsonl",
        text_field: str = "text",
    ) -> None:
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.source = source
        self.text_field = text_field
        for p in self.paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"JSONL input not found: {p}")

    def _iter_records(self) -> Iterator[Record]:
        for path in self.paths:
            open_fn = gzip.open if path.endswith(".gz") else open
            with open_fn(path, "rt", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        log.warning(
                            "skipping malformed JSON at %s:%d (%s)", path, line_no, exc
                        )
                        continue
                    if not isinstance(obj, dict):
                        log.warning(
                            "skipping non-object row at %s:%d", path, line_no
                        )
                        continue
                    text = obj.pop(self.text_field, None)
                    if not isinstance(text, str):
                        log.warning(
                            "skipping row with missing text at %s:%d", path, line_no
                        )
                        continue
                    record = dict(obj)  # remaining keys are metadata
                    record["text"] = text
                    record["source"] = record.get("source", self.source)
                    yield record

    def num_records(self) -> Optional[int]:
        """Count lines cheaply by streaming (non-empty, non-gzip fast path)."""
        total = 0
        for path in self.paths:
            open_fn = gzip.open if path.endswith(".gz") else open
            with open_fn(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        total += 1
        return total


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
class TextReader(DatasetReader):
    """Read one or more UTF-8 text files.

    By default each *file* is a single document (record references the
    filename in ``url``/``source``). With ``split_paragraphs=True`` each
    blank-line-separated paragraph becomes its own record.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        source: str = "text",
        split_paragraphs: bool = False,
    ) -> None:
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.source = source
        self.split_paragraphs = split_paragraphs
        for p in self.paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"text input not found: {p}")

    def _iter_records(self) -> Iterator[Record]:
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if self.split_paragraphs:
                for para in content.split("\n\n"):
                    para = para.strip()
                    if para:
                        yield {"text": para, "url": path}
            else:
                yield {"text": content.strip(), "url": path}


# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
class ParquetReader(DatasetReader):
    """Stream rows from a Parquet file using optional ``pyarrow``.

    Records use ``text_column`` as the document text; remaining columns are
    attached as metadata.
    """

    def __init__(
        self,
        path: str,
        source: str = "parquet",
        text_column: str = "text",
        columns: Optional[List[str]] = None,
    ) -> None:
        pa = _import_pyarrow()
        self._pa = pa
        if not os.path.isfile(path):
            raise FileNotFoundError(f"parquet input not found: {path}")
        self.path = path
        self.source = source
        self.text_column = text_column
        self.columns = columns
        self._table = pa.parquet.read_table(path, columns=columns)
        if text_column not in self._table.column_names:
            raise ValueError(
                f"text column {text_column!r} not in {self._table.column_names}"
            )

    def _iter_records(self) -> Iterator[Record]:
        table = self._table
        names = table.column_names
        for row in zip(*[table.column(n).to_pylist() for n in names]):
            d = dict(zip(names, row))
            text = d.get(self.text_column)
            if not isinstance(text, str):
                continue
            meta = {k: v for k, v in d.items() if k != self.text_column}
            meta["text"] = text
            meta["source"] = meta.get("source", self.source)
            yield meta

    def num_records(self) -> Optional[int]:
        return self._table.num_rows

    def close(self) -> None:
        self._table = None


class HuggingFaceReader(DatasetReader):
    """Stream records from a HuggingFace ``datasets`` dataset (optional).

    ``dataset`` is an HF dataset id or a loaded ``Dataset``/``IterableDataset``.
    Uses ``datasets`` if a string id is given (needs network for remote ids);
    callers training offline can pass an already-loaded ``Dataset`` object.
    """

    def __init__(
        self,
        dataset: Any,
        split: str = "train",
        source: Optional[str] = None,
        text_field: str = "text",
    ) -> None:
        self._ds = _require_datasets()
        if isinstance(dataset, str):
            self._data = self._ds.load_dataset(dataset, split=split)
        else:
            self._data = dataset
        self.source = source or getattr(self._data, "info", None) and getattr(
            self._data.info, "dataset_name", "hf"
        ) or "hf"
        self.text_field = text_field

    def _iter_records(self) -> Iterator[Record]:
        for row in self._data:
            rec = dict(row)
            text = rec.get(self.text_field)
            if isinstance(text, str):
                rec["text"] = text
                rec["source"] = rec.get("source", self.source)
                yield rec

    def num_records(self) -> Optional[int]:
        try:
            return len(self._data)
        except TypeError:
            return None


def _import_pyarrow():
    try:
        import pyarrow  # type: ignore
        return pyarrow
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "pyarrow is required for ParquetReader; pip install pyarrow"
        ) from exc


def _require_datasets():
    try:
        import datasets  # type: ignore
        return datasets
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "datasets is required for HuggingFaceReader; pip install datasets"
        ) from exc


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------
_READER_TYPES = {
    "jsonl": JSONLReader,
    "text": TextReader,
    "parquet": ParquetReader,
    "huggingface": HuggingFaceReader,
}


def reader_from_config(config: Dict[str, Any]) -> DatasetReader:
    """Build a reader from a declarative dict.

    Example::

        {"type": "jsonl", "path": "data/web.jsonl", "source": "web"}

    ``type`` selects the format; the remaining keys are passed to the
    constructor minus the reserved ``type`` key.
    """
    cfg = dict(config)
    rtype = cfg.pop("type", None)
    if rtype not in _READER_TYPES:
        raise ValueError(
            f"unknown reader type {rtype!r}; expected one of {sorted(_READER_TYPES)}"
        )
    cls = _READER_TYPES[rtype]
    # HuggingFaceReader accepts a dataset object, not a path — handle specially.
    if rtype == "huggingface" and "dataset" in cfg:
        target = cfg["dataset"]
        path_args = {"dataset": target}
        cfg = {k: v for k, v in cfg.items() if k != "dataset"}
        return cls(**path_args, **cfg)
    if "path" in cfg:
        path = cfg.pop("path")
        return cls(path, **cfg)
    return cls(**cfg)
