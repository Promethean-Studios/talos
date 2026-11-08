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
* :class:`ParquetStreamReader` / :class:`HuggingFaceParquetReader` — the
  **bounded** remote-parquet path (HTTP range reads, no ``datasets``, no
  read-ahead/pre-buffer cache). Used by ``scripts/prepare_corpus.py``, whose
  memory budget is the reason it exists; see :class:`HttpRangeFile`.

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
            if self.split_paragraphs:
                # Streaming paragraph mode: buffer only the *current* paragraph
                # and yield it as soon as its terminating blank line is seen.
                # Never fh.read() + split("\n\n") + strip() the whole file —
                # that materialised 3 full copies plus a list of every
                # paragraph at once (the classic multi-GB OOM on large .txt).
                buf: List[str] = []
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            if buf:
                                yield {"text": "".join(buf).strip(), "url": path}
                                buf.clear()
                        else:
                            buf.append(line)
                if buf:
                    yield {"text": "".join(buf).strip(), "url": path}
            else:
                # One record per file (documented semantics): the text is
                # inherently O(file), so read once and strip once — no split
                # list, no extra full copies.
                with open(path, "r", encoding="utf-8") as fh:
                    content = fh.read()
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
        batch_size: int = 10_000,
    ) -> None:
        pa = _import_pyarrow()
        self._pa = pa
        if not os.path.isfile(path):
            raise FileNotFoundError(f"parquet input not found: {path}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if columns is not None and text_column not in columns:
            raise ValueError(
                f"text column {text_column!r} must be included in columns"
            )
        self.path = path
        self.source = source
        self.text_column = text_column
        self.columns = columns
        self.batch_size = batch_size
        # Open the file metadata only and stream record batches; never
        # read_table() the whole file into RAM (multi-GB parquet would OOM).
        self._pf = pa.parquet.ParquetFile(path)
        schema_names = list(self._pf.schema_arrow.names)
        if text_column not in schema_names:
            raise ValueError(f"text column {text_column!r} not in {schema_names}")

    def _iter_records(self) -> Iterator[Record]:
        names: Optional[List[str]] = None
        for batch in self._pf.iter_batches(
            batch_size=self.batch_size, columns=self.columns
        ):
            if names is None:
                names = list(batch.schema.names)
            # One batch at a time: columnar -> python lists bounded by
            # batch_size rows, then discarded before the next batch.
            cols = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
            for row in zip(*cols):
                d = dict(zip(names, row))
                text = d.get(self.text_column)
                if not isinstance(text, str):
                    continue
                meta = {k: v for k, v in d.items() if k != self.text_column}
                meta["text"] = text
                meta["source"] = meta.get("source", self.source)
                yield meta

    def num_records(self) -> Optional[int]:
        return self._pf.metadata.num_rows

    def close(self) -> None:
        self._pf = None


class HuggingFaceReader(DatasetReader):
    """Stream records from a HuggingFace ``datasets`` dataset (optional).

    ``dataset`` is an HF dataset id or a loaded ``Dataset``/``IterableDataset``.
    Uses ``datasets`` if a string id is given (needs network for remote ids);
    callers training offline can pass an already-loaded ``Dataset`` object.

    ``streaming=True`` (default) loads string ids via ``load_dataset(...,
    streaming=True)`` so rows are fetched lazily instead of downloading and
    materialising the entire split in RAM/scratch disk first. Pass
    ``streaming=False`` to opt out (e.g. when shuffling requires full material).

    ``config`` selects a named dataset configuration (e.g. ``"sample-10BT"``
    for ``HuggingFaceFW/fineweb-edu``); ``revision`` pins a git revision of
    the dataset. Both are forwarded to ``load_dataset`` only for string ids —
    ``config`` as its ``name=`` parameter (``load_dataset`` has no ``config``
    kwarg: passing one leaks into the builder config and raises
    ``BuilderConfig ... doesn't have a 'config' key``), ``revision`` as
    ``revision=``.
    """

    def __init__(
        self,
        dataset: Any,
        split: str = "train",
        source: Optional[str] = None,
        text_field: str = "text",
        streaming: bool = True,
        config: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        self._ds = _require_datasets()
        if isinstance(dataset, str):
            kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
            if config is not None:
                # ``load_dataset`` names the dataset configuration ``name``;
                # a ``config=`` kwarg is not in its signature and gets folded
                # into the builder config, which has no ``config`` attribute
                # (``BuilderConfig ... doesn't have a 'config' key``).
                kwargs["name"] = config
            if revision is not None:
                kwargs["revision"] = revision
            self._data = self._ds.load_dataset(dataset, **kwargs)
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
        import pyarrow.parquet  # noqa: F401  (ParquetFile lives in the submodule; the
        # bare ``import pyarrow`` above does not load it, and nothing else in the
        # bounded path imports ``datasets`` — which is what previously loaded it.)
        return pyarrow
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "pyarrow is required for ParquetReader; pip install pyarrow"
        ) from exc


# ---------------------------------------------------------------------------
# Bounded remote parquet (HF Hub) — no ``datasets``, no read-ahead cache
# ---------------------------------------------------------------------------
#: Rows converted to Python strings per batch in the bounded read path. This is
#: the unit of transient memory: one batch's text is materialised, handed to the
#: consumer, and dropped before the next batch is fetched.
DEFAULT_PARQUET_BATCH_SIZE = 1_000
#: Per-request timeout (seconds) for range reads.
DEFAULT_HTTP_TIMEOUT = 120.0
#: Attempts per range request (the Hub rate-limits anonymous reads with 429s).
DEFAULT_HTTP_RETRIES = 5
#: Base backoff (seconds) between range-request retries; doubles per attempt.
DEFAULT_HTTP_BACKOFF = 1.0


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


class HttpRangeFile:
    """Read-only, seekable file object backed by HTTP range requests.

    Why this exists: a parquet reader needs random access to a remote file, and
    every convenience layer that provides it (``fsspec``'s ``readahead`` cache,
    ``datasets``' streaming wrapper, ``requests``' ``Response.content``) keeps
    fetched bytes alive for longer than the reader needs them. On a 4 GB box
    packing a 220 M-token corpus that is the difference between finishing and
    being OOM-killed.

    The contract here is deliberately dumb and bounded:

    * ``read(n)`` fetches exactly ``n`` bytes with one ``Range: bytes=a-b``
      request, hands them back, and keeps **no** copy: the response body is
      streamed off the socket (``stream=True``, so ``Response.content`` is never
      populated) and the connection is closed in a ``finally``.
    * nothing is cached between reads — the packer consumes each source file
      strictly forward, so there is no re-read for a cache to serve.
    * the only state that grows with progress is a counter (``requests_made``).

    Peak resident memory is therefore one range request's payload (one parquet
    column chunk, ~3 MB for the FineWeb-Edu layout) plus whatever the parquet
    reader holds for the row group it is decoding.

    Args:
        url: absolute ``http(s)`` URL of the file.
        size: known content length; when ``None`` a ``HEAD`` (falling back to a
            1-byte ranged ``GET``) discovers it.
        timeout: per-request timeout in seconds.
        retries: attempts per read (retries 429/5xx and connection errors with
            exponential backoff, honouring ``Retry-After``).
        session: optional pre-configured ``requests.Session`` (reused across
            files to keep the TLS connection pool warm).
    """

    def __init__(
        self,
        url: str,
        *,
        size: Optional[int] = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retries: int = DEFAULT_HTTP_RETRIES,
        backoff: float = DEFAULT_HTTP_BACKOFF,
        session: Any = None,
    ) -> None:
        import requests

        self.url = url
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self.backoff = float(backoff)
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self.pos = 0
        self.closed = False
        #: Observability: exactly what the read path fetched (no hidden copies).
        self.requests_made = 0
        self.bytes_fetched = 0
        self.max_request_bytes = 0
        self.size = int(size) if size is not None else self._discover_size()

    # -- size discovery -----------------------------------------------------
    def _discover_size(self) -> int:
        head = self._session.head(self.url, allow_redirects=True, timeout=self.timeout)
        if head.status_code == 200 and head.headers.get("Content-Length"):
            size = int(head.headers["Content-Length"])
            head.close()
            return size
        head.close()
        # Some CDN responses omit Content-Length on HEAD: ask for one byte.
        resp = self._session.get(
            self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout, stream=True
        )
        try:
            content_range = resp.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise ValueError(
                    f"cannot determine the size of {self.url!r} "
                    f"(HTTP {resp.status_code}, Content-Range {content_range!r})"
                )
            return int(content_range.rsplit("/", 1)[1])
        finally:
            resp.close()

    # -- file-object protocol (pyarrow's PythonFile needs all of these) -----
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = int(offset)
        elif whence == 1:
            self.pos += int(offset)
        elif whence == 2:
            self.pos = self.size + int(offset)
        else:  # pragma: no cover - defensive
            raise ValueError(f"invalid whence {whence}")
        if self.pos < 0:
            raise ValueError("negative seek position")
        return self.pos

    def read(self, n: int = -1) -> bytes:
        """Fetch and return exactly the next ``n`` bytes (no caching)."""
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if n is None or n < 0:
            n = self.size - self.pos
        n = int(n)
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = self._range_get(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:  # pragma: no cover - pyarrow uses read()
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def readall(self) -> bytes:  # pragma: no cover - never used by the packer
        return self.read(-1)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._owns_session:
            self._session.close()

    # -- internals ----------------------------------------------------------
    def _range_get(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}"}
        want = end - start + 1
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries):
            resp = None
            try:
                resp = self._session.get(
                    self.url, headers=headers, timeout=self.timeout, stream=True
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableStatus(resp.status_code, resp.headers.get("Retry-After"))
                if resp.status_code not in (200, 206):
                    raise IOError(
                        f"range request for {self.url!r} bytes {start}-{end} "
                        f"returned HTTP {resp.status_code}"
                    )
                # Read the body straight off the socket: ``Response.content`` is
                # never populated, so nothing is retained on the response object.
                chunks: List[bytes] = []
                got = 0
                while got < want:
                    chunk = resp.raw.read(want - got, decode_content=False)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    got += len(chunk)
                data = b"".join(chunks)
                if not data:
                    raise IOError(
                        f"empty range response for {self.url!r} bytes {start}-{end}"
                    )
                self.requests_made += 1
                self.bytes_fetched += len(data)
                self.max_request_bytes = max(self.max_request_bytes, len(data))
                return data
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                if attempt + 1 >= self.retries:
                    break
                time.sleep(self._retry_delay(attempt, exc))
            finally:
                if resp is not None:
                    resp.close()
                    del resp
        raise IOError(
            f"range request for {self.url!r} bytes {start}-{end} failed after "
            f"{self.retries} attempts: {last_error}"
        ) from last_error

    def _retry_delay(self, attempt: int, exc: BaseException) -> float:
        if isinstance(exc, _RetryableStatus) and exc.retry_after is not None:
            try:
                return min(60.0, max(0.0, float(exc.retry_after)))
            except (TypeError, ValueError):
                pass
        return min(60.0, self.backoff * (2 ** attempt))


class _RetryableStatus(Exception):
    """HTTP 429/5xx — worth retrying after a backoff."""

    def __init__(self, status: int, retry_after: Optional[str] = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


class ParquetStreamReader(DatasetReader):
    """Stream one text column from parquet files (local paths or URLs), bounded.

    This is the read path the corpus packer uses. It replaces
    ``HuggingFaceReader``'s ``datasets`` streaming wrapper with three
    deliberately boring pieces:

    1. one file at a time — a local path is opened by pyarrow directly, a remote
       URL through :class:`HttpRangeFile` (range requests, no cache, nothing
       retained between reads);
    2. ``pyarrow.parquet.ParquetFile(..., pre_buffer=False)`` — pyarrow's
       ``pre_buffer`` default is ``True``, which enables Arrow's
       ``ReadRangeCache``: it retains **every** range it reads for the life of
       the reader, so resident memory grows with corpus progress (measured:
       +0.6 bytes per streamed character, i.e. ~1 GB by 1 G characters). With
       ``pre_buffer=False`` each row group is fetched, decoded, and released;
    3. ``iter_batches(batch_size=..., columns=[text_field])`` — only the text
       column is decoded, and at most one batch's strings exist at a time.

    Records are ``{"text": ..., "source": ...}``, the same contract
    ``HuggingFaceReader`` yields, so ``prepare_corpus`` is reader-agnostic.

    Args:
        paths: parquet file paths or ``http(s)`` URLs, consumed **in order**.
        text_field: column holding the document text (default ``"text"``).
        batch_size: rows per batch — the transient-memory unit.
        source: value of each record's ``source`` field.
        pre_buffer: pyarrow's range-prefetch cache; ``False`` (default) is what
            keeps memory bounded. Exposed so the difference can be measured.
    """

    def __init__(
        self,
        paths: Union[str, Iterable[str]],
        *,
        text_field: str = "text",
        batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
        source: str = "hf-parquet",
        pre_buffer: bool = False,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retries: int = DEFAULT_HTTP_RETRIES,
    ) -> None:
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        if not self.paths:
            raise ValueError("ParquetStreamReader needs at least one path")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.text_field = text_field
        self.batch_size = int(batch_size)
        self.source = source
        self.pre_buffer = bool(pre_buffer)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._pa = _import_pyarrow()
        self._session: Any = None
        self._closed = False
        #: Files fully consumed so far (observability for the packer's log).
        self.files_done = 0
        #: Bytes fetched over the network by this reader (0 for local files).
        self.bytes_fetched = 0

    # -- public API ---------------------------------------------------------
    def _iter_records(self) -> Iterator[Record]:
        for index, path in enumerate(self.paths):
            handle = None
            pf = None
            try:
                if _is_url(path):
                    if self._session is None:
                        import requests

                        self._session = requests.Session()
                    handle = HttpRangeFile(
                        path,
                        timeout=self.timeout,
                        retries=self.retries,
                        session=self._session,
                    )
                    pf = self._pa.parquet.ParquetFile(handle, pre_buffer=self.pre_buffer)
                else:
                    pf = self._pa.parquet.ParquetFile(path, pre_buffer=self.pre_buffer)
                log.info(
                    "parquet source %d/%d %s: %d rows, %d row groups",
                    index + 1, len(self.paths), os.path.basename(path),
                    pf.metadata.num_rows, pf.metadata.num_row_groups,
                )
                yield from self._iter_file_records(pf)
                self.files_done += 1
            finally:
                # Drop the reader before the next file: its metadata, decoded
                # row group and (for URLs) the range-reader state all go away.
                pf = None
                if handle is not None:
                    self.bytes_fetched += getattr(handle, "bytes_fetched", 0)
                    handle.close()
                    handle = None

    def _iter_file_records(self, pf: Any) -> Iterator[Record]:
        # ``use_threads=False``: Arrow's threaded row-group reader allocates each
        # row group's buffers on a worker thread, so the freed pages land in that
        # thread's allocator arena instead of the main one and are not returned
        # (measured on this box: RSS grew ~0.6 bytes per streamed character with
        # threads on, flat with them off — the corpus packer is I/O bound anyway,
        # so the thread pool bought nothing).
        for batch in pf.iter_batches(
            batch_size=self.batch_size,
            columns=[self.text_field],
            use_threads=False,
        ):
            texts = batch.column(0).to_pylist()
            del batch
            for text in texts:
                if isinstance(text, str):
                    yield {"text": text, "source": self.source}
            del texts

    def num_records(self) -> Optional[int]:
        """Total rows across all files, or ``None`` if the metadata is unreadable."""
        total = 0
        for path in self.paths:
            try:
                if _is_url(path):
                    return None  # would cost a network round-trip per file
                total += self._pa.parquet.ParquetFile(
                    path, pre_buffer=self.pre_buffer
                ).metadata.num_rows
            except Exception:  # noqa: BLE001 - length is optional
                return None
        return total

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            self._session.close()
            self._session = None


def resolve_hf_parquet_files(
    dataset: str,
    *,
    config: Optional[str] = None,
    split: str = "train",
    revision: Optional[str] = None,
) -> List[str]:
    """Resolve an HF dataset split to its parquet files, in the order ``datasets`` uses.

    Two sources, tried in order:

    1. the dataset repository's own ``*.parquet`` files (what ``datasets`` globs
       for a parquet-backed dataset — FineWeb-Edu stores ``sample/10BT/*.parquet``
       natively). Files are filtered by configuration and split when the paths
       let us, then sorted — the same order ``datasets`` builds its file list in;
    2. the Hub's parquet-conversion endpoint
       (``/api/datasets/<id>/parquet/<config>/<split>``) for datasets that are
       not stored as parquet (csv/json/...), which returns the auto-converted
       files in split order.

    Raises ``ValueError`` when neither yields a file, naming the dataset/config/
    split so a typo fails loudly instead of packing an empty corpus.
    """
    files: List[str] = []
    try:
        from huggingface_hub import HfApi

        paths = HfApi().list_repo_files(
            dataset, repo_type="dataset", revision=revision
        )
        files = _select_parquet_paths(paths, config=config, split=split)
        if files:
            rev = revision or "main"
            return [
                f"https://huggingface.co/datasets/{dataset}/resolve/{rev}/{p}"
                for p in files
            ]
    except Exception as exc:  # noqa: BLE001 - fall through to the API endpoint
        log.debug("repo listing for %s failed (%s); trying the parquet API", dataset, exc)

    try:
        import requests

        url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config or 'default'}/{split}"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, dict):
                payload = payload.get("parquet_files", [])
            for entry in payload:
                if isinstance(entry, str):
                    files.append(entry)
                elif isinstance(entry, dict) and entry.get("url"):
                    files.append(entry["url"])
        resp.close()
    except Exception as exc:  # noqa: BLE001 - reported below
        log.debug("parquet API for %s failed: %s", dataset, exc)

    if not files:
        raise ValueError(
            f"no parquet files found for dataset {dataset!r} "
            f"(config={config!r}, split={split!r}) — check the dataset id/config/"
            "split, or pass explicit files"
        )
    return files


def _select_parquet_paths(
    paths: Iterable[str], *, config: Optional[str], split: str
) -> List[str]:
    """Pick the parquet files of one (config, split), sorted like ``datasets`` does."""
    parquet = sorted(p for p in paths if p.endswith(".parquet"))
    if config:
        # ``sample-10BT`` lives at ``sample/10BT/...`` on the Hub, so try the
        # literal config first and then its slash-separated form.
        for needle in (config, config.replace("-", "/")):
            subset = [p for p in parquet if needle in p]
            if subset:
                parquet = subset
                break
    if split:
        subset = [p for p in parquet if split in os.path.basename(p)]
        if subset:
            parquet = subset
    return parquet


class HuggingFaceParquetReader(DatasetReader):
    """Stream an HF dataset split's text through :class:`ParquetStreamReader`.

    The ``datasets``-free alternative to :class:`HuggingFaceReader` for corpus
    packing: resolve the split's parquet files (see
    :func:`resolve_hf_parquet_files`), then read them one at a time with bounded
    memory. Doc order is the file order ``datasets`` uses (files sorted, rows in
    file order), so the packed corpus is identical.

    Resolution is lazy (first iteration), so constructing the reader never
    touches the network — a bad CLI argument fails before any download.

    Args:
        dataset: HF dataset id, e.g. ``"HuggingFaceFW/fineweb-edu"``.
        split / config / revision: as for ``load_dataset``.
        text_field: column holding the document text.
        batch_size: rows per batch (the transient-memory unit).
        files: explicit file paths/URLs; skips resolution entirely.
    """

    def __init__(
        self,
        dataset: str,
        *,
        split: str = "train",
        config: Optional[str] = None,
        revision: Optional[str] = None,
        text_field: str = "text",
        batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
        files: Optional[Iterable[str]] = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retries: int = DEFAULT_HTTP_RETRIES,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.config = config
        self.revision = revision
        self.text_field = text_field
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.source = dataset
        self._files: Optional[List[str]] = list(files) if files is not None else None
        self._reader: Optional[ParquetStreamReader] = None

    @property
    def files(self) -> List[str]:
        """The resolved parquet files (resolves on first access)."""
        if self._files is None:
            self._files = resolve_hf_parquet_files(
                self.dataset,
                config=self.config,
                split=self.split,
                revision=self.revision,
            )
            log.info(
                "resolved %d parquet file(s) for %s/%s/%s",
                len(self._files), self.dataset, self.config, self.split,
            )
        return self._files

    def _iter_records(self) -> Iterator[Record]:
        reader = ParquetStreamReader(
            self.files,
            text_field=self.text_field,
            batch_size=self.batch_size,
            source=self.source,
            timeout=self.timeout,
            retries=self.retries,
        )
        self._reader = reader
        try:
            yield from reader._iter_records()
        finally:
            self.bytes_fetched = reader.bytes_fetched
            reader.close()
            self._reader = None

    #: Bytes fetched over the network during the last iteration.
    bytes_fetched: int = 0

    def num_records(self) -> Optional[int]:
        return None

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None


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
    "hf-parquet": HuggingFaceParquetReader,
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
