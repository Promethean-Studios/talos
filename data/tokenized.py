"""Streaming tokenized-dataset loader: sharded JSONL -> fixed-length int32 batches.

This is the bridge between the Phase 3 data pipeline (sharded JSONL on disk)
and the training loop. It reads **one document at a time**, tokenizes it,
converts the token ids to a compact ``numpy.int32`` array immediately, and
yields fixed-length ``(batch, seq_len)`` batches.

Why this exists (memory): a Python ``int`` costs ~28-36 bytes vs 4 bytes for
``int32``, and a ``list`` adds ~8 bytes of pointer per element — so holding a
corpus as ``list[list[int]]`` is a ~10-18x RAM blowup versus an ``int32``
tensor of the same tokens. Materializing even a modest corpus that way OOMs a
12GB Colab; this module's peak memory is O(batch + max document) instead of
O(corpus), so it streams corpora far larger than RAM.

Two layouts:

* ``mode="pack"`` (default) — concatenate every document's tokens into one
  stream (``eos``-separated) and cut contiguous ``seq_len`` blocks. This is
  the standard layout for base-LM pretraining: no padding waste, and every
  position trains.
* ``mode="padded"`` — each document becomes one row, truncated/padded with
  ``pad_id`` to ``seq_len`` (documents longer than ``seq_len`` are truncated).

The :class:`StreamingTokenizedDataset` wrapper is a ``torch.utils.data``
``IterableDataset`` yielding ready-made batches, sharding its shard files
across DataLoader workers when ``num_workers > 0``.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Iterator, List, Optional, Sequence

import numpy as np

try:  # torch is only needed by the IterableDataset wrapper below; the
    # numpy-level generators (the memory-critical path) run without it.
    import torch
    from torch.utils.data import IterableDataset as _IterableDataset
    from torch.utils.data import get_worker_info as _get_worker_info
except ImportError:  # pragma: no cover - torch is a core repo dep, but the
    torch = None  # numpy generators must still work in slim environments
    _IterableDataset = object  # type: ignore[assignment, misc]
    _get_worker_info = None

from data._logging import get_logger

log = get_logger("data.tokenized")

__all__ = [
    "StreamingTokenizedDataset",
    "iter_documents",
    "iter_token_arrays",
    "resolve_shard_paths",
]


def resolve_shard_paths(
    data_path: str,
    manifest_name: str = "manifest.json",
) -> List[str]:
    """Resolve ``data_path`` to an ordered list of shard files.

    Accepts:
      * a ``manifest.json`` written by :class:`~data.writers.ShardedWriter`
        (shards are used in manifest order),
      * a directory containing a ``manifest.json`` (same as above) or, failing
        that, all ``*.jsonl`` files sorted by name,
      * a single ``.jsonl`` / ``.jsonl.gz`` file.
    """
    if os.path.isfile(data_path):
        if data_path.endswith((".jsonl", ".jsonl.gz")):
            return [data_path]
        if os.path.basename(data_path) == manifest_name:
            return _paths_from_manifest(data_path)
        raise ValueError(f"unsupported data file: {data_path}")
    if os.path.isdir(data_path):
        manifest = os.path.join(data_path, manifest_name)
        if os.path.isfile(manifest):
            return _paths_from_manifest(manifest)
        shards = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".jsonl")
        )
        if shards:
            return shards
        raise FileNotFoundError(f"no manifest.json or *.jsonl under {data_path}")
    raise FileNotFoundError(f"data path not found: {data_path}")


def _paths_from_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    base = os.path.dirname(os.path.abspath(manifest_path))
    paths: List[str] = []
    for entry in payload.get("shards", []):
        p = os.path.join(base, entry["shard"])
        if os.path.isfile(p):
            paths.append(p)
        else:
            log.warning("manifest shard missing on disk, skipping: %s", p)
    if not paths:
        raise FileNotFoundError(f"manifest lists no existing shards: {manifest_path}")
    return paths


def iter_documents(
    paths: Sequence[str],
    text_field: str = "text",
    limit_docs: Optional[int] = None,
) -> Iterator[str]:
    """Yield document text from sharded JSONL one document at a time.

    Streams line-by-line (supports ``.gz`` shards); never materializes more
    than one record. Malformed lines are skipped with a warning.
    """
    seen = 0
    for path in paths:
        if limit_docs is not None and seen >= limit_docs:
            return
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("skipping malformed JSON %s:%d (%s)", path, line_no, exc)
                    continue
                text = obj.get(text_field) if isinstance(obj, dict) else None
                if not isinstance(text, str):
                    continue
                yield text
                seen += 1
                if limit_docs is not None and seen >= limit_docs:
                    return


def iter_token_arrays(
    paths: Sequence[str],
    tokenizer,
    *,
    seq_len: int,
    batch_size: int = 1,
    mode: str = "pack",
    bos: bool = False,
    eos: bool = True,
    drop_last: bool = True,
    text_field: str = "text",
    limit_docs: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield ``(batch, seq_len)`` ``int32`` numpy arrays of token ids.

    Token ids are converted to ``int32`` numpy arrays the moment each document
    is encoded (the Python id list is released immediately), and batches are
    emitted as soon as ``batch_size`` rows are ready. Peak memory is
    O(batch * seq_len + carry + max document tokens) — independent of corpus
    size.

    Args:
        paths: shard files (see :func:`resolve_shard_paths`).
        tokenizer: object with ``encode(text, bos=, eos=)`` and ``pad_id``.
        seq_len: fixed sequence length per row.
        batch_size: rows per yielded batch.
        mode: ``"pack"`` (concatenate + cut blocks) or ``"padded"``
            (one doc per row, padded/truncated to ``seq_len``).
        bos / eos: whether to wrap each document's token ids.
        drop_last: if True, drop a final partial batch (and, in ``"pack"``
            mode, the trailing partial block). If False, pad the final block
            with ``pad_id`` and yield a short final batch.
        text_field: JSON field holding the document text.
        limit_docs: stop after this many documents (``None`` = all).
    """
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mode not in ("pack", "padded"):
        raise ValueError("mode must be 'pack' or 'padded'")

    pad_id = int(tokenizer.pad_id)
    if mode == "padded":
        yield from _iter_padded(
            paths, tokenizer, seq_len=seq_len, batch_size=batch_size,
            bos=bos, eos=eos, drop_last=drop_last, text_field=text_field,
            limit_docs=limit_docs, pad_id=pad_id,
        )
        return

    carry = np.empty(0, dtype=np.int32)  # tokens not yet cut into blocks
    pending: List[np.ndarray] = []       # blocks waiting to fill a batch
    for text in iter_documents(paths, text_field=text_field, limit_docs=limit_docs):
        ids = tokenizer.encode(text, bos=bos, eos=eos)
        # Convert immediately: int32 = 4 bytes/token vs ~36 for list[int].
        arr = np.asarray(ids, dtype=np.int32)
        del ids
        if arr.size == 0:
            continue
        carry = np.concatenate((carry, arr))
        n_blocks = carry.size // seq_len
        if n_blocks == 0:
            continue
        blocks = carry[: n_blocks * seq_len].reshape(n_blocks, seq_len)
        carry = carry[n_blocks * seq_len :].copy()
        for block in blocks:
            pending.append(block)
            if len(pending) == batch_size:
                yield np.stack(pending)
                pending = []
    if not drop_last and carry.size:
        pad = np.full(seq_len - carry.size, pad_id, dtype=np.int32)
        pending.append(np.concatenate((carry, pad)))
        carry = np.empty(0, dtype=np.int32)
    if pending and not drop_last:
        yield np.stack(pending)


def _iter_padded(
    paths: Sequence[str],
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    bos: bool,
    eos: bool,
    drop_last: bool,
    text_field: str,
    limit_docs: Optional[int],
    pad_id: int,
) -> Iterator[np.ndarray]:
    pending: List[np.ndarray] = []
    for text in iter_documents(paths, text_field=text_field, limit_docs=limit_docs):
        ids = tokenizer.encode(text, bos=bos, eos=eos)
        arr = np.asarray(ids, dtype=np.int32)
        del ids
        row = np.full(seq_len, pad_id, dtype=np.int32)
        n = min(arr.size, seq_len)
        row[:n] = arr[:n]
        del arr
        pending.append(row)
        if len(pending) == batch_size:
            yield np.stack(pending)
            pending = []
    if pending and not drop_last:
        yield np.stack(pending)


class StreamingTokenizedDataset(_IterableDataset):
    """A ``torch`` ``IterableDataset`` of ready-made token batches.

    Yields ``torch.Tensor`` batches of shape ``(batch, seq_len)`` and dtype
    ``dtype`` (default ``torch.int32`` — 4 bytes/token; ``nn.Embedding``
    accepts int32 indices directly, or cast with ``.long()`` if preferred).

    Args:
        data_path: manifest.json / shard directory / single .jsonl file
            (resolved via :func:`resolve_shard_paths`), or an explicit list
            of shard paths.
        tokenizer: a :class:`~tokenizer.tokenizer.ByteLevelBPETokenizer` (or
            anything with ``encode``/``pad_id``).
        seq_len / batch_size: batch shape.
        mode: ``"pack"`` (default) or ``"padded"`` — see :func:`iter_token_arrays`.
        bos / eos / drop_last / text_field / limit_docs: forwarded.
        dtype: torch dtype of yielded batches (default ``torch.int32``).
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        *,
        seq_len: int,
        batch_size: int = 1,
        mode: str = "pack",
        bos: bool = False,
        eos: bool = True,
        drop_last: bool = True,
        text_field: str = "text",
        limit_docs: Optional[int] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> None:
        if torch is None:  # pragma: no cover - guarded import
            raise ImportError("torch is required for StreamingTokenizedDataset")
        super().__init__()
        if isinstance(data_path, (str, os.PathLike)):
            paths = resolve_shard_paths(str(data_path))
        else:
            paths = list(data_path)
            if not paths:
                raise ValueError("data_path resolved to an empty shard list")
        self.paths = paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.mode = mode
        self.bos = bos
        self.eos = eos
        self.drop_last = drop_last
        self.text_field = text_field
        self.limit_docs = limit_docs
        self.dtype = dtype if dtype is not None else torch.int32

    def __iter__(self) -> Iterator["torch.Tensor"]:
        # With DataLoader workers, split shard files across workers so each
        # yields disjoint batches (document-level sharding).
        worker = _get_worker_info() if _get_worker_info is not None else None
        paths = self.paths
        if worker is not None and worker.num_workers > 1:
            paths = [
                p for i, p in enumerate(paths) if i % worker.num_workers == worker.id
            ]
            if not paths:  # more workers than shards: this worker idles
                return
        for batch in iter_token_arrays(
            paths,
            self.tokenizer,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            mode=self.mode,
            bos=self.bos,
            eos=self.eos,
            drop_last=self.drop_last,
            text_field=self.text_field,
            limit_docs=self.limit_docs,
        ):
            yield torch.from_numpy(batch).to(self.dtype)
