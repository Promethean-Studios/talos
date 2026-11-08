"""Corpus preparation for Talos pretraining: streamed HF corpus -> packed .npy shards.

This is the corpus-prep stage of the tiny_100m pretraining program (pass A).
It streams document text from a HuggingFace ``datasets`` corpus (default
``HuggingFaceFW/fineweb-edu`` / ``sample-10BT`` / ``train``, streaming=True, no
auth), tokenizes each document with the *established* Talos tokenizer (an
existing ``tokenizer.json`` — passed with ``--tokenizer-json``; BPE is NOT
trained here), and packs the token stream into fixed ``seq``-length rows with
EOS separators between documents, following the packing convention of
``data.tokenized.py`` (``mode="pack"``, ``eos=True``: every document — including
the last — is ``encode(text) + [eos]``; ids are validated at encode time against
the **loaded tokenizer's own id space** via ``model.utils.validate_token_ids``).

The token-id bound is derived from the ``tokenizer.json`` that is loaded
(``tokenizer.config.vocab_size``, the exclusive upper bound on every id the file
can assign — base bytes, merges and the top-of-range special ids), never from a
hard-coded constant. The canonical ladder's vocab-1024 tokenizers therefore keep
their old bound of 1024, while the 1B-track assets (``talos_1b_32k`` v2 =
32,000 ids, ``talos_1b_49k`` = 49,152 ids) pack under their own bound. The
``--dtype`` capacity check follows the same rule: ``uint16`` is accepted only
when every id fits in 16 bits (vocab_size <= 65,536), ``int32`` always is, and a
tokenizer that outgrows the requested dtype fails loudly instead of truncating.

A held-out validation slice (``--val-tokens``) is carved from a DISJOINT region
of the stream: the first ``--val-skip-docs`` documents are skipped entirely,
then val documents are collected until ``--val-tokens`` is reached; training
documents are collected from *after* the val region. The two regions therefore
never share a source document at any budget, and the val region's location is
independent of ``--target-tokens`` (stable across runs with different budgets).

Output layout (``--out-dir``)::

    out_dir/
      shard-val-0000.npy      # (rows, seq) packed token rows, int32 or uint16
      shard-train-0000.npy    # ...
      manifest.json           # packing manifest (shards + embedded metadata)
      run_metadata.json       # the reproducibility record (same dict as the
                              # manifest's "metadata" section)

Token accounting (recorded in the metadata):

* ``val_tokens`` / ``train_tokens`` — *real* tokens written: per document
  ``len(encode(text)) + 1`` (the EOS separator), summed per region. This is the
  number a downstream trainer would consume.
* ``*_tokens_incl_padding`` — ``rows * seq``: what is physically stored. The
  final partial row of each region is padded with the tokenizer's ``pad_id``
  (recorded as ``pad_tokens``); real token counts exclude padding.
* ``target_tokens`` is the TOTAL budget, inclusive of the val slice: the run
  stops once ``val_tokens + train_tokens >= --target-tokens``. If the stream is
  exhausted first, ``truncated`` is set in the metadata.

The script is streaming end-to-end: at most ``--stream-buffer-docs`` documents'
tokens and one shard's rows are in RAM at once, independent of corpus size.

Usage::

    python -m scripts.prepare_corpus --tokenizer-json /path/to/tokenizer.json \\
        --target-tokens 2000000000 --val-tokens 15000000 --out-dir runs/corpus-2b

This is the reproducible-materialization entry point for the real Colab run; the
local test suite exercises the same code path with synthetic JSONL readers (no
network, no corpus download).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the script
# also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data._logging import get_logger  # noqa: E402
from data.readers import DatasetReader, HuggingFaceReader  # noqa: E402
from tokenizer.tokenizer import (  # noqa: E402
    ByteLevelBPETokenizer,
    tokenizer_file_sha256,
)

log = get_logger("scripts.prepare_corpus")

#: Bumped on any change to the output format or the metadata schema.
#: 1.1.0: token-id bound derived from the loaded tokenizer (was the hard-coded
#: canonical VOCAB_SIZE); ``tokenizer.configured_vocab_size`` and
#: ``tokenizer.exact_bpe`` added to the metadata.
SCRIPT_VERSION = "1.1.0"
#: Packing manifest format tag (analogous to ShardedWriter's JSONL manifests).
MANIFEST_FORMAT = "talos-packed-tokens-v1"
#: Top-level metadata schema tag.
METADATA_SCHEMA = "talos-corpus-run-metadata-v1"

VALID_DTYPES = ("int32", "uint16")

#: Shard naming mirrors ShardedWriter's ``{prefix}-{index:04d}.{ext}`` pattern.
def _shard_name(phase: str, index: int) -> str:
    return f"shard-{phase}-{index:04d}.npy"


class NpyShardWriter:
    """Write ``(rows, seq_len)`` token rows to numbered ``*.npy`` shards.

    Follows :class:`data.writers.ShardedWriter`'s patterns: numbered shards, a
    row-count cap per shard, an eager doc-count flush (the streaming memory
    bound), and atomic writes (``*.tmp`` + ``os.replace``) so an interrupted
    run never leaves a half-written shard under its real name.

    Memory is *pre-allocated and bounded*: one ``(rows_per_shard, seq_len)``
    buffer is reserved once and rows are copied into it, so the writer never
    holds more than one shard's worth of tokens and never needs the
    ``np.stack`` copy the old list-of-rows design required (which momentarily
    doubled the shard buffer). Copying each row also drops the reference the
    old code kept to the document-sized array a row was sliced from — a row
    view used to pin its whole source array for as long as the shard buffer
    lived.
    """

    def __init__(
        self,
        out_dir: str,
        phase: str,
        *,
        seq_len: int,
        dtype: np.dtype,
        rows_per_shard: int,
        stream_buffer_docs: int,
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        if stream_buffer_docs <= 0:
            raise ValueError("stream_buffer_docs must be positive")
        self.out_dir = out_dir
        self.phase = phase
        self.seq_len = seq_len
        self.dtype = dtype
        self.rows_per_shard = rows_per_shard
        self.stream_buffer_docs = stream_buffer_docs
        #: Fixed-size row buffer — the writer's entire RAM footprint per shard.
        self._buffer = np.empty((rows_per_shard, seq_len), dtype=dtype)
        self._pending = 0
        self._docs_since_flush = 0
        self._index = 0
        self._entries: List[Dict[str, Any]] = []
        self._closed = False

    # -- public API ---------------------------------------------------------
    def write_row(self, row: np.ndarray) -> None:
        """Append one validated ``(seq_len,)`` row to the pending buffer."""
        if self._closed:
            raise RuntimeError("writer already closed")
        if row.shape != (self.seq_len,):
            raise ValueError(f"row shape {row.shape} != ({self.seq_len},)")
        if self._pending >= self.rows_per_shard:
            self._flush()
        self._buffer[self._pending] = row  # copy: never pin the source array
        self._pending += 1
        if self._pending >= self.rows_per_shard:
            self._flush()

    def doc_seen(self) -> None:
        """Note one encoded document — bounds the docs held before a flush."""
        self._docs_since_flush += 1
        if self._docs_since_flush >= self.stream_buffer_docs:
            self._flush()

    def close(self) -> List[Dict[str, Any]]:
        """Flush pending rows and return the per-shard manifest entries."""
        if self._closed:
            return self._entries
        self._flush()
        self._closed = True
        return self._entries

    # -- internals ----------------------------------------------------------
    def _flush(self) -> None:
        if self._pending == 0:
            self._docs_since_flush = 0
            return
        # A view, not a copy: the pre-allocated buffer already holds the rows
        # contiguously, so flushing costs no extra shard-sized allocation.
        array = self._buffer[: self._pending]
        final = os.path.join(self.out_dir, _shard_name(self.phase, self._index))
        tmp = final + ".tmp"
        # np.save appends ".npy" to *paths* unless they already end in ".npy",
        # so write through an open handle to keep the suffix exactly ".tmp".
        with open(tmp, "wb") as fh:
            np.save(fh, array)
        os.replace(tmp, final)  # atomic rename
        self._entries.append(
            {
                "shard": os.path.basename(final),
                "phase": self.phase,
                "index": self._index,
                "rows": int(array.shape[0]),
                "tokens_incl_padding": int(array.shape[0]) * self.seq_len,
            }
        )
        self._pending = 0
        self._docs_since_flush = 0
        self._index += 1


def _np_dtype(dtype: str) -> np.dtype:
    if dtype == "int32":
        return np.dtype(np.int32)
    if dtype == "uint16":
        return np.dtype(np.uint16)
    raise ValueError(f"unsupported dtype {dtype!r}: choose from {VALID_DTYPES}")


def tokenizer_id_bound(tokenizer: ByteLevelBPETokenizer) -> int:
    """Exclusive upper bound on the token ids the LOADED tokenizer can assign.

    Derived from the loaded ``tokenizer.json`` itself (its
    ``TokenizerConfig.vocab_size``) — the same object whose sha256/vocab counts
    the metadata records — so the packer follows the tokenizer it was handed
    instead of a hard-coded canonical constant. The canonical vocab-1024
    tokenizers keep bound 1024; the 1B-track assets get 32,000 / 49,152.

    ``config.vocab_size`` (not the *realized* ``tokenizer.vocab_size``) is the
    right bound: special-token ids are assigned at the TOP of the configured
    range, so a partially-trained tokenizer can have ``vocab_size == 260`` while
    its EOS/PAD live at 1021/1022 — bounding by the realized size would reject
    the canonical untrained fixtures. ``max(config, realized)`` keeps that safe
    even for a hand-built tokenizer whose two numbers disagree the other way.
    """
    config = getattr(tokenizer, "config", None)
    configured = getattr(config, "vocab_size", None)
    realized = int(tokenizer.vocab_size)
    if not isinstance(configured, int) or configured <= 0:
        bound = realized
    else:
        bound = max(int(configured), realized)
    if bound <= 0:
        raise ValueError(
            f"tokenizer reports vocab_size {bound} — cannot derive a token-id bound"
        )
    return bound


def check_tokenizer_dtype(
    tokenizer: ByteLevelBPETokenizer, dtype: str
) -> int:
    """Validate ``dtype`` can hold every id of ``tokenizer``; return the bound.

    ``uint16`` is accepted iff the whole id space fits in 16 bits
    (``vocab_size <= 65,536``, i.e. max id 65,535); ``int32`` is always
    accepted. A tokenizer that outgrows the requested dtype raises a
    ``ValueError`` naming both numbers — never a silent truncation, which would
    corrupt the packed corpus.
    """
    np_dtype = _np_dtype(dtype)  # raises on an unknown dtype name
    bound = tokenizer_id_bound(tokenizer)
    max_id = bound - 1
    capacity = int(np.iinfo(np_dtype).max)
    if max_id > capacity:
        raise ValueError(
            f"dtype {dtype!r} cannot hold token ids up to {max_id} "
            f"(tokenizer vocab_size {bound}; {dtype} maxes out at {capacity}) "
            "— use --dtype int32 for this tokenizer"
        )
    return bound


class RegionPacker:
    """Pack one region's documents into seq-length rows (EOS-separated).

    Encodes each document with the loaded tokenizer (plain encode followed by
    an explicit EOS id — no other special tokens), range-checks every id
    against the tokenizer's own id bound (``vocab_bound``, derived by
    :func:`tokenizer_id_bound`) at encode time, cuts contiguous ``seq_len`` rows
    and forwards them to the region's :class:`NpyShardWriter`. A final partial
    row is padded with ``pad_id`` on :meth:`finish` (recorded as ``pad_tokens``).
    """

    def __init__(
        self,
        writer: NpyShardWriter,
        tokenizer: ByteLevelBPETokenizer,
        *,
        phase: str,
        seq_len: int,
        vocab_bound: int,
    ) -> None:
        self.writer = writer
        self.tokenizer = tokenizer
        self.phase = phase
        self.seq_len = seq_len
        #: Exclusive upper bound on token ids for the LOADED tokenizer.
        self.vocab_bound = int(vocab_bound)
        self.eos_id = int(tokenizer.eos_id)
        self.pad_id = int(tokenizer.pad_id)
        self.dtype = writer.dtype
        #: Tokens not yet cut into rows (always < seq_len at rest).
        self._carry = np.empty(0, dtype=self.dtype)
        self.tokens = 0  # real tokens: doc ids + one EOS per document
        self.docs = 0
        self.chars = 0
        self.rows = 0
        self.pad_tokens = 0
        self.empty_docs = 0
        self.first_doc: Optional[int] = None
        self.last_doc: Optional[int] = None

    def _validate_ids(self, arr: np.ndarray, where: str) -> None:
        """Range-check numpy token ids against the loaded tokenizer's bound.

        The check is the repo's canonical one — ``model.utils.validate_token_ids``
        (two reductions, ``[0, vocab_bound)``, ``ValueError`` naming the id) — but
        it is *reached* only when numpy says something is actually out of range.
        Importing ``torch`` costs ~250 MB of resident memory and this process
        packs a 220M-token corpus on a 4 GB box, so the happy path stays in
        numpy and the canonical guard is imported (and raises) only on a
        violation. Valid input therefore behaves exactly as before — no error,
        no torch import; invalid input still raises the canonical
        ``ValueError`` from ``validate_token_ids`` itself.
        """
        if arr.size == 0:
            return
        if int(arr.min()) >= 0 and int(arr.max()) < self.vocab_bound:
            return
        _validate_ids_canonical(arr, self.vocab_bound, where)

    def add_doc(self, doc_index: int, text: str) -> None:
        """Encode + pack one document; token id bounds are enforced here."""
        self.chars += len(text)
        ids64 = np.asarray(self.tokenizer.encode(text), dtype=np.int64)
        if ids64.size == 0:
            self.empty_docs += 1
            return
        where = f"{self.phase} document {doc_index}"
        self._validate_ids(ids64, where)
        ids = ids64.astype(self.dtype)
        # EOS separator after every document (data/tokenized.py pack convention).
        full = np.concatenate(
            (self._carry, ids, np.asarray([self.eos_id], dtype=self.dtype))
        )
        n_rows = full.size // self.seq_len
        if n_rows:
            blocks = full[: n_rows * self.seq_len].reshape(n_rows, self.seq_len)
            for row in blocks:
                self._validate_ids(
                    row, where=f"{self.phase} row of doc {doc_index}"
                )
                self.writer.write_row(row)
            self.rows += n_rows
        self._carry = full[n_rows * self.seq_len :].copy()
        self.tokens += ids.size + 1
        self.docs += 1
        if self.first_doc is None:
            self.first_doc = doc_index
        self.last_doc = doc_index
        self.writer.doc_seen()

    def finish(self) -> Dict[str, int]:
        """Flush the padded tail row (if any) and return the region's counts."""
        if self._carry.size:
            pad_len = self.seq_len - int(self._carry.size)
            tail = np.concatenate(
                (self._carry, np.full(pad_len, self.pad_id, dtype=self.dtype))
            )
            self._validate_ids(tail, where=f"{self.phase} padded tail")
            self.writer.write_row(tail)
            self.rows += 1
            self.pad_tokens += pad_len
            self._carry = np.empty(0, dtype=self.dtype)
        self.writer.close()
        return {
            "phase": self.phase,
            "docs": self.docs,
            "empty_docs": self.empty_docs,
            "chars": self.chars,
            "tokens": self.tokens,
            "rows": self.rows,
            "pad_tokens": self.pad_tokens,
            "tokens_incl_padding": self.rows * self.seq_len,
            "first_doc": self.first_doc,
            "last_doc": self.last_doc,
        }


def _validate_ids_canonical(arr: np.ndarray, vocab_bound: int, where: str) -> None:
    """Raise the canonical out-of-range ``ValueError`` (slow path only).

    torch has no uint16 tensor support, so non-``int32``/``int64`` arrays (the
    ``uint16`` packed layout) are widened to ``int64`` first — the same
    conversion the pre-fix code always performed. Importing torch here (rather
    than at module scope) is what keeps ~250 MB of RSS out of the packer.
    """
    import torch

    from model.utils import validate_token_ids

    if arr.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        arr = arr.astype(np.int64)
    validate_token_ids(torch.from_numpy(arr), vocab_bound, where=where)


def limit_stream_readahead() -> bool:
    """Make fsspec file reads pass-through, so a sequential scan retains nothing.

    Why: ``datasets`` streams a remote parquet corpus through ``fsspec``, which
    opens every source file with ``cache_type="readahead"``. That cache keeps
    the *last fetched chunk* alive on the file object, and a parquet reader
    asks for a whole column chunk at a time — measured here at ~26–31 MB per
    source file. ``datasets`` holds those file objects for the life of the
    stream, so resident memory grew by one multi-MB buffer per source file,
    i.e. **unboundedly with progress**: packing 761 M characters (≈15 parquet
    files) reached 1.12 GB RSS on a 3.9 GB box and the full 220 M-token run was
    OOM-killed at 1.5 GB. The same run with read-ahead disabled peaked at
    0.96 GB — the remaining growth is ~0.6 bytes per character streamed, well
    inside the budget.

    Nothing is lost by disabling it: the packer consumes each file strictly
    forward, so there is no re-read for a cache to serve. ``fsspec`` resolves
    cache classes through the plain ``fsspec.caching.caches`` registry at file
    open time; ``BaseCache`` is its documented pass-through ("doesn't keep
    anything, calls every time").

    Idempotent and a no-op when fsspec is not installed. Returns ``True`` when
    the override is in place.
    """
    try:
        import fsspec.caching as caching
    except ImportError:  # pragma: no cover - fsspec ships with datasets
        return False
    caching.caches["readahead"] = caching.BaseCache
    return caching.caches.get("readahead") is caching.BaseCache


def _fetch_hf_dataset_meta(dataset_id: str, config: Optional[str]) -> Dict[str, Any]:
    """Best-effort HF revision/sha metadata for a *string* dataset id.

    Returns a dict with ``dataset``/``config``/``revision``/``sha`` plus any
    license found in the dataset card. Never raises and never blocks on a
    missing network: any failure records ``"error"`` and ``revision=None``.
    Only called from ``main()``; tests pass reader objects and never hit this.
    """
    meta: Dict[str, Any] = {
        "dataset": dataset_id,
        "config": config,
        "revision": None,
        "sha": None,
        "license": None,
    }
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id)
        sha = getattr(info, "sha", None)
        meta["revision"] = sha
        meta["sha"] = sha
        card = getattr(info, "cardData", None)
        if isinstance(card, dict) and card.get("license"):
            meta["license"] = card.get("license")
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        meta["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("could not fetch HF metadata for %s: %s", dataset_id, exc)
    return meta


def prepare_corpus(
    reader: DatasetReader,
    tokenizer: ByteLevelBPETokenizer,
    *,
    out_dir: str,
    target_tokens: int,
    val_tokens: int,
    tokenizer_path: str,
    seq_len: int = 512,
    dtype: str = "int32",
    val_skip_docs: int = 10_000,
    stream_buffer_docs: int = 1_000,
    rows_per_shard: int = 16_384,
    text_field: str = "text",
    dataset_meta: Optional[Dict[str, Any]] = None,
    args_echo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stream ``reader``, tokenize with the loaded tokenizer, write packed shards.

    Args:
        reader: any :class:`data.readers.DatasetReader` (the CLI wires a
            ``HuggingFaceReader``; tests use synthetic JSONL readers).
        tokenizer: the established Talos tokenizer (plain ``encode``, EOS
            separators inserted by this function). Its own ``vocab_size`` is the
            token-id bound every packed id is validated against.
        tokenizer_path: path of the loaded ``tokenizer.json`` — its sha256
            fingerprint, realized/configured vocab sizes, merge count and encoder
            variant are recorded in the metadata.
        out_dir: destination for ``*.npy`` shards + ``manifest.json`` +
            ``run_metadata.json`` (created if missing).
        target_tokens: TOTAL token budget — the run stops once
            ``val_tokens + train_tokens >= target_tokens`` (val included).
        val_tokens: tokens carved for the held-out val region (must be
            ``< target_tokens``).
        seq_len / dtype: packed row width and on-disk dtype (``int32`` always
            accepted; ``uint16`` iff the loaded tokenizer's ``vocab_size`` is
            ``<= 65,536``, so every id fits in 16 bits).
        val_skip_docs: number of documents skipped at the head of the stream
            before the val region starts — this gap is what makes the val
            region DISJOINT from the train region (train starts after val).
        stream_buffer_docs: cap on encoded docs held in RAM between shard
            flushes (the streaming memory bound).
        rows_per_shard: max ``(rows, seq_len)`` arrays per ``*.npy`` shard.
        text_field: record key holding the document text.
        dataset_meta: ``{dataset, config, split, revision, sha, ...}`` dict
            recorded verbatim into the metadata (HF revision/sha when known).
        args_echo: the raw CLI args dict, recorded for reproducibility.

    Returns:
        The full metadata dict (also written to ``run_metadata.json`` and
        embedded in ``manifest.json``).
    """
    np_dtype = _np_dtype(dtype)
    # The id bound comes from the LOADED tokenizer (canonical 1024, 1B-track
    # 32K/49K, ...) — never a hard-coded constant. The dtype must be able to
    # hold every id that tokenizer can assign; refuse loudly otherwise.
    vocab_bound = check_tokenizer_dtype(tokenizer, dtype)
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if target_tokens < 1 or val_tokens < 1:
        raise ValueError("target_tokens and val_tokens must both be >= 1")
    if val_tokens >= target_tokens:
        raise ValueError(
            f"val_tokens ({val_tokens}) must be strictly less than "
            f"target_tokens ({target_tokens}) — the val slice is part of the "
            "total budget"
        )
    if val_skip_docs < 0:
        raise ValueError("val_skip_docs must be >= 0")
    eos_id = int(tokenizer.eos_id)
    pad_id = int(tokenizer.pad_id)
    if not (0 <= eos_id < vocab_bound) or not (0 <= pad_id < vocab_bound):
        raise ValueError(
            f"tokenizer special ids outside [0, {vocab_bound}): eos={eos_id} "
            f"pad={pad_id} — the tokenizer's own vocab_size is {vocab_bound}, so "
            "its special ids must fall inside that range"
        )

    os.makedirs(out_dir, exist_ok=True)
    val_writer = NpyShardWriter(
        out_dir, "val",
        seq_len=seq_len, dtype=np_dtype,
        rows_per_shard=rows_per_shard, stream_buffer_docs=stream_buffer_docs,
    )
    train_writer = NpyShardWriter(
        out_dir, "train",
        seq_len=seq_len, dtype=np_dtype,
        rows_per_shard=rows_per_shard, stream_buffer_docs=stream_buffer_docs,
    )
    val = RegionPacker(
        val_writer, tokenizer, phase="val", seq_len=seq_len, vocab_bound=vocab_bound
    )
    train = RegionPacker(
        train_writer, tokenizer, phase="train", seq_len=seq_len, vocab_bound=vocab_bound
    )

    docs_streamed = 0
    skipped_gap = 0
    truncated = False
    val_done = False
    t0 = time.monotonic()
    for record in reader:
        docs_streamed += 1
        text = record.get(text_field, "")
        if not isinstance(text, str):
            continue
        if docs_streamed <= val_skip_docs:
            skipped_gap += 1
            continue
        if not val_done:
            val.add_doc(docs_streamed - 1, text)
            if val.tokens >= val_tokens:
                val_done = True
            continue
        train.add_doc(docs_streamed - 1, text)
        if val.tokens + train.tokens >= target_tokens:
            break
    else:  # reader exhausted before the budget was reached
        truncated = True

    val_counts = val.finish()
    train_counts = train.finish()
    if val_counts["tokens"] == 0:
        raise ValueError(
            f"val region is empty: {val_skip_docs} skip docs consumed the whole "
            "stream (or the stream is too short) — lower --val-skip-docs or "
            "check the dataset/split"
        )
    elapsed_s = time.monotonic() - t0

    tokenizer_fingerprint = tokenizer_file_sha256(tokenizer_path)
    # Normalize the dataset record: schema keys (dataset/config/split and the
    # best-effort HF revision/sha) are always present, NULL when unavailable;
    # any extra keys the caller provided (license, error, ...) are kept.
    provided_ds = dataset_meta if dataset_meta is not None else {}
    dataset_entry: Dict[str, Any] = {
        "dataset": None,
        "config": None,
        "split": None,
        "revision": None,
        "sha": None,
    }
    dataset_entry.update({k: v for k, v in provided_ds.items() if v is not None})
    metadata: Dict[str, Any] = {
        "schema": METADATA_SCHEMA,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": args_echo if args_echo is not None else {},
        "dataset": dataset_entry,
        "tokenizer": {
            "path": os.path.abspath(tokenizer_path),
            "sha256": tokenizer_fingerprint,
            #: realized size: base bytes + specials + learned merges (the count
            #: ``tokenizer.vocab_size`` reports; equals the configured size for a
            #: fully-trained tokenizer such as talos_1b_49k -> 49,152).
            "vocab_size": tokenizer.vocab_size,
            #: configured id space — the exclusive bound every packed id obeys.
            "configured_vocab_size": int(tokenizer.config.vocab_size),
            "merge_count": tokenizer.merge_count,
            #: encoder variant persisted in the tokenizer file ("exact_bpe" vs
            #: the legacy heap encoder) — part of the asset's identity.
            "exact_bpe": bool(getattr(tokenizer, "exact_bpe", False)),
        },
        "packing": {
            "seq": seq_len,
            "eos_id": eos_id,
            "pad_id": pad_id,
            "dtype": dtype,
            "vocab_size_bound": vocab_bound,
            "rows_per_shard": rows_per_shard,
            "stream_buffer_docs": stream_buffer_docs,
        },
        "val_region": {
            "skip_docs": val_skip_docs,
            "first_doc": val_counts["first_doc"],
            "last_doc": val_counts["last_doc"],
            "docs": val_counts["docs"],
            "tokens": val_counts["tokens"],
        },
        "train_region": {
            "first_doc": train_counts["first_doc"],
            "last_doc": train_counts["last_doc"],
            "docs": train_counts["docs"],
            "tokens": train_counts["tokens"],
        },
        "counts": {
            "target_tokens": target_tokens,
            "docs_streamed": docs_streamed,
            "skipped_gap_docs": skipped_gap,
            "empty_docs_skipped": int(val_counts["empty_docs"]) + int(train_counts["empty_docs"]),
            "val_docs": val_counts["docs"],
            "val_chars": val_counts["chars"],
            "val_tokens": val_counts["tokens"],
            "val_rows": val_counts["rows"],
            "val_pad_tokens": val_counts["pad_tokens"],
            "val_tokens_incl_padding": val_counts["tokens_incl_padding"],
            "train_docs": train_counts["docs"],
            "train_chars": train_counts["chars"],
            "train_tokens": train_counts["tokens"],
            "train_rows": train_counts["rows"],
            "train_pad_tokens": train_counts["pad_tokens"],
            "train_tokens_incl_padding": train_counts["tokens_incl_padding"],
            "total_tokens": int(val_counts["tokens"]) + int(train_counts["tokens"]),
            "total_rows": int(val_counts["rows"]) + int(train_counts["rows"]),
            "total_chars": int(val_counts["chars"]) + int(train_counts["chars"]),
            "truncated": truncated,
        },
        "wall_s": round(elapsed_s, 3),
    }

    manifest = {
        "format": MANIFEST_FORMAT,
        "dtype": dtype,
        "seq_len": seq_len,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "num_shards": len(val_writer._entries) + len(train_writer._entries),
        "shards": list(val_writer._entries) + list(train_writer._entries),
        "metadata": metadata,
    }
    _atomic_write_json(os.path.join(out_dir, "manifest.json"), manifest)
    _atomic_write_json(os.path.join(out_dir, "run_metadata.json"), metadata)
    log.info(
        "prepared %s: %d train tokens + %d val tokens (target %d, truncated=%s) "
        "-> %d shards in %.1fs",
        out_dir, train_counts["tokens"], val_counts["tokens"],
        target_tokens, truncated, manifest["num_shards"], elapsed_s,
    )
    return metadata


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.prepare_corpus",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset id (streaming, ungated/no auth) — default "
                        "HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT",
                   help="dataset config/subset (default sample-10BT)")
    p.add_argument("--split", default="train", help="dataset split (default train)")
    p.add_argument("--tokenizer-json", required=True, metavar="path",
                   help="established Talos tokenizer.json to LOAD (BPE is not "
                        "trained here); sha256/vocab/merges are recorded in the metadata")
    p.add_argument("--target-tokens", type=int, required=True,
                   help="TOTAL token budget — stop once val+train tokens reach "
                        "this (includes the val slice)")
    p.add_argument("--val-tokens", type=int, default=15_000_000,
                   help="tokens carved from a DISJOINT val region (default 15e6)")
    p.add_argument("--val-skip-docs", type=int, default=10_000,
                   help="docs skipped at the stream head before the val region "
                        "(the disjoint-region gap; default 10000)")
    p.add_argument("--seq", type=int, default=512, help="packed sequence length (default 512)")
    p.add_argument("--dtype", choices=VALID_DTYPES, default="int32",
                   help="on-disk token dtype (default int32). uint16 is accepted "
                        "only when the loaded tokenizer's vocab_size <= 65536 "
                        "(every id fits in 16 bits; the 49K/32K 1B assets "
                        "qualify); int32 always works")
    p.add_argument("--out-dir", required=True, help="output directory (shards + manifests)")
    p.add_argument("--stream-buffer-docs", type=int, default=1_000,
                   help="max encoded docs held in RAM between shard flushes (default 1000)")
    p.add_argument("--rows-per-shard", type=int, default=16_384,
                   help="max rows per .npy shard (default 16384)")
    p.add_argument("--text-field", default="text", help="record field holding the document text")
    p.add_argument("--no-hf-metadata", action="store_true",
                   help="skip the best-effort HF dataset revision/sha lookup")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        tokenizer_path = os.path.abspath(args.tokenizer_json)
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"--tokenizer-json not found: {tokenizer_path}")
        tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
        # Pre-flight against the LOADED tokenizer: any vocab it declares is
        # packable (canonical 1024, 1B-track 32K/49K, ...) as long as the
        # requested dtype can hold its ids — fail before streaming otherwise.
        vocab_bound = check_tokenizer_dtype(tokenizer, args.dtype)
        log.info(
            "packing with tokenizer %s (vocab_size %d, bound %d, %d merges, dtype %s)",
            os.path.basename(tokenizer_path), tokenizer.vocab_size, vocab_bound,
            tokenizer.merge_count, args.dtype,
        )
        # Bound the streaming reader's resident memory before it opens anything:
        # fsspec's read-ahead cache would otherwise retain a multi-MB buffer per
        # source file for the life of the stream (see limit_stream_readahead).
        if limit_stream_readahead():
            log.info("streaming reads are pass-through (fsspec read-ahead disabled)")
        reader = HuggingFaceReader(
            args.dataset, split=args.split, text_field=args.text_field,
            streaming=True, config=args.config,
        )
        dataset_meta: Dict[str, Any] = {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
        }
        if not args.no_hf_metadata:
            dataset_meta = _fetch_hf_dataset_meta(args.dataset, args.config)
            dataset_meta["split"] = args.split
        prepare_corpus(
            reader,
            tokenizer,
            out_dir=args.out_dir,
            target_tokens=args.target_tokens,
            val_tokens=args.val_tokens,
            tokenizer_path=tokenizer_path,
            seq_len=args.seq,
            dtype=args.dtype,
            val_skip_docs=args.val_skip_docs,
            stream_buffer_docs=args.stream_buffer_docs,
            rows_per_shard=args.rows_per_shard,
            text_field=args.text_field,
            dataset_meta=dataset_meta,
            args_echo=vars(args),
        )
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())