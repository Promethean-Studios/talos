"""Packed-token shard reading + validation for the Talos trainer.

Consumes the output of ``scripts/prepare_corpus.py`` (pass A of the 100M
program): ``*.npy`` shards of exactly-``seq``-length token rows plus a
``manifest.json`` recording the packing parameters, per-shard entry metadata
(shard name / phase / row count) and a full reproducibility metadata block
(the same dict that ``run_metadata.json`` holds — dataset, tokenizer sha256,
counts, regions, args).

The contract enforced here (DELIVERABLE 1 of the token-budget trainer pass):

* rows are used **directly** — they are already ``seq``-wide, so no
  re-splitting or re-tokenization happens;
* manifest-vs-shard consistency is checked eagerly and cheaply (numpy header
  only): every shard's real ``(rows, seq)`` shape and dtype must match its
  manifest entry;
* every token id is range-checked against ``[0, vocab_size)`` per shard when
  the shard is streamed (the same fail-fast-at-source guarantee the JSONL
  path has);
* any mismatch raises a ``ValueError`` that names the shard, the manifest
  value and the on-disk value — loud, never silent.

Memory stays bounded: one shard's rows are in RAM at a time (rows-per-shard
caps the width; a 16 384-row seq-512 int32 shard is ~32 MiB).

Only numpy + torch are imported here, so both ``data.tokenized`` and the
trainer can use this module without pulling in the corpus-prep CLI's reader
stack.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

#: Mirrors ``scripts.prepare_corpus.MANIFEST_FORMAT`` — duplicated here so the
#: data layer never imports the CLI module (keep the dependency direction
#: one-way: prepare_corpus -> data.*, never the reverse).
MANIFEST_FORMAT = "talos-packed-tokens-v1"
VALID_DTYPES = ("int32", "uint16")

_NP_DTYPES = {
    "int32": np.dtype(np.int32),
    "uint16": np.dtype(np.uint16),
}


def _load_npy_header(path: str) -> np.ndarray:
    """Return a mmap view of ``path`` (header parsed, body not read)."""
    arr = np.load(path, mmap_mode="r")
    return arr


def _check_rows_and_ids(
    arr: np.ndarray,
    *,
    path: str,
    seq_len: int,
    max_id: int,
) -> None:
    """Loud shape/value validation of one loaded shard (values force a read).

    The shape/dtype checks are cheap header facts; the id-value check touches
    every element (vectorized min/max — microseconds per shard, regardless of
    width) so a drifted/corrupt shard can never feed the model.
    """
    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise ValueError(
            f"shard {os.path.basename(path)} has shape {arr.shape} "
            f"(expected (rows, {seq_len})) — mismatched seq/packing; the "
            "shard was not produced by this manifest's packing config"
        )
    if arr.size == 0:
        return
    if int(arr.min()) < 0 or int(arr.max()) >= max_id:
        raise ValueError(
            f"shard {os.path.basename(path)} contains token ids outside "
            f"[0, {max_id}): min={int(arr.min())} max={int(arr.max())} — "
            "corrupt shard or a tokenizer/model vocab mismatch"
        )


def load_packed_manifest(
    packed_dir: str,
    *,
    expected_seq: Optional[int] = None,
    expected_vocab: Optional[int] = None,
    expected_tokenizer_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate ``packed_dir/manifest.json``.

    Validates (raising ``ValueError``/``FileNotFoundError`` with the offending
    value named in every message):

    * the manifest exists and carries the expected format tag;
    * ``seq_len``/``dtype``/``eos_id``/``pad_id`` are present and sane;
    * every shard entry is internally consistent (rows ``>= 1``,
      ``tokens_incl_padding == rows * seq``) and the file exists on disk;
    * phase row sums match the metadata counts (``train_rows``/``val_rows``);
    * ``expected_seq`` (the trainer's effective seq) matches ``seq_len``;
    * ``expected_vocab`` is large enough for the recorded
      ``vocab_size_bound``; and
    * ``expected_tokenizer_sha256`` (the trainer's sidecar tokenizer) matches
      the recorded ``metadata.tokenizer.sha256``.

    Returns the parsed manifest dict (callers read ``shards`` for paths and
    ``metadata`` for the provenance block).
    """
    manifest_path = os.path.join(packed_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"packed corpus has no manifest.json: {packed_dir}"
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    problems: List[str] = []
    if manifest.get("format") != MANIFEST_FORMAT:
        problems.append(
            f"manifest format {manifest.get('format')!r} — expected "
            f"{MANIFEST_FORMAT!r}"
        )
    seq_len = manifest.get("seq_len")
    if not isinstance(seq_len, int) or seq_len < 2:
        problems.append(f"manifest seq_len {seq_len!r} is not a valid integer")
    dtype = manifest.get("dtype")
    if dtype not in VALID_DTYPES:
        problems.append(f"manifest dtype {dtype!r} — expected one of {VALID_DTYPES}")
    eos_id, pad_id = manifest.get("eos_id"), manifest.get("pad_id")
    for label, val in (("eos_id", eos_id), ("pad_id", pad_id)):
        if not isinstance(val, int) or val < 0:
            problems.append(f"manifest {label} {val!r} is not a valid id")

    metadata = manifest.get("metadata") or {}
    packing = metadata.get("packing") or {}
    tok_meta = metadata.get("tokenizer") or {}
    counts = metadata.get("counts") or {}
    vocab_bound = packing.get("vocab_size_bound")
    manifest_tok_sha = tok_meta.get("sha256")

    shards: List[Dict[str, Any]] = list(manifest.get("shards") or [])
    if not shards:
        problems.append("manifest lists no shards — the corpus is empty")
    phase_rows: Dict[str, int] = {}
    missing_files: List[str] = []
    for entry in shards:
        phase, rows = entry.get("phase"), entry.get("rows")
        shard_name = entry.get("shard")
        if shard_name is None:
            problems.append(f"shard entry missing 'shard': {entry!r}")
            continue
        if not isinstance(rows, int) or rows < 1:
            problems.append(
                f"shard {shard_name} rows {rows!r} — expected a positive int"
            )
        if entry.get("tokens_incl_padding") != rows * seq_len:
            problems.append(
                f"shard {shard_name} tokens_incl_padding "
                f"{entry.get('tokens_incl_padding')} != rows*seq "
                f"({rows}*{seq_len}) — inconsistent manifest entry"
            )
        if not os.path.isfile(os.path.join(packed_dir, shard_name)):
            missing_files.append(shard_name)
        phase_rows[phase] = phase_rows.get(phase, 0) + (rows if isinstance(rows, int) else 0)

    if missing_files:
        problems.append(
            "shard file(s) missing on disk: " + ", ".join(missing_files)
        )

    if expected_seq is not None and seq_len != expected_seq:
        problems.append(
            f"manifest seq_len {seq_len} != trained sequence length "
            f"{expected_seq} — the packed rows are {seq_len}-wide; train with "
            f"--seq {seq_len} (or omit --seq)"
        )
    if expected_vocab is not None and vocab_bound is not None:
        if not isinstance(vocab_bound, int) or vocab_bound > expected_vocab:
            problems.append(
                f"manifest vocab_size_bound {vocab_bound} exceeds the model's "
                f"{expected_vocab} — the corpus was packed for a larger vocab"
            )
    if expected_tokenizer_sha256 is not None:
        if not manifest_tok_sha:
            problems.append(
                "manifest records no tokenizer sha256 — cannot verify the "
                "--tokenizer-json identity"
            )
        elif manifest_tok_sha != expected_tokenizer_sha256:
            problems.append(
                f"tokenizer identity mismatch: manifest records sha256 "
                f"{manifest_tok_sha[:12]}… but the provided tokenizer.json "
                f"hashes to {expected_tokenizer_sha256[:12]}… — the rows were "
                "packed with a different tokenizer; refusing to train with "
                "the wrong vocabulary"
            )

    for phase in ("val", "train"):
        recorded = counts.get(f"{phase}_rows")
        if isinstance(recorded, int) and phase_rows.get(phase, 0) != recorded:
            problems.append(
                f"per-phase row mismatch for {phase!r}: manifest entries sum to "
                f"{phase_rows.get(phase, 0)} but metadata records {recorded}"
            )

    if problems:
        raise ValueError(
            f"packed corpus manifest {manifest_path} failed validation: "
            + "; ".join(problems)
        )
    if metadata.get("schema") is None:
        raise ValueError(
            f"packed corpus manifest {manifest_path} has no metadata block — "
            "not a prepare_corpus output"
        )
    return manifest


def manifest_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """A stable identity of the *data* a packed run trained on.

    Used on ``--resume``: the resumed run must point at the same corpus. Paths
    are excluded so a checkout on a different machine (Colab vs CI) still
    matches; everything that changes the token stream is compared.
    """
    metadata = manifest["metadata"]
    packing = metadata["packing"]
    counts = metadata["counts"]
    return {
        "format": manifest["format"],
        "seq_len": manifest["seq_len"],
        "dtype": manifest["dtype"],
        "eos_id": manifest["eos_id"],
        "pad_id": manifest["pad_id"],
        "packing": {
            "seq": packing.get("seq"),
            "vocab_size_bound": packing.get("vocab_size_bound"),
        },
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "counts": {
            k: counts.get(k)
            for k in (
                "train_rows",
                "train_tokens",
                "val_rows",
                "val_tokens",
                "total_rows",
            )
        },
        "num_shards": manifest["num_shards"],
    }


def packed_phase_shard_paths(
    manifest: Dict[str, Any], packed_dir: str, phase: str
) -> List[str]:
    """Every on-disk shard path of one phase, in manifest order."""
    paths: List[str] = []
    for entry in manifest["shards"]:
        if entry["phase"] != phase:
            continue
        paths.append(os.path.join(packed_dir, entry["shard"]))
    return paths


class PackedTokenDataset(IterableDataset):
    """A ``torch`` ``IterableDataset`` over packed ``*.npy`` token shards.

    Rows are taken **verbatim** (they are already exactly ``seq_len`` wide —
    the packing convention of ``scripts.prepare_corpus``); the dataset only
    validates and batches them. Yields ``torch.Tensor`` batches of shape
    ``(batch, seq_len)`` and dtype ``torch.int32`` — the same contract as
    :class:`data.tokenized.StreamingTokenizedDataset` — so the trainer's
    ``validate_token_ids`` + ``x[:, :-1] -> x[:, 1:]`` objective works
    unchanged.

    Args:
        shard_paths: ordered ``*.npy`` shard files of one phase.
        seq_len: row width (must equal every shard's ``shape[1]``).
        batch_size: rows per yielded batch.
        expected_dtype: manifest dtype (``"int32"``/``"uint16"``) — a shard
            whose on-disk dtype differs is rejected (manifest-vs-shard check).
        max_id: hard upper bound on token ids (the model's ``vocab_size``);
            every id must be in ``[0, max_id)`` or the shard fails loudly.
        drop_last: drop a trailing partial batch (default True — keeps every
            yielded batch exactly ``(batch, seq_len)``).
    """

    def __init__(
        self,
        shard_paths: Sequence[str],
        *,
        seq_len: int,
        batch_size: int = 1,
        expected_dtype: str = "int32",
        max_id: Optional[int] = None,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        if not shard_paths:
            raise ValueError("PackedTokenDataset needs at least one shard")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.paths = [str(p) for p in shard_paths]
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.expected_dtype = expected_dtype
        self.max_id = int(max_id) if max_id is not None else None
        self.drop_last = drop_last

    def __iter__(self) -> Iterator["torch.Tensor"]:
        for path in self.paths:
            arr = _load_npy_header(path)
            if arr.dtype != _NP_DTYPES[self.expected_dtype]:
                raise ValueError(
                    f"shard {os.path.basename(path)} dtype {arr.dtype} does "
                    f"not match manifest dtype {self.expected_dtype!r} — "
                    "corrupt or mismatched shard"
                )
            if self.max_id is not None:
                _check_rows_and_ids(
                    arr, path=path, seq_len=self.seq_len, max_id=self.max_id
                )
            else:
                if arr.ndim != 2 or arr.shape[1] != self.seq_len:
                    raise ValueError(
                        f"shard {os.path.basename(path)} has shape {arr.shape} "
                        f"(expected (rows, {self.seq_len}))"
                    )
            for start in range(0, len(arr), self.batch_size):
                n = min(self.batch_size, len(arr) - start)
                if n < self.batch_size and self.drop_last:
                    continue
                # Copy into a fresh writable buffer: views of a mode="r" memmap
                # are read-only, which torch.from_numpy rejects with a warning
                # (and would break any future in-place op on the batch).
                batch = np.empty((n, self.seq_len), dtype=np.int32)
                batch[:] = arr[start : start + n]
                yield torch.from_numpy(batch)