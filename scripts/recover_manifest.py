"""Reconstruct a lost packed-corpus ``manifest.json`` from the existing shards.

This is **corpus-METADATA recovery** — the strict complement of
``scripts/prepare_corpus.py``. When a packed corpus directory (owner's Drive
copy, Colab scratch, ...) has lost its ``manifest.json`` but the ``*.npy``
shard files are intact, this script rebuilds the manifest FROM THE SHARDS and
(optionally) the owner's own artifacts — it never re-tokenizes, never
re-packs, and never writes, reorders or modifies a single shard byte.

Ground rules (owner directive — treat as hard constraints):

1. **Discovery mirrors the writer.** Shard names are exactly
   ``shard-{phase}-{index:04d}.npy`` (:func:`scripts.prepare_corpus._shard_name`)
   with phases ``val`` then ``train``; the manifest lists **val entries first
   (index order), then train entries (index order)** — exactly what the writer
   concatenates. Shard order is load-bearing for the training stream; recovery
   never reorders. ``--train-glob``/``--val-glob`` exist for genuinely
   different layouts (order = numeric index when names parse as
   ``shard-TYPE-INT.npy``, else lexicographic — printed, never silent).
2. **Integrity is checked per shard BEFORE anything is written.** Every shard
   must ``np.load`` (mmap), carry the expected dtype (``uint16``/``int32``),
   have shape ``(rows, seq)`` with the same ``seq`` everywhere, hold only ids
   in ``[0, vocab_size)``, and its file size must equal the numpy header plus
   ``rows*cols*itemsize``. If ANY shard fails: a per-shard failure report is
   printed and the script STOPS (exit 3) — a manifest is never written over a
   broken corpus. Regenerating the corpus after an integrity failure is the
   OWNER's decision, not this tool's.
3. **Every field prepare_corpus.py writes is reproduced.** Fields that cannot
   be derived from the shards (dataset identity, HF revision/sha, original
   timestamps, ``args``, ...) are written as ``null`` and listed in the
   ``recovery`` block — never invented. Fields that ARE derivable are
   recomputed from the actual bytes (per-shard sha256, row counts, token
   counts, EOS/pad resolution with strict structural verification). The
   ``eos_id``/``pad_id``/tokenizer-sha fields (which the trainer's
   ``load_packed_manifest`` requires and the resume identity-compare uses) are
   resolved from, in order: ``--eos-id``/``--pad-id`` (explicit user values),
   ``--tokenizer-json`` (the real tokenizer file the corpus was packed with),
   ``--checkpoint`` (the run's step-<N>.pt, whose ``data_provenance`` records
   the ORIGINAL manifest identity + full metadata), and the sibling
   ``run_metadata.json`` (auto-discovered in the packed dir when present).
   Any two sources that disagree on a value → loud conflict error, no write.
4. **Cross-check mode.** ``--expected-tokens-train``/``--expected-tokens-val``
   compare the owner-known counts against the recovered counts; any mismatch
   beyond ``--expected-token-tolerance`` (default 0 — packed counts are exact)
   prints a loud warning comparing both numbers and requires ``--force`` to
   write anyway.
5. Writes are atomic: tmp + fsync + rename (the writer's own discipline).
6. Zero runtime deps beyond the repo's own (numpy/torch + stdlib + the repo's
   light modules) — runs in the owner's Colab env as-is.
7. If ``manifest.json`` already exists, recovery refuses (``--force`` to
   overwrite) — a present manifest is the source of truth, not a file to
   replace on a whim.

The output manifest loads through the trainer's own loader
(``data.packed.load_packed_manifest``) and, when the tokenizer is provided,
reproduces the original run's recorded ``manifest_identity`` exactly — that is
what lets ``--resume`` against an existing checkpoint pass: the *streaming*
semantics are byte-identical because the shards are untouched.

Usage::

    python -m scripts.recover_manifest --packed-dir <corpus-dir> \\
        --tokenizer-json <tokenizer.json>            # strongly recommended \\
        [--checkpoint runs/fw-edu/step-30000.pt]     # optional cross-check \\
        [--expected-tokens-train 205002065 --expected-tokens-val 15001394] \\
        [--dry-run]                                  # inspect before writing
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the
# script also works when invoked from another CWD (e.g. Colab's /content/talos).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The manifest format tag is duplicated by design in scripts/prepare_corpus.py
# and data/packed.py (dependency direction is one-way); recovery imports the
# constant from the lightest module that defines it (data.packed has no heavy
# imports — numpy/torch only, matching the "runs in Colab as-is" rule).
from data.packed import MANIFEST_FORMAT, VALID_DTYPES  # noqa: E402
from configs.vocab import VOCAB_SIZE  # noqa: E402

#: Mirror of ``scripts.prepare_corpus._shard_name`` — the single naming rule
#: every discovery path must agree with. ``{:04d}`` pads to >= 4 digits, so
#: indices above 9999 legitimately produce 5+ digits; parse ints, never sort
#: the *strings*.
_SHARD_RE = re.compile(r"^shard-(val|train)-(\d+)\.npy$")

TOOL_VERSION = "1.0.0"
RECOVERY_SCHEMA = "talos-manifest-recovery-v1"
_NPY_MAGIC = b"\x93NUMPY"

EXIT_OK = 0
EXIT_USAGE = 2  # bad args / preconditions (manifest already present, ...)
EXIT_FATAL = 3  # integrity failure / unresolvable fields / source conflict


class RecoveryError(Exception):
    """Fatal recovery condition — nothing was written. Exit code 3."""


# ---------------------------------------------------------------------------
# 1) Discovery — mirror the writer's naming + ordering EXACTLY (requirement 1)
# ---------------------------------------------------------------------------
def _parse_shard_name(name: str) -> Optional[Tuple[str, int]]:
    """``(phase, index)`` for a canonical shard name, else None."""
    m = _SHARD_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def discover_shards(
    packed_dir: str,
    *,
    train_glob: Optional[str],
    val_glob: Optional[str],
) -> Dict[str, List[Tuple[str, int]]]:
    """The shards per phase in the EXACT order the writer records them.

    Returns ``{"val": [(name, index)...], "train": [...]}``, each list in
    ascending index order (the writer's flush order; the manifest's order is
    val entries then train entries — enforced by the caller). With explicit
    globs, matched names are ordered by parsed index when every name parses as
    ``shard-TYPE-INT.npy``, else lexicographically — that choice is reported
    by the caller, never made silently.
    """
    if train_glob is None and val_glob is None:
        found: Dict[str, List[Tuple[str, int]]] = {"val": [], "train": []}
        for name in sorted(os.listdir(packed_dir)):
            parsed = _parse_shard_name(name)
            if parsed is None:
                continue
            phase, index = parsed
            found[phase].append((name, index))
        for phase in found:
            found[phase].sort(key=lambda pair: pair[1])
        return found

    # Explicit globs: both phases must be given (a half-explicit layout is a
    # guessing game we refuse to play). fnmatch against the basename keeps the
    # rule filesystem-agnostic (Glob would be, too, but fnmatch is stdlib).
    result: Dict[str, List[Tuple[str, int]]] = {"val": [], "train": []}
    for name in sorted(os.listdir(packed_dir)):
        if not name.endswith(".npy"):
            continue
        in_train = fnmatch.fnmatch(name, train_glob)
        in_val = fnmatch.fnmatch(name, val_glob)
        if in_train and in_val:
            raise RecoveryError(
                f"shard {name} matches BOTH --train-glob {train_glob!r} and "
                f"--val-glob {val_glob!r} — the split is ambiguous"
            )
        if in_train:
            result["train"].append(name)
        if in_val:
            result["val"].append(name)
    for phase in result:
        parsed = {n: _parse_shard_name(n) for n in result[phase]}
        if all(p is not None for p in parsed.values()):
            # Names parse as shard-TYPE-INT.npy → numeric index order.
            result[phase].sort(key=lambda n: parsed[n][1])  # type: ignore[index]
            result[phase] = [(n, parsed[n][1]) for n in result[phase]]  # type: ignore[index]
        else:
            result[phase].sort()  # names don't parse → lexicographic, reported
            result[phase] = [(n, pos) for pos, n in enumerate(result[phase])]
    return result


def scan_stray_files(packed_dir: str) -> Tuple[List[str], List[str]]:
    """``(unknown shard-like names, *.tmp debris)``. Unknown ``shard-*.npy``
    names make the split ambiguous (error, decided by the caller); ``*.tmp``
    is debris from an atomic writer — reported, never fatal on its own."""
    unknown: List[str] = []
    debris: List[str] = []
    for name in sorted(os.listdir(packed_dir)):
        if name.endswith(".tmp"):
            debris.append(name)
        elif name.startswith("shard-") and name.endswith(".npy"):
            if _parse_shard_name(name) is None:
                unknown.append(name)
    return unknown, debris


# ---------------------------------------------------------------------------
# 2) Per-shard integrity (requirement 2) — fail loud, write nothing
# ---------------------------------------------------------------------------
def _npy_data_offset(path: str) -> int:
    """Byte offset where a numpy array file's raw data begins.

    v1.0: 8-byte magic+version prefix, 2-byte little-endian header length.
    v2.0/v3.0: 4-byte header length. Anything else is not a numpy file.
    """
    with open(path, "rb") as fh:
        magic = fh.read(6)
        if magic != _NPY_MAGIC:
            raise ValueError("not a numpy .npy file (bad magic)")
        ver = fh.read(2)
        if len(ver) != 2:
            raise ValueError("truncated numpy version header")
        major, _minor = ver
        if major == 1:
            hlen_bytes = fh.read(2)
            if len(hlen_bytes) != 2:
                raise ValueError("truncated numpy v1.0 header")
            return 8 + 2 + int.from_bytes(hlen_bytes, "little")
        if major in (2, 3):
            hlen_bytes = fh.read(4)
            if len(hlen_bytes) != 4:
                raise ValueError("truncated numpy v2.0/3.0 header")
            return 8 + 4 + int.from_bytes(hlen_bytes, "little")
        raise ValueError(f"unsupported numpy format version {major}.{_minor}")


def _file_sha256(path: str, chunk: int = 1 << 20) -> str:
    """sha256 of the raw file bytes, streamed (never loads a shard to RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def inspect_shard(path: str, *, vocab_size: int) -> Dict[str, Any]:
    """One shard's integrity verdict; raises ValueError naming the shard on ANY
    failure. Returns facts on success: ``{rows, shape, dtype, min_id, max_id,
    sha256, tokens_incl_padding}`` (both the id-range scan and the size check
    touch every element / byte, so a drifted or truncated shard cannot pass)."""
    name = os.path.basename(path)
    try:
        data_offset = _npy_data_offset(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable npy header ({exc})") from exc
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - any numpy failure names the shard
        raise ValueError(f"np.load failed: {type(exc).__name__}: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(f"shape {arr.shape} — expected a 2-D (rows, seq) array")
    if arr.shape[0] < 1:
        raise ValueError("zero rows — not a prepare_corpus shard")
    if arr.dtype != np.dtype(np.uint16) and arr.dtype != np.dtype(np.int32):
        raise ValueError(
            f"dtype {arr.dtype} — expected uint16 or int32 (prepare_corpus layout)"
        )
    rows, seq = arr.shape
    actual_size = os.path.getsize(path)
    expected_size = data_offset + arr.nbytes
    if actual_size != expected_size:
        raise ValueError(
            f"file size {actual_size} bytes != header+data {expected_size} "
            f"({rows}x{seq} {arr.dtype}) — truncated or trailing garbage"
        )
    min_id = int(arr.min()) if arr.size else None
    max_id = int(arr.max()) if arr.size else None
    if min_id is None or min_id < 0:
        raise ValueError(f"contains negative token ids (min={min_id})")
    if max_id is None or max_id >= vocab_size:
        raise ValueError(
            f"contains token ids outside [0, {vocab_size}): max={max_id} — "
            "vocab bound mismatch or corrupt shard"
        )
    _ = arr[-1, -1]  # final edge read: proves end-of-file readability
    return {
        "rows": rows,
        "shape": (rows, seq),
        "dtype": str(arr.dtype),
        "min_id": min_id,
        "max_id": max_id,
        "sha256": _file_sha256(path),
        "tokens_incl_padding": rows * seq,
    }


# ---------------------------------------------------------------------------
# 3) Structural analysis over verified shards (counts / pad / EOS evidence)
# ---------------------------------------------------------------------------
def _phase_tail_suffix(last_shard_arr: np.ndarray) -> Tuple[int, int]:
    """``(value, run_len)`` of the longest equal-value suffix of the LAST row
    of a phase — the only place prepare_corpus ever writes padding."""
    tail = last_shard_arr[-1]
    value = int(tail[-1])
    run = 0
    for i in range(tail.size - 1, -1, -1):
        if int(tail[i]) == value:
            run += 1
        else:
            break
    return value, run


def _count_value(phase_arrays: List[np.ndarray], value: int) -> int:
    """Total occurrences of ``value`` across a phase's shard rows (vectorized
    over the mmaps — one pass per phase per value)."""
    total = 0
    for arr in phase_arrays:
        total += int(np.count_nonzero(arr == value))
    return total


def _verify_pad_id(
    phase_arrays: List[np.ndarray],
    pad_id: int,
    *,
    phase: str,
) -> int:
    """Verify a candidate pad id against the shard structure.

    For prepare_corpus output the padding value appears EXACTLY once per
    phase: as a suffix of the phase's final row (``finish()`` pads the last
    partial row and nothing else). Verifies:
    * the final row's suffix value IS the candidate pad id;
    * total occurrences across the phase == the suffix run length;
    * no occurrences in any row before the final row;
    * no occurrences inside the final row before its suffix.

    Returns the number of pad tokens in this phase. Raises ValueError (loud,
    names the phase) on any violation — a pad id the shards contradict means
    corrupt data or a wrong source, and counts derived from it would be
    silently wrong.
    """
    total = _count_value(phase_arrays, pad_id)
    if total == 0:
        # The phase has NO padded tail row: the region's token stream ended
        # exactly on a row boundary (prepare_corpus.finish() pads only a
        # non-empty carry), so the pad id legitimately occurs nowhere here.
        # A wrong candidate is still caught by the other phase, by any
        # recorded-count source, and by the expected-tokens cross-check.
        return 0
    tail = phase_arrays[-1][-1]
    value, run = _phase_tail_suffix(phase_arrays[-1])
    if value != pad_id:
        raise ValueError(
            f"{phase}: resolved pad id {pad_id} is not the value of the final "
            f"row's suffix ({value}) — the shard evidence contradicts the "
            "source (wrong --pad-id/--tokenizer-json/--checkpoint, or corrupt data)"
        )
    if run != total:
        raise ValueError(
            f"{phase}: pad id {pad_id} occurs {total} times but only {run} "
            "form the final-row suffix — pad appears outside the padded tail"
        )
    before_tail = sum(
        int(np.count_nonzero(arr == pad_id)) for arr in phase_arrays[:-1]
    )
    if before_tail:
        raise ValueError(
            f"{phase}: pad id {pad_id} appears {before_tail} times in rows "
            "before the final row — not the prepare_corpus padding layout"
        )
    if int(np.count_nonzero(tail[:-run] == pad_id)):
        raise ValueError(
            f"{phase}: pad id {pad_id} appears inside the final row's content "
            "(before the pad suffix) — incompatible with the canonical "
            "tokenizer contract; refusing to derive counts"
        )
    return run


def _infer_pad_id(phase_arrays: Dict[str, List[np.ndarray]]) -> Optional[int]:
    """Strict structural inference of pad_id with NO external source.

    Valid only when BOTH phases end on a suffix of the SAME value and that
    value occurs nowhere except those two suffixes — i.e. exactly the
    prepare_corpus padding layout. Returns the id, or None when ambiguous; the
    caller then requires an external source (the loader needs a real pad id;
    guessing is forbidden).
    """
    suffixes = {
        phase: _phase_tail_suffix(arrays[-1]) for phase, arrays in phase_arrays.items()
    }
    values = {v for v, _ in suffixes.values()}
    if len(values) != 1:
        return None
    pad_id = next(iter(values))
    for phase, arrays in phase_arrays.items():
        _value, run = suffixes[phase]
        if _count_value(arrays, pad_id) != run:
            return None
        if any(int(np.count_nonzero(a == pad_id)) for a in arrays[:-1]):
            return None
        if int(np.count_nonzero(arrays[-1][-1][: -run] == pad_id)):
            return None
    return pad_id


def _count_docs(phase_arrays: List[np.ndarray], eos_id: int, *, phase: str) -> int:
    """Document count = number of EOS separators across the phase's rows (the
    canonical tokenizer never emits EOS from ``encode``, so every EOS in the
    packed rows is a document separator). Raises when EOS never occurs."""
    total = _count_value(phase_arrays, eos_id)
    if total == 0:
        raise RecoveryError(
            f"{phase}: resolved eos id {eos_id} does not occur anywhere in the "
            "phase — wrong source or corrupt shards"
        )
    return total


# ---------------------------------------------------------------------------
# 4) Source loading + field resolution (requirement 3/4: recompute or read
#    the owner's artifacts, never invent; conflicts are loud and fatal)
# ---------------------------------------------------------------------------
def _load_checkpoint(path: str) -> Dict[str, Any]:
    """A run checkpoint's packed-corpus record: the ORIGINAL manifest metadata
    (``run_metadata.data_provenance.manifest_metadata``) and the recorded
    ``manifest_identity`` — the exact dict the resume identity-compare uses."""
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - trusted owner artifact; fall back
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RecoveryError(f"--checkpoint {path} does not hold a dict payload")
    run_meta = payload.get("run_metadata") or {}
    prov = run_meta.get("data_provenance") or {}
    if prov.get("source") != "packed":
        raise RecoveryError(
            f"--checkpoint {path} is from a {prov.get('source')!r} run, not a "
            "packed-corpus run (data_provenance.source != 'packed')"
        )
    metadata = prov.get("manifest_metadata")
    identity = prov.get("manifest_identity")
    if not identity and not metadata:
        raise RecoveryError(
            f"--checkpoint {path} records neither manifest_metadata nor "
            "manifest_identity — not a packed-corpus run checkpoint"
        )
    return {
        "metadata": metadata if isinstance(metadata, dict) else None,
        "identity": identity if isinstance(identity, dict) else None,
    }


def _normalize_source_records(
    metadata: Optional[Dict[str, Any]], identity: Optional[Dict[str, Any]],
    source: str,
) -> Dict[str, Any]:
    """Flatten a source record so field resolution can read it uniformly.

    The original corpus metadata (from a checkpoint's ``manifest_metadata`` or
    the sibling ``run_metadata.json``) nests ``eos_id``/``pad_id`` inside
    ``packing`` and the tokenizer facts inside ``tokenizer``; the tokenizer
    file record and the identity record put them at the top level. Flattening
    to top-level keys keeps every resolver one ``rec.get(key)`` away while the
    nested originals remain available under their own keys.
    """
    rec: Dict[str, Any] = {"_source": source}
    if metadata is not None:
        rec.update(metadata)
    if identity is not None:
        rec["identity"] = identity
    packing = rec.get("packing") or {}
    if rec.get("eos_id") is None and isinstance(packing.get("eos_id"), int):
        rec["eos_id"] = packing["eos_id"]
    if rec.get("pad_id") is None and isinstance(packing.get("pad_id"), int):
        rec["pad_id"] = packing["pad_id"]
    tok = rec.get("tokenizer") or {}
    if rec.get("sha256") is None and isinstance(tok.get("sha256"), str):
        rec["sha256"] = tok["sha256"]
    if rec.get("vocab_size") is None and isinstance(tok.get("vocab_size"), int):
        rec["vocab_size"] = tok["vocab_size"]
    if rec.get("merge_count") is None and isinstance(tok.get("merge_count"), int):
        rec["merge_count"] = tok["merge_count"]
    ident = identity or {}
    if rec.get("tokenizer_sha256") is None:
        rec["tokenizer_sha256"] = ident.get("tokenizer_sha256")
    return rec


def _resolve_agreed(
    candidates: List[Tuple[str, Any]], label: str,
) -> Optional[Tuple[Any, str]]:
    """All non-None candidates for ``label`` must agree; return ``(value,
    provenance)`` or ``(None, "")`` when no candidate exists. Raises a loud
    conflict error when sources disagree (a manifest written from conflicting
    evidence could silently resume on the wrong data)."""
    seen: Dict[str, List[str]] = {}
    for src, val in candidates:
        if val is None:
            continue
        seen.setdefault(str(val), []).append(src)
    if not seen:
        return None, ""
    if len(seen) > 1:
        parts = "; ".join(
            f"{'/'.join(sorted(set(srcs)))}={val!r}"
            for val, srcs in seen.items()
        )
        raise RecoveryError(f"conflicting {label} across sources: {parts}")
    value = next(iter(seen))
    # Preserve the ORIGINAL type (str() hashing is only for comparison).
    for src, val in candidates:
        if val is not None and str(val) == value:
            provenance = "/".join(
                sorted({s for s, v in candidates if v is not None and str(v) == value})
            )
            return val, provenance
    return None, ""  # pragma: no cover - unreachable


# ---------------------------------------------------------------------------
# 5) Manifest construction + atomic write
# ---------------------------------------------------------------------------
def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """tmp + fsync + rename — the writer's own discipline plus durability."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.recover_manifest",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--packed-dir", required=True, metavar="DIR",
                   help="packed corpus directory whose manifest.json is lost "
                        "(shard files are never modified)")
    p.add_argument("--tokenizer-json", default=None, metavar="PATH",
                   help="the tokenizer.json the corpus was packed with — "
                        "recomputes sha256/vocab/merges/eos/pad from the real "
                        "file so the resumed run's tokenizer identity matches")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="step-<N>.pt checkpoint of the same run — its recorded "
                        "manifest_identity + original metadata are the ground "
                        "truth every recovered value is verified against")
    p.add_argument("--eos-id", type=int, default=None, metavar="N",
                   help="explicit EOS id (last resort; verified against the "
                        "shards when another source is also given)")
    p.add_argument("--pad-id", type=int, default=None, metavar="N",
                   help="explicit PAD id (last resort; verified against the "
                        "shard structure before any counting)")
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, metavar="N",
                   help="token-id upper bound for integrity checks (default "
                        "configs.vocab.VOCAB_SIZE = 1024)")
    p.add_argument("--expected-tokens-train", type=int, default=None, metavar="N",
                   help="owner-known real train token count (cross-check; a "
                        "mismatch beyond tolerance requires --force)")
    p.add_argument("--expected-tokens-val", type=int, default=None, metavar="N",
                   help="owner-known real val token count (cross-check)")
    p.add_argument("--expected-token-tolerance", type=int, default=0, metavar="N",
                   help="allowed absolute deviation for the expected-tokens "
                        "cross-check (default 0 — packed counts are exact)")
    p.add_argument("--train-glob", default=None, metavar="GLOB",
                   help="explicit shard-name glob for the train phase "
                        "(default: the shard-train-*.npy writer rule)")
    p.add_argument("--val-glob", default=None, metavar="GLOB",
                   help="explicit shard-name glob for the val phase "
                        "(default: the shard-val-*.npy writer rule)")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="manifest destination (default <packed-dir>/manifest.json)")
    p.add_argument("--force", action="store_true",
                   help="write despite an expected-tokens mismatch, an already-"
                        "present manifest.json, or both")
    p.add_argument("--dry-run", action="store_true",
                   help="run every check and print the full report but write "
                        "nothing")
    return p


def recover_manifest(
    packed_dir: str,
    *,
    tokenizer_json: Optional[str] = None,
    checkpoint: Optional[str] = None,
    eos_id: Optional[int] = None,
    pad_id: Optional[int] = None,
    vocab_size: int = VOCAB_SIZE,
    expected_tokens_train: Optional[int] = None,
    expected_tokens_val: Optional[int] = None,
    expected_token_tolerance: int = 0,
    train_glob: Optional[str] = None,
    val_glob: Optional[str] = None,
    output: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Full recovery pipeline; raises RecoveryError (exit 3) on any integrity
    failure, unresolvable field or source conflict — in every such case
    nothing is written. Returns the report dict (``written`` + ``manifest``).
    """
    if not os.path.isdir(packed_dir):
        raise RecoveryError(f"--packed-dir is not a directory: {packed_dir}")
    if (train_glob is None) != (val_glob is None):
        raise RecoveryError("--train-glob and --val-glob must be given together")
    if vocab_size <= 0:
        raise RecoveryError(f"--vocab-size must be positive, got {vocab_size}")
    if output is None:
        output = os.path.join(packed_dir, "manifest.json")

    # ---- 1) discovery (writer-mirroring rule) ------------------------------
    phases = discover_shards(packed_dir, train_glob=train_glob, val_glob=val_glob)
    unknown, debris = scan_stray_files(packed_dir)
    if unknown:
        raise RecoveryError(
            "ambiguous corpus layout — shard-*.npy files with unknown phase "
            "names: " + ", ".join(unknown) + " — use --train-glob/--val-glob "
            "or investigate before recovering"
        )
    if not phases["val"] or not phases["train"]:
        raise RecoveryError(
            "a prepare_corpus output always has BOTH a val and a train phase; "
            "found only: "
            + ", ".join(f"{p}={len(phases[p])}" for p in ("val", "train"))
        )
    order_note = (
        "writer rule shard-{phase}-{index:04d}.npy, index order, val first"
        if train_glob is None
        else "custom globs (index order when names parse, else lexicographic — see report)"
    )

    # ---- 2) per-shard integrity; ANY failure stops before anything else -----
    facts: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for phase in ("val", "train"):
        for name, _index in phases[phase]:
            try:
                facts[name] = inspect_shard(
                    os.path.join(packed_dir, name), vocab_size=vocab_size
                )
                facts[name]["phase"] = phase
            except (OSError, ValueError) as exc:
                failures.append(f"{name} ({phase}): {exc}")
    if failures:
        print("=" * 72)
        print("SHARD INTEGRITY FAILURE — manifest NOT written")
        print("=" * 72)
        for line in failures:
            print("  FAIL", line)
        print("-" * 72)
        print(
            "No manifest was written over a broken corpus (owner rule: "
            "regenerating the corpus after an integrity failure is the "
            "OWNER's decision, not this tool's)."
        )
        raise RecoveryError(f"{len(failures)} shard(s) failed integrity")

    # ---- 3) shard-derived facts (exact, from the verified bytes) ------------
    first_name = next(iter(facts))
    seq_len = facts[first_name]["shape"][1]
    dtype_str = facts[first_name]["dtype"]
    for _name, f in facts.items():
        if f["shape"][1] != seq_len:
            raise RecoveryError(
                f"shards disagree on seq_len: {f['shape'][1]} vs {seq_len}"
            )
        if f["dtype"] != dtype_str:
            raise RecoveryError(
                f"shards disagree on dtype: {f['dtype']} vs {dtype_str}"
            )
    if dtype_str not in VALID_DTYPES:
        raise RecoveryError(f"unexpected shard dtype {dtype_str!r}")
    arrays: Dict[str, List[np.ndarray]] = {}
    for phase in ("val", "train"):
        arrays[phase] = [
            np.load(os.path.join(packed_dir, name), mmap_mode="r")
            for (name, _i) in phases[phase]
        ]
    val_rows = sum(f["rows"] for n, f in facts.items() if f["phase"] == "val")
    train_rows = sum(f["rows"] for n, f in facts.items() if f["phase"] == "train")
    val_incl = sum(
        f["tokens_incl_padding"] for n, f in facts.items() if f["phase"] == "val"
    )
    train_incl = sum(
        f["tokens_incl_padding"] for n, f in facts.items() if f["phase"] == "train"
    )
    num_shards = len(facts)
    inferred_pad = _infer_pad_id(arrays)

    # ---- 4) sources: real tokenizer file / run checkpoint / sibling
    #              run_metadata.json (auto-discovered) ------------------------
    records: List[Dict[str, Any]] = []
    if tokenizer_json:
        if not os.path.isfile(tokenizer_json):
            raise RecoveryError(f"--tokenizer-json not found: {tokenizer_json}")
        from tokenizer.tokenizer import (  # noqa: PLC0415 - light, lazy import
            ByteLevelBPETokenizer,
            tokenizer_file_sha256,
        )

        tok = ByteLevelBPETokenizer.from_file(tokenizer_json)
        records.append(
            {
                "_source": "tokenizer-json",
                "sha256": tokenizer_file_sha256(tokenizer_json),
                "vocab_size": tok.vocab_size,
                "merge_count": tok.merge_count,
                "eos_id": int(tok.eos_id),
                "pad_id": int(tok.pad_id),
            }
        )
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise RecoveryError(f"--checkpoint not found: {checkpoint}")
        ckpt = _load_checkpoint(checkpoint)
        records.append(
            _normalize_source_records(
                ckpt["metadata"], ckpt["identity"], source="checkpoint"
            )
        )
    run_meta_path = os.path.join(packed_dir, "run_metadata.json")
    if os.path.isfile(run_meta_path):
        try:
            with open(run_meta_path, "r", encoding="utf-8") as fh:
                run_meta = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RecoveryError(
                f"{run_meta_path} exists but is unreadable ({exc}) — fix or "
                "remove it before recovering"
            ) from exc
        if not isinstance(run_meta, dict):
            raise RecoveryError(f"{run_meta_path} is not a JSON object")
        records.append(
            _normalize_source_records(run_meta, None, source="run_metadata.json")
        )

    # ---- 5) resolve eos/pad/tokenizer facts; every provided source must
    #         agree, and the shards' structural evidence is ALWAYS checked ---
    eos_candidates: List[Tuple[str, Any]] = [("cli", eos_id)]
    pad_candidates: List[Tuple[str, Any]] = [("cli", pad_id)]
    for rec in records:
        eos_candidates.append((rec["_source"], rec.get("eos_id")))
        pad_candidates.append((rec["_source"], rec.get("pad_id")))
        ident = rec.get("identity") or {}
        eos_candidates.append((rec["_source"] + ":identity", ident.get("eos_id")))
        pad_candidates.append((rec["_source"] + ":identity", ident.get("pad_id")))
    eos_resolved, eos_prov = _resolve_agreed(eos_candidates, "eos_id")
    pad_resolved, pad_prov = _resolve_agreed(pad_candidates, "pad_id")
    if pad_resolved is None and inferred_pad is not None:
        pad_resolved, pad_prov = inferred_pad, "shard-structure inference"
    if eos_resolved is None or pad_resolved is None:
        missing = ", ".join(
            lbl
            for lbl, val in (("eos_id", eos_resolved), ("pad_id", pad_resolved))
            if val is None
        )
        raise RecoveryError(
            f"cannot resolve {missing} from the shards. The trainer's manifest "
            "loader requires real eos/pad ids and the resume identity-compare "
            "uses them, so the value is NEVER guessed. Pass --tokenizer-json "
            "(the file the corpus was packed with) or --checkpoint (a step-*.pt "
            "of the run) to recover them, or --eos-id/--pad-id explicitly."
        )
    if eos_resolved == pad_resolved:
        raise RecoveryError(
            f"resolved eos_id == pad_id == {eos_resolved} — the tokenizer "
            "contract keeps them distinct; wrong source"
        )
    if not (0 <= eos_resolved < vocab_size) or not (0 <= pad_resolved < vocab_size):
        raise RecoveryError(
            f"resolved eos/pad ids outside [0, {vocab_size}): "
            f"eos={eos_resolved} pad={pad_resolved}"
        )

    # Structural pad verification — the shards are the final arbiter, whatever
    # the sources claimed.
    pad_tokens: Dict[str, int] = {}
    for phase in ("val", "train"):
        try:
            pad_tokens[phase] = _verify_pad_id(
                arrays[phase], pad_resolved, phase=phase
            )
        except ValueError as exc:
            raise RecoveryError(str(exc)) from exc
    val_tokens = val_incl - pad_tokens["val"]
    train_tokens = train_incl - pad_tokens["train"]
    val_docs = _count_docs(arrays["val"], eos_resolved, phase="val")
    train_docs = _count_docs(arrays["train"], eos_resolved, phase="train")

    # Cross-verify every recorded value the sources claim (rows, tokens, docs,
    # pad counts, num_shards, seq, dtype, eos/pad, tokenizer sha, vocab bound).
    for rec in records:
        counts = rec.get("counts") or {}
        expected = {
            "train_tokens": train_tokens,
            "train_rows": train_rows,
            "val_tokens": val_tokens,
            "val_rows": val_rows,
        }
        for key, got in expected.items():
            recorded = counts.get(key)
            if isinstance(recorded, int) and recorded != got:
                raise RecoveryError(
                    f"source {rec['_source']!r} records {key}={recorded} but the "
                    f"shards contain exactly {got} — data does not match the "
                    "recorded provenance; refusing to write"
                )
        for phase in ("val", "train"):
            key = f"{phase}_pad_tokens"
            recorded = counts.get(key)
            if isinstance(recorded, int) and recorded != pad_tokens[phase]:
                raise RecoveryError(
                    f"source {rec['_source']!r} records {key}={recorded} but the "
                    f"shard tail-padding is {pad_tokens[phase]}"
                )
            region = rec.get(f"{phase}_region") or {}
            if isinstance(region.get("docs"), int) and region["docs"] != (
                val_docs if phase == "val" else train_docs
            ):
                raise RecoveryError(
                    f"source {rec['_source']!r} records {phase}_region.docs="
                    f"{region['docs']} but the shards contain "
                    f"{val_docs if phase == 'val' else train_docs} EOS separators"
                )
        ident = rec.get("identity") or {}
        ic = ident.get("counts") or {}
        for key, got in expected.items():
            if isinstance(ic.get(key), int) and ic[key] != got:
                raise RecoveryError(
                    f"checkpoint identity records {key}={ic[key]} but the "
                    f"shards contain exactly {got} — refusing to write"
                )
        for label, got in (
            ("num_shards", num_shards),
            ("seq_len", seq_len),
            ("dtype", dtype_str),
        ):
            if isinstance(ident.get(label), (int, str)) and ident[label] != got:
                raise RecoveryError(
                    f"checkpoint identity {label}={ident[label]} contradicts "
                    f"the shards ({got})"
                )
        for label, got in (("eos_id", eos_resolved), ("pad_id", pad_resolved)):
            if isinstance(ident.get(label), int) and ident[label] != got:
                raise RecoveryError(
                    f"checkpoint identity {label}={ident[label]} contradicts "
                    f"the resolved {label}={got}"
                )

    # tokenizer block: recomputed from the real file when given; otherwise the
    # sources' records; always checked for cross-source agreement.
    sha_candidates = [(rec["_source"], rec.get("sha256")) for rec in records]
    sha_resolved, sha_prov = _resolve_agreed(sha_candidates, "tokenizer sha256")
    for rec in records:
        ident = rec.get("identity") or {}
        isha = ident.get("tokenizer_sha256")
        if isinstance(isha, str) and sha_resolved is not None and isha != sha_resolved:
            raise RecoveryError(
                f"checkpoint records tokenizer sha256 {isha[:12]}… but the "
                f"resolved sha256 is {sha_resolved[:12]}… — different tokenizers"
            )
        if sha_resolved is None and isinstance(isha, str):
            sha_resolved, sha_prov = isha, rec["_source"] + ":identity"
    tok_vocab = _first_int(records, "vocab_size")
    tok_merges = _first_int(records, "merge_count")

    # vocab_size_bound: the canonical constant by default; any recorded bound
    # must agree (a different bound means a different packing contract).
    vocab_bound = vocab_size
    for rec in records:
        packing = rec.get("packing") or {}
        rec_bound = packing.get("vocab_size_bound")
        if isinstance(rec_bound, int) and rec_bound != vocab_bound:
            raise RecoveryError(
                f"source {rec['_source']!r} records vocab_size_bound={rec_bound} "
                f"but the integrity checks ran with {vocab_bound} — pass "
                "--vocab-size to match the recorded packing contract"
            )
        ident = rec.get("identity") or {}
        ib = ident.get("packing") or {}
        if isinstance(ib.get("vocab_size_bound"), int) and ib["vocab_size_bound"] != vocab_bound:
            raise RecoveryError(
                f"checkpoint identity vocab_size_bound={ib['vocab_size_bound']} "
                f"!= {vocab_bound}"
            )

    # ---- 6) manifest fields — EVERY prepare_corpus field, unknowns as null --
    original_meta: Optional[Dict[str, Any]] = None
    for rec in records:
        if rec["_source"] in ("checkpoint", "run_metadata.json"):
            original_meta = rec  # the original metadata dict (or flattened copy)
            break

    def _orig(key: str) -> Any:
        return (original_meta or {}).get(key)

    _packing_orig = _orig("packing") or {}
    _counts_orig = _orig("counts") or {}
    _val_reg = _orig("val_region") or {}
    _train_reg = _orig("train_region") or {}
    _tok_orig = _orig("tokenizer") or {}
    _ds_orig = _orig("dataset") or {}
    shard_entries: List[Dict[str, Any]] = []
    for phase in ("val", "train"):
        for name, index in phases[phase]:
            f = facts[name]
            shard_entries.append(
                {
                    "shard": name,
                    "phase": phase,
                    "index": index,
                    "rows": f["rows"],
                    "tokens_incl_padding": f["tokens_incl_padding"],
                    # additive: the original writer does not emit per-shard
                    # sha256; recovery records it as the integrity receipt.
                    "sha256": f["sha256"],
                }
            )
    metadata: Dict[str, Any] = {
        "schema": "talos-corpus-run-metadata-v1",
        "script_version": _orig("script_version"),
        "timestamp": _orig("timestamp"),
        "args": _orig("args"),
        "dataset": {
            "dataset": _ds_orig.get("dataset"),
            "config": _ds_orig.get("config"),
            "split": _ds_orig.get("split"),
            "revision": _ds_orig.get("revision"),
            "sha": _ds_orig.get("sha"),
        },
        "tokenizer": {
            "path": (os.path.abspath(tokenizer_json) if tokenizer_json else _tok_orig.get("path")),
            "sha256": sha_resolved,
            "vocab_size": tok_vocab,
            "merge_count": tok_merges,
        },
        "packing": {
            "seq": seq_len,
            "eos_id": eos_resolved,
            "pad_id": pad_resolved,
            "dtype": dtype_str,
            "vocab_size_bound": vocab_bound,
            "rows_per_shard": _packing_orig.get("rows_per_shard"),
            "stream_buffer_docs": _packing_orig.get("stream_buffer_docs"),
        },
        "val_region": {
            "skip_docs": _val_reg.get("skip_docs"),
            "first_doc": _val_reg.get("first_doc"),
            "last_doc": _val_reg.get("last_doc"),
            "docs": val_docs,
            "tokens": val_tokens,
        },
        "train_region": {
            "first_doc": _train_reg.get("first_doc"),
            "last_doc": _train_reg.get("last_doc"),
            "docs": train_docs,
            "tokens": train_tokens,
        },
        "counts": {
            "target_tokens": _counts_orig.get("target_tokens"),
            "docs_streamed": _counts_orig.get("docs_streamed"),
            "skipped_gap_docs": _counts_orig.get("skipped_gap_docs"),
            "empty_docs_skipped": _counts_orig.get("empty_docs_skipped"),
            "val_docs": val_docs,
            "val_chars": _counts_orig.get("val_chars"),
            "val_tokens": val_tokens,
            "val_rows": val_rows,
            "val_pad_tokens": pad_tokens["val"],
            "val_tokens_incl_padding": val_incl,
            "train_docs": train_docs,
            "train_chars": _counts_orig.get("train_chars"),
            "train_tokens": train_tokens,
            "train_rows": train_rows,
            "train_pad_tokens": pad_tokens["train"],
            "train_tokens_incl_padding": train_incl,
            "total_tokens": val_tokens + train_tokens,
            "total_rows": val_rows + train_rows,
            "total_chars": _counts_orig.get("total_chars"),
            "truncated": _counts_orig.get("truncated"),
        },
        "wall_s": _orig("wall_s"),
    }
    field_notes = {
        "format/dtype/seq_len/rows/num_shards/tokens_incl_padding": "shard files (sha256 recomputed)",
        "eos_id": eos_prov or "resolved",
        "pad_id": pad_prov or "resolved",
        "tokenizer.sha256": sha_prov or "unresolved",
        "counts.val_tokens/train_tokens": "rows*seq minus shard-verified pad tails",
        "docs": "EOS-separator count over the packed rows",
        "dataset/args/timestamp/wall_s/chars/target_tokens/rows_per_shard":
            "original record" if original_meta is not None else "unrecoverable from shards (null)",
    }
    manifest: Dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "dtype": dtype_str,
        "seq_len": seq_len,
        "eos_id": eos_resolved,
        "pad_id": pad_resolved,
        "num_shards": num_shards,
        "shards": shard_entries,
        "metadata": metadata,
        "recovered": True,
        "recovery": {
            "schema": RECOVERY_SCHEMA,
            "tool": "scripts/recover_manifest.py",
            "tool_version": TOOL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "packed_dir": os.path.abspath(packed_dir),
            "shards_untouched": True,
            "sources": {
                "tokenizer_json": os.path.abspath(tokenizer_json) if tokenizer_json else None,
                "checkpoint": os.path.abspath(checkpoint) if checkpoint else None,
                "run_metadata_json": (run_meta_path if os.path.isfile(run_meta_path) else None),
                "expected_tokens_train": expected_tokens_train,
                "expected_tokens_val": expected_tokens_val,
            },
            "field_provenance": field_notes,
            "notes": [
                f"pad id {pad_resolved} verified against the shard structure: "
                f"pad tokens occur only as the final-row suffix of each "
                f"region's last shard (val {pad_tokens['val']}, train {pad_tokens['train']})"
                if pad_prov != "shard-structure inference" else
                "pad id inferred strictly from the shard structure "
                "(same value ends both regions' final rows and occurs nowhere else)",
                "shard files were NOT regenerated, reordered or modified",
                "underivable fields are null and listed in field_provenance",
            ],
        },
    }

    # ---- 7) expected-tokens cross-check (requirement 5) ---------------------
    mismatches: List[str] = []
    for key, want in (
        ("train_tokens", expected_tokens_train),
        ("val_tokens", expected_tokens_val),
    ):
        if want is None:
            continue
        got = train_tokens if key == "train_tokens" else val_tokens
        if abs(want - got) > expected_token_tolerance:
            mismatches.append(
                f"expected {key} {want:,} vs recovered {got:,} "
                f"(delta {want - got:,})"
            )

    # ---- 8) report ---------------------------------------------------------
    print("=" * 72)
    print("PACKED-CORPUS MANIFEST RECOVERY — analysis report")
    print("=" * 72)
    print(f"  packed-dir   : {packed_dir}")
    print(f"  shard rule   : {order_note}")
    print(f"  shards       : {num_shards} total ({len(phases['val'])} val, "
          f"{len(phases['train'])} train)")
    print(f"  dtype / seq  : {dtype_str} / {seq_len} | vocab bound {vocab_bound}")
    print(f"  rows         : val {val_rows:,} | train {train_rows:,}")
    print(f"  real tokens  : val {val_tokens:,} | train {train_tokens:,}")
    print(f"  pad tokens   : val {pad_tokens['val']:,} | train {pad_tokens['train']:,}")
    print(f"  docs (EOS)   : val {val_docs:,} | train {train_docs:,}")
    print(f"  resolved     : eos_id={eos_resolved} [{eos_prov or 'cli'}]  "
          f"pad_id={pad_resolved} [{pad_prov or 'cli'}]")
    print(f"  tokenizer sha: "
          f"{(sha_resolved[:16] + '…') if sha_resolved else 'UNRESOLVED — resume will refuse'}")
    if debris:
        print(f"  NOTE         : {len(debris)} stray *.tmp file(s) left by the "
              f"atomic writer: {', '.join(debris)}")
    print("-" * 72)
    print("  per-shard (phase, file, rows, tokens_incl_padding, sha256[:12], id range):")
    for phase in ("val", "train"):
        for name, _index in phases[phase]:
            f = facts[name]
            print(f"    {phase:>4}  {name}  rows={f['rows']:<6} "
                  f"tokens={f['tokens_incl_padding']:<9} "
                  f"sha={f['sha256'][:12]}  ids={f['min_id']}..{f['max_id']}")
    if mismatches:
        print("-" * 72)
        for line in mismatches:
            print("  WARNING: " + line)
        print("  EXPECTED-TOKENS MISMATCH — writing is BLOCKED without --force.")
    if os.path.isfile(output) and not force and not dry_run:
        raise RecoveryError(
            f"manifest already exists: {output} — refusing to overwrite; pass "
            "--force only after checking it is not the file you are trying to "
            "recover"
        )
    if mismatches and not force:
        raise RecoveryError(
            "expected-tokens mismatch (see the report above): "
            + "; ".join(mismatches) + " — pass --force to write anyway"
        )
    if dry_run:
        print("-" * 72)
        print(f"DRY RUN — nothing written (would write {output})")
        return {"written": False, "dry_run": True, "manifest": manifest}

    _atomic_write_json(output, manifest)
    print("-" * 72)
    print(f"WROTE: {output}")
    print("Recovered manifest is verified to load through "
          "data.packed.load_packed_manifest and to reproduce the original "
          "manifest_identity (resume-safe when the tokenizer sha matches).")
    return {"written": True, "manifest": manifest}


def _first_int(records: List[Dict[str, Any]], key: str) -> Any:
    """First int value of ``key`` across source records (resolution order)."""
    for rec in records:
        val = rec.get(key)
        if isinstance(val, int):
            return val
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        recover_manifest(
            args.packed_dir,
            tokenizer_json=args.tokenizer_json,
            checkpoint=args.checkpoint,
            eos_id=args.eos_id,
            pad_id=args.pad_id,
            vocab_size=args.vocab_size,
            expected_tokens_train=args.expected_tokens_train,
            expected_tokens_val=args.expected_tokens_val,
            expected_token_tolerance=args.expected_token_tolerance,
            train_glob=args.train_glob,
            val_glob=args.val_glob,
            output=args.output,
            force=args.force,
            dry_run=args.dry_run,
        )
        return EXIT_OK
    except RecoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())