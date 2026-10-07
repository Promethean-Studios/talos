"""Tests for ``scripts.recover_manifest.py`` (packed-corpus manifest recovery).

Covers the full delegation contract:

* discovery mirrors the writer (index order, val-then-train) — the recovered
  manifest's shard list equals the original's (modulo the additive per-shard
  ``sha256``) and the packed dataset streams IDENTICALLY before/after recovery
  (classic + fast paths);
* per-shard integrity checks stop the recovery with a per-shard failure report
  and write NOTHING (truncated file; out-of-vocab ids);
* train/val separation follows the writer's naming rule, custom
  ``--train-glob``/``--val-glob`` overrides preserve order, and ambiguous
  layouts (both globs matching one file; single-glob invocation) fail loudly;
* every prepare_corpus manifest field is reproduced — from the shards, the
  real tokenizer file, the sibling ``run_metadata.json`` (auto-discovered) or
  the run checkpoint; underivable fields are null + recovery markers when no
  original record exists; conflicting sources refuse loudly;
* ``--expected-tokens-*`` cross-check: mismatch requires ``--force`` and the
  written manifest always carries the TRUE recovered counts;
* resume glue: an end-to-end packed ``train_run`` → delete manifest → recover
  → ``--resume`` continues against the recovered manifest (identity compare
  passes), and an absurd ``tokens_consumed`` counter is refused loudly.

Fixtures are REAL ``prepare_corpus`` outputs (synthetic JSONL docs, a BPE-
trained canonical tokenizer, uint16 dtype — the owner's fw-edu-250m layout).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.packed import (
    PackedTokenDataset,
    load_packed_manifest,
    manifest_identity,
    packed_phase_shard_paths,
)
from data.readers import JSONLReader
from scripts.prepare_corpus import prepare_corpus
from scripts.recover_manifest import (
    RecoveryError,
    _parse_shard_name,
    recover_manifest,
)
from scripts.train_oasst1 import train_run, train_tokenizer_for_run
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256
from tools.make_synthetic_oasst1 import generate

SEQ = 8  # packed row width (tiny for speed)
BATCH = 2
TOKENS_PER_STEP = BATCH * (SEQ - 1)  # the trainer's accounting


# ---------------------------------------------------------------------------
# Fixtures — one real prepare_corpus output, reused read-only by every test
# ---------------------------------------------------------------------------
def _write_synthetic_jsonl(path: str, n_docs: int = 60, seed: int = 4) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for doc in generate(n_docs, seed):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("recover-fixture")
    src = str(tmp / "docs.jsonl")
    _write_synthetic_jsonl(src)
    tok_path = str(tmp / "tokenizer.json")
    train_tokenizer_for_run(src, tok_path, preset="tiny", num_merges=150)
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    packed_dir = str(tmp / "packed")
    prepare_corpus(
        JSONLReader(src), tokenizer,
        out_dir=packed_dir, target_tokens=5000, val_tokens=1200,
        tokenizer_path=tok_path, seq_len=SEQ, dtype="uint16",
        val_skip_docs=4, stream_buffer_docs=6, rows_per_shard=8,
        dataset_meta={"dataset": "synthetic", "config": "sample", "split": "train"},
        args_echo={"smoke": True},
    )
    with open(os.path.join(packed_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert len(manifest["shards"]) >= 6  # multi-shard, both phases
    train_rows = sum(
        e["rows"] for e in manifest["shards"] if e["phase"] == "train"
    )
    assert train_rows >= 8, "fixture too small for training"
    return {"packed_dir": packed_dir, "tok_path": tok_path, "manifest": manifest}


def _corpus_copy(base: dict, tmp_path: Path, *, keep_manifest: bool = False) -> dict:
    """Copy the corpus shards (cheap) into a fresh dir for one test to mutate.
    The manifest is removed by default — that is the recovery scenario."""
    dst = str(tmp_path / "packed")
    shutil.copytree(base["packed_dir"], dst)
    if not keep_manifest:
        os.remove(os.path.join(dst, "manifest.json"))
    return {
        "packed_dir": dst,
        "tok_path": base["tok_path"],
        "manifest": base["manifest"],
    }


def _read_manifest(packed_dir: str) -> dict:
    with open(os.path.join(packed_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _stream(manifest: dict, packed_dir: str, **kw) -> list:
    ds = PackedTokenDataset(
        packed_phase_shard_paths(manifest, packed_dir, "train"),
        seq_len=manifest["seq_len"], batch_size=BATCH,
        expected_dtype=manifest["dtype"], max_id=1024, **kw,
    )
    return [b.tolist() for b in ds]


def _packed_args(
    base: dict, out_dir: str, *, resume=None, token_budget=None, epochs=1,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=None, packed_dir=base["packed_dir"], out_dir=out_dir, seed=0,
        preset="tiny", resume=resume, split_ratio=0.9, split_max_docs=None,
        bpe_num_merges=None, bpe_minfreq=2, bpe_max_docs=None, bpe_max_chars=None,
        tokenizer_json=base["tok_path"], epochs=epochs, seq=None, batch=BATCH,
        lr=3e-3, token_budget=token_budget, warmup_tokens=0, lr_decay="none",
        max_steps_per_epoch=None, val_max_steps=3, device="cpu",
    )


# ---------------------------------------------------------------------------
# 1) Recovery round-trip: identity + metadata + stream equality
# ---------------------------------------------------------------------------
def test_recover_roundtrip_identity_and_stream(base: dict, tmp_path: Path) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir, tok_path = rec["packed_dir"], rec["tok_path"]
    # run_metadata.json is deliberately LEFT in place: the tool must
    # auto-discover it so every originally-recorded field comes back.
    report = recover_manifest(packed_dir, tokenizer_json=tok_path)
    assert report["written"] is True
    recovered = _read_manifest(packed_dir)
    original = rec["manifest"]

    assert recovered["recovered"] is True
    assert recovered["recovery"]["schema"] == "talos-manifest-recovery-v1"
    assert recovered["recovery"]["shards_untouched"] is True
    assert recovered["recovery"]["sources"]["run_metadata_json"] is not None

    # The full metadata block is reproduced verbatim (run_metadata source).
    assert recovered["metadata"] == original["metadata"]
    # Shard list: identical entries, in the writer's order, plus the additive
    # per-shard sha256 receipt.
    assert [e["shard"] for e in recovered["shards"]] == [
        e["shard"] for e in original["shards"]
    ]
    for oe, re_ in zip(original["shards"], recovered["shards"]):
        for key in ("shard", "phase", "index", "rows", "tokens_incl_padding"):
            assert re_[key] == oe[key], (key, re_, oe)
        assert len(re_["sha256"]) == 64 and "sha256" not in oe
    # Identity is byte-for-byte the recorded one (resume-safe).
    assert manifest_identity(recovered) == manifest_identity(original)
    # The trainer's own loader validates the recovered manifest in place,
    # including the tokenizer-identity expectation.
    load_packed_manifest(
        packed_dir, expected_tokenizer_sha256=tokenizer_file_sha256(tok_path)
    )

    # Stream equality: the same packed token stream before and after recovery,
    # on both the classic and the fast (prefetch/mmap-cache) paths.
    for kw in ({"prefetch": 0}, {"prefetch": 1, "cache_mmaps": True}):
        assert _stream(recovered, packed_dir, **kw) == _stream(original, packed_dir, **kw)


def test_recover_refuses_without_real_ids_and_marks_unknowns(
    base: dict, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir = rec["packed_dir"]
    os.remove(os.path.join(packed_dir, "run_metadata.json"))  # no original record

    # No sources at all: pad_id is structurally inferable but eos_id is not —
    # the tool refuses (a value is never guessed).
    with pytest.raises(RecoveryError, match="eos_id"):
        recover_manifest(packed_dir)
    assert not os.path.exists(os.path.join(packed_dir, "manifest.json"))

    # Explicit ids (the canonical tokenizer's, read by the test from the real
    # tokenizer file) unlock the write; unknowns stay null + marked.
    tok = ByteLevelBPETokenizer.from_file(rec["tok_path"])
    recover_manifest(packed_dir, eos_id=int(tok.eos_id), pad_id=int(tok.pad_id))
    recovered = _read_manifest(packed_dir)
    original = rec["manifest"]
    assert recovered["metadata"]["dataset"]["dataset"] is None
    assert recovered["metadata"]["args"] is None
    assert recovered["metadata"]["timestamp"] is None
    assert recovered["metadata"]["counts"]["train_chars"] is None
    assert recovered["metadata"]["packing"]["rows_per_shard"] is None
    assert "UNRESOLVED" in capsys.readouterr().out  # resume will refuse
    # Counts and rows are exact even without the original record.
    for key in ("train_tokens", "train_rows", "val_tokens", "val_rows",
                "train_pad_tokens", "val_pad_tokens", "total_rows"):
        assert recovered["metadata"]["counts"][key] == \
            original["metadata"]["counts"][key], key
    # Identity fields that ARE resolved match; only tokenizer_sha256 is null.
    rid, oid = manifest_identity(recovered), manifest_identity(original)
    assert rid["format"] == oid["format"] and rid["seq_len"] == oid["seq_len"]
    assert rid["eos_id"] == oid["eos_id"] and rid["pad_id"] == oid["pad_id"]
    assert rid["counts"] == oid["counts"] and rid["num_shards"] == oid["num_shards"]
    assert rid["tokenizer_sha256"] is None
    load_packed_manifest(packed_dir)  # loader accepts the recovery markers


# ---------------------------------------------------------------------------
# 2) Integrity failures: per-shard report, no write (owner rule)
# ---------------------------------------------------------------------------
def test_integrity_failure_truncated_shard_stops_with_report(
    base: dict, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir = rec["packed_dir"]
    # Truncate the first val shard: the size check must catch it.
    shard_path = os.path.join(packed_dir, rec["manifest"]["shards"][0]["shard"])
    with open(shard_path, "rb") as fh:
        data = fh.read()
    with open(shard_path, "wb") as fh:
        fh.write(data[:-64])
    with pytest.raises(RecoveryError, match="integrity"):
        recover_manifest(packed_dir, tokenizer_json=rec["tok_path"])
    out = capsys.readouterr().out
    assert "SHARD INTEGRITY FAILURE" in out
    assert "FAIL" in out and "shard-val-0000.npy" in out
    assert not os.path.exists(os.path.join(packed_dir, "manifest.json"))
    # No shard bytes were touched by the tool itself.
    with open(shard_path, "rb") as fh:
        assert len(fh.read()) == len(data) - 64


def test_integrity_failure_out_of_vocab_ids_stops(
    base: dict, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir = rec["packed_dir"]
    # Replace a train shard's content with an out-of-vocab id (valid size,
    # valid shape — the id-range check must catch it).
    train_shard = next(
        e["shard"] for e in rec["manifest"]["shards"] if e["phase"] == "train"
    )
    path = os.path.join(packed_dir, train_shard)
    arr = np.load(path)
    arr[0, 0] = 9999
    np.save(path, arr)
    with pytest.raises(RecoveryError, match="integrity"):
        recover_manifest(packed_dir, tokenizer_json=rec["tok_path"])
    out = capsys.readouterr().out
    assert "SHARD INTEGRITY FAILURE" in out and "9999" in out
    assert not os.path.exists(os.path.join(packed_dir, "manifest.json"))


# ---------------------------------------------------------------------------
# 3) Expected-tokens cross-check: mismatch requires --force
# ---------------------------------------------------------------------------
def test_expected_tokens_mismatch_requires_force(
    base: dict, tmp_path: Path,
) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir = rec["packed_dir"]
    true_train = rec["manifest"]["metadata"]["counts"]["train_tokens"]
    true_val = rec["manifest"]["metadata"]["counts"]["val_tokens"]

    with pytest.raises(RecoveryError, match="--force"):
        recover_manifest(
            packed_dir, tokenizer_json=rec["tok_path"],
            expected_tokens_train=true_train + 123456,
        )
    assert not os.path.exists(os.path.join(packed_dir, "manifest.json"))

    # Exact numbers pass without force; a within-tolerance delta also passes.
    # (Every call AFTER the first write needs --force: an existing manifest is
    # the source of truth and is never overwritten on a whim — that is the
    # tool's own guard, not a test artifact.)
    recover_manifest(
        packed_dir, tokenizer_json=rec["tok_path"],
        expected_tokens_train=true_train, expected_tokens_val=true_val,
    )
    recover_manifest(
        packed_dir, tokenizer_json=rec["tok_path"],
        expected_tokens_train=true_train + 500, expected_tokens_val=true_val,
        expected_token_tolerance=1000, force=True,
    )

    # Forced write goes through but NEVER invents counts: the manifest carries
    # the true recovered numbers, not the mismatching expected ones.
    before = _read_manifest(packed_dir)
    recover_manifest(
        packed_dir, tokenizer_json=rec["tok_path"],
        expected_tokens_train=true_train + 123456, force=True,
    )
    after = _read_manifest(packed_dir)
    assert after["metadata"]["counts"]["train_tokens"] == true_train
    assert after["metadata"]["counts"]["val_tokens"] == true_val
    assert before["metadata"]["counts"] == after["metadata"]["counts"]


# ---------------------------------------------------------------------------
# 4) Split/layout handling: custom globs, ambiguity errors
# ---------------------------------------------------------------------------
def test_custom_globs_recover_ordering(base: dict, tmp_path: Path) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir = rec["packed_dir"]
    os.remove(os.path.join(packed_dir, "run_metadata.json"))
    for name in sorted(os.listdir(packed_dir)):
        parsed = _parse_shard_name(name)
        if parsed:
            phase, index = parsed
            # zero-padded rename keeps lexicographic order == numeric order
            # (the glob path must report its order rule; never rely on
            # unpadded lexicographic sorting == numeric).
            os.rename(
                os.path.join(packed_dir, name),
                os.path.join(packed_dir, f"tok-{phase}-{index:04d}.npy"),
            )
    # One-sided glob is refused (never guess the other phase).
    with pytest.raises(RecoveryError, match="--train-glob and --val-glob"):
        recover_manifest(packed_dir, train_glob="tok-train-*.npy")
    # A file matching both globs is an ambiguous split — refused.
    with pytest.raises(RecoveryError, match="matches BOTH"):
        recover_manifest(
            packed_dir, train_glob="*.npy", val_glob="tok-val-*.npy",
        )
    tok = ByteLevelBPETokenizer.from_file(rec["tok_path"])
    recover_manifest(
        packed_dir, train_glob="tok-train-*.npy", val_glob="tok-val-*.npy",
        eos_id=int(tok.eos_id), pad_id=int(tok.pad_id),
    )
    recovered = _read_manifest(packed_dir)
    original = rec["manifest"]
    assert [e["shard"] for e in recovered["shards"]] == [
        e["shard"].replace("shard-", "tok-") for e in original["shards"]
    ]
    assert [e["rows"] for e in recovered["shards"]] == [
        e["rows"] for e in original["shards"]
    ]
    load_packed_manifest(packed_dir)


# ---------------------------------------------------------------------------
# 5) Conflicting evidence: a tampered run_metadata is refused loudly
# ---------------------------------------------------------------------------
def test_source_conflict_is_loud(base: dict, tmp_path: Path) -> None:
    rec = _corpus_copy(base, tmp_path)
    packed_dir = rec["packed_dir"]
    meta_path = os.path.join(packed_dir, "run_metadata.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["counts"]["train_tokens"] += 1  # tampered: contradicts the shards
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with pytest.raises(RecoveryError, match="train_tokens"):
        recover_manifest(packed_dir, tokenizer_json=rec["tok_path"])
    assert not os.path.exists(os.path.join(packed_dir, "manifest.json"))


# ---------------------------------------------------------------------------
# 6) Resume glue: recovered manifest resumes; absurd counters refuse
# ---------------------------------------------------------------------------
def test_resume_with_recovered_manifest_and_counter_guard(
    base: dict, tmp_path: Path,
) -> None:
    rec = _corpus_copy(base, tmp_path, keep_manifest=True)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir)

    # Run 1: 3 steps on the ORIGINAL manifest, budget-capped (token-accounting
    # untouched: batch*(seq-1) = 14 tokens/step).
    metrics1 = train_run(
        _packed_args(rec, out_dir, token_budget=3 * TOKENS_PER_STEP)
    )
    assert metrics1["steps"] == 3
    step3 = os.path.join(out_dir, "step-3.pt")
    assert os.path.isfile(step3)

    # The corpus loses its manifest (the owner's scenario); recovery rebuilds
    # it from the shards + the real tokenizer file.
    os.remove(os.path.join(rec["packed_dir"], "manifest.json"))
    recover_manifest(rec["packed_dir"], tokenizer_json=rec["tok_path"])
    load_packed_manifest(rec["packed_dir"])

    # Resume: identity compare against the checkpoint's recorded identity
    # passes (recovered == original), so training continues on the recovered
    # manifest for the remaining 3 steps of the budget. --epochs must be the
    # SAME TOTAL as the original run (the resume contract) — the checkpoint is
    # already at epoch 1, so the total must be >= 2.
    metrics2 = train_run(
        _packed_args(rec, out_dir, resume=step3,
                     token_budget=6 * TOKENS_PER_STEP, epochs=2)
    )
    assert metrics2["steps"] == 3
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as fh:
        final_metrics = json.load(fh)
    # metrics["steps"] is session-scoped on resume (3 more); the cumulative
    # truth lives in tokens_consumed (42 -> 84) and the step-6 checkpoint.
    assert final_metrics["tokens_consumed"] == 6 * TOKENS_PER_STEP
    assert os.path.isfile(os.path.join(out_dir, "step-6.pt"))

    # Guard: a checkpoint claiming absurd consumption is impossible for this
    # corpus (epochs=1, train_rows x (seq-1) capacity) — refused loudly BEFORE
    # any training continues.
    ckpt = torch.load(step3, map_location="cpu", weights_only=False)
    ckpt["tokens_consumed"] = 10**12
    fake = os.path.join(out_dir, "step-999.pt")
    torch.save(ckpt, fake)
    with pytest.raises(ValueError, match="tokens consumed"):
        train_run(_packed_args(rec, out_dir, resume=fake,
                               token_budget=6 * TOKENS_PER_STEP, epochs=2))


def test_manifest_already_present_refuses_without_force(
    base: dict, tmp_path: Path,
) -> None:
    rec = _corpus_copy(base, tmp_path, keep_manifest=True)
    with pytest.raises(RecoveryError, match="already exists"):
        recover_manifest(rec["packed_dir"], tokenizer_json=rec["tok_path"])
    # The existing manifest is untouched.
    assert _read_manifest(rec["packed_dir"]) == rec["manifest"]