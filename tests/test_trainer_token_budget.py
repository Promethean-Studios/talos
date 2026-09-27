"""Token-budget / packed-corpus trainer pass (100M program pass B).

Exercises the six deliverables end-to-end on SYNTHETIC, memory-lean fixtures
(no network, no real corpus): a packed-token corpus is produced through the
real ``scripts.prepare_corpus`` packing path (JSONLReader + a BPE-trained
tokenizer) and the trainer runs ``train_run()`` on the canonical 254,272-param
``tiny`` preset.

Covered here:

* DELIVERABLE 1: packed-dir consumption — rows used verbatim, val from val
  shards, loud manifest-vs-shard validation (seq, row counts, tokenizer sha);
* DELIVERABLE 2: token-budget stop — stops at the budget incl. resumed tokens,
  metrics record budget/consumed/steps;
* DELIVERABLE 3: LR schedule math — monotone warmup then cosine decay to
  10% of lr; fixed-LR unchanged when the schedule is off;
* DELIVERABLE 4: run-metadata sidecar — train_run_metadata.json at run start,
  same dict embedded in every checkpoint, provenance merge on resume;
* DELIVERABLE 5: numeric (never lexicographic) checkpoint selection + corrupt
  checkpoint fallback on resume from a directory;
* DELIVERABLE 6: the regression tests themselves.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from data.packed import PackedTokenDataset, manifest_identity
from data.readers import JSONLReader
from scripts.prepare_corpus import prepare_corpus
from scripts.train_oasst1 import (
    RUN_METADATA_FILENAME,
    TokenSchedule,
    list_checkpoint_candidates,
    load_checkpoint,
    train_run,
    train_tokenizer_for_run,
)
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256
from tools.make_synthetic_oasst1 import generate

SEQ = 8        # packed row width (tiny for speed)
BATCH = 2
#: tokens consumed per step: batch * (seq - 1) — the trainer's accounting.
TOKENS_PER_STEP = BATCH * (SEQ - 1)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _write_synthetic_jsonl(path, n_docs: int = 60, seed: int = 4) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for doc in generate(n_docs, seed):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _make_packed_corpus(tmp_path, *, seq: int = SEQ, num_merges: int = 150,
                        docs_seed: int = 4, tokenizer_path=None):
    """A real prepare_corpus packed corpus (shards + manifest + run_metadata).

    With ``tokenizer_path`` set, that established tokenizer.json is loaded
    instead of BPE-training a fresh one (used to build a *different* corpus on
    the same tokenizer for the resume-identity test).
    """
    src = tmp_path / "docs.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_jsonl(str(src), seed=docs_seed)
    if tokenizer_path is None:
        tok_path = str(tmp_path / "tokenizer.json")
        train_tokenizer_for_run(
            str(src), tok_path, preset="tiny", num_merges=num_merges
        )
    else:
        tok_path = tokenizer_path
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    packed_dir = str(tmp_path / "packed")
    prepare_corpus(
        JSONLReader(str(src)), tokenizer,
        out_dir=packed_dir, target_tokens=5000, val_tokens=1200,
        tokenizer_path=tok_path, seq_len=seq, dtype="int32",
        val_skip_docs=4, stream_buffer_docs=6, rows_per_shard=8,
        dataset_meta={"dataset": "synthetic", "config": "sample", "split": "train"},
        args_echo={"smoke": True},
    )
    with open(os.path.join(packed_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    train_rows = sum(
        e["rows"] for e in manifest["shards"] if e["phase"] == "train"
    )
    assert train_rows >= 8, f"packed fixture too small for training: {train_rows}"
    return packed_dir, tok_path


def _packed_args(
    tmp_path, out_dir, packed_dir, tok_path, *, resume=None, **overrides
):
    kwargs = dict(
        data=None, packed_dir=packed_dir, out_dir=out_dir, seed=0, preset="tiny",
        resume=resume, split_ratio=0.9, split_max_docs=None,
        bpe_num_merges=None, bpe_minfreq=2, bpe_max_docs=None, bpe_max_chars=None,
        tokenizer_json=tok_path, epochs=1, seq=None, batch=BATCH, lr=3e-3,
        token_budget=None, warmup_tokens=0, lr_decay="none",
        max_steps_per_epoch=None, val_max_steps=3, device="cpu",
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# DELIVERABLE 1 — packed-dir consumption + manifest validation
# ---------------------------------------------------------------------------
def test_packed_train_e2e_loss_decreases_and_checkpoints_carry_metadata(
    tmp_path,
) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    out_dir = str(tmp_path / "run")
    args = _packed_args(
        tmp_path, out_dir, packed_dir, tok_path,
        epochs=3, max_steps_per_epoch=60,
    )
    metrics = train_run(args)

    assert metrics["data_source"] == "packed"
    assert metrics["val_rows"] and metrics["train_rows"]
    # Loss decreases over the run (fixed tiny corpus, enough steps).
    losses = [r["train_loss"] for r in metrics["epochs"]]
    assert losses[0] > losses[-1] + 0.2, f"loss did not decrease: {losses}"
    assert metrics["final_val_loss"] is not None  # val loss from val shards

    # Checkpoint payload: v1 format + the new additive keys.
    ckpt_file = os.path.join(out_dir, f"step-{metrics['steps']}.pt")
    ckpt = load_checkpoint(ckpt_file)
    assert ckpt["run_metadata"]["schema"] == "talos-training-run-metadata-v1"
    prov = ckpt["run_metadata"]["data_provenance"]
    assert prov["source"] == "packed"
    assert prov["manifest_identity"] == manifest_identity(
        _read_json(os.path.join(packed_dir, "manifest.json"))
    )
    assert ckpt["run_metadata"]["n_params"] == 254272
    assert ckpt["run_metadata"]["git"]["commit"]  # running code commit sha
    assert isinstance(ckpt["tokens_consumed"], int) and ckpt["tokens_consumed"] > 0

    # Sidecar file == the dict embedded in the checkpoint (invariant parts).
    sidecar = _read_json(os.path.join(out_dir, RUN_METADATA_FILENAME))
    for key in ("schema", "preset", "n_params", "data_provenance", "training_config"):
        assert sidecar[key] == ckpt["run_metadata"][key], key

    # Packed runs where the manifest's tokenizer .json is NOT on disk still
    # train: identity is recorded from the manifest sha256 alone.
    out2 = str(tmp_path / "run-no-tok")
    os.remove(tok_path)  # manifest still records its sha256
    args2 = _packed_args(tmp_path, out2, packed_dir, None)
    m2 = train_run(args2)
    assert m2["tokenizer_origin"] == "packed-manifest"
    assert m2["tokenizer_sha256"] == metrics["tokenizer_sha256"]
    assert m2["final_val_loss"] is not None


def test_packed_seq_mismatch_fails(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    args = _packed_args(
        tmp_path, str(tmp_path / "run"), packed_dir, tok_path, seq=SEQ + 1
    )
    with pytest.raises(ValueError, match="seq_len"):
        train_run(args)


def test_packed_tokenizer_mismatch_fails(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    # A different (valid) tokenizer: different merge count -> different sha256.
    other_src = tmp_path / "other.jsonl"
    _write_synthetic_jsonl(str(other_src), seed=9)
    other_path = str(tmp_path / "other-tokenizer.json")
    train_tokenizer_for_run(
        str(other_src), other_path, preset="tiny", num_merges=60
    )
    assert tokenizer_file_sha256(other_path) != tokenizer_file_sha256(tok_path)
    args = _packed_args(
        tmp_path, str(tmp_path / "run"), packed_dir, other_path
    )
    with pytest.raises(ValueError, match="tokenizer identity mismatch"):
        train_run(args)


def test_packed_manifest_row_count_mismatch_fails(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    manifest_path = os.path.join(packed_dir, "manifest.json")
    manifest = _read_json(manifest_path)
    train_entry = next(e for e in manifest["shards"] if e["phase"] == "train")
    train_entry["rows"] += 1  # tamper: manifest row count no longer on disk
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    args = _packed_args(tmp_path, str(tmp_path / "run"), packed_dir, tok_path)
    with pytest.raises(ValueError, match="failed validation"):
        train_run(args)


def test_packed_missing_shard_file_fails(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    shard = next(
        e["shard"] for e in _read_json(os.path.join(packed_dir, "manifest.json"))["shards"]
    )
    os.remove(os.path.join(packed_dir, shard))
    args = _packed_args(tmp_path, str(tmp_path / "run"), packed_dir, tok_path)
    with pytest.raises(ValueError, match="missing on disk"):
        train_run(args)


def test_packed_dataset_rejects_out_of_vocab_ids(tmp_path) -> None:
    arr = np.zeros((4, SEQ), dtype=np.int32)
    arr[1, 2] = 5000  # token id beyond the 1024-vocab model
    shard = str(tmp_path / "bad.npy")
    with open(shard, "wb") as fh:
        np.save(fh, arr)
    ds = PackedTokenDataset([shard], seq_len=SEQ, batch_size=2,
                            expected_dtype="int32", max_id=1024)
    with pytest.raises(ValueError, match="outside"):
        for _ in ds:
            pass


def test_packed_dataset_dtype_mismatch_fails(tmp_path) -> None:
    arr = np.zeros((4, SEQ), dtype=np.int32)
    shard = str(tmp_path / "s.npy")
    with open(shard, "wb") as fh:
        np.save(fh, arr)
    ds = PackedTokenDataset([shard], seq_len=SEQ, batch_size=2,
                            expected_dtype="uint16", max_id=1024)
    with pytest.raises(ValueError, match="dtype"):
        for _ in ds:
            pass


# ---------------------------------------------------------------------------
# DELIVERABLE 2 — token-budget stop
# ---------------------------------------------------------------------------
def test_token_budget_stops_training_and_metrics_are_consistent(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    out_dir = str(tmp_path / "run")
    budget = 3 * TOKENS_PER_STEP  # exactly 3 steps
    args = _packed_args(
        tmp_path, out_dir, packed_dir, tok_path, token_budget=budget
    )
    metrics = train_run(args)

    assert metrics["token_budget"] == budget
    assert metrics["tokens_consumed"] == budget
    assert metrics["steps"] == 3
    assert metrics["budget_reached"] is True
    assert len(metrics["epochs"]) == 1 and metrics["epochs"][0]["steps"] == 3
    # The partial epoch still got a checkpoint with the consumed counter.
    ckpt = load_checkpoint(os.path.join(out_dir, "step-3.pt"))
    assert ckpt["tokens_consumed"] == budget


def test_resume_token_budget_counts_prior_tokens(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    out_dir = str(tmp_path / "run")
    budget_a = 2 * TOKENS_PER_STEP
    train_run(_packed_args(
        tmp_path, out_dir, packed_dir, tok_path, token_budget=budget_a
    ))
    step2 = os.path.join(out_dir, "step-2.pt")
    assert load_checkpoint(step2)["tokens_consumed"] == budget_a

    # Resume to a larger budget: exactly one more step (14 tokens) is trained,
    # then the run stops WITH the accumulated total.
    budget_b = 3 * TOKENS_PER_STEP
    metrics = train_run(_packed_args(
        tmp_path, out_dir, packed_dir, tok_path,
        resume=step2, epochs=3, token_budget=budget_b,
    ))
    assert metrics["steps"] == 1
    assert metrics["tokens_consumed"] == budget_b
    assert metrics["budget_reached"] is True
    assert metrics["resumed_from"] == os.path.abspath(step2)
    ckpt3 = load_checkpoint(os.path.join(out_dir, "step-3.pt"))
    assert ckpt3["tokens_consumed"] == budget_b

    # Resume with the budget ALREADY consumed: zero new steps, no crash.
    metrics_done = train_run(_packed_args(
        tmp_path, out_dir, packed_dir, tok_path,
        resume=step2, epochs=3, token_budget=budget_a,
    ))
    assert metrics_done["steps"] == 0
    assert metrics_done["tokens_consumed"] == budget_a
    assert metrics_done["budget_reached"] is True
    assert metrics_done["final_train_loss"] is None  # nothing new was trained


# ---------------------------------------------------------------------------
# DELIVERABLE 3 — LR schedule math
# ---------------------------------------------------------------------------
def test_cosine_warmup_schedule_is_monotone_and_hits_targets() -> None:
    s = TokenSchedule(lr=1.0, warmup_tokens=100, decay="cosine", budget=1000)
    assert s.lr_at(0) == 0.0
    assert s.lr_at(50) == pytest.approx(0.5)
    assert s.lr_at(100) == pytest.approx(1.0)
    assert s.lr_at(550) == pytest.approx(0.55, abs=1e-9)  # cosine midpoint
    assert s.lr_at(1000) == pytest.approx(0.1)            # 10% of lr at budget
    # Monotone: strictly climbing during warmup, strictly falling afterwards.
    for t in (20, 40, 60, 80):
        assert s.lr_at(t) > s.lr_at(t - 1)
    for t in (200, 400, 600, 800):
        assert s.lr_at(t) < s.lr_at(t - 1)
    # Warmup-only schedule stays at full lr once warmed up.
    w = TokenSchedule(lr=2e-3, warmup_tokens=50, decay="none")
    assert w.lr_at(0) == 0.0 and w.lr_at(25) == pytest.approx(1e-3)
    assert w.lr_at(50) == 2e-3 and w.lr_at(10_000_000) == 2e-3


def test_fixed_lr_unchanged_when_schedule_off() -> None:
    s = TokenSchedule(lr=3e-3, warmup_tokens=0, decay="none", budget=None)
    for t in (0, 1, 17, 123456):
        assert s.lr_at(t) == 3e-3  # EXACTLY the caller's lr (no FP drift)


def test_cosine_without_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="token-budget"):
        TokenSchedule(lr=1e-3, decay="cosine", budget=None)
    with pytest.raises(ValueError, match="warmup"):
        TokenSchedule(lr=1e-3, warmup_tokens=100, decay="cosine", budget=100)


# ---------------------------------------------------------------------------
# DELIVERABLE 4 — run-metadata sidecar (fresh + resume provenance merge)
# ---------------------------------------------------------------------------
def test_metadata_sidecar_fresh_and_merged_on_resume(tmp_path) -> None:
    packed_dir, tok_path = _make_packed_corpus(tmp_path)
    out_dir = str(tmp_path / "run")
    train_run(_packed_args(
        tmp_path, out_dir, packed_dir, tok_path,
        token_budget=2 * TOKENS_PER_STEP, warmup_tokens=8,
    ))

    sidecar_path = os.path.join(out_dir, RUN_METADATA_FILENAME)
    fresh = _read_json(sidecar_path)
    assert fresh["schema"] == "talos-training-run-metadata-v1"
    assert fresh["preset"] == "tiny" and fresh["n_params"] == 254272
    assert fresh["model_config"]["vocab_size"] == 1024
    tc = fresh["training_config"]
    assert tc["batch"] == BATCH and tc["seq"] == SEQ
    assert tc["lr"] == 3e-3 and tc["warmup_tokens"] == 8
    assert tc["token_budget"] == 2 * TOKENS_PER_STEP
    assert tc["device"] == "cpu" and "seed" in tc
    assert fresh["data_provenance"]["source"] == "packed"
    assert fresh["tokenizer"]["sha256"]
    assert fresh["git"]["commit"]
    assert fresh["timestamps"]["started"] and fresh["timestamps"]["finished"]
    assert fresh["resume_history"] == []

    # Resume: provenance merge — original model/training/data kept, the new
    # session's git + resume trail stamped, nothing destructive.
    step = os.path.join(out_dir, "step-2.pt")
    train_run(_packed_args(
        tmp_path, out_dir, packed_dir, tok_path,
        resume=step, epochs=2, token_budget=3 * TOKENS_PER_STEP,
    ))
    merged = _read_json(sidecar_path)
    assert merged["data_provenance"] == fresh["data_provenance"]
    assert merged["model_config"] == fresh["model_config"]
    assert merged["n_params"] == fresh["n_params"]
    assert merged["resumed_from"] == os.path.abspath(step)
    assert merged["resume_history"] == [os.path.abspath(step)]
    assert merged["timestamps"]["resumed_at"]
    assert merged["timestamps"]["started"] == fresh["timestamps"]["started"]
    assert merged["git"]["commit"]  # re-stamped for the resumed session
    # Every epoch checkpoint of the resumed run embeds the merged sidecar.
    ckpt3 = load_checkpoint(os.path.join(out_dir, "step-3.pt"))
    assert ckpt3["run_metadata"]["resume_history"] == [os.path.abspath(step)]


def test_packed_resume_rejects_different_corpus(tmp_path) -> None:
    packed_a, tok_path = _make_packed_corpus(tmp_path)
    out_dir = str(tmp_path / "run")
    train_run(_packed_args(
        tmp_path, out_dir, packed_a, tok_path,
        token_budget=2 * TOKENS_PER_STEP,
    ))
    step = os.path.join(out_dir, "step-2.pt")

    # Same tokenizer, DIFFERENT corpus (different docs -> different row/token
    # counts in the manifest identity).
    packed_b, tok_b = _make_packed_corpus(
        tmp_path / "b", docs_seed=7, tokenizer_path=tok_path
    )
    assert tok_b == tok_path
    assert os.path.abspath(packed_b) != os.path.abspath(packed_a)
    with pytest.raises(ValueError, match="corpus identity"):
        train_run(_packed_args(
            tmp_path, str(tmp_path / "run2"), packed_b, tok_path,
            resume=step, epochs=2,
        ))


# ---------------------------------------------------------------------------
# DELIVERABLE 5 — numeric checkpoint selection + corrupt fallback
# ---------------------------------------------------------------------------
def test_numeric_checkpoint_selection_regression(tmp_path) -> None:
    """step-<N> selection must be NUMERIC: step-100 > step-10 > step-9.

    Lexicographic string order would pick step-9 as "newest" ('9' > '1') —
    the exact bug this regression guards against.
    """
    d = tmp_path / "ckpts"
    d.mkdir()
    for name in ("step-9.pt", "step-10.pt", "step-100.pt", "notes.txt"):
        (d / name).write_text("x")
    picked = [os.path.basename(p) for p in list_checkpoint_candidates(str(d))]
    assert picked == ["step-100.pt", "step-10.pt", "step-9.pt"]


def _short_jsonl_run(tmp_path, out_dir: str, epochs: int = 1) -> str:
    """A quick JSONL trainer run returning the out_dir (real checkpoint at
    step-<epochs*2>.pt)."""
    src = tmp_path / "src.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=30, seed=0)
    args = SimpleNamespace(
        data=str(src), packed_dir=None, out_dir=out_dir, seed=0, preset="tiny",
        resume=None, split_ratio=0.9, split_max_docs=None,
        bpe_num_merges=None, bpe_minfreq=2, bpe_max_docs=None, bpe_max_chars=None,
        tokenizer_json=None, epochs=epochs, seq=16, batch=2, lr=3e-3,
        token_budget=None, warmup_tokens=0, lr_decay="none",
        max_steps_per_epoch=2, val_max_steps=2, device="cpu",
    )
    train_run(args)
    return out_dir


def test_resume_from_directory_skips_corrupt_checkpoints(tmp_path) -> None:
    out_dir = _short_jsonl_run(tmp_path, str(tmp_path / "run"))
    # Real checkpoint at step-2. Corrupt the "newest" candidates so the scan
    # must fall back: step-9 garbage bytes, step-5 a valid dict but missing
    # optimizer state (fails validate_resume_checkpoint).
    (tmp_path / "run" / "step-9.pt").write_bytes(b"\x00garbage-not-a-pickle\xff")
    torch.save({"step": 5, "format": "talos-training-checkpoint-v1"}, 
               str(tmp_path / "run" / "step-5.pt"))

    args = SimpleNamespace(
        data=str(tmp_path / "src.jsonl"), packed_dir=None, out_dir=out_dir,
        seed=0, preset="tiny", resume=out_dir, split_ratio=0.9,
        split_max_docs=None, bpe_num_merges=None, bpe_minfreq=2,
        bpe_max_docs=None, bpe_max_chars=None, tokenizer_json=None,
        epochs=2, seq=16, batch=2, lr=3e-3, token_budget=None,
        warmup_tokens=0, lr_decay="none", max_steps_per_epoch=2,
        val_max_steps=2, device="cpu",
    )
    metrics = train_run(args)
    # Fell through step-9 (corrupt) and step-5 (invalid) to the numeric-next
    # VALID checkpoint, step-2 — and continued to step-4.
    assert metrics["resumed_from"] == os.path.join(out_dir, "step-2.pt")
    assert os.path.isfile(os.path.join(out_dir, "step-4.pt"))


def test_resume_from_directory_all_corrupt_fails_loudly(tmp_path) -> None:
    out_dir = _short_jsonl_run(tmp_path, str(tmp_path / "run"))
    # Corrupt BOTH surviving checkpoints (step-2 is the real one, step-9 the
    # planted garbage) — the scan must then find nothing valid.
    (tmp_path / "run" / "step-9.pt").write_bytes(b"\x00corrupt\xff")
    (tmp_path / "run" / "step-2.pt").write_bytes(b"\x00corrupt\xff")
    args = SimpleNamespace(
        data=str(tmp_path / "src.jsonl"), packed_dir=None, out_dir=out_dir,
        seed=0, preset="tiny", resume=out_dir, split_ratio=0.9,
        split_max_docs=None, bpe_num_merges=None, bpe_minfreq=2,
        bpe_max_docs=None, bpe_max_chars=None, tokenizer_json=None,
        epochs=2, seq=16, batch=2, lr=3e-3, token_budget=None,
        warmup_tokens=0, lr_decay="none", max_steps_per_epoch=2,
        val_max_steps=2, device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="no VALID resume checkpoint"):
        train_run(args)