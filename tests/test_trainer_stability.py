"""Trainer numerical-stability hardening (100M program — the §12 audit gaps).

Synthetic, memory-lean, no-network tests on the canonical `tiny` (254,272-param)
preset — identical machinery the `tiny_100m` T4 run uses (`train_epochs` +
`train_run`), just smaller:

* DELIVERABLE 1 — `--save-every-tokens`: intra-epoch periodic checkpoints at
  token intervals; resume from a MID-epoch periodic checkpoint restores
  ``tokens_consumed`` exactly (the packed/JSONL IterableDataset cannot restore
  its intra-epoch position, so the next epoch re-streams from its start — the
  remainder of the partial epoch is skipped, token accounting stays exact);
* DELIVERABLE 2 — `--grad-clip`: a spiked gradient is bounded to max_norm, the
  PRE-clip total norm is recorded in history/metrics/checkpoints, and the
  off-default (0) is a no-op;
* DELIVERABLE 3 — NaN/Inf loss + gradient detection: a bad step skips the
  optimizer update entirely, logs a structured event, and writes a safety
  checkpoint of the last-good state immediately; K consecutive bad steps abort
  with a recoverable checkpoint AND metrics.json/metadata are still written;
* metadata regression — the trainer now self-reports the truth:
  ``gradient_clip_type``/``grad_clip_max_norm``/``nan_inf_detection``.
"""
from __future__ import annotations

import json
import os
import types
from types import SimpleNamespace

import pytest
import torch

import scripts.train_oasst1 as train_mod
from data.tokenized import StreamingTokenizedDataset
from scripts.train_oasst1 import (
    RUN_METADATA_FILENAME,
    BadStepsAbort,
    build_tiny_model,
    load_checkpoint,
    split_jsonl,
    train_epochs,
    train_run,
    train_tokenizer_for_run,
    validate_resume_checkpoint,
)
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tools.make_synthetic_oasst1 import generate

SEQ = 16
BATCH = 2
#: tokens consumed per good step: batch * (seq - 1).
TOKENS_PER_STEP = BATCH * (SEQ - 1)


# ---------------------------------------------------------------------------
# Synthetic fixtures (JSONL path — same trainer machinery as packed)
# ---------------------------------------------------------------------------
def _write_synthetic_jsonl(path, n_docs: int = 50, seed: int = 0) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for doc in generate(n_docs, seed):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _jsonl_env(tmp_path):
    """src.jsonl -> deterministic split + tokenizer + streamed datasets."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    split = split_jsonl(src, str(tmp_path / "data"), ratio=0.9, seed=0)
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(split.train_path, tok_path)
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    train_ds = StreamingTokenizedDataset(
        split.train_path, tokenizer, seq_len=SEQ, batch_size=BATCH,
        mode="pack", eos=True,
    )
    val_ds = StreamingTokenizedDataset(
        split.val_path, tokenizer, seq_len=SEQ, batch_size=BATCH,
        mode="pack", eos=True,
    )
    return split, tok_path, train_ds, val_ds


def _args(tmp_path, out_dir, *, resume=None, **overrides):
    kwargs = dict(
        data=str(tmp_path / "src.jsonl"), packed_dir=None, out_dir=out_dir,
        seed=0, preset="tiny", resume=resume, split_ratio=0.9,
        split_max_docs=None, bpe_num_merges=None, bpe_minfreq=2,
        bpe_max_docs=None, bpe_max_chars=None, tokenizer_json=None,
        epochs=1, seq=SEQ, batch=BATCH, lr=3e-3, token_budget=None,
        warmup_tokens=0, lr_decay="none", save_every_tokens=0, grad_clip=0.0,
        max_consecutive_bad_steps=3, max_steps_per_epoch=None,
        val_max_steps=3, device="cpu",
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _wrap_forward_nan_on_calls(model, nan_calls):
    """Return `model` whose forward emits NaN logits on the given forward-call
    numbers (1-based). Deterministic per-run injection of a NaN loss."""
    orig_forward = model.forward
    calls = {"n": 0}

    def forward(self, x, *a, **kw):
        calls["n"] += 1
        logits, state = orig_forward(x, *a, **kw)
        if calls["n"] in nan_calls:
            logits = torch.full_like(logits, float("nan"))
        return logits, state

    model.forward = types.MethodType(forward, model)
    return model


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# DELIVERABLE 1 — intra-epoch periodic checkpoints + mid-epoch resume
# ---------------------------------------------------------------------------
def test_save_every_tokens_writes_interval_checkpoints(tmp_path) -> None:
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    # 2 epochs x 3 steps = 6 steps; step tokens = 30. Periodic at the first
    # step crossing each absolute multiple of 60: steps 2 and 4 (both
    # mid-epoch; step 3 is an epoch end).
    metrics = train_run(_args(
        tmp_path, out_dir, epochs=2, max_steps_per_epoch=3,
        save_every_tokens=2 * TOKENS_PER_STEP,
    ))

    assert metrics["steps"] == 6
    assert metrics["tokens_consumed"] == 6 * TOKENS_PER_STEP
    ckpts = {n: load_checkpoint(os.path.join(out_dir, f"step-{n}.pt"))
             for n in (2, 3, 4, 6)}
    # Periodic (mid-epoch) checkpoints carry exact token accounting + metadata.
    # Step 2 is inside epoch 1; step 4 is inside epoch 2 (steps 1-3 = epoch 1).
    for n, expected_tokens, expected_epoch in (
        (2, 2 * TOKENS_PER_STEP, 1),
        (4, 4 * TOKENS_PER_STEP, 2),
    ):
        c = ckpts[n]
        assert c["step"] == n and c["tokens_consumed"] == expected_tokens
        assert c["val_loss"] is None                    # no val pass mid-epoch
        assert c["epoch"] == expected_epoch
        assert c["run_metadata"]["schema"] == "talos-training-run-metadata-v1"
        tc = c["run_metadata"]["training_config"]
        assert tc["save_every_tokens"] == 2 * TOKENS_PER_STEP
        assert tc["nan_inf_detection"] is True and tc["gradient_clip_type"] is None
    # Epoch-end checkpoints still behave identically (val loss present).
    assert ckpts[3]["val_loss"] is not None and ckpts[6]["val_loss"] is not None
    assert ckpts[6]["tokens_consumed"] == 6 * TOKENS_PER_STEP
    assert metrics["periodic_checkpoint"] == os.path.join(out_dir, "step-6.pt")
    # No stray tmp files: every checkpoint is fully written.
    assert [f for f in os.listdir(out_dir) if f.endswith(".tmp")] == []


def test_resume_from_mid_epoch_periodic_checkpoint_keeps_token_accounting(
    tmp_path,
) -> None:
    out_dir = str(tmp_path / "run")
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    # Run 1: 2 epochs x 3 steps, periodic every 2 steps (mid-epoch ckpts at
    # steps 2 and 4). No budget: runs all 6 steps, tokens 180.
    train_run(_args(
        tmp_path, out_dir, epochs=2, max_steps_per_epoch=3,
        save_every_tokens=2 * TOKENS_PER_STEP,
    ))
    step2 = os.path.join(out_dir, "step-2.pt")
    assert load_checkpoint(step2)["tokens_consumed"] == 2 * TOKENS_PER_STEP

    # Run 2: resume from the MID-EPOCH periodic checkpoint at step 2 with a
    # budget of 150 tokens. The packed/JSONL IterableDataset cannot restore an
    # intra-epoch position, so the next epoch re-streams from its start and the
    # remainder of the partial epoch is skipped — tokens_consumed is restored
    # exactly and counts every newly-executed good step.
    budget = 5 * TOKENS_PER_STEP
    metrics = train_run(_args(
        tmp_path, out_dir, resume=step2, epochs=3,
        max_steps_per_epoch=3, save_every_tokens=2 * TOKENS_PER_STEP,
        token_budget=budget,
    ))
    assert metrics["resumed_from"] == os.path.abspath(step2)
    assert metrics["tokens_consumed"] == budget   # 60 restored + 3*30 new
    assert metrics["steps"] == 3
    assert metrics["budget_reached"] is True
    # The last checkpoint of the resumed run records the exact budget.
    ckpt5 = load_checkpoint(os.path.join(out_dir, "step-5.pt"))
    assert ckpt5["tokens_consumed"] == budget
    assert ckpt5["epoch"] == 2                     # continued at the NEXT epoch
    # No double-counting drift: every checkpoint's tokens are monotone and
    # exactly step * TOKENS_PER_STEP.
    for n, expected in ((2, 60), (3, 90), (4, 120), (5, 150), (6, 180)):
        assert load_checkpoint(os.path.join(out_dir, f"step-{n}.pt"))[
            "tokens_consumed"
        ] == expected


# ---------------------------------------------------------------------------
# DELIVERABLE 2 — gradient clipping
# ---------------------------------------------------------------------------
def test_grad_clip_engages_records_preclip_norm_and_bounds_grads(tmp_path) -> None:
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    model = build_tiny_model()
    # Measured unclipped total grad norms on this fixture are ~2.2-2.9, so
    # max_norm=0.8 engages clipping on EVERY step.
    history = train_epochs(
        model, train_ds, val_ds, out_dir=out_dir, tokenizer_path=tok_path,
        lr=3e-3, epochs=1, device=torch.device("cpu"), seed=0,
        max_steps_per_epoch=6, val_max_steps=3, grad_clip=0.8,
    )
    # The PRE-clip norm is recorded in history and must exceed the cap (i.e.
    # clipping really engaged).
    assert history.last_grad_norm is not None and history.last_grad_norm > 0.8
    # POST-clip gradients of the final step are bounded by max_norm (the raw
    # total norm of the grads left after the last optimizer step == 0.8).
    post_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    )
    assert post_norm == pytest.approx(0.8, abs=1e-4)
    # The last checkpoint records the last pre-clip norm.
    ckpt = load_checkpoint(os.path.join(out_dir, "step-6.pt"))
    assert ckpt["last_grad_norm"] is not None and ckpt["last_grad_norm"] > 0.8


def test_grad_clip_off_default_is_a_noop(tmp_path) -> None:
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    model = build_tiny_model()
    history = train_epochs(
        model, train_ds, val_ds, out_dir=out_dir, tokenizer_path=tok_path,
        lr=3e-3, epochs=1, device=torch.device("cpu"), seed=0,
        max_steps_per_epoch=3, val_max_steps=3,   # grad_clip default 0.0
    )
    assert history.last_grad_norm is None
    assert history.last_good_loss is not None and torch.isfinite(
        torch.tensor(history.last_good_loss)
    )
    ckpt = load_checkpoint(os.path.join(out_dir, "step-3.pt"))
    assert ckpt["last_grad_norm"] is None


# ---------------------------------------------------------------------------
# DELIVERABLE 3 — NaN/Inf detection: single bad step
# ---------------------------------------------------------------------------
def test_nan_loss_skips_step_logs_event_and_writes_safety_checkpoint(
    tmp_path,
) -> None:
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    model = _wrap_forward_nan_on_calls(build_tiny_model(), nan_calls={3})

    history = train_epochs(
        model, train_ds, val_ds, out_dir=out_dir, tokenizer_path=tok_path,
        lr=3e-3, epochs=1, device=torch.device("cpu"), seed=0,
        max_steps_per_epoch=6, val_max_steps=3, max_consecutive_bad_steps=3,
    )

    # Exactly one skipped step, fully recorded, consecutive counter reset by
    # the later good steps.
    assert history.total_bad_steps == 1
    assert history.consecutive_bad_steps == 0   # reset on the next good step
    assert len(history.bad_step_events) == 1
    ev = history.bad_step_events[0]
    assert ev["step"] == 3
    assert ev["tokens_consumed"] == 2 * TOKENS_PER_STEP   # pre-step good tokens
    assert ev["tensor"] == "loss" and ev["stat"] == "nan"
    assert ev["loss"] is not None and ev["consecutive_bad_steps"] == 1
    assert ev["checkpoint"] == "step-3.pt"

    # The safety checkpoint holds the LAST-GOOD state, labeled step 3.
    safety = load_checkpoint(os.path.join(out_dir, "step-3.pt"))
    assert safety["step"] == 3 and safety["tokens_consumed"] == 2 * TOKENS_PER_STEP
    assert safety["epoch"] == 1 and safety["optimizer_state_dict"] is not None
    # ... and training CONTINUED: 6 attempts, 5 good steps -> tokens 150.
    row = history.row(1)
    assert row.steps == 6 and row.global_step == 6
    assert history.tokens_processed == 5 * TOKENS_PER_STEP
    end = load_checkpoint(os.path.join(out_dir, "step-6.pt"))
    assert end["tokens_consumed"] == 5 * TOKENS_PER_STEP
    assert row.val_loss is not None
    # The epoch mean excludes the NaN step (it is not in epoch_losses).
    assert torch.isfinite(torch.tensor(row.train_loss))


def test_nan_gradient_skips_step_and_identifies_tensor(tmp_path, monkeypatch) -> None:
    """A finite-loss step with a non-finite GRADIENT is caught after backward.

    Injected deterministically: the real backward runs, then one element of
    ``lm_head.weight.grad`` is corrupted to NaN on the 3rd step."""
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    model = build_tiny_model()

    orig_backward = torch.Tensor.backward
    calls = {"n": 0}

    def backward_spike(self, *a, **kw):
        calls["n"] += 1
        orig_backward(self, *a, **kw)
        if calls["n"] == 3:
            assert model.lm_head.weight.grad is not None
            model.lm_head.weight.grad[0, 0] = float("nan")

    monkeypatch.setattr(torch.Tensor, "backward", backward_spike)
    history = train_epochs(
        model, train_ds, val_ds, out_dir=out_dir, tokenizer_path=tok_path,
        lr=3e-3, epochs=1, device=torch.device("cpu"), seed=0,
        max_steps_per_epoch=6, val_max_steps=3, max_consecutive_bad_steps=3,
    )

    assert history.total_bad_steps == 1
    ev = history.bad_step_events[0]
    assert ev["step"] == 3 and ev["tensor"] == "grad:lm_head.weight"
    assert ev["stat"] == "nan" and ev["loss"] is None
    assert ev["checkpoint"] == "step-3.pt"
    assert os.path.isfile(os.path.join(out_dir, "step-3.pt"))
    assert history.tokens_processed == 5 * TOKENS_PER_STEP


# ---------------------------------------------------------------------------
# DELIVERABLE 3 — K consecutive bad steps -> abort + recoverable checkpoint
# ---------------------------------------------------------------------------
def test_consecutive_bad_steps_abort_and_checkpoint_is_resumable(tmp_path) -> None:
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    # Steps 3, 4, 5 go NaN — the 3rd consecutive bad step (step 5) aborts.
    model = _wrap_forward_nan_on_calls(build_tiny_model(), nan_calls={3, 4, 5})

    with pytest.raises(BadStepsAbort) as exc:
        train_epochs(
            model, train_ds, val_ds, out_dir=out_dir, tokenizer_path=tok_path,
            lr=3e-3, epochs=1, device=torch.device("cpu"), seed=0,
            max_steps_per_epoch=8, val_max_steps=3, max_consecutive_bad_steps=3,
        )
    hist = exc.value.history
    assert hist.aborted and "3 consecutive" in hist.abort_reason
    assert hist.total_bad_steps == 3 and hist.consecutive_bad_steps == 3
    assert [e["step"] for e in hist.bad_step_events] == [3, 4, 5]
    # Every bad step wrote a safety checkpoint with the pre-step token count.
    for n in (3, 4, 5):
        c = load_checkpoint(os.path.join(out_dir, f"step-{n}.pt"))
        assert c["tokens_consumed"] == 2 * TOKENS_PER_STEP
        assert c["epoch"] == 1 and c["run_metadata"] is None   # no sidecar here
    # The LAST safety checkpoint is the last-good state and passes every
    # resume-validation rule (format, preset, params, optimizer present).
    last_ckpt = load_checkpoint(os.path.join(out_dir, "step-5.pt"))
    validate_resume_checkpoint(last_ckpt, "tiny")  # raises on any violation

    # --- resume from the abort site: tokens restored, budget honored --------
    fresh = build_tiny_model()
    hist2 = train_epochs(
        fresh, train_ds, val_ds, out_dir=out_dir, tokenizer_path=tok_path,
        lr=3e-3, epochs=2, device=torch.device("cpu"), seed=0,
        max_steps_per_epoch=4, val_max_steps=3, resume_from=last_ckpt,
        token_budget=5 * TOKENS_PER_STEP,   # 60 restored + 3 more steps
    )
    assert hist2.tokens_processed == 5 * TOKENS_PER_STEP
    assert hist2.budget_reached and hist2.row(2).global_step == 8
    ckpt8 = load_checkpoint(os.path.join(out_dir, "step-8.pt"))
    assert ckpt8["tokens_consumed"] == 5 * TOKENS_PER_STEP


# ---------------------------------------------------------------------------
# End-to-end: train_run writes metrics.json + metadata on abort, and the run
# is resumable after the abort to a clean finish.
# ---------------------------------------------------------------------------
def test_train_run_abort_writes_metrics_metadata_and_is_recoverable(
    tmp_path, monkeypatch,
) -> None:
    out_dir = str(tmp_path / "run")
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)

    # Inject NaN logits on forward calls 4, 5, 6 (steps 4, 5, 6) so the run
    # aborts at its 3rd consecutive bad step. Steps 1-3 train normally.
    real_build = train_mod.build_preset_model
    calls = {"n": 0}

    def build_with_nan_injector(preset, **kw):
        model = real_build(preset, **kw)
        orig_forward = model.forward

        def forward(self, x, *a, **kw):
            calls["n"] += 1
            logits, state = orig_forward(x, *a, **kw)
            if calls["n"] in (4, 5, 6):
                logits = torch.full_like(logits, float("nan"))
            return logits, state

        model.forward = types.MethodType(forward, model)
        return model

    monkeypatch.setattr(train_mod, "build_preset_model", build_with_nan_injector)
    args = _args(
        tmp_path, out_dir, epochs=1, max_steps_per_epoch=8,
        token_budget=6 * TOKENS_PER_STEP, save_every_tokens=2 * TOKENS_PER_STEP,
        max_consecutive_bad_steps=3,
    )
    with pytest.raises(BadStepsAbort):
        train_run(args)

    # metrics.json exists and records every event + the abort reason.
    metrics = _read_json(os.path.join(out_dir, "metrics.json"))
    assert metrics["total_bad_steps"] == 3
    assert metrics["consecutive_bad_steps"] == 3
    assert metrics["aborted"] is not None
    assert "3 consecutive" in metrics["aborted"]["reason"]
    assert metrics["aborted"]["last_checkpoint"] == "step-6.pt"
    assert [e["step"] for e in metrics["bad_steps"]] == [4, 5, 6]
    assert [e["tensor"] for e in metrics["bad_steps"]] == ["loss"] * 3
    # 3 good steps trained (tokens 90) before the NaN tail; steps 4-6 each
    # record the pre-step token count.
    assert metrics["tokens_consumed"] == 3 * TOKENS_PER_STEP
    assert [e["tokens_consumed"] for e in metrics["bad_steps"]] == [
        3 * TOKENS_PER_STEP
    ] * 3
    assert metrics["steps"] == 6          # attempts
    assert metrics["final_train_loss"] is None   # no completed epoch
    assert metrics["last_good_loss"] is not None

    # The sidecar is finish-stamped even on abort.
    sidecar = _read_json(os.path.join(out_dir, RUN_METADATA_FILENAME))
    assert sidecar["timestamps"]["finished"]
    assert sidecar["training_config"]["nan_inf_detection"] is True
    assert sidecar["training_config"]["gradient_clip_type"] is None

    # Periodic checkpoint at step 2 exists (tokens 60) and the safety
    # checkpoints at 4/5/6 (tokens 90) all load + validate.
    assert load_checkpoint(os.path.join(out_dir, "step-2.pt"))[
        "tokens_consumed"
    ] == 2 * TOKENS_PER_STEP
    for n in (4, 5, 6):
        c = load_checkpoint(os.path.join(out_dir, f"step-{n}.pt"))
        assert c["tokens_consumed"] == 3 * TOKENS_PER_STEP
        validate_resume_checkpoint(c, "tiny")

    # --- recovery: a CLEAN run resumes from the aborted dir and finishes ----
    monkeypatch.undo()  # NaN injector is gone; the real builder is back
    metrics2 = train_run(_args(
        tmp_path, out_dir, resume=out_dir, epochs=2,
        max_steps_per_epoch=8, token_budget=6 * TOKENS_PER_STEP,
        save_every_tokens=2 * TOKENS_PER_STEP,
    ))
    assert metrics2["resumed_from"] == os.path.join(out_dir, "step-6.pt")
    assert metrics2["tokens_consumed"] == 6 * TOKENS_PER_STEP == 180
    assert metrics2["budget_reached"] is True
    assert metrics2["total_bad_steps"] == 0
    assert metrics2["bad_steps"] == [] and metrics2["aborted"] is None
    assert metrics2["final_train_loss"] is not None
    assert load_checkpoint(os.path.join(out_dir, "step-9.pt"))[
        "tokens_consumed"
    ] == 6 * TOKENS_PER_STEP


# ---------------------------------------------------------------------------
# Metadata truth + flag validation
# ---------------------------------------------------------------------------
def test_metadata_records_stability_settings(tmp_path) -> None:
    out_dir = str(tmp_path / "run")
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    metrics = train_run(_args(
        tmp_path, out_dir, max_steps_per_epoch=2,
        grad_clip=1.0, save_every_tokens=TOKENS_PER_STEP,
    ))
    assert metrics["nan_inf_detection"] is True
    assert metrics["grad_clip_max_norm"] == 1.0
    assert metrics["save_every_tokens"] == TOKENS_PER_STEP
    assert metrics["max_consecutive_bad_steps"] == 3
    assert metrics["bad_steps"] == [] and metrics["aborted"] is None
    tc = _read_json(os.path.join(out_dir, RUN_METADATA_FILENAME))[
        "training_config"
    ]
    assert tc["gradient_clip_type"] == "max_grad_norm"
    assert tc["grad_clip_max_norm"] == 1.0
    assert tc["nan_inf_detection"] is True
    assert tc["max_consecutive_bad_steps"] == 3
    assert tc["save_every_tokens"] == TOKENS_PER_STEP
    # Every checkpoint embeds the same config record.
    ckpt = load_checkpoint(os.path.join(out_dir, "step-2.pt"))
    assert ckpt["run_metadata"]["training_config"] == tc


def test_invalid_stability_flags_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="save-every-tokens"):
        train_run(_args(tmp_path, str(tmp_path / "r1"), save_every_tokens=-1))
    with pytest.raises(ValueError, match="grad-clip"):
        train_run(_args(tmp_path, str(tmp_path / "r2"), grad_clip=-0.5))
    with pytest.raises(ValueError, match="max-consecutive-bad-steps"):
        train_run(_args(
            tmp_path, str(tmp_path / "r3"), max_consecutive_bad_steps=-2
        ))