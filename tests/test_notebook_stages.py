"""Tests for the notebook's stage/budget/schedule logic (``talos_stages``).

No torch, no GPU, no network: the budget cap, the LR schedule, the checkpoint
stage contract and the accounting invariant are all pure functions.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import talos_stages as stages  # noqa: E402

SEQ = 2048
MB = 4
GA = 16
TPS = MB * GA * SEQ          # 131,072


# ---------------------------------------------------------------------------
# Stage specs
# ---------------------------------------------------------------------------
def test_stage_a_is_c4_pretraining_and_stage_b_is_ultrachat_tuning():
    a = stages.stage_spec(stages.STAGE_A, token_budget=100_000_000)
    b = stages.stage_spec(stages.STAGE_B, token_budget=100_000_000)
    assert (a.dataset_id, a.dataset_config, a.train_split, a.val_split) == (
        "allenai/c4", "en", "train", "validation")
    assert (b.dataset_id, b.train_split, b.val_split) == (
        "HuggingFaceH4/ultrachat_200k", "train_sft", "test_sft")
    # Stage B supervises assistant turns only; Stage A supervises every token.
    assert b.mask_prompt_tokens is True
    assert a.mask_prompt_tokens is False
    # Stage B is capped at 3 epochs (no unbounded re-cycling of the chat set).
    assert b.max_epochs == 3
    assert b.peak_lr < a.peak_lr


def test_stage_spec_rejects_a_bad_stage_or_budget():
    with pytest.raises(ValueError):
        stages.stage_spec("C", token_budget=10)
    with pytest.raises(ValueError):
        stages.stage_spec(stages.STAGE_A, token_budget=0)
    with pytest.raises(ValueError):
        stages.stage_spec(stages.STAGE_A, token_budget=10, seq_len=0)


def test_step_and_warmup_arithmetic():
    spec = stages.stage_spec(stages.STAGE_A, token_budget=10 * TPS + 1, seq_len=SEQ)
    assert spec.tokens_per_step(MB, GA) == TPS
    assert spec.total_steps(MB, GA) == 11          # ceil(10*TPS+1 / TPS)
    warmup = spec.warmup_steps(MB, GA)
    assert 0 <= warmup < spec.total_steps(MB, GA)
    with pytest.raises(ValueError):
        spec.tokens_per_step(0, GA)


def test_run_metadata_records_stage_dataset_and_settings():
    spec = stages.stage_spec(stages.STAGE_B, token_budget=TPS, seq_len=SEQ)
    meta = spec.run_metadata(micro_batch=MB, grad_accum=GA, precision="bf16", gpu="L4")
    assert meta[stages.STAGE_METADATA_KEY] == "B"
    assert meta[stages.DATASET_METADATA_KEY] == spec.dataset_identity()
    assert meta["tokens_per_step"] == TPS
    assert meta["precision"] == "bf16" and meta["gpu"] == "L4"
    assert meta["mask_prompt_tokens"] is True
    assert meta["stage_name"] == stages.STAGE_NAMES["B"]


def test_stage_out_dir_is_per_stage():
    assert stages.stage_out_dir("/runs/1b", "A") == "/runs/1b/stage-a"
    assert stages.stage_out_dir("/runs/1b/", "B") == "/runs/1b/stage-b"
    with pytest.raises(ValueError):
        stages.stage_out_dir("/runs/1b", "Z")


# ---------------------------------------------------------------------------
# The budget cap (owner directive: overshoot must be structurally impossible)
# ---------------------------------------------------------------------------
def test_plan_step_full_partial_and_stop():
    full = stages.plan_step(10 * TPS, micro_batch=MB, grad_accum=GA, seq_len=SEQ)
    assert (full.micro_batches, full.tokens_this_step, full.partial) == (GA, TPS, False)
    assert not full.should_stop

    partial = stages.plan_step(TPS // 2 + 1, micro_batch=MB, grad_accum=GA, seq_len=SEQ)
    assert partial.partial is True
    assert partial.micro_batches == 8                    # 8 * 4 * 2048 = 65,536
    assert partial.tokens_this_step <= TPS // 2 + 1

    stop = stages.plan_step(MB * SEQ - 1, micro_batch=MB, grad_accum=GA, seq_len=SEQ)
    assert stop.should_stop and stop.micro_batches == 0 and stop.tokens_this_step == 0


def test_plan_step_validates_its_inputs():
    with pytest.raises(ValueError):
        stages.plan_step(-1, micro_batch=MB, grad_accum=GA, seq_len=SEQ)
    with pytest.raises(ValueError):
        stages.plan_step(10, micro_batch=0, grad_accum=GA, seq_len=SEQ)


@pytest.mark.parametrize("budget", [
    1, TPS - 1, TPS, TPS + 1, 3 * TPS, 10_000_013, 131_072 * 7 + 5,
])
def test_simulated_run_never_overshoots_the_budget(budget):
    result = stages.simulate_run(budget, micro_batch=MB, grad_accum=GA, seq_len=SEQ)
    assert result["overshoot"] == 0
    assert result["tokens_consumed"] <= budget
    # It must also not stop early: less than one micro-batch may be left over.
    assert budget - result["tokens_consumed"] < MB * SEQ


def test_token_accountant_refuses_an_overshoot():
    accountant = stages.TokenAccountant(1000)
    accountant.add(600, steps=0)
    assert accountant.remaining == 400 and not accountant.exhausted
    accountant.add(400, steps=1)
    assert accountant.exhausted and accountant.steps == 1
    with pytest.raises(ValueError):
        accountant.add(1)


def test_token_accountant_rejects_a_bad_initial_state():
    with pytest.raises(ValueError):
        stages.TokenAccountant(0)
    with pytest.raises(ValueError):
        stages.TokenAccountant(10, consumed=11)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def test_lr_warmup_peak_and_floor():
    peak, total, warmup = 3e-4, 1000, 100
    assert stages.lr_at(0, peak_lr=peak, total_steps=total, warmup=warmup,
                        min_lr_ratio=0.1) == pytest.approx(peak / warmup)
    assert stages.lr_at(warmup - 1, peak_lr=peak, total_steps=total, warmup=warmup,
                        min_lr_ratio=0.1) == pytest.approx(peak)
    assert stages.lr_at(total, peak_lr=peak, total_steps=total, warmup=warmup,
                        min_lr_ratio=0.1) == pytest.approx(peak * 0.1)
    # past the end it stays on the floor (a resumed run cannot "re-warm")
    assert stages.lr_at(total + 500, peak_lr=peak, total_steps=total, warmup=warmup,
                        min_lr_ratio=0.1) == pytest.approx(peak * 0.1)


def test_lr_is_monotone_during_warmup_and_decay():
    values = [stages.lr_at(step, peak_lr=1.0, total_steps=200, warmup=20,
                           min_lr_ratio=0.1) for step in range(200)]
    assert all(later >= earlier for later, earlier in zip(values[1:20], values[:19]))
    assert all(later <= earlier for later, earlier in zip(values[21:], values[20:-1]))


def test_lr_is_a_function_of_the_step_only_so_resume_matches():
    # The resumed run asks for the same LR at the same absolute step.
    for step in (0, 37, 500, 999):
        a = stages.lr_at(step, peak_lr=2e-4, total_steps=1000, warmup=50, min_lr_ratio=0.1)
        b = stages.lr_at(step, peak_lr=2e-4, total_steps=1000, warmup=50, min_lr_ratio=0.1)
        assert a == b


def test_warmup_steps_is_bounded_by_the_run():
    assert stages.warmup_steps(1000, 0.01) == 20        # minimum
    assert stages.warmup_steps(10, 0.5) <= 9            # never the whole run
    assert stages.warmup_steps(1, 0.5) == 0
    with pytest.raises(ValueError):
        stages.warmup_steps(0, 0.1)


# ---------------------------------------------------------------------------
# Checkpoint / resume contract
# ---------------------------------------------------------------------------
def _ckpt(stage="A", *, tokens=TPS, optimizer=True, rng=True, fingerprint="abc",
          dataset=None):
    spec = stages.stage_spec(stage, token_budget=10 * TPS, seq_len=SEQ)
    metadata = spec.run_metadata(micro_batch=MB, grad_accum=GA, precision="bf16", gpu="L4")
    if dataset is not None:
        metadata[stages.DATASET_METADATA_KEY] = dataset
    return spec, {
        "format": stages.CHECKPOINT_FORMAT,
        "step": 3,
        "tokens_consumed": tokens,
        "train_loss": 2.5,
        "val_loss": 2.6,
        "optimizer_state_dict": {"state": {}} if optimizer else None,
        "rng_state": {"torch": b""} if rng else None,
        "tokenizer_fingerprint": fingerprint,
        "run_metadata": metadata,
    }


def test_validate_stage_resume_accepts_a_matching_checkpoint():
    spec, ckpt = _ckpt("A")
    stages.validate_stage_resume(ckpt, spec, tokenizer_fingerprint="abc")


def test_validate_stage_resume_refuses_the_other_stage():
    _, ckpt_b = _ckpt("B")
    spec_a = stages.stage_spec("A", token_budget=10 * TPS, seq_len=SEQ)
    with pytest.raises(ValueError, match="stage"):
        stages.validate_stage_resume(ckpt_b, spec_a)


def test_validate_stage_resume_refuses_a_checkpoint_without_a_stage():
    spec, ckpt = _ckpt("A")
    del ckpt["run_metadata"][stages.STAGE_METADATA_KEY]
    with pytest.raises(ValueError, match="does not record a training stage"):
        stages.validate_stage_resume(ckpt, spec)


def test_validate_stage_resume_refuses_a_different_dataset():
    spec, ckpt = _ckpt("A", dataset={"id": "HuggingFaceFW/fineweb", "config": "sample-10BT",
                                     "train_split": "train", "val_split": "train",
                                     "seq_len": SEQ})
    with pytest.raises(ValueError, match="dataset identity"):
        stages.validate_stage_resume(ckpt, spec)


def test_validate_stage_resume_refuses_a_different_tokenizer():
    spec, ckpt = _ckpt("A", fingerprint="deadbeef")
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        stages.validate_stage_resume(ckpt, spec, tokenizer_fingerprint="abc")


@pytest.mark.parametrize("flag,message", [
    ("optimizer", "optimizer state"),
    ("rng", "RNG state"),
])
def test_validate_stage_resume_refuses_a_checkpoint_without_resume_state(flag, message):
    spec, ckpt = _ckpt("A", **{flag: False})
    with pytest.raises(ValueError, match=message):
        stages.validate_stage_resume(ckpt, spec)


def test_validate_stage_resume_refuses_a_checkpoint_over_budget():
    spec = stages.stage_spec("A", token_budget=TPS, seq_len=SEQ)
    _, ckpt = _ckpt("A", tokens=2 * TPS)
    with pytest.raises(ValueError, match="budget"):
        stages.validate_stage_resume(ckpt, spec)


def test_validate_stage_resume_refuses_a_wrong_format():
    spec, ckpt = _ckpt("A")
    ckpt["format"] = "talos-training-checkpoint-v0"
    with pytest.raises(ValueError, match="format"):
        stages.validate_stage_resume(ckpt, spec)


def test_resume_position_reads_stage_tokens_and_dataset_position():
    spec, ckpt = _ckpt("B")
    ckpt["epoch"] = 2
    ckpt["run_metadata"][stages.POSITION_METADATA_KEY] = {
        "epoch": 2, "docs_consumed": 17, "rows_in_epoch": 40,
        "pending_ids": [1, 2, 3], "pending_mask": [1, 1, 1],
    }
    position = stages.resume_position(ckpt)
    assert position["stage"] == "B"
    assert position["step"] == 3
    assert position["tokens_consumed"] == TPS
    assert position["epoch"] == 2
    assert position["docs_consumed"] == 17
    assert position["rows_in_epoch"] == 40


def test_steps_for_budget_is_ceil():
    assert stages.steps_for_budget(TPS, TPS) == 1
    assert stages.steps_for_budget(TPS + 1, TPS) == 2
    with pytest.raises(ValueError):
        stages.steps_for_budget(0, TPS)
