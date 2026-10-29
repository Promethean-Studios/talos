"""W2 training-engine capabilities (1B program): bf16 AMP, activation
checkpointing, gradient accumulation — CPU-run and memory-lean on the tiny
preset (the same machinery the 1B run uses, just smaller):

* ``--amp bf16`` — autocast bf16 on CUDA **and CPU**, NO GradScaler (direct
  backward/step); fp16 path unchanged; on T4/Turing (SM7.5, no bf16) use
  fp16 or none;
* ``--grad-checkpoint`` — per-block ``torch.utils.checkpoint.checkpoint(
  use_reentrant=False)`` during TRAINING only; final weights/loss identical
  to the non-checkpointed run (recompute is deterministic); eval/val/generation
  never checkpointed;
* ``--grad-accum N`` — N micro-batches (``--batch`` each) per optimizer step,
  loss scaled by 1/N; ``N=1`` is the legacy single-step path (bit-identical
  default regression test); remainder windows at epoch end are dropped and
  checkpoints land on optimizer-step boundaries; fp16 unscale happens once
  per window BEFORE the clip and the always-on finiteness guard.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch

from model.attention import build_attention_backend
from scripts.train_oasst1 import (
    AMP_AUTOCAST_DTYPES,
    AMP_CHOICES,
    RUN_METADATA_FILENAME,
    build_tiny_model,
    load_checkpoint,
    make_amp_autocast,
    make_grad_scaler,
    train_epochs,
    train_run,
)
from tests.test_trainer_stability import _args, _write_synthetic_jsonl


def _weights(ckpt_path: str) -> dict:
    return load_checkpoint(ckpt_path)["model_state_dict"]


def _max_abs_diff(wa: dict, wb: dict) -> float:
    return max((wa[k] - wb[k]).abs().max().item() for k in wa)


# ---------------------------------------------------------------------------
# W2-1 — --amp bf16
# ---------------------------------------------------------------------------
def test_amp_choices_include_bf16():
    assert AMP_CHOICES == ("none", "fp16", "bf16")
    assert AMP_AUTOCAST_DTYPES == {"fp16": torch.float16, "bf16": torch.bfloat16}


def test_make_amp_autocast_mode_dtypes():
    dev = torch.device("cpu")
    for amp, want in (("none", torch.float32), ("fp16", torch.float16),
                      ("bf16", torch.bfloat16)):
        a, b = torch.randn(4, 8), torch.randn(8, 4)
        with make_amp_autocast(amp, dev):
            assert (a @ b).dtype == want, amp
    with pytest.raises(ValueError, match="--amp must be one of"):
        make_amp_autocast("bogus", dev)


def test_make_grad_scaler_bf16_disabled():
    # bf16 needs NO scaling: the scaler must be disabled on cuda and cpu alike.
    assert make_grad_scaler("bf16", torch.device("cuda")).is_enabled() is False
    assert make_grad_scaler("bf16", torch.device("cpu")).is_enabled() is False


def test_bf16_train_tracks_fp32(tmp_path):
    """bf16 end-to-end on CPU: engages the autocast path, produces no bad
    steps, converges alongside the fp32 run within bf16 rounding."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    outs = {}
    for amp in ("none", "bf16"):
        out = str(tmp_path / f"out_{amp}")
        metrics = train_run(_args(
            tmp_path, out, data=src, max_steps_per_epoch=2, val_max_steps=1,
            seq=16, attention_backend="sdpa", amp=amp, log_every_steps=1,
        ))
        assert metrics["engine"]["amp"] == amp
        assert metrics["total_bad_steps"] == 0
        assert metrics["final_train_loss"] is not None
        outs[amp] = _weights(os.path.join(out, "step-2.pt"))
    # CPU autocast genuinely casts matmuls to bf16 (~8-bit mantissa): the
    # bound is the fp16-test analogue — a broken path diverges far more.
    assert torch.allclose(
        outs["none"]["layers.0.attention.q_proj.weight"],
        outs["bf16"]["layers.0.attention.q_proj.weight"],
        atol=1e-2, rtol=1e-2,
    ), "bf16 must track fp32 within bf16 rounding"


# ---------------------------------------------------------------------------
# W2-2 — --grad-checkpoint
# ---------------------------------------------------------------------------
def test_grad_checkpoint_gradients_and_loss_identical(tmp_path):
    """Same seed, same data: checkpointed vs plain train to identical final
    weights and the identical loss trajectory (recompute is deterministic)."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    results = {}
    for flag in (False, True):
        out = str(tmp_path / f"out_ck{int(flag)}")
        m = train_run(_args(
            tmp_path, out, data=src, max_steps_per_epoch=3, val_max_steps=1,
            seq=16, attention_backend="sdpa", grad_checkpoint=flag,
            log_every_steps=1,
        ))
        assert m["total_bad_steps"] == 0
        results[flag] = (m, _weights(os.path.join(out, "step-3.pt")))
    (m0, w0), (m1, w1) = results[False], results[True]
    # metrics.json holds floats, not tensors — compare with abs(); the weight
    # and gradient comparisons below stay on actual tensors (torch.allclose).
    assert abs(m0["final_train_loss"] - m1["final_train_loss"]) < 1e-6
    for k in w0:
        assert torch.allclose(w0[k], w1[k], atol=1e-6, rtol=1e-6), k
    # measured: max |ck_on - ck_off| = 0.0 on this box (exact recompute)
    assert _max_abs_diff(w0, w1) < 1e-6


def test_grad_checkpoint_training_only(monkeypatch):
    """Block forwards are checkpointed while train() is set, never in eval
    (the val/eval/generation contract)."""
    from configs.presets import tiny_config
    from model.gpt import TalosGPT
    model = TalosGPT(tiny_config().derive(),
                     attention_backend=build_attention_backend("plain"))
    assert model.gradient_checkpointing is False
    model.set_gradient_checkpointing(True)
    assert model.gradient_checkpointing is True

    import torch.utils.checkpoint as ckpt_mod
    calls = {"n": 0}
    orig = ckpt_mod.checkpoint

    def fake(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(ckpt_mod, "checkpoint", fake)
    ids = torch.randint(0, 1024, (2, 16))
    model.train()
    with torch.enable_grad():
        model(ids)
    assert calls["n"] == model.config.num_layers
    calls["n"] = 0
    model.eval()
    with torch.no_grad():
        model(ids)
    assert calls["n"] == 0, "eval/val forward must never be checkpointed"


# ---------------------------------------------------------------------------
# W2-3 — --grad-accum
# ---------------------------------------------------------------------------
def test_grad_accum_equals_full_batch(tmp_path):
    """grad_accum=2 with micro-batch B/2 converges to the SAME final weights
    as single-step batch B on identical data (per-step row identity)."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out_a = str(tmp_path / "A_batch4")
    ma = train_run(_args(
        tmp_path, out_a, data=src, max_steps_per_epoch=2, val_max_steps=1,
        seq=16, batch=4, attention_backend="sdpa", log_every_steps=1,
    ))
    out_b = str(tmp_path / "B_accum")
    mb = train_run(_args(
        tmp_path, out_b, data=src, max_steps_per_epoch=2, val_max_steps=1,
        seq=16, batch=2, grad_accum=2, attention_backend="sdpa",
        log_every_steps=1,
    ))
    assert mb["steps"] == 2 and ma["tokens_consumed"] == mb["tokens_consumed"]
    assert mb["engine"]["grad_accum"] == 2
    wa, wb = _weights(os.path.join(out_a, "step-2.pt")), \
        _weights(os.path.join(out_b, "step-2.pt"))
    # measured max diff ~8e-6 (fp32 summation order only); 1e-4 is a
    # generous CI-safe bound that still fails on any real divergence.
    assert _max_abs_diff(wa, wb) < 1e-4
    for k in wa:
        assert torch.allclose(wa[k], wb[k], atol=1e-4, rtol=1e-4), k


def test_grad_accum_remainder_dropped_at_epoch_end(tmp_path):
    """When N does not divide the loader, the partial window at epoch end is
    DROPPED: no step, no token credit, and the checkpoint lands exactly on
    the last completed optimizer step (step boundary)."""
    from tests.test_trainer_stability import BATCH, SEQ, _jsonl_env
    split, tok_path, train_ds, val_ds = _jsonl_env(tmp_path)
    n_batches = sum(1 for _ in train_ds)
    assert n_batches % 3 != 0, "fixture must leave a remainder"
    out_dir = str(tmp_path / "run")
    os.makedirs(out_dir, exist_ok=True)
    history = train_epochs(
        build_tiny_model(), train_ds, val_ds,
        out_dir=out_dir, tokenizer_path=tok_path, lr=3e-3, epochs=1,
        device=torch.device("cpu"), seed=0, max_steps_per_epoch=None,
        val_max_steps=1, grad_accum=3, log_every_steps=1,
    )
    done = n_batches // 3
    # Pinned semantics (the ONE coherent rule): `attempts` counts every loop
    # window that reaches its completion/void DECISION — completed optimizer
    # steps plus bad-step-voided windows (each consuming one step number, the
    # N=1 bad-batch rule of §12) — while the epoch-end PARTIAL window consumes
    # no step number, no tokens and no checkpoint. On this healthy run (no bad
    # steps) attempts == completed steps == the final global_step.
    assert history.attempts == done == history.rows[-1].global_step
    assert history.tokens_processed == done * 3 * BATCH * (SEQ - 1)
    assert history.last_good_loss is not None
    ckpt = load_checkpoint(os.path.join(out_dir, f"step-{done}.pt"))
    assert ckpt["tokens_consumed"] == history.tokens_processed
    # the voided remainder window produced NO checkpoint (no step-{done+1}.pt)
    assert not os.path.exists(os.path.join(out_dir, f"step-{done + 1}.pt"))


def test_grad_accum_fp16_and_bf16_run_clean(tmp_path):
    """Accumulation windows with the fp16 scaler plumbing (unscale/step/update
    once per window) and the bf16 direct path both run clean on CPU."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    for amp in ("fp16", "bf16"):
        out = str(tmp_path / f"out_{amp}")
        metrics = train_run(_args(
            tmp_path, out, data=src, max_steps_per_epoch=3, val_max_steps=1,
            seq=16, batch=2, grad_accum=2, attention_backend="sdpa", amp=amp,
            log_every_steps=1,
        ))
        assert metrics["steps"] == 3, amp
        assert metrics["total_bad_steps"] == 0, amp
        assert metrics["final_train_loss"] is not None, amp


def test_grad_accum_window_overflow_is_a_bad_step(tmp_path, monkeypatch):
    """A scaling overflow at WINDOW end (surfaced by unscale_ as an inf grad)
    turns the whole window into a bad step: no update, no token credit, and
    the safety checkpoint names the attempted step."""
    import scripts.train_oasst1 as train_mod
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out = str(tmp_path / "out")

    class OverflowScaler:
        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss * 8.0

        def unscale_(self, opt):
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad[0] = float("inf")

        def step(self, opt):
            pass

        def update(self):
            pass

    monkeypatch.setattr(train_mod, "make_grad_scaler",
                        lambda amp, device: OverflowScaler() if amp == "fp16"
                        else make_grad_scaler(amp, device))
    metrics = train_run(_args(
        tmp_path, out, data=src, max_steps_per_epoch=1, val_max_steps=1,
        seq=16, batch=2, grad_accum=2, attention_backend="sdpa", amp="fp16",
        max_consecutive_bad_steps=2, log_every_steps=1,
    ))
    assert metrics["total_bad_steps"] >= 1
    assert metrics["bad_steps"][0]["tensor"].startswith("grad:")
    # the voided window gained no tokens and the step-1 safety ckpt exists
    assert metrics["tokens_consumed"] == 0
    assert os.path.isfile(os.path.join(out, "step-1.pt"))


def test_default_flags_bit_identical(tmp_path):
    """fp32 / no-flag default behavior regression: a run with no W2 flags and
    a run with the explicit defaults (grad_accum=1, grad_checkpoint=False,
    amp=none) are bit-identical — weights equal elementwise and the loss
    trajectory matches to the last bit."""
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out_a = str(tmp_path / "A_default")
    ma = train_run(_args(
        tmp_path, out_a, data=src, max_steps_per_epoch=3, val_max_steps=1,
        seq=16, attention_backend="sdpa", log_every_steps=1,
    ))
    out_b = str(tmp_path / "B_explicit")
    mb = train_run(_args(
        tmp_path, out_b, data=src, max_steps_per_epoch=3, val_max_steps=1,
        seq=16, attention_backend="sdpa", log_every_steps=1,
        grad_accum=1, grad_checkpoint=False, amp="none",
    ))
    assert ma["engine"]["grad_accum"] == mb["engine"]["grad_accum"] == 1
    wa = _weights(os.path.join(out_a, "step-3.pt"))
    wb = _weights(os.path.join(out_b, "step-3.pt"))
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
    for la, lb in zip(ma["interval_logs"], mb["interval_logs"]):
        assert la["loss"] == lb["loss"]
    assert ma["final_train_loss"] == mb["final_train_loss"]
    assert ma["tokens_consumed"] == mb["tokens_consumed"]
    assert ma["steps"] == mb["steps"]


def test_sidecar_records_w2_engine_fields(tmp_path):
    src = str(tmp_path / "src.jsonl")
    _write_synthetic_jsonl(src)
    out = str(tmp_path / "out")
    train_run(_args(
        tmp_path, out, data=src, max_steps_per_epoch=2, val_max_steps=1,
        seq=16, batch=2, grad_accum=2, grad_checkpoint=True,
        attention_backend="sdpa", log_every_steps=1,
    ))
    sidecar = json.load(open(os.path.join(out, RUN_METADATA_FILENAME)))
    cfg = sidecar["training_config"]
    assert cfg["grad_accum"] == 2 and cfg["grad_checkpoint"] is True
    metrics = json.load(open(os.path.join(out, "metrics.json")))
    engine = metrics["engine"]
    assert engine["grad_accum"] == 2
    assert engine["grad_checkpoint"] is True
    assert engine["micro_batches_per_step"] == 2
    assert metrics["checkpoint"].endswith("step-2.pt")


# ---------------------------------------------------------------------------
# W2 bench lane — bf16 A/B opt-in
# ---------------------------------------------------------------------------
def test_bench_lanes_bf16_opt_in():
    from scripts.bench_train import DEFAULT_LANES, LANES
    assert "bf16" in LANES and LANES["bf16"]["amp"] == "bf16"
    assert "bf16" not in DEFAULT_LANES, (
        "bf16 is an opt-in lane: the T4 default run must stay old/fp32/fp16"
    )


def test_bench_train_bf16_lane_dry_run(tmp_path):
    from scripts.bench_train import main
    from tests.test_trainer_engine import _packed_mini
    packed_dir, tok_path, _paths, _manifest = _packed_mini(tmp_path)
    out = str(tmp_path / "bench")
    rc = main([
        "--packed-dir", packed_dir, "--preset", "tiny", "--dry-run",
        "--batch", "2", "--tokenizer-json", tok_path, "--out-dir", out,
        "--lanes", "old,bf16",
    ])
    assert rc == 0
    rows = {r["lane"]: r for r in json.load(
        open(os.path.join(out, "bench.json")))["rows"]}
    assert set(rows) == {"old", "bf16"}
    assert rows["bf16"]["amp"] == "bf16"
    assert rows["bf16"]["tok_s_run_wide"] is not None