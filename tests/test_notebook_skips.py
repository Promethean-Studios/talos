"""Fix 4 — non-finite loss/gradient handling: skip the update, never poison.

A single NaN/Inf batch must not reach the optimizer, and the run must not spin
silently either. The contract the notebook relies on:

* the loss is checked **before** ``backward()``;
* the pre-clip gradient norm is checked **before** the optimizer step;
* an anomaly skips the whole update — gradients dropped, no optimizer step, no
  LR advance, no step-counter advance — while the tokens the forward pass
  consumed are still counted (matching the partial-step accounting, and
  guaranteeing forward progress);
* ``skipped_updates`` increments and is carried in the metrics row, the
  checkpoint metadata and the tracker's ``status.json``;
* a burst of consecutive skips aborts loudly with ``NonFiniteAbort`` after
  writing a checkpoint.

The tests drive the real ``train_stage`` loop with the canonical ``tiny`` preset
(254,272 parameters) on the CPU — the same code path the 1B run uses.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOTEBOOKS = os.path.join(_ROOT, "notebooks")
for _path in (_ROOT, _NOTEBOOKS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from configs.presets import ALL_PRESETS  # noqa: E402
from model.gpt import TalosGPT  # noqa: E402
from talos_stages import (NonFiniteAbort, SkipTracker, TokenAccountant,  # noqa: E402
                          lr_at)
from talos_train_loop import TrainLoopConfig, train_stage  # noqa: E402

SEQ = 8
MICRO_BATCH = 1
GRAD_ACCUM = 1
TOKENS_PER_STEP = MICRO_BATCH * GRAD_ACCUM * SEQ
PEAK_LR = 1e-3
NAN = float("nan")


class FakeStream:
    """Minimal row stream: the loop only needs ``iter()`` and a position."""

    def __init__(self, rows):
        self._rows = list(rows)
        self.position = SimpleNamespace(epoch=1, rows_in_epoch=0, pending_ids=[])

    def __iter__(self):
        return iter(self._rows)


class RecordingAdamW(torch.optim.AdamW):
    """AdamW that records the LR actually used for every applied update."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_lrs = []

    def step(self, closure=None):
        self.step_lrs.append(float(self.param_groups[0]["lr"]))
        return super().step(closure)


def _rows(count: int = 40):
    """Packed rows with the ``.ids`` / ``.mask`` shape ``rows_to_tensor`` expects."""
    return [SimpleNamespace(ids=[(i * 7 + j) % 1000 for j in range(SEQ)], mask=None)
            for i in range(count)]


def _build(*, max_consecutive_skips: int = 5):
    torch.manual_seed(0)
    model = TalosGPT(ALL_PRESETS["tiny"]().derive())
    optimizer = RecordingAdamW(model.parameters(), lr=PEAK_LR)
    stream = FakeStream(_rows())
    return model, optimizer, stream


def _run(*, poison, steps: int = 3, max_consecutive_skips: int = 5,
         model=None, optimizer=None, stream=None, checkpoints=None):
    """Train ``steps`` optimizer steps with ``poison(call_index, model)`` applied."""
    if model is None:
        model, optimizer, stream = _build(max_consecutive_skips=max_consecutive_skips)
    budget = steps * TOKENS_PER_STEP
    accountant = TokenAccountant(budget)
    config = TrainLoopConfig(
        micro_batch=MICRO_BATCH, grad_accum=GRAD_ACCUM, seq_len=SEQ,
        total_steps=steps, warmup_steps=0, peak_lr=PEAK_LR,
        log_every=1, eval_every=10_000, checkpoint_every=10_000,
        checkpoint_every_min=0.0, max_consecutive_skips=max_consecutive_skips,
    )
    calls = {"n": 0}

    def forward_loss(ids, mask=None):
        calls["n"] += 1
        poisoned = poison(calls["n"], model)
        if poisoned is not None:
            return poisoned
        logits, _ = model(ids[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))

    def on_checkpoint(info):
        if checkpoints is not None:
            checkpoints.append(info.as_dict())
        return "step.pt"

    result = train_stage(
        model=model, optimizer=optimizer, scaler=None, accountant=accountant,
        stream=stream, forward_loss=forward_loss, config=config,
        stage_key="A", device="cpu", on_checkpoint=on_checkpoint,
        evaluate=None, on_log=None, on_skip=None, say=lambda *a, **k: None,
    )
    return result, accountant, model, optimizer


def _nan_on_first_call(call_index, model):
    return torch.tensor(NAN, requires_grad=True) if call_index == 1 else None


# ---------------------------------------------------------------------------
# NaN loss: skip the update, keep the tokens, keep the schedule
# ---------------------------------------------------------------------------
def test_a_nan_loss_skips_the_update_but_not_the_tokens():
    result, accountant, _, optimizer = _run(poison=_nan_on_first_call, steps=3)

    assert result.skipped_updates == 1
    assert result.steps == 2                      # the skipped step did not count
    assert accountant.steps == 2
    assert result.tokens_consumed == 3 * TOKENS_PER_STEP   # consumed either way
    assert accountant.consumed == 3 * TOKENS_PER_STEP
    assert result.reached_budget is True
    assert "non-finite loss" in result.last_skip_reason
    assert result.consecutive_skips == 0          # reset by the next good update
    assert len(optimizer.step_lrs) == 2


def test_the_schedule_does_not_advance_on_a_skipped_update():
    _, _, _, optimizer = _run(poison=_nan_on_first_call, steps=3)

    expected = [lr_at(step, peak_lr=PEAK_LR, total_steps=3, warmup=0,
                      min_lr_ratio=0.1)
                for step in (0, 1)]                  # steps 0 and 1, not 1 and 2
    assert optimizer.step_lrs == pytest.approx(expected)


def test_a_skipped_update_does_not_touch_the_weights():
    model, optimizer, stream = _build()
    before = {name: tensor.detach().clone()
              for name, tensor in model.state_dict().items()}

    result, _, _, _ = _run(poison=_nan_on_first_call, steps=1, model=model,
                           optimizer=optimizer, stream=stream)

    assert result.skipped_updates == 1
    assert result.steps == 0
    assert optimizer.step_lrs == []                # the optimizer never stepped
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), f"{name} changed on a skipped update"


# ---------------------------------------------------------------------------
# Infinite gradients (finite loss): same treatment
# ---------------------------------------------------------------------------
def test_an_infinite_gradient_skips_the_update():
    def poison(call_index, model):
        if call_index != 1:
            return None
        # A finite loss whose gradient is infinite: d/dx sqrt(|x|) at x == 0.
        target = next(p for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            target.view(-1)[0] = 0.0
        return target.abs().sqrt().sum()

    result, accountant, _, optimizer = _run(poison=poison, steps=2)

    assert result.skipped_updates == 1
    assert result.steps == 1
    assert optimizer.step_lrs and len(optimizer.step_lrs) == 1
    assert "non-finite gradient" in result.last_skip_reason
    assert accountant.consumed == 2 * TOKENS_PER_STEP


# ---------------------------------------------------------------------------
# A burst of skips aborts loudly (after a checkpoint) instead of spinning
# ---------------------------------------------------------------------------
def test_a_burst_of_consecutive_skips_aborts_loudly():
    checkpoints: list = []

    def always_nan(call_index, model):
        return torch.tensor(NAN, requires_grad=True)

    with pytest.raises(NonFiniteAbort, match="consecutive"):
        _run(poison=always_nan, steps=10, max_consecutive_skips=3,
             checkpoints=checkpoints)

    assert checkpoints, "a checkpoint must be written before the abort"


# ---------------------------------------------------------------------------
# The counter itself
# ---------------------------------------------------------------------------
def test_skip_tracker_counts_and_resets():
    tracker = SkipTracker(max_consecutive=3)
    tracker.note_skip("bad loss")
    assert (tracker.total, tracker.consecutive) == (1, 1)
    tracker.note_update()
    assert tracker.consecutive == 0
    assert tracker.total == 1
    tracker.note_skip("bad grad")
    assert tracker.as_dict() == {"skipped_updates": 2, "consecutive_skips": 1,
                                 "last_skip_reason": "bad grad"}


def test_skip_tracker_aborts_on_the_configured_burst():
    tracker = SkipTracker(max_consecutive=2)
    tracker.note_skip("first")
    with pytest.raises(NonFiniteAbort, match="2 consecutive"):
        tracker.note_skip("second")


def test_skip_tracker_rejects_a_useless_limit():
    with pytest.raises(ValueError):
        SkipTracker(max_consecutive=0)
