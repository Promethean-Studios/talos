"""Regression: the whole SkipTracker state survives a session boundary (fix 3).

``SkipTracker`` holds three counters (``total``, ``consecutive``,
``last_reason``) and ``train_stage`` used to build a **fresh** tracker on every
call, so a resumed run restarted its skip accounting at zero: the published
``skipped_updates`` under-reported the run, and the 20-consecutive-skip abort
guard reset at every session boundary.

Now all three fields go into the checkpoint metadata and back into a fresh
tracker through ``train_stage(skip_state=...)``:

* ``total`` continues across sessions;
* ``consecutive`` is preserved, so a session that resumes with the counter at
  the limit aborts on its next skip (correct — the abort message already says to
  resume from an older step);
* ``last_reason`` is restored;
* an old checkpoint without the fields starts at zero with one loud warning and
  never crashes.

Torch-free: the loop is driven with the budget already spent (no forward pass),
and the torch-only end-to-end case skips where torch is absent.
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

from talos_stages import (MAX_CONSECUTIVE_SKIPPED, NonFiniteAbort,  # noqa: E402
                          SkipTracker, TokenAccountant, skip_state_from_metadata)
from talos_train_loop import TrainLoopConfig, train_stage  # noqa: E402

class _Stream:
    """Minimal row stream: the loop only needs ``iter()`` and a position."""

    def __init__(self, rows=()):
        self._rows = list(rows)
        self.position = SimpleNamespace(epoch=1, rows_in_epoch=0)

    def __iter__(self):
        return iter(self._rows)


STATE = {"total": 7, "consecutive": 2, "last_reason": "non-finite loss (nan)"}


# ---------------------------------------------------------------------------
# Reading the counters out of a checkpoint's metadata
# ---------------------------------------------------------------------------
def test_all_three_fields_are_read_from_the_metadata():
    metadata = {"skipped_updates": 7, "consecutive_skips": 2,
                "last_skip_reason": "non-finite loss (nan)"}
    assert skip_state_from_metadata(metadata) == STATE


def test_a_checkpoint_without_the_fields_reports_none():
    # Old checkpoint: no skip counters at all -> the caller warns and starts at 0.
    assert skip_state_from_metadata({}) is None
    assert skip_state_from_metadata(None) is None
    assert skip_state_from_metadata({"stage": "A"}) is None
    # Only the total present (the pre-fix field): still a usable state.
    assert skip_state_from_metadata({"skipped_updates": 3}) == {
        "total": 3, "consecutive": 0, "last_reason": ""}


def test_a_tracker_is_restored_from_that_state():
    tracker = SkipTracker.from_state(STATE, max_consecutive=5)
    assert (tracker.total, tracker.consecutive) == (7, 2)
    assert tracker.last_reason == STATE["last_reason"]
    # The limit always comes from the current config, never the checkpoint.
    assert tracker.max_consecutive == 5
    with pytest.raises(ValueError):
        SkipTracker.from_state("nonsense", max_consecutive=5)


def test_a_restored_counter_at_the_limit_aborts_on_the_next_skip():
    tracker = SkipTracker.from_state(
        {"total": 19, "consecutive": MAX_CONSECUTIVE_SKIPPED, "last_reason": "x"},
        max_consecutive=MAX_CONSECUTIVE_SKIPPED)
    with pytest.raises(NonFiniteAbort, match="consecutive"):
        tracker.note_skip("still not finite")
    assert tracker.total == 20


# ---------------------------------------------------------------------------
# The loop: a fresh train_stage continues the restored counters
# ---------------------------------------------------------------------------
def _loop_config(max_consecutive_skips: int = 5) -> TrainLoopConfig:
    return TrainLoopConfig(
        micro_batch=1, grad_accum=1, seq_len=8, total_steps=1, warmup_steps=0,
        peak_lr=1e-3, log_every=1, eval_every=10, checkpoint_every=10,
        checkpoint_every_min=0.0, max_consecutive_skips=max_consecutive_skips,
    )


def _run(*, skip_state=None, budget: int = 1, max_consecutive_skips: int = 5):
    """Run one stage whose budget is already spent: no forward pass happens."""
    checkpoints = []
    lines = []
    result = train_stage(
        model=SimpleNamespace(train=lambda: None), optimizer=None, scaler=None,
        accountant=TokenAccountant(budget),
        stream=_Stream(),
        forward_loss=lambda ids, mask=None: pytest.fail("no forward pass expected"),
        config=_loop_config(max_consecutive_skips),
        stage_key="A", device="cpu",
        on_checkpoint=lambda info: checkpoints.append(info) or "step-0.pt",
        evaluate=None, on_log=None, on_skip=None,
        say=lines.append, skip_state=skip_state,
    )
    return result, checkpoints, lines


def test_a_fresh_train_stage_restores_all_three_counters():
    result, checkpoints, lines = _run(skip_state=STATE)
    assert result.skipped_updates == 7
    assert result.consecutive_skips == 2
    assert result.last_skip_reason == STATE["last_reason"]
    # The checkpoint written at the end carries them on, so the NEXT session
    # restores the same state.
    assert checkpoints, "a checkpoint must be written at the end of the stage"
    info = checkpoints[-1]
    assert (info.skipped_updates, info.consecutive_skips) == (7, 2)
    assert info.last_skip_reason == STATE["last_reason"]
    assert info.as_dict()["consecutive_skips"] == 2
    assert any("skip counters restored" in line for line in lines)


def test_no_skip_state_starts_at_zero():
    result, _, lines = _run(skip_state=None)
    assert (result.skipped_updates, result.consecutive_skips) == (0, 0)
    assert result.last_skip_reason == ""
    assert not any("restored" in line for line in lines)


def test_an_old_checkpoint_starts_at_zero_without_crashing():
    """`skip_state_from_metadata` returns None -> the notebook warns, loop is 0."""
    state = skip_state_from_metadata({"stage": "A", "skipped_updates": None})
    result, _, _ = _run(skip_state=state)
    assert (result.skipped_updates, result.consecutive_skips) == (0, 0)


# ---------------------------------------------------------------------------
# End to end with real tensors: the restored burst guard spans sessions
# ---------------------------------------------------------------------------
def test_a_restored_consecutive_burst_aborts_on_the_first_skip_of_the_session():
    torch = pytest.importorskip("torch")
    from configs.presets import ALL_PRESETS
    from model.gpt import TalosGPT

    torch.manual_seed(0)
    model = TalosGPT(ALL_PRESETS["tiny"]().derive())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = [SimpleNamespace(ids=[(i * 7 + j) % 1000 for j in range(8)], mask=None)
            for i in range(4)]
    config = TrainLoopConfig(
        micro_batch=1, grad_accum=1, seq_len=8, total_steps=4, warmup_steps=0,
        peak_lr=1e-3, log_every=1, eval_every=10, checkpoint_every=10,
        checkpoint_every_min=0.0, max_consecutive_skips=3)

    def forward_loss(ids, mask=None):
        return torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(NonFiniteAbort, match="consecutive"):
        train_stage(
            model=model, optimizer=optimizer, scaler=None,
            accountant=TokenAccountant(4 * 8),
            stream=_Stream(rows),
            forward_loss=forward_loss, config=config, stage_key="A", device="cpu",
            on_checkpoint=lambda info: "step.pt", evaluate=None, on_log=None,
            on_skip=None, say=lambda *a, **k: None,
            # Two skips were already recorded before this session started, so the
            # THIRD consecutive skip (this session's first) trips the limit.
            skip_state={"total": 2, "consecutive": 2, "last_reason": "nan"},
        )
