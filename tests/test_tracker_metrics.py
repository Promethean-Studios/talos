"""The four tracker metrics, after a resume.

The owner reported four broken metrics on their L4 run: the training-loss graph,
the ETA, tokens/sec and the elapsed time. They share one root cause — accounting
that lives only inside a session:

* ``elapsed_s`` was measured from the session's own start, so it restarted at
  zero on every resume (and the ETA, derived from it, was wrong in the opposite
  direction);
* ``tokens_per_sec`` divided the run's LIFETIME token count by the SESSION's
  elapsed time, so a resumed session's first row spiked (5B tokens "in 30
  seconds") and then decayed for hours;
* ``train_loss`` started as ``nan`` in every session, so a resumed session that
  skipped its first updates published a ``nan`` point;
* the published history (``metrics.jsonl`` / ``history.json``) was rewritten from
  each session's in-memory rows, so everything before the resume was erased —
  the loss graph restarted at the resume point.

These tests pin the fixed behaviour: cumulative elapsed, a rolling
current-session rate, a finite loss on every row, a merged history, and the
tracker's field names/order unchanged.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOKS = _ROOT / "notebooks"
for _path in (str(_ROOT), str(_NOTEBOOKS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import talos_training_lib as tlib  # noqa: E402

#: Every field a synced metrics row must carry, in order (unchanged schema).
REQUIRED_ROW_FIELDS = (
    "step", "epoch", "tokens_so_far", "total_tokens", "tokens_per_sec",
    "elapsed_s", "eta_s", "train_loss", "val_loss", "checkpoint", "gpu",
    "model", "params", "vocab", "seq_len", "timestamp_utc", "stage",
    "skipped_updates",
)

OWNER_STATUS_FIELDS = (
    "status", "model_name", "parameters", "step", "step_target",
    "tokens_consumed", "token_target", "train_loss", "val_loss", "checkpoint",
    "checkpoint_timestamp", "gpu", "vocab_size", "sequence_length", "timestamp",
)


def _row(step: int, *, stage: str = "A", tokens: int = 65_536, elapsed: float = 60.0,
         loss: float = 2.5) -> "tlib.MetricsRow":
    return tlib.build_metrics_row(
        step=step, epoch=1, tokens_so_far=tokens * step, total_tokens=10 ** 9,
        elapsed_s=elapsed, train_loss=loss, stage=stage)


# ---------------------------------------------------------------------------
# tokens/sec: the rolling current-session rate, not lifetime-from-zero
# ---------------------------------------------------------------------------
def test_the_window_measures_the_current_sessions_delta():
    """A resume at 5B tokens reports THIS session's pace, not 5B/seconds."""
    window = tlib.ThroughputWindow(window_s=300.0)
    window.reset(0.0, 5_000_000_000)          # the resume point
    window.update(10.0, 5_000_065_536)        # one 65,536-token step in 10 s

    assert window.tokens_per_sec() == pytest.approx(6_553.6)

    # The old formula (lifetime tokens / session seconds) is a 500-million
    # tokens/sec spike on the very same numbers — that was the bug.
    assert tlib.compute_tokens_per_sec(5_000_065_536, 10.0) > 1e8


def test_the_window_is_the_trailing_span_and_never_a_lifetime_mean():
    window = tlib.ThroughputWindow(window_s=60.0)
    window.reset(0.0, 0)
    for second in range(1, 11):               # 1 step/s, 8 tokens/step
        window.update(float(second) * 10.0, second * 8)
    # 100 s elapsed, window 60 s: the two oldest samples are pruned.
    assert window.tokens_per_sec() == pytest.approx(8.0 / 10.0)


def test_the_window_needs_two_samples_and_rejects_impossible_inputs():
    window = tlib.ThroughputWindow()
    window.reset(0.0, 100)
    assert window.tokens_per_sec() is None    # nothing measured yet
    assert window.eta_s(100, 1_000) is None
    with pytest.raises(ValueError):
        tlib.ThroughputWindow(window_s=0)
    with pytest.raises(ValueError):
        window.update(-1.0, 0)
    with pytest.raises(ValueError):
        window.update(1.0, -1)


def test_eta_is_remaining_tokens_over_the_rolling_rate_and_never_negative():
    window = tlib.ThroughputWindow()
    window.reset(0.0, 1_000)
    window.update(10.0, 3_000)                # 200 tok/s
    assert window.eta_s(3_000, 5_000) == pytest.approx(10.0)
    assert window.eta_s(5_000, 5_000) == 0.0  # at the budget: 0, never negative
    assert window.eta_s(6_000, 5_000) == 0.0
    with pytest.raises(ValueError):
        window.eta_s(0, 0)


def test_the_row_uses_a_supplied_rate_and_eta_and_keeps_the_schema():
    row = tlib.build_metrics_row(
        step=7, epoch=1, tokens_so_far=3_000, total_tokens=5_000,
        elapsed_s=12_000.0, train_loss=2.0, tokens_per_sec=200.0, eta_s=10.0)

    assert tuple(row.as_dict()) == REQUIRED_ROW_FIELDS
    assert row.tokens_per_sec == 200.0
    assert row.eta_s == 10.0
    # ... and the derived path is untouched when no rate is supplied.
    derived = tlib.build_metrics_row(
        step=7, epoch=1, tokens_so_far=3_000, total_tokens=5_000,
        elapsed_s=12_000.0, train_loss=2.0)
    assert derived.tokens_per_sec == pytest.approx(3_000 / 12_000.0)


@pytest.mark.parametrize("rate,eta", [(-1.0, None), (float("nan"), None),
                                      (1.0, -5.0), (1.0, float("inf"))])
def test_the_row_refuses_an_impossible_rate_or_eta(rate, eta):
    with pytest.raises(ValueError):
        tlib.build_metrics_row(
            step=1, epoch=1, tokens_so_far=10, total_tokens=100, elapsed_s=1.0,
            train_loss=1.0, tokens_per_sec=rate, eta_s=eta)


# ---------------------------------------------------------------------------
# The published history survives a resume (the loss graph)
# ---------------------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.calls = []
        self.payload = ""

    def __call__(self, repo_id, local_path, path_in_repo, token):
        self.calls.append((repo_id, local_path, path_in_repo, token))
        with open(local_path, encoding="utf-8") as handle:
            self.payload = handle.read()


def test_metrics_history_is_merged_across_sessions(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = _Recorder()
    first = tlib.MetricsSync("team/metrics", local_path=str(path),
                             uploader=recorder, min_interval_s=0)
    for step in (1, 2, 3):
        first.maybe_push(_row(step))

    # A NEW session (a Colab resume) that re-treads step 3 and continues.
    second = tlib.MetricsSync("team/metrics", local_path=str(path),
                              uploader=recorder, min_interval_s=0)
    for step in (3, 4, 5):
        second.maybe_push(_row(step))

    on_disk = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in on_disk] == [1, 2, 3, 4, 5]
    uploaded = [json.loads(line) for line in recorder.payload.splitlines()]
    assert [row["step"] for row in uploaded] == [1, 2, 3, 4, 5]
    for row in on_disk:
        assert tuple(row) == REQUIRED_ROW_FIELDS


def test_a_poisoned_metrics_history_heals_on_the_next_push(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "stage": "A", "train_loss": NaN}\n',
                    encoding="utf-8")
    sync = tlib.MetricsSync("team/metrics", local_path=str(path),
                            uploader=_Recorder(), min_interval_s=0)
    sync.maybe_push(_row(2))

    text = path.read_text()
    assert "NaN" not in text
    assert [json.loads(line)["step"] for line in text.splitlines()] == [2]


def test_tracker_history_is_merged_across_sessions(tmp_path):
    def publisher():
        return tlib.TrackerPublisher(local_dir=str(tmp_path), dry_run=True,
                                     min_interval_s=0)

    first = publisher()
    for step in (1, 2, 3):
        first.note_checkpoint(step, step * 65_536)
        first.push(_row(step))

    second = publisher()                    # a resumed session: empty in memory
    second.note_checkpoint(4, 4 * 65_536)
    second.push(_row(4))

    history = json.loads((tmp_path / "history.json").read_text())
    assert tuple(history) == tlib.TRACKER_HISTORY_FIELDS
    assert [row["step"] for row in history["epochs"]] == [1, 2, 3, 4]
    assert [entry["step"] for entry in history["checkpoints"]] == [1, 2, 3, 4]

    status = json.loads((tmp_path / "status.json").read_text())
    assert tuple(status) == tlib.TRACKER_STATUS_FIELDS + tlib.TRACKER_EXTRA_STATUS_FIELDS
    assert status["step"] == 4
    assert status["train_loss"] == pytest.approx(2.5)


def test_status_never_carries_a_nan_loss(tmp_path):
    tracker = tlib.TrackerPublisher(local_dir=str(tmp_path), dry_run=True)
    status = tracker.build_status(step=1, tokens_consumed=10,
                                  train_loss=float("nan"),
                                  val_loss=float("inf"))
    assert status["train_loss"] is None and status["val_loss"] is None
    tracker.push()
    assert "NaN" not in (tmp_path / "status.json").read_text()


# ---------------------------------------------------------------------------
# A real (CPU) two-session run through the training loop
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from configs.presets import ALL_PRESETS  # noqa: E402
from model.gpt import TalosGPT  # noqa: E402
from talos_stages import TokenAccountant  # noqa: E402
from talos_train_loop import TrainLoopConfig, train_stage  # noqa: E402

SEQ = 8
TOKENS_PER_STEP = SEQ
BUDGET_STEPS = 4


class FakeStream:
    def __init__(self, rows):
        self._rows = list(rows)
        self.position = SimpleNamespace(epoch=1, rows_in_epoch=0, pending_ids=[])

    def __iter__(self):
        return iter(self._rows)


def _rows(count: int = 64):
    return [SimpleNamespace(ids=[(i * 7 + j) % 1000 for j in range(SEQ)], mask=None)
            for i in range(count)]


def _config() -> TrainLoopConfig:
    return TrainLoopConfig(
        micro_batch=1, grad_accum=1, seq_len=SEQ, total_steps=BUDGET_STEPS,
        warmup_steps=0, peak_lr=1e-3, log_every=1, eval_every=10 ** 6,
        checkpoint_every=10 ** 6, checkpoint_every_min=0.0,
        throughput_window_s=300.0)


def _session(*, seconds_per_step: float, now: list, accountant: TokenAccountant,
             model, optimizer, infos: list, skips: list, poison_first: bool = False,
             elapsed_offset_s: float = 0.0, initial_train_loss=None):
    calls = {"n": 0}

    def forward_loss(ids, mask=None):
        calls["n"] += 1
        now[0] += seconds_per_step
        if poison_first and calls["n"] == 1:
            return torch.tensor(float("nan"), requires_grad=True)
        logits, _ = model(ids[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))

    return train_stage(
        model=model, optimizer=optimizer, scaler=None, accountant=accountant,
        stream=FakeStream(_rows()), forward_loss=forward_loss, config=_config(),
        stage_key="A", device="cpu", evaluate=None,
        on_log=infos.append, on_skip=lambda reason, info: skips.append(info),
        say=lambda *args, **kwargs: None, clock=lambda: now[0],
        elapsed_offset_s=elapsed_offset_s, initial_train_loss=initial_train_loss,
    )


def _build():
    torch.manual_seed(0)
    model = TalosGPT(ALL_PRESETS["tiny"]().derive())
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def _by_step(infos):
    """One info per step (the loop logs once more at the end)."""
    seen = {}
    for info in infos:
        seen.setdefault(info.step, info)
    return [seen[step] for step in sorted(seen)]


def test_the_four_metrics_stay_correct_across_a_resume():
    model, optimizer = _build()
    now = [0.0]

    # --- session 1: 2 steps at 5 s/step --------------------------------
    accountant = TokenAccountant(TOKENS_PER_STEP * 2)
    infos_1: list = []
    result_1 = _session(seconds_per_step=5.0, now=now, accountant=accountant,
                        model=model, optimizer=optimizer, infos=infos_1, skips=[])
    assert result_1.steps == 2 and result_1.tokens_consumed == 2 * TOKENS_PER_STEP
    rows_1 = _by_step(infos_1)
    assert [row.elapsed_s for row in rows_1] == pytest.approx([5.0, 10.0])
    assert all(row.train_loss == row.train_loss for row in rows_1)   # finite

    # --- session 2: resume at step 2, 2 more steps at 1 s/step ---------
    accountant_2 = TokenAccountant(TOKENS_PER_STEP * BUDGET_STEPS)
    accountant_2.consumed = result_1.tokens_consumed
    accountant_2.steps = result_1.steps
    infos_2: list = []
    result_2 = _session(seconds_per_step=1.0, now=now, accountant=accountant_2,
                        model=model, optimizer=optimizer, infos=infos_2, skips=[],
                        elapsed_offset_s=result_1.elapsed_s,
                        initial_train_loss=result_1.train_loss)

    rows_2 = _by_step(infos_2)
    assert [row.step for row in rows_2] == [3, 4]
    # elapsed time is CUMULATIVE: it continues from session 1, never resets.
    assert [row.elapsed_s for row in rows_2] == pytest.approx([11.0, 12.0])
    assert result_2.elapsed_s == pytest.approx(12.0)
    # tokens/sec is this session's pace (8 tokens per 1 s step), not a spike.
    for row in rows_2:
        assert row.tokens_per_sec == pytest.approx(8.0)
        assert row.eta_s is not None and row.eta_s >= 0.0
    # The old formula on the same numbers: 32 lifetime tokens / 2 session
    # seconds = 16 tok/s, a 2x spike that grew with the resume point.
    assert tlib.compute_tokens_per_sec(result_2.tokens_consumed, 2.0) == 16.0
    # Every row is finite and plottable, before and after the boundary.
    for row in rows_1 + rows_2:
        assert math.isfinite(row.train_loss) and math.isfinite(row.tokens_per_sec)
        assert row.elapsed_s > 0 and row.tokens_consumed > 0
    assert result_2.tokens_consumed == TOKENS_PER_STEP * BUDGET_STEPS


def test_a_resumed_session_seeds_the_loss_so_a_skip_publishes_a_real_number():
    model, optimizer = _build()
    now = [0.0]
    accountant = TokenAccountant(TOKENS_PER_STEP * BUDGET_STEPS)
    accountant.consumed = 2 * TOKENS_PER_STEP
    accountant.steps = 2
    skips: list = []

    result = _session(seconds_per_step=1.0, now=now, accountant=accountant,
                      model=model, optimizer=optimizer, infos=[], skips=skips,
                      poison_first=True, elapsed_offset_s=10.0,
                      initial_train_loss=3.25)

    assert result.skipped_updates == 1
    assert skips, "the skipped update must be reported"
    # Without the seed this is nan — the point the owner's loss graph choked on.
    assert skips[0].train_loss == pytest.approx(3.25)
    assert math.isfinite(skips[0].elapsed_s) and skips[0].elapsed_s >= 10.0
