"""Tests for the 1B training notebook's testable helpers and the notebook itself.

Covers ``notebooks/talos_training_lib.py`` (metrics row construction, ETA math,
display table, chat→packed-token adapter, HF Dataset sync incl. the dry-run
no-op and the never-raise failure path) and validates that
``notebooks/talos_colab_train.ipynb`` is well-formed JSON whose code cells
compile and which carries the owner's required configuration and display
fields.

No training, no GPU, no network: the sync uploader is injected.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOTEBOOKS_DIR = os.path.join(_REPO_ROOT, "notebooks")
if _NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, _NOTEBOOKS_DIR)

import talos_training_lib as tlib  # noqa: E402

NOTEBOOK_PATH = os.path.join(_NOTEBOOKS_DIR, "talos_colab_train.ipynb")

#: Every field a synced metrics row must carry (spec §7).
REQUIRED_ROW_FIELDS = (
    "step", "epoch", "tokens_so_far", "total_tokens", "tokens_per_sec",
    "elapsed_s", "eta_s", "train_loss", "val_loss", "checkpoint", "gpu",
    "model", "params", "vocab", "seq_len", "timestamp_utc",
    # Additive: the two-stage run records which stage produced the row.
    "stage",
    # Additive: optimizer updates skipped on NaN/Inf loss or gradients.
    "skipped_updates",
)


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
def _row(**overrides):
    kwargs = dict(
        step=500,
        epoch=3,
        tokens_so_far=5_000_000_000,
        total_tokens=20_000_000_000,
        elapsed_s=10_000.0,
        train_loss=2.5,
        val_loss=2.75,
        checkpoint="step-500.pt",
        gpu="NVIDIA L4",
    )
    kwargs.update(overrides)
    return tlib.build_metrics_row(**kwargs)


def test_metrics_row_has_every_required_field_with_the_right_type():
    row = _row()
    payload = row.as_dict()
    assert tuple(payload) == REQUIRED_ROW_FIELDS
    assert isinstance(payload["step"], int) and payload["step"] == 500
    assert isinstance(payload["epoch"], int) and payload["epoch"] == 3
    assert isinstance(payload["tokens_so_far"], int)
    assert isinstance(payload["total_tokens"], int)
    assert isinstance(payload["tokens_per_sec"], float)
    assert isinstance(payload["elapsed_s"], float)
    assert isinstance(payload["eta_s"], float)
    assert isinstance(payload["train_loss"], float)
    assert isinstance(payload["val_loss"], float)
    assert isinstance(payload["checkpoint"], str)
    assert isinstance(payload["gpu"], str)
    assert isinstance(payload["model"], str)
    assert isinstance(payload["params"], int)
    assert isinstance(payload["vocab"], int)
    assert isinstance(payload["seq_len"], int)
    assert isinstance(payload["timestamp_utc"], str)


def test_metrics_row_pins_the_talos_1b_facts():
    row = _row()
    assert row.model == "talos_1b"
    assert row.params == 1_036_060_672
    assert row.vocab == 49_152
    assert row.seq_len == 2048
    assert row.total_tokens == 20_000_000_000


def test_metrics_row_timestamp_is_exact_utc():
    when = datetime(2026, 11, 17, 14, 3, 22, tzinfo=timezone.utc)
    row = _row(timestamp=when)
    assert row.timestamp_utc == "2026-11-17T14:03:22Z"
    assert tlib.utc_display(row.timestamp_utc) == "2026-11-17 14:03:22 UTC"


def test_metrics_row_val_loss_may_be_none_before_the_first_eval():
    row = _row(val_loss=None)
    assert row.val_loss is None
    assert row.as_dict()["val_loss"] is None
    assert "null" in row.to_jsonl()


@pytest.mark.parametrize(
    "overrides",
    [
        {"step": -1},
        {"epoch": 0},
        {"tokens_so_far": -5},
        {"total_tokens": 0},
        {"elapsed_s": -1.0},
        {"train_loss": float("nan")},
        {"val_loss": float("inf")},
        {"seq_len": 0},
        {"params": 0},
    ],
)
def test_metrics_row_rejects_impossible_values(overrides):
    with pytest.raises(ValueError):
        _row(**overrides)


# ---------------------------------------------------------------------------
# ETA / throughput math
# ---------------------------------------------------------------------------
def test_eta_is_the_remaining_fraction_of_the_elapsed_time():
    # Half the budget in 10,000 s -> the other half needs 10,000 s.
    assert tlib.compute_eta_s(10_000_000_000, 20_000_000_000, 10_000.0) == pytest.approx(10_000.0)
    # A quarter of the budget in 400 s -> the remaining 3/4 needs 1,200 s.
    assert tlib.compute_eta_s(5_000_000_000, 20_000_000_000, 400.0) == pytest.approx(1_200.0)


def test_eta_is_zero_at_or_past_the_budget():
    assert tlib.compute_eta_s(20_000_000_000, 20_000_000_000, 10.0) == 0.0
    assert tlib.compute_eta_s(21_000_000_000, 20_000_000_000, 10.0) == 0.0


def test_eta_is_none_when_it_cannot_be_estimated_yet():
    assert tlib.compute_eta_s(0, 20_000_000_000, 10.0) is None
    assert tlib.compute_eta_s(1_000, 20_000_000_000, 0.0) is None


def test_eta_rejects_a_non_positive_budget():
    with pytest.raises(ValueError):
        tlib.compute_eta_s(10, 0, 1.0)


def test_tokens_per_sec_and_its_zero_guards():
    assert tlib.compute_tokens_per_sec(65_536, 2.0) == pytest.approx(32_768.0)
    assert tlib.compute_tokens_per_sec(65_536, 0.0) == 0.0
    assert tlib.compute_tokens_per_sec(0, 5.0) == 0.0


def test_row_throughput_and_eta_agree_with_the_helpers():
    row = _row()
    assert row.tokens_per_sec == pytest.approx(5_000_000_000 / 10_000.0)
    assert row.eta_s == pytest.approx(30_000.0)   # 3/4 of the budget left at the same rate


# ---------------------------------------------------------------------------
# Formatting + display table
# ---------------------------------------------------------------------------
def test_format_helpers():
    assert tlib.format_count(1_036_060_672) == "1,036,060,672"
    assert tlib.format_billions(20_000_000_000) == "20.00B"
    assert tlib.format_duration(0) == "00m 00s"
    assert tlib.format_duration(3_661) == "1h 01m 01s"
    assert tlib.format_duration(90_061) == "1d 01h 01m"
    assert tlib.format_duration(None) == "—"
    assert tlib.format_loss(None) == "—"
    assert tlib.format_loss(2.5) == "2.5000"


def test_display_table_has_every_owner_field_and_the_live_values():
    html = tlib.render_display_table(_row())
    for label in (
        "Token progress", "Training step", "Tokens/sec", "Elapsed time", "ETA",
        "Train loss", "Val loss", "Checkpoint", "GPU", "Model", "Parameters",
        "Vocab", "Sequence length", "Epoch", "Last update",
    ):
        assert label in html, label
    assert "out of 20.00B" in html          # token progress, x out of 20b
    assert "5,000,000,000" in html
    assert "step-500.pt" in html            # checkpoint filename
    assert "1,036,060,672" in html          # parameters
    assert "49,152" in html                 # vocab
    assert "2,048" in html                  # sequence length
    assert "2026-11-17 14:03:22 UTC" not in html  # timestamp is generated, not pinned
    assert html.count("<tr>") == len(tlib.DISPLAY_LABELS)


def test_display_table_renders_missing_values_as_placeholders():
    html = tlib.render_display_table(_row(val_loss=None, checkpoint="", gpu=""))
    assert "—" in html
    assert "step-500.pt" not in html


# ---------------------------------------------------------------------------
# Chat -> packed token stream
# ---------------------------------------------------------------------------
def test_linearize_dialog_matches_the_documented_format():
    text = tlib.linearize_dialog([
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": " hello "},
    ])
    assert text == "User: hi there\n\nAssistant: hello"


def test_linearize_dialog_degrades_on_junk_rows():
    assert tlib.linearize_dialog([]) == ""
    assert tlib.linearize_dialog([{"content": "no role"}]) == "Unknown: no role"
    assert tlib.linearize_dialog([{"role": "user"}]) == "User:"


def test_pack_token_stream_packs_end_to_end_and_drops_the_tail():
    rows = list(tlib.pack_token_stream([[1, 2, 3], [4, 5, 6, 7]], seq_len=4))
    assert rows == [[1, 2, 3, 4]]
    rows = list(tlib.pack_token_stream([[1, 2, 3], [4, 5, 6, 7]], seq_len=4,
                                       drop_last=False))
    assert rows == [[1, 2, 3, 4], [5, 6, 7, 0]]


def test_pack_token_stream_rejects_ids_outside_the_vocab():
    with pytest.raises(ValueError, match="outside"):
        list(tlib.pack_token_stream([[0, 49_152]], seq_len=2, vocab_size=49_152))
    with pytest.raises(ValueError, match="outside"):
        list(tlib.pack_token_stream([[-1, 5]], seq_len=2, vocab_size=49_152))


# ---------------------------------------------------------------------------
# HF Dataset sync
# ---------------------------------------------------------------------------
class _Recorder:
    """Stand-in uploader: records every call, optionally raising."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def __call__(self, repo_id, local_path, path_in_repo, token):
        self.calls.append((repo_id, local_path, path_in_repo, token))
        if self.fail:
            raise RuntimeError("hub is down")
        assert os.path.isfile(local_path)
        with open(local_path, encoding="utf-8") as handle:
            self.payload = handle.read()


def _sync(tmp_path, **kwargs):
    kwargs.setdefault("repo_id", "talos-team/talos-1b-training-metrics")
    kwargs.setdefault("local_path", str(tmp_path / "metrics.jsonl"))
    return tlib.MetricsSync(**kwargs)


def test_sync_dry_run_appends_locally_and_never_uploads(tmp_path):
    recorder = _Recorder()
    sync = _sync(tmp_path, dry_run=True, uploader=recorder, min_interval_s=0)
    assert sync.maybe_push(_row()) is True     # dry run is a no-op success
    assert recorder.calls == []
    assert sync.push_count == 0
    assert len(sync.rows) == 1
    assert os.path.isfile(sync.local_path)
    with open(sync.local_path, encoding="utf-8") as handle:
        assert json.loads(handle.readline())["step"] == 500


def test_sync_pushes_one_jsonl_file_with_every_required_field(tmp_path):
    recorder = _Recorder()
    sync = _sync(tmp_path, uploader=recorder, token="hf_test", min_interval_s=0)
    assert sync.push(_row(step=1)) is True
    assert sync.push(_row(step=2)) is True
    assert sync.push_count == 2
    assert recorder.calls[0][0] == "talos-team/talos-1b-training-metrics"
    assert recorder.calls[0][2] == "metrics.jsonl"
    assert recorder.calls[0][3] == "hf_test"
    lines = [json.loads(line) for line in recorder.payload.splitlines()]
    assert [line["step"] for line in lines] == [1, 2]
    for line in lines:
        assert tuple(line) == REQUIRED_ROW_FIELDS


def test_sync_is_throttled_by_the_interval(tmp_path):
    recorder = _Recorder()
    now = [0.0]
    sync = _sync(tmp_path, uploader=recorder, min_interval_s=300.0,
                 clock=lambda: now[0])
    assert sync.maybe_push(_row(step=1)) is True     # first push goes out
    now[0] = 120.0
    assert sync.maybe_push(_row(step=2)) is False    # too soon
    assert sync.push_count == 1
    now[0] = 400.0
    assert sync.maybe_push(_row(step=3)) is True     # interval elapsed
    assert sync.push_count == 2
    assert len(sync.rows) == 3                       # every row is kept locally
    assert sync.flush() is True                      # forced final push
    assert sync.push_count == 3


def test_sync_failure_never_raises_and_is_recorded(tmp_path):
    recorder = _Recorder(fail=True)
    sync = _sync(tmp_path, uploader=recorder, min_interval_s=0)
    assert sync.maybe_push(_row()) is False          # no exception escapes
    assert sync.last_error is not None and "hub is down" in sync.last_error
    assert len(recorder.calls) == 1
    assert len(sync.rows) == 1                       # local history survives

    recorder.fail = False
    assert sync.maybe_push(_row(step=501)) is True   # next interval recovers
    assert sync.last_error is None


def test_sync_rejects_a_repo_id_that_is_not_owner_slash_name(tmp_path):
    with pytest.raises(ValueError):
        _sync(tmp_path, repo_id="talos-1b-training-metrics")


def test_sync_creates_the_local_directory(tmp_path):
    target = tmp_path / "nested" / "dir" / "metrics.jsonl"
    sync = _sync(tmp_path, dry_run=True, local_path=str(target))
    sync.maybe_push(_row())
    assert target.is_file()


def test_rows_to_jsonl_round_trips():
    payload = tlib.rows_to_jsonl([_row(step=1), _row(step=2, val_loss=None)])
    lines = payload.splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["val_loss"] is None


# ---------------------------------------------------------------------------
# Public training tracker (status.json + history.json)
# ---------------------------------------------------------------------------
#: The owner's 100M ``status.json`` keys, in their order — the 1B publisher must
#: emit exactly these, so the existing dashboard keeps parsing.
OWNER_STATUS_FIELDS = (
    "status", "model_name", "parameters", "step", "step_target",
    "tokens_consumed", "token_target", "train_loss", "val_loss", "checkpoint",
    "checkpoint_timestamp", "gpu", "vocab_size", "sequence_length", "timestamp",
)


def _publisher(tmp_path, **kwargs):
    kwargs.setdefault("local_dir", str(tmp_path))
    kwargs.setdefault("min_interval_s", 0)
    return tlib.TrackerPublisher(**kwargs)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_tracker_defaults_to_the_owner_public_repo():
    tracker = tlib.TrackerPublisher()
    assert tracker.repo_id == "PrometheanStudio/talos-training-tracker"
    assert tlib.TRACKER_REPO_ID == tracker.repo_id
    assert tracker.model_name == "Argo 1B"


def test_tracker_status_keeps_the_owner_schema_and_adapts_the_1b_values(tmp_path):
    recorder = _Recorder()
    tracker = _publisher(tmp_path, uploader=recorder)
    assert tracker.push(_row()) is True

    status = _read_json(tracker.status_path)
    # The owner's keys keep their names and order; "stage" is appended after
    # them (additive), so their dashboard keeps parsing.
    assert tuple(OWNER_STATUS_FIELDS) == tlib.TRACKER_STATUS_FIELDS
    assert tuple(status) == OWNER_STATUS_FIELDS + tlib.TRACKER_EXTRA_STATUS_FIELDS
    assert status["stage"] == ""
    assert status["status"] == "training"
    assert status["model_name"] == "Argo 1B"
    assert status["parameters"] == 1_036_060_672
    assert status["vocab_size"] == 49_152
    assert status["sequence_length"] == 2048
    assert status["token_target"] == 20_000_000_000
    assert status["step_target"] == 305_176        # ceil(20B / 65,536)
    # counters come from the row, not from step * tokens_per_step
    assert status["step"] == 500
    assert status["tokens_consumed"] == 5_000_000_000
    assert status["train_loss"] == pytest.approx(2.5)
    assert status["val_loss"] == pytest.approx(2.75)
    assert status["checkpoint"] == "step-500.pt"
    assert status["gpu"] == tracker.gpu and status["gpu"]
    # both timestamps are ISO-8601 wall clock, and the same instant
    assert status["timestamp"] == status["checkpoint_timestamp"]
    parsed = datetime.fromisoformat(status["timestamp"])
    assert parsed.tzinfo is not None


def test_tracker_reports_the_visible_gpu_or_cpu(tmp_path):
    tracker = _publisher(tmp_path, uploader=_Recorder())
    assert isinstance(tlib.resolve_gpu_name(), str)
    assert tracker.gpu == tlib.resolve_gpu_name() and tracker.gpu
    tracker.push(_row())
    assert _read_json(tracker.status_path)["gpu"] == tracker.gpu


def test_tracker_history_carries_checkpoints_and_the_epoch_rows(tmp_path):
    tracker = _publisher(tmp_path, uploader=_Recorder())
    tracker.note_checkpoint(1000, 65_536_000)
    tracker.note_checkpoint(500, 32_768_000)          # out of order on purpose
    tracker.push(_row())

    history = _read_json(tracker.history_path)
    assert tuple(history) == ("checkpoints", "epochs") == tlib.TRACKER_HISTORY_FIELDS
    assert history["checkpoints"] == [
        {"step": 500, "tokens_consumed": 32_768_000},
        {"step": 1000, "tokens_consumed": 65_536_000},
    ]
    assert history["epochs"] == [_row().as_dict()]


def test_tracker_history_takes_the_full_metric_history_when_given_it(tmp_path):
    sync = _sync(tmp_path, dry_run=True, min_interval_s=0)
    sync.maybe_push(_row(step=1))
    sync.maybe_push(_row(step=2))
    tracker = _publisher(tmp_path, uploader=_Recorder())
    tracker.push(_row(step=2), history_rows=sync.rows)
    history = _read_json(tracker.history_path)
    assert [row["step"] for row in history["epochs"]] == [1, 2]


def test_tracker_uploads_only_when_something_changed(tmp_path):
    recorder = _Recorder()
    tracker = _publisher(tmp_path, uploader=recorder)
    row = _row()

    assert tracker.push(row) is True
    assert [call[2] for call in recorder.calls] == ["status.json", "history.json"]
    assert tracker.published is True and tracker.push_count == 1
    assert recorder.calls[0][0] == "PrometheanStudio/talos-training-tracker"

    # same content on a later tick: nothing is uploaded (the wall clock moved)
    assert tracker.push(row) is True
    assert len(recorder.calls) == 2
    assert tracker.skipped_unchanged == 1
    assert tracker.published is False and tracker.push_count == 1

    # a new step is a change -> both files go out again
    assert tracker.push(_row(step=520, tokens_so_far=5_100_000_000)) is True
    assert len(recorder.calls) == 4
    assert tracker.published is True and tracker.push_count == 2
    assert _read_json(tracker.status_path)["step"] == 520


def test_tracker_is_throttled_like_the_metrics_sync(tmp_path):
    recorder = _Recorder()
    now = [0.0]
    tracker = _publisher(tmp_path, uploader=recorder, min_interval_s=60.0,
                         clock=lambda: now[0])
    assert tracker.maybe_push(_row(step=1)) is True       # first tick publishes
    now[0] = 30.0
    assert tracker.maybe_push(_row(step=2)) is False      # too soon
    assert tracker.push_count == 1
    now[0] = 90.0
    assert tracker.maybe_push(_row(step=3)) is True       # interval elapsed
    assert tracker.push_count == 2
    # every row is remembered even while throttled (history.json is complete)
    assert [row["step"] for row in tracker.epoch_rows] == [1, 2, 3]
    assert tracker.flush() is True                        # forced final publish
    assert tracker.push_count == 2                        # ...nothing had changed


def test_tracker_failure_never_raises_and_is_recorded(tmp_path):
    recorder = _Recorder(fail=True)
    tracker = _publisher(tmp_path, uploader=recorder)
    assert tracker.push(_row()) is False                  # no exception escapes
    assert tracker.last_error is not None and "hub is down" in tracker.last_error
    assert len(recorder.calls) == 1
    # the local copies are written first, so the payload survives a bad push
    assert _read_json(tracker.status_path)["step"] == 500
    assert _read_json(tracker.history_path)["epochs"]

    recorder.fail = False
    assert tracker.push() is True                         # next tick recovers
    assert tracker.last_error is None
    assert tracker.push_count == 1


def test_tracker_dry_run_writes_locally_and_never_uploads(tmp_path):
    recorder = _Recorder()
    tracker = _publisher(tmp_path, uploader=recorder, dry_run=True)
    assert tracker.push(_row()) is True
    assert recorder.calls == []
    assert tracker.push_count == 0
    assert os.path.isfile(tracker.status_path)
    assert os.path.isfile(tracker.history_path)
    assert _read_json(tracker.status_path)["model_name"] == "Argo 1B"


def test_tracker_rejects_a_repo_id_that_is_not_owner_slash_name(tmp_path):
    with pytest.raises(ValueError):
        _publisher(tmp_path, repo_id="talos-training-tracker")


def test_tracker_creates_the_local_directory(tmp_path):
    target = tmp_path / "nested" / "dir"
    tracker = _publisher(tmp_path, dry_run=True, local_dir=str(target))
    assert tracker.push(_row()) is True
    assert (target / "status.json").is_file()
    assert (target / "history.json").is_file()


# ---------------------------------------------------------------------------
# The notebook itself
# ---------------------------------------------------------------------------
def _load_notebook():
    with open(NOTEBOOK_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_notebook_is_valid_json_with_cells():
    nb = _load_notebook()
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    assert sum(1 for cell in nb["cells"] if cell["cell_type"] == "code") >= 8
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)


def test_every_notebook_code_cell_compiles():
    nb = _load_notebook()
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"<notebook cell {index}>", "exec")


def test_notebook_carries_the_owner_required_configuration():
    source = "\n".join("".join(cell["source"]) for cell in _load_notebook()["cells"])
    required = (
        # Two-stage config: no hard-coded 20B budget any more.
        'STAGE = "A"',
        "STAGE_A_HOURS",
        "STAGE_B_EPOCHS",
        "SEQ_LEN = 2048",
        "talos_1b_49k",                       # tokenizer asset
        "TALOS_1B_PARAMS",                    # 1,036,060,672 pinned by assert
        "assert n_params == TALOS_1B_PARAMS",
        "HuggingFaceH4/ultrachat_200k",
        "allenai/c4",
        "set_gradient_checkpointing(GRADIENT_CHECKPOINTING)",
        "PRECISION",
        "USE_8BIT_ADAM",
        "from talos_training_lib import",
        "MetricsSync(",
        "render_display_table",
        "RESUME",
        "list_checkpoint_candidates",
        "validate_resume_checkpoint",
        "validate_stage_resume",
        "save_checkpoint",
        "step-{info.step}.pt",
    )
    for needle in required:
        assert needle in source, needle
    # The old single-stage 20B default must be gone from the notebook.
    assert "20_000_000_000" not in source


def test_notebook_displays_exactly_the_owner_fields():
    source = "\n".join("".join(cell["source"]) for cell in _load_notebook()["cells"])
    assert "build_metrics_row(" in source
    for label in tlib.DISPLAY_LABELS:
        # labels live in the lib's renderer, which the notebook calls per interval
        assert label in tlib.render_display_table(_row())


def test_notebook_wires_the_public_tracker_and_the_drive_checkpoint_dir():
    source = "\n".join("".join(cell["source"]) for cell in _load_notebook()["cells"])
    # the owner's existing tracker repo, built from the lib and fed from the loop
    assert "PrometheanStudio/talos-training-tracker" in source
    assert "TrackerPublisher(" in source
    assert "tracker.maybe_push(row" in source
    assert "tracker.note_checkpoint(" in source
    assert "tracker.flush()" in source
    # checkpoints on Drive by default, TALOS_1B_OUT still overrides; the stage
    # directory is appended per stage (stage-a / stage-b).
    assert "/content/drive/MyDrive/Talos/Argo_1B" in source
    assert '"TALOS_1B_OUT"' in source
    assert "stage-{spec.key.lower()}" in source
    assert "argo_1b_runs" in source


def test_notebook_documents_resume_and_the_corpus_boundary():
    text = "\n".join(
        "".join(cell["source"])
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "markdown"
    ).lower()
    assert "resume" in text
    assert "stage a" in text and "pretraining" in text
    assert "stage b" in text and "tuning" in text
    assert "chinchilla" in text
    # The team's own packed corpus is not the notebook's input.
    assert "corpus-1b-49k" not in text
