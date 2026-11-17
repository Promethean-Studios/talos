"""Testable helpers for ``notebooks/talos_colab_train.ipynb`` (Talos 1B, 20B tokens).

The notebook is the owner's live 1B Chinchilla run (RTX 3060 12 GB). Everything
in this module is **pure logic** — no ``torch``, no GPU, no network at import
time — so the notebook's metrics row, ETA math, HTML display and Hugging Face
Dataset sync can be unit-tested on CI with only ``pytest`` installed:

* :func:`build_metrics_row` / :class:`MetricsRow` — one typed record per log
  interval, with exactly the fields the notebook displays and pushes.
* :func:`compute_eta_s` / :func:`compute_tokens_per_sec` — ETA math.
* :func:`render_display_table` — the single updated HTML table (the owner's
  exact display fields), not a scrolling text log.
* :func:`linearize_dialog` / :func:`pack_token_stream` — the chat → packed
  seq-2048 token stream adapter (dataset format, no torch).
* :class:`MetricsSync` — append a metrics row locally and push the jsonl to a
  HF **dataset** repo, throttled, with every failure caught and logged so a
  broken sync can never kill a 20B-token run.
* :class:`TrackerPublisher` — publish ``status.json`` + ``history.json`` in the
  owner's existing public tracker schema (the same ``PrometheanStudio/
  talos-training-tracker`` dataset repo their 100M run feeds), same throttling
  and same never-raise guarantee, uploading only when something changed.

The previous notebook's sync "never worked"; the reasons this one is different
are structural, not cosmetic: the row is built from a frozen dataclass with a
fixed field list (nothing can be silently dropped), the payload is a plain
``jsonl`` file uploaded as a single file (HF dataset repos have no append RPC),
``dry_run`` skips the network entirely, and :meth:`MetricsSync.maybe_push`
returns a bool and stores :attr:`MetricsSync.last_error` instead of raising.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger("talos.notebook.training_lib")

#: Legacy default token budget for ``talos_1b`` (1,036,060,672 params × ~20
#: tokens/param ≈ 20.7B). Kept only as a default for callers that do not pass a
#: budget: the two-stage notebook derives Stage A's budget from the available
#: compute and Stage B's from the measured tokens/epoch (see
#: ``notebooks/talos_stages.py`` and ``docs/training-1b-notebook.md``).
TOTAL_TOKENS = 20_000_000_000

#: Model facts the notebook prints and records. Pinned by ``configs.presets``
#: (``talos_1b_config`` / ``talos_1b_tokenizer_config``) and by
#: ``tests/test_talos_1b.py``; repeated here so a metrics row can be built and
#: tested without importing torch.
TALOS_1B_PARAMS = 1_036_060_672
TALOS_1B_VOCAB = 49_152
TALOS_1B_SEQ_LEN = 2048
TALOS_1B_NAME = "talos_1b"
TALOS_1B_TOKENIZER_ASSET = "talos_1b_49k"

#: Display labels, in the owner's requested order. The notebook renders exactly
#: these rows; :func:`render_display_table` is the only place they are spelled.
DISPLAY_LABELS: tuple = (
    "Token progress",
    "Stage",
    "Training step",
    "Tokens/sec",
    "Elapsed time",
    "ETA",
    "Train loss",
    "Val loss",
    "Checkpoint",
    "GPU",
    "Model",
    "Parameters",
    "Vocab",
    "Sequence length",
    "Epoch",
    "Last update",
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_count(n: int) -> str:
    """``1234567`` -> ``'1,234,567'``."""
    return f"{int(n):,}"


def format_billions(n: float) -> str:
    """``20_000_000_000`` -> ``'20.00B'``; ``49_152`` -> ``'49.15K'``."""
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.0f}"


def format_progress(tokens_so_far: int, total_tokens: int) -> str:
    """``'3,000,000,000 out of 20.00B (15.0%)'`` — the token-progress cell."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    pct = 100.0 * float(tokens_so_far) / float(total_tokens)
    return f"{format_count(tokens_so_far)} out of {format_billions(total_tokens)} ({pct:.1f}%)"


def format_duration(seconds: Optional[float]) -> str:
    """Human duration: ``'3d 04h 12m'``, ``'1h 02m 09s'``, ``'07m 41s'``.

    ``None`` / non-finite / negative renders as ``'—'`` (used for an ETA that
    cannot be estimated yet, e.g. before the first log interval).
    """
    if seconds is None:
        return "—"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return "—"
    total = int(round(value))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def format_loss(value: Optional[float]) -> str:
    """``4.1234`` -> ``'4.1234'``; ``None`` -> ``'—'`` (not evaluated yet)."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


#: Human labels for the two training stages (``notebooks/talos_stages.py``).
STAGE_LABELS: Dict[str, str] = {
    "A": "A — language pretraining",
    "B": "B — conversational tuning",
}


def stage_label(stage: str) -> str:
    """``'A'`` -> ``'A — language pretraining'``; unknown/blank -> ``'—'``."""
    return STAGE_LABELS.get(str(stage or "").strip().upper(), str(stage or "—") or "—")


def utc_timestamp(when: Optional[datetime] = None) -> str:
    """Exact date+time in UTC, ISO-8601 seconds precision: ``2026-11-17T14:03:22Z``."""
    when = datetime.now(timezone.utc) if when is None else when
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_display(timestamp: str) -> str:
    """``'2026-11-17T14:03:22Z'`` -> ``'2026-11-17 14:03:22 UTC'``."""
    try:
        when = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return str(timestamp)
    return when.strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# ETA / throughput math
# ---------------------------------------------------------------------------
def compute_tokens_per_sec(tokens_so_far: int, elapsed_s: float) -> float:
    """Mean tokens/sec since run start (0.0 while nothing has been counted)."""
    if tokens_so_far < 0:
        raise ValueError("tokens_so_far must be >= 0")
    if elapsed_s <= 0:
        return 0.0
    return float(tokens_so_far) / float(elapsed_s)


def compute_eta_s(tokens_so_far: int, total_tokens: int, elapsed_s: float) -> Optional[float]:
    """Seconds remaining at the run's *mean* throughput so far.

    Returns ``None`` when no estimate exists yet (no tokens counted, no elapsed
    time, or a non-finite throughput) — the display then shows ``—`` rather than
    a fabricated number. Returns ``0.0`` once the budget is reached.
    """
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if tokens_so_far < 0:
        raise ValueError("tokens_so_far must be >= 0")
    if tokens_so_far >= total_tokens:
        return 0.0
    if tokens_so_far == 0 or elapsed_s <= 0:
        return None
    rate = float(tokens_so_far) / float(elapsed_s)
    if rate != rate or rate <= 0 or rate == float("inf"):
        return None
    return (float(total_tokens) - float(tokens_so_far)) / rate


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricsRow:
    """One log interval's metrics — the display row *and* the synced payload.

    Field order is the push order (and the jsonl key order), so a row on the
    HF dataset reads the same as a row on screen.
    """

    step: int
    epoch: int
    tokens_so_far: int
    total_tokens: int
    tokens_per_sec: float
    elapsed_s: float
    eta_s: Optional[float]
    train_loss: float
    val_loss: Optional[float]
    checkpoint: str
    gpu: str
    model: str
    params: int
    vocab: int
    seq_len: int
    timestamp_utc: str
    #: Training stage ("A" = language pretraining, "B" = conversational tuning).
    #: Additive: every pre-existing field keeps its name and position, so a
    #: reader of the old schema keeps working and a new row gains one key.
    stage: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Plain ``dict`` in field order (json-ready)."""
        return asdict(self)

    def to_jsonl(self) -> str:
        """One JSON line, no trailing newline."""
        return json.dumps(self.as_dict(), ensure_ascii=False, allow_nan=False)

    @property
    def eta_display(self) -> str:
        return format_duration(self.eta_s)


def build_metrics_row(
    *,
    step: int,
    epoch: int,
    tokens_so_far: int,
    elapsed_s: float,
    train_loss: float,
    total_tokens: int = TOTAL_TOKENS,
    val_loss: Optional[float] = None,
    checkpoint: str = "",
    gpu: str = "",
    model: str = TALOS_1B_NAME,
    params: int = TALOS_1B_PARAMS,
    vocab: int = TALOS_1B_VOCAB,
    seq_len: int = TALOS_1B_SEQ_LEN,
    stage: str = "",
    timestamp: Optional[datetime] = None,
) -> MetricsRow:
    """Build a validated :class:`MetricsRow`; throughput and ETA are derived.

    Raises ``ValueError`` on impossible inputs (negative step/tokens, a
    non-positive budget, a non-finite loss) so a corrupt log line is caught at
    the source instead of silently polluting the metrics dataset.
    """
    if step < 0:
        raise ValueError("step must be >= 0")
    if epoch < 1:
        raise ValueError("epoch is 1-based and must be >= 1")
    if tokens_so_far < 0:
        raise ValueError("tokens_so_far must be >= 0")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if params <= 0 or vocab <= 0:
        raise ValueError("params and vocab must be positive")
    if elapsed_s < 0:
        raise ValueError("elapsed_s must be >= 0")
    train = float(train_loss)
    if train != train or train in (float("inf"), float("-inf")):
        raise ValueError(f"train_loss must be finite, got {train_loss!r}")
    if val_loss is not None:
        val = float(val_loss)
        if val != val or val in (float("inf"), float("-inf")):
            raise ValueError(f"val_loss must be finite when set, got {val_loss!r}")
        val_loss = val
    return MetricsRow(
        step=int(step),
        epoch=int(epoch),
        tokens_so_far=int(tokens_so_far),
        total_tokens=int(total_tokens),
        tokens_per_sec=compute_tokens_per_sec(tokens_so_far, elapsed_s),
        elapsed_s=float(elapsed_s),
        eta_s=compute_eta_s(tokens_so_far, total_tokens, elapsed_s),
        train_loss=train,
        val_loss=val_loss,
        checkpoint=str(checkpoint),
        gpu=str(gpu),
        model=str(model),
        params=int(params),
        vocab=int(vocab),
        seq_len=int(seq_len),
        timestamp_utc=utc_timestamp(timestamp),
        stage=str(stage),
    )


# ---------------------------------------------------------------------------
# HTML display
# ---------------------------------------------------------------------------
def render_display_table(row: MetricsRow) -> str:
    """The notebook's single live output: one HTML table, all owner fields.

    Returned as a string so the training cell can update *one* display handle
    per log interval (``handle.update(HTML(...))``) instead of printing a
    scrolling text log. Renders identically in Jupyter, Colab and VS Code.
    """
    values = (
        format_progress(row.tokens_so_far, row.total_tokens),
        stage_label(row.stage),
        f"{format_count(row.step)}  ({(row.tokens_so_far / max(row.seq_len, 1)):,.0f} rows seen)",
        f"{row.tokens_per_sec:,.0f} tok/s",
        format_duration(row.elapsed_s),
        row.eta_display,
        format_loss(row.train_loss),
        format_loss(row.val_loss),
        row.checkpoint or "—",
        row.gpu or "—",
        row.model,
        format_count(row.params),
        format_count(row.vocab),
        format_count(row.seq_len),
        f"epoch {row.epoch}",
        utc_display(row.timestamp_utc),
    )
    rows_html = "\n".join(
        f'<tr><th style="text-align:left;padding:3px 14px 3px 0;'
        f'font-weight:600;color:#444;">{label}</th>'
        f'<td style="text-align:left;padding:3px 0;font-family:ui-monospace,'
        f'Menlo,monospace;">{value}</td></tr>'
        for label, value in zip(DISPLAY_LABELS, values)
    )
    return (
        '<div style="border:1px solid #d0d7de;border-radius:6px;padding:10px 14px;'
        'display:inline-block;background:#fafbfc;">'
        '<div style="font-weight:700;margin-bottom:6px;color:#0b3d91;">'
        "Talos 1B — training</div>"
        f'<table style="border-collapse:collapse;font-size:13px;">{rows_html}</table>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Chat dataset → packed token stream
# ---------------------------------------------------------------------------
def linearize_dialog(
    messages: Sequence[Mapping[str, Any]],
    *,
    eos: str = "\n\n",
) -> str:
    """Linearize one chat dialog into a single document.

    ``[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]``
    becomes ``"User: hi\\n\\nAssistant: hello"``. Unknown roles are kept
    (capitalised) rather than dropped, and non-string / missing content yields
    an empty turn, so a malformed row degrades instead of crashing the stream.
    """
    lines: List[str] = []
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "unknown").strip() or "unknown"
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        lines.append(f"{role.capitalize()}: {text.strip()}")
    return eos.join(lines).strip()


def pack_token_stream(
    token_lists: Iterable[Sequence[int]],
    *,
    seq_len: int,
    vocab_size: int = TALOS_1B_VOCAB,
    pad_id: int = 0,
    drop_last: bool = True,
) -> Iterable[List[int]]:
    """Concatenate per-document token id lists into fixed ``seq_len`` rows.

    Documents are joined end-to-end (no padding between documents — a packed
    stream, as the 1B pretraining corpus is packed) and sliced into rows of
    ``seq_len``; a short trailing row is padded with ``pad_id`` when
    ``drop_last`` is false and dropped otherwise. **Every** id is bounds-checked
    against ``vocab_size`` before it is emitted: the repo convention
    (``data.tokenized.validate_id_array``) is that an out-of-range id must fail
    at the data source with context, never as a device-side assert mid-step.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not (0 <= pad_id < vocab_size):
        raise ValueError(f"pad_id {pad_id} outside [0, {vocab_size})")

    buffer: List[int] = []
    for ids in token_lists:
        for token_id in ids:
            value = int(token_id)
            if value < 0 or value >= vocab_size:
                raise ValueError(
                    f"token id {value} outside [0, {vocab_size}) — refusing to "
                    "pack a document whose tokenizer does not match the model vocab"
                )
            buffer.append(value)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]
    if buffer and not drop_last:
        yield buffer + [pad_id] * (seq_len - len(buffer))


# ---------------------------------------------------------------------------
# HF Dataset metrics sync
# ---------------------------------------------------------------------------
def rows_to_jsonl(rows: Iterable[MetricsRow]) -> str:
    """Serialize rows as newline-delimited JSON (trailing newline included)."""
    return "".join(row.to_jsonl() + "\n" for row in rows)


#: Uploader signature: ``(repo_id, local_path, path_in_repo, token) -> None``.
Uploader = Callable[[str, str, str, Optional[str]], None]


def _hf_upload(repo_id: str, local_path: str, path_in_repo: str,
               token: Optional[str]) -> None:
    """Default uploader: create the dataset repo if missing, then upload the file.

    HF dataset repos have no "append a row" RPC, so the notebook keeps the full
    metrics history in one local ``jsonl`` (appended every log interval) and
    uploads that single file — the remote copy is always the complete history,
    which is also what makes a resumed run's metrics continuous.
    """
    from huggingface_hub import HfApi  # imported lazily: CI has no HF client

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True,
                    private=False, token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"talos-1b metrics: +{os.path.basename(local_path)}",
    )


class MetricsSync:
    """Append metrics rows locally and push them to a HF dataset repo.

    Guarantees, in order of importance:

    1. **A sync failure never kills training.** :meth:`maybe_push` and
       :meth:`flush` catch everything, log a warning, record
       :attr:`last_error` and return ``False``.
    2. **``dry_run=True`` touches no network at all** (used by the notebook's
       smoke run and by the test suite).
    3. **Throttled.** :meth:`maybe_push` uploads at most once per
       ``min_interval_s`` (default 300 s = 5 min); ``LOG_EVERY`` steps at 20B
       tokens is minutes apart, so the default is "every 5 min, whichever is
       coarser" in practice. :meth:`flush` forces a final upload.

    Args:
        repo_id: dataset repo, e.g. ``"<whoami>/talos-1b-training-metrics"``.
        token: HF token; ``None`` lets ``huggingface_hub`` use the ambient
            login (``notebook_login()`` / ``HF_TOKEN``).
        dry_run: skip all network calls (rows are still appended locally).
        min_interval_s: minimum seconds between uploads.
        local_path: jsonl history file; defaults to ``talos-1b-metrics.jsonl``
            in the current directory.
        path_in_repo: filename inside the dataset repo.
        uploader: injectable upload function (tests pass a recorder).
        clock: injectable monotonic clock (tests advance it by hand).
    """

    def __init__(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        dry_run: bool = False,
        min_interval_s: float = 300.0,
        local_path: Optional[str] = None,
        path_in_repo: str = "metrics.jsonl",
        uploader: Optional[Uploader] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not repo_id or "/" not in str(repo_id):
            raise ValueError(
                f"repo_id must look like '<user>/<dataset-name>', got {repo_id!r}"
            )
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        self.repo_id = str(repo_id)
        self.token = token
        self.dry_run = bool(dry_run)
        self.min_interval_s = float(min_interval_s)
        self.path_in_repo = path_in_repo
        self.local_path = local_path or "talos-1b-metrics.jsonl"
        self._uploader: Uploader = uploader or _hf_upload
        self._clock = clock
        self._last_push_at: Optional[float] = None
        self.rows: List[MetricsRow] = []
        self.push_count = 0
        self.last_error: Optional[str] = None

    # -- local history ------------------------------------------------------
    def append(self, row: MetricsRow) -> None:
        """Append one row to the in-memory history and to the local jsonl."""
        self.rows.append(row)
        try:
            parent = os.path.dirname(os.path.abspath(self.local_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.local_path, "a", encoding="utf-8") as handle:
                handle.write(row.to_jsonl() + "\n")
        except OSError as exc:  # a read-only dir must not stop training
            self.last_error = f"local append failed: {exc}"
            logger.warning("metrics local append failed (%s): %s",
                           self.local_path, exc)

    # -- uploads ------------------------------------------------------------
    def push(self, rows: Optional[Sequence[MetricsRow]] = None) -> bool:
        """Upload the metrics jsonl now; never raises.

        ``rows`` may be a single :class:`MetricsRow`, a sequence (they are added
        to the history first, so nothing is lost), or ``None`` to publish the
        full history. Returns ``True`` on success (or on a dry run, where the
        network step is skipped by design), ``False`` when the upload failed.
        """
        if isinstance(rows, MetricsRow):
            rows = [rows]
        if rows is not None:
            for row in rows:
                if row not in self.rows:
                    self.rows.append(row)
        if self.dry_run:
            logger.info("metrics sync dry_run=True — %d row(s) not uploaded",
                        len(self.rows))
            return True
        if not self.rows:
            return True
        try:
            parent = os.path.dirname(os.path.abspath(self.local_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.local_path, "w", encoding="utf-8") as handle:
                handle.write(rows_to_jsonl(self.rows))
            self._uploader(self.repo_id, self.local_path, self.path_in_repo, self.token)
        except Exception as exc:  # noqa: BLE001 — sync must never kill training
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("metrics sync to %s failed (%s) — training continues",
                           self.repo_id, self.last_error)
            return False
        self._last_push_at = self._clock()
        self.push_count += 1
        self.last_error = None
        logger.info("metrics sync: pushed %d row(s) to %s", len(self.rows), self.repo_id)
        return True

    def maybe_push(self, row: MetricsRow) -> bool:
        """Append ``row`` and upload only if the throttle interval has elapsed."""
        self.append(row)
        now = self._clock()
        if (self._last_push_at is not None
                and (now - self._last_push_at) < self.min_interval_s):
            return False
        return self.push()

    def flush(self) -> bool:
        """Force a final upload (end of run / before resume); never raises."""
        return self.push()


# ---------------------------------------------------------------------------
# Public training tracker (owner's PrometheanStudio/talos-training-tracker)
# ---------------------------------------------------------------------------
#: The owner's public tracker **dataset** repo — the same one their 100M run
#: feeds, so the 1B run shows up on the tracker they already watch.
TRACKER_REPO_ID = "PrometheanStudio/talos-training-tracker"

#: ``status.json`` keys in the owner's 100M order. The 1B *values* are adapted
#: (model name, params, vocab, sequence length, token target); the keys and
#: their order are the owner's, so an existing dashboard keeps parsing.
TRACKER_STATUS_FIELDS: tuple = (
    "status",
    "model_name",
    "parameters",
    "step",
    "step_target",
    "tokens_consumed",
    "token_target",
    "train_loss",
    "val_loss",
    "checkpoint",
    "checkpoint_timestamp",
    "gpu",
    "vocab_size",
    "sequence_length",
    "timestamp",
)

#: ``history.json`` keys, in the owner's order.
TRACKER_HISTORY_FIELDS: tuple = ("checkpoints", "epochs")

#: Extra ``status.json`` keys appended AFTER the owner's. Additive by design:
#: the owner's 15 keys keep their names, order and types, so their dashboard
#: keeps parsing, and a two-stage run gains one extra key it can ignore.
TRACKER_EXTRA_STATUS_FIELDS: tuple = ("stage",)

#: The owner's tracker labels the 100M model ``"Talos 100M"``; the 1B run is
#: ``"Talos 1B"`` (the display name, not the internal preset key ``talos_1b``).
TRACKER_MODEL_NAME = "Talos 1B"

#: Tokens per optimizer step in the 1B notebook (MICRO_BATCH 2 × GRAD_ACCUM 16 ×
#: SEQ_LEN 2048). Only used to derive ``step_target``; the notebook passes its
#: own live value so an OOM-driven MICRO_BATCH change stays truthful.
TALOS_1B_TOKENS_PER_STEP = 65_536


def resolve_gpu_name() -> str:
    """``torch.cuda.get_device_name(0)`` when a GPU is present, else ``"CPU"``.

    Never raises and never imports torch at module scope: on a CPU-only box
    (or in CI, where torch may be absent) the tracker simply reports ``"CPU"``.
    """
    try:
        import torch  # local import: the rest of this module stays torch-free

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 — a missing/broken CUDA build is not fatal
        logger.debug("could not read the CUDA device name", exc_info=True)
    return "CPU"


def _tracker_signature(status: Mapping[str, Any],
                       history: Mapping[str, Any]) -> str:
    """Change signature: the payloads *without* their wall-clock timestamps.

    The owner's 100M cell signed the full payload, which contains
    ``timestamp``/``checkpoint_timestamp`` — those change on every tick, so the
    comparison never matched and it uploaded every 60 s regardless. Dropping the
    two timestamps makes "only on change" mean what it says: a tick where no
    step, token count, loss, checkpoint or metric row moved uploads nothing.
    """
    volatile = ("timestamp", "checkpoint_timestamp")
    stable_status = {key: value for key, value in status.items() if key not in volatile}
    return json.dumps([stable_status, history], sort_keys=True, default=str)


class TrackerPublisher:
    """Publish ``status.json`` + ``history.json`` to the owner's tracker repo.

    Same contract as :class:`MetricsSync`, because the same failure modes apply
    to a 20B-token run: throttled to one upload per ``min_interval_s``, an
    injectable uploader and clock (so CI never touches the network), ``dry_run``
    writing the files locally but uploading nothing, and every failure caught,
    logged and stored on :attr:`last_error` — a tracker hiccup can never kill
    training.

    What it publishes is the owner's own schema (:data:`TRACKER_STATUS_FIELDS`,
    :data:`TRACKER_HISTORY_FIELDS`) with the 1B constants substituted, so the
    existing public dashboard keeps reading it unchanged.

    Args:
        repo_id: dataset repo; defaults to the owner's
            :data:`TRACKER_REPO_ID`.
        token: HF token; ``None`` lets ``huggingface_hub`` use the ambient login.
        dry_run: write locally, never touch the network.
        min_interval_s: minimum seconds between uploads (the owner's 100M cell
            used 60 s).
        local_dir: where ``status.json``/``history.json`` are written (the
            notebook points this at ``OUT_DIR``, i.e. Drive when mounted).
        uploader: injectable upload function ``(repo_id, local_path,
            path_in_repo, token) -> None``; tests pass a recorder.
        clock: injectable monotonic clock; tests advance it by hand.
        tokens_per_step: used only for ``step_target``.
        gpu: GPU label; defaults to :func:`resolve_gpu_name`.
    """

    def __init__(
        self,
        repo_id: str = TRACKER_REPO_ID,
        *,
        token: Optional[str] = None,
        dry_run: bool = False,
        min_interval_s: float = 60.0,
        local_dir: str = ".",
        status_filename: str = "status.json",
        history_filename: str = "history.json",
        uploader: Optional[Uploader] = None,
        clock: Callable[[], float] = time.monotonic,
        model_name: str = TRACKER_MODEL_NAME,
        parameters: int = TALOS_1B_PARAMS,
        vocab_size: int = TALOS_1B_VOCAB,
        sequence_length: int = TALOS_1B_SEQ_LEN,
        token_target: int = TOTAL_TOKENS,
        tokens_per_step: int = TALOS_1B_TOKENS_PER_STEP,
        gpu: Optional[str] = None,
        status: str = "training",
        stage: str = "",
    ) -> None:
        if not repo_id or "/" not in str(repo_id):
            raise ValueError(
                f"repo_id must look like '<user>/<dataset-name>', got {repo_id!r}"
            )
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        if token_target <= 0:
            raise ValueError("token_target must be positive")
        if tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive")
        if vocab_size <= 0 or sequence_length <= 0 or parameters <= 0:
            raise ValueError("parameters, vocab_size and sequence_length must be positive")
        self.repo_id = str(repo_id)
        self.token = token
        self.dry_run = bool(dry_run)
        self.min_interval_s = float(min_interval_s)
        self.local_dir = str(local_dir)
        self.status_filename = status_filename
        self.history_filename = history_filename
        self.model_name = str(model_name)
        self.parameters = int(parameters)
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.token_target = int(token_target)
        self.tokens_per_step = int(tokens_per_step)
        self.gpu = str(gpu) if gpu is not None else resolve_gpu_name()
        self.status = str(status)
        self.stage = str(stage)
        self._uploader: Uploader = uploader or _hf_upload
        self._clock = clock
        self._last_push_at: Optional[float] = None
        self._last_row: Optional[MetricsRow] = None
        self._signature: Optional[str] = None
        self.checkpoints: List[Dict[str, int]] = []
        self.epoch_rows: List[Dict[str, Any]] = []
        self.push_count = 0
        self.skipped_unchanged = 0
        self.published = False
        self.last_error: Optional[str] = None

    # -- paths --------------------------------------------------------------
    @property
    def status_path(self) -> str:
        return os.path.join(self.local_dir, self.status_filename)

    @property
    def history_path(self) -> str:
        return os.path.join(self.local_dir, self.history_filename)

    # -- state --------------------------------------------------------------
    def note_checkpoint(self, step: int, tokens_consumed: int) -> None:
        """Record a checkpoint for ``history.json``'s ``checkpoints`` list.

        ``tokens_consumed`` is the counter the checkpoint was written at — the
        truth, not ``step × tokens_per_step`` (the owner's 100M cell derived it,
        which is only right while the tokens-per-step never changes; the 1B
        notebook's MICRO_BATCH may drop to 1 on a CUDA OOM).
        """
        entry = {"step": int(step), "tokens_consumed": int(tokens_consumed)}
        self.checkpoints = [item for item in self.checkpoints
                            if item["step"] != entry["step"]]
        self.checkpoints.append(entry)
        self.checkpoints.sort(key=lambda item: item["step"])

    def _remember(self, row: MetricsRow) -> None:
        payload = row.as_dict()
        if payload not in self.epoch_rows:
            self.epoch_rows.append(payload)
        self._last_row = row

    # -- payloads -----------------------------------------------------------
    def build_status(
        self,
        *,
        step: int = 0,
        tokens_consumed: int = 0,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        checkpoint: str = "",
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """The owner's ``status.json`` object with the 1B values filled in."""
        now = datetime.now(timezone.utc).isoformat() if when is None else (
            when if isinstance(when, str) else when.isoformat())
        status = {
            "status": self.status,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "step": int(step),
            "step_target": math.ceil(self.token_target / self.tokens_per_step),
            "tokens_consumed": int(tokens_consumed),
            "token_target": self.token_target,
            "train_loss": None if train_loss is None else float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "checkpoint": str(checkpoint),
            "checkpoint_timestamp": now,
            "gpu": self.gpu,
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "timestamp": now,
        }
        payload = {key: status[key] for key in TRACKER_STATUS_FIELDS}
        # Additive extras (never reordered into the owner's keys above).
        payload["stage"] = self.stage
        return payload

    def build_history(
        self,
        epoch_rows: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """The owner's ``history.json``: checkpoint list + metric rows."""
        if epoch_rows is None:
            rows: List[Any] = list(self.epoch_rows)
        else:
            rows = list(epoch_rows)
        serialized = [
            row.as_dict() if isinstance(row, MetricsRow) else dict(row) for row in rows
        ]
        history = {
            "checkpoints": [dict(item) for item in self.checkpoints],
            "epochs": serialized,
        }
        return {key: history[key] for key in TRACKER_HISTORY_FIELDS}

    # -- uploads ------------------------------------------------------------
    def push(
        self,
        row: Optional[MetricsRow] = None,
        *,
        history_rows: Optional[Iterable[Any]] = None,
        when: Optional[datetime] = None,
    ) -> bool:
        """Write both files locally and upload them; **never raises**.

        Returns ``True`` when the tracker is up to date afterwards — uploaded
        now, unchanged since the last upload, or a dry run — and ``False`` when
        the local write or the upload failed (:attr:`last_error` then says
        why). :attr:`published` is ``True`` only when this call actually put new
        content on the hub, and :attr:`skipped_unchanged` counts no-op ticks.
        """
        self.published = False
        if row is not None:
            self._remember(row)
        last = self._last_row
        state = {
            "step": 0 if last is None else last.step,
            "tokens_consumed": 0 if last is None else last.tokens_so_far,
            "train_loss": None if last is None else last.train_loss,
            "val_loss": None if last is None else last.val_loss,
            "checkpoint": "" if last is None else last.checkpoint,
        }
        status = self.build_status(when=when, **state)
        history = self.build_history(history_rows)
        signature = _tracker_signature(status, history)

        try:
            os.makedirs(self.local_dir or ".", exist_ok=True)
            with open(self.status_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(status, indent=2))
            with open(self.history_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(history, indent=2))
        except (OSError, TypeError, ValueError) as exc:
            self.last_error = f"local write failed: {exc}"
            logger.warning("tracker local write failed (%s): %s",
                           self.local_dir, exc)
            return False

        if self.dry_run:
            self._signature = signature
            logger.info("tracker dry_run=True — status/history written to %s, "
                        "nothing uploaded", self.local_dir)
            return True

        if signature == self._signature:
            self.skipped_unchanged += 1
            return True

        try:
            self._uploader(self.repo_id, self.status_path, self.status_filename,
                           self.token)
            self._uploader(self.repo_id, self.history_path, self.history_filename,
                           self.token)
        except Exception as exc:  # noqa: BLE001 — tracking must never kill training
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("tracker upload to %s failed (%s) — training continues",
                           self.repo_id, self.last_error)
            return False
        self._signature = signature
        self._last_push_at = self._clock()
        self.push_count += 1
        self.published = True
        self.last_error = None
        logger.info("tracker: published status.json + history.json to %s "
                    "(step %s)", self.repo_id, state["step"])
        return True

    def maybe_push(
        self,
        row: MetricsRow,
        *,
        history_rows: Optional[Iterable[Any]] = None,
        when: Optional[datetime] = None,
    ) -> bool:
        """Record ``row`` and publish only if the throttle interval elapsed."""
        self._remember(row)
        now = self._clock()
        if (self._last_push_at is not None
                and (now - self._last_push_at) < self.min_interval_s):
            return False
        return self.push(history_rows=history_rows, when=when)

    def flush(self, *, when: Optional[datetime] = None) -> bool:
        """Force a final publish (end of run / before resume); never raises."""
        return self.push(when=when)


__all__ = [
    "DISPLAY_LABELS",
    "MetricsRow",
    "MetricsSync",
    "TALOS_1B_NAME",
    "TALOS_1B_TOKENS_PER_STEP",
    "STAGE_LABELS",
    "TRACKER_EXTRA_STATUS_FIELDS",
    "TRACKER_HISTORY_FIELDS",
    "TRACKER_MODEL_NAME",
    "TRACKER_REPO_ID",
    "TRACKER_STATUS_FIELDS",
    "TrackerPublisher",
    "TALOS_1B_PARAMS",
    "TALOS_1B_SEQ_LEN",
    "TALOS_1B_TOKENIZER_ASSET",
    "TALOS_1B_VOCAB",
    "TOTAL_TOKENS",
    "build_metrics_row",
    "compute_eta_s",
    "compute_tokens_per_sec",
    "format_billions",
    "format_count",
    "format_duration",
    "format_loss",
    "format_progress",
    "linearize_dialog",
    "pack_token_stream",
    "render_display_table",
    "stage_label",
    "resolve_gpu_name",
    "rows_to_jsonl",
    "utc_display",
    "utc_timestamp",
]
