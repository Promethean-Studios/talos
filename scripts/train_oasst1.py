"""Reproducible OASST1-style JSONL -> Talos-native tokenizer -> tiny-model training.

This is the owner-scoped reliability pass for the JSONL training path. It wires
**existing** Talos components end-to-end — no new data pipeline, no new training
framework, no new architecture:

    JSONL (OASST1-shaped, any path)
      -> deterministic train/val split        split_jsonl()  (fixed seed, disjoint files)
      -> Talos-native byte-level BPE          tokenizer.train.train_tokenizer()
         (vocab 1024 shared by every          preset configs.presets.preset_tokenizer_config,
          canonical preset —                     max_docs/max_chars/num_merges/minfreq
          configs.vocab.VOCAB_SIZE,             knobs exposed as CLI flags)
          TRAIN SPLIT ONLY)                   OR LOAD an established tokenizer.json via
                                              --tokenizer-json (the corpus-pretraining
                                              path: no BPE training; the sidecar copy's
                                              sha256 fingerprint is recorded in every
                                              checkpoint + metrics.json)
      -> canonical preset by --preset           configs.presets.<preset>_config()
         (registry-pinned exact params+vocab,   fails fast on any config drift
          any of tiny/tiny_1m/tiny_10m/tiny_100m)
      -> streaming training                   data.tokenized.StreamingTokenizedDataset
         (same objective/components as         + AdamW + CrossEntropyLoss, pack mode,
          examples/tiny_train.py)              x[:, :-1] -> x[:, 1:]
      -> per-epoch checkpoint artifact        weights + optimizer + RNG state + step/epoch
         + tokenizer.json next to it           + tokenizer fingerprint + losses in one .pt
      -> train + validation loss saved        out_dir/metrics.json (losses, wall time,
                                              peak RSS, params, tokenizer vocab + sha256)
      -> resume from any checkpoint           --resume step-<N>.pt continues the run with
         (bit-exact vs uninterrupted runs)     optimizer + RNG state restored

Why the loop lives here instead of examples/tiny_train.py: the existing
``train_stream`` helper is step-based (no epoch boundaries, no evaluation, no
checkpointing). Everything it uses is reused verbatim; this module only adds the
epoch/eval/checkpoint orchestration the task requires. The artifact layout is
flat so a checkpoint and its tokenizer always sit side by side::

    out_dir/
      data/train.jsonl        data/val.jsonl      # disjoint, deterministic split
      tokenizer.json                             # BPE-trained on train split, or a
                                                 # copy of the --tokenizer-json file
      step-<N>.pt                                # checkpoint (weights+opt+RNG+losses)
      metrics.json                               # run report

Usage::

    python -m tools.make_synthetic_oasst1 --docs 300 --seed 0 --output /tmp/oasst1.jsonl
    python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir runs/oasst1-tiny \\
        --epochs 3 --seq 64 --batch 4 --seed 0
    # resume a 100K-step Colab campaign from the last checkpoint:
    python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir runs/oasst1-tiny \\
        --resume runs/oasst1-tiny/step-7680.pt --epochs 20000 --seq 64 --batch 4 --seed 0

Pipeline integrity guards (all cheap, all always-on):

* every batch is token-id-range-checked at the data source (per encoded
  document, ``data.tokenized.validate_id_array``), at batch assembly
  (``model.utils.validate_token_ids`` on the train/eval loops), and at the
  embedding seam (``TalosGPT.forward``) — an out-of-range id can never reach a
  CUDA index kernel and poison the T4 context (audit P0 fix 1);
* the checkpoint records the sha256 of the trained ``tokenizer.json`` and
  resumed runs verify it, so a swapped/re-trained sidecar tokenizer fails
  loudly instead of silently mis-tokenizing (audit P0 fix 4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the script
# also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.canonical import CANONICAL_PRESETS, resolve_preset  # noqa: E402
from configs.presets import ALL_PRESETS, preset_tokenizer_config  # noqa: E402
from data.packed import (  # noqa: E402
    PackedTokenDataset,
    load_packed_manifest,
    manifest_identity,
    packed_phase_shard_paths,
)
from data.tokenized import StreamingTokenizedDataset  # noqa: E402
from model import ModelConfig, TalosGPT  # noqa: E402
from model.utils import get_logger, set_seed, validate_token_ids  # noqa: E402
from tokenizer.corpus import iter_text_documents  # noqa: E402
from tokenizer.tokenizer import (  # noqa: E402
    ByteLevelBPETokenizer,
    tokenizer_file_sha256,
)
from tokenizer.train import train_tokenizer  # noqa: E402

log = get_logger("scripts.train_oasst1")

CHECKPOINT_FORMAT = "talos-training-checkpoint-v1"

RUN_METADATA_FILENAME = "train_run_metadata.json"
RUN_METADATA_SCHEMA = "talos-training-run-metadata-v1"

#: LR-schedule knobs (additive; defaults reproduce the fixed-LR behavior).
LR_DECAY_CHOICES = ("none", "cosine")
#: Cosine decays to this fraction of ``--lr`` over the token budget.
MIN_LR_RATIO = 0.1


# ---------------------------------------------------------------------------
# Token-budget LR schedule (DELIVERABLE 3 — additive, default = fixed LR)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenSchedule:
    """LR as a function of tokens consumed. ``lr_at(t)`` is monotone.

    * ``warmup_tokens`` > 0: linear ramp ``0 -> lr`` over the first W tokens.
    * ``decay="cosine"``: after warmup, cosine from ``lr`` to
      ``min_lr_ratio * lr`` over the remaining budget (``budget - warmup``).
    * ``decay="none"`` (default): constant ``lr`` after warmup — bit-identical
      to the pre-schedule fixed-LR trainer when warmup is 0.
    """

    lr: float
    warmup_tokens: int = 0
    decay: str = "none"
    budget: Optional[int] = None
    min_lr_ratio: float = MIN_LR_RATIO

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.warmup_tokens < 0:
            raise ValueError(
                f"warmup_tokens must be >= 0, got {self.warmup_tokens}"
            )
        if self.decay not in LR_DECAY_CHOICES:
            raise ValueError(
                f"lr_decay must be one of {LR_DECAY_CHOICES}, got "
                f"{self.decay!r}"
            )
        if self.decay == "cosine" and self.budget is None:
            raise ValueError(
                "--lr-decay cosine needs --token-budget (the decay spans the "
                "token budget); there is no finite schedule to decay over "
                "otherwise"
            )
        if self.budget is not None:
            if self.budget < 1:
                raise ValueError(
                    f"token budget must be >= 1, got {self.budget}"
                )
            if self.warmup_tokens >= self.budget:
                raise ValueError(
                    f"warmup_tokens ({self.warmup_tokens}) must be strictly "
                    f"less than the token budget ({self.budget}) — the decay "
                    "phase would be empty"
                )
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError(
                f"min_lr_ratio must be in (0, 1], got {self.min_lr_ratio}"
            )

    def lr_at(self, tokens: int) -> float:
        """The LR for the step whose *pre-step* token count is ``tokens``."""
        t = max(int(tokens), 0)
        if self.warmup_tokens and t < self.warmup_tokens:
            return self.lr * t / self.warmup_tokens
        if self.decay == "cosine":
            assert self.budget is not None
            span = self.budget - self.warmup_tokens
            x = (t - self.warmup_tokens) / span if span > 0 else 1.0
            x = min(max(x, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * x))
            return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)
        return self.lr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.decay,
            "warmup_tokens": self.warmup_tokens,
            "token_budget": self.budget,
            "min_lr_ratio": self.min_lr_ratio,
        }


# ---------------------------------------------------------------------------
# Checkpoint selection by NUMERIC step (DELIVERABLE 5)
# ---------------------------------------------------------------------------
_STEP_CKPT_RE = re.compile(r"^step-(\d+)\.pt$")


def list_checkpoint_candidates(checkpoint_dir: str) -> List[str]:
    """``step-<N>.pt`` files under ``checkpoint_dir``, NUMERIC step desc.

    Regression-guarded: sorting must be numeric, never lexicographic
    (lexicographic order puts ``step-9.pt`` "newest" ahead of
    ``step-100.pt`` — the exact bug this replaces).
    """
    entries: List[tuple] = []
    for name in os.listdir(checkpoint_dir):
        m = _STEP_CKPT_RE.match(name)
        if m:
            entries.append((int(m.group(1)), os.path.join(checkpoint_dir, name)))
    entries.sort(key=lambda pair: pair[0], reverse=True)
    return [path for _, path in entries]


def validate_resume_checkpoint(ckpt: dict, preset: str) -> None:
    """Sanity checks a checkpoint must pass to resume from (shared by the
    explicit-file and directory-scan paths). Raises ``ValueError`` naming the
    failing contract — reused verbatim by the corrupt-fallback scanner."""
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint has unsupported format {ckpt.get('format')!r}: "
            f"expected {CHECKPOINT_FORMAT!r}"
        )
    ckpt_cfg = ModelConfig(**ckpt["model_config"]).derive()
    ckpt_preset = resolve_preset(ckpt_cfg)
    if ckpt_preset != preset:
        raise ValueError(
            f"checkpoint is the {ckpt_preset} preset but --preset is "
            f"{preset!r} — the preset must match the resumed run"
        )
    exp_params, exp_vocab = CANONICAL_PRESETS[ckpt_preset]
    if int(ckpt["n_params"]) != exp_params:
        raise ValueError(
            f"checkpoint records {ckpt['n_params']:,} params but the canonical "
            f"{ckpt_preset} preset is exactly {exp_params:,} — corrupted or "
            "tampered checkpoint"
        )
    if ckpt.get("optimizer_state_dict") is None:
        raise ValueError(
            f"checkpoint has no optimizer_state_dict — it predates resume "
            "support (old v1 format); use a checkpoint written by the current "
            "script"
        )


# ---------------------------------------------------------------------------
# Run-metadata sidecar (DELIVERABLE 4)
# ---------------------------------------------------------------------------
def git_repo_state() -> Dict[str, Any]:
    """``{commit, branch, dirty}`` of the running code, best-effort.

    ``None`` when the repo is unavailable (e.g. a Colab checkout without
    ``.git``) — the run must never fail on git introspection.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        branch = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "-C", _REPO_ROOT, "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:  # pragma: no cover - environment dependent
        return {"commit": None, "branch": None, "dirty": None}


def build_run_metadata(
    *,
    preset: str,
    model_cfg: ModelConfig,
    n_params: int,
    training_config: Dict[str, Any],
    data_provenance: Dict[str, Any],
    tokenizer: Dict[str, Any],
    device: str,
    resumed_from: Optional[str] = None,
) -> Dict[str, Any]:
    """The reproducibility record written to ``train_run_metadata.json`` and
    embedded (the *same dict*) in every checkpoint payload."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema": RUN_METADATA_SCHEMA,
        "script": "scripts/train_oasst1.py",
        "preset": preset,
        "model_config": asdict(model_cfg),
        "n_params": int(n_params),
        "training_config": training_config,
        "data_provenance": data_provenance,
        "tokenizer": tokenizer,
        "git": git_repo_state(),
        "timestamps": {"started": now, "finished": None},
        "resumed_from": resumed_from,
        "resume_history": [resumed_from] if resumed_from else [],
    }


def write_run_metadata(out_dir: str, run_metadata: Dict[str, Any]) -> str:
    """Persist the sidecar (atomic) and return its absolute path."""
    path = os.path.join(out_dir, RUN_METADATA_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def merge_run_metadata(
    base: Dict[str, Any], *, resumed_from: str
) -> Dict[str, Any]:
    """Provenance merge for a resume: keep the ORIGINAL run's model/training
    config and data provenance (they are what produced the weights), stamp the
    new session's git state, timestamp, and resume trail."""
    merged = deepcopy(base)
    merged["git"] = git_repo_state()
    merged["timestamps"] = dict(merged.get("timestamps") or {})
    merged["timestamps"]["resumed_at"] = datetime.now(timezone.utc).isoformat()
    merged["resumed_from"] = os.path.abspath(resumed_from)
    history = list(merged.get("resume_history") or [])
    if os.path.abspath(resumed_from) not in history:
        history.append(os.path.abspath(resumed_from))
    merged["resume_history"] = history
    return merged


def capture_rng_state() -> dict:
    """Snapshot every RNG source (torch CPU/GPU, numpy, python random).

    Stored in each checkpoint so a resumed run continues with the exact RNG
    state the uninterrupted run would have had (the repo's determinism
    guarantees make resume bit-exact — asserted by the test suite).
    """
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot from :func:`capture_rng_state` (or its old pair)."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Step 1 — deterministic train/val split (disjoint, no leakage)
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    """Outcome of :func:`split_jsonl`."""

    train_path: str
    val_path: str
    total_docs: int
    train_docs: int
    val_docs: int
    seed: int


def split_jsonl(
    src_path: str,
    out_dir: str,
    ratio: float = 0.9,
    seed: int = 0,
    max_docs: Optional[int] = None,
) -> SplitResult:
    """Deterministically split one JSONL corpus into disjoint train/val files.

    Reads the source **once** into memory (the task scope is a small subset —
    a few hundred documents; the heavy tokenization/training paths afterwards
    stay streaming). Lines are preserved verbatim so the written files are
    byte-identical to the source records. The split is a fixed-seed shuffle of
    document indices, so the same ``(src, ratio, seed)`` always yields the same
    two files, and a document can never appear in both (no leakage).

    Raises ``ValueError`` if the corpus is too small to split (needs at least
    one train and one validation document).
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"corpus JSONL not found: {src_path}")
    docs: List[str] = []
    with open(src_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                docs.append(line)
                if max_docs is not None and len(docs) >= max_docs:
                    break
    if len(docs) < 2:
        raise ValueError(
            f"cannot split {src_path}: need at least 2 documents for a "
            f"train/val split, found {len(docs)}"
        )
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    n_train = max(1, round(len(docs) * ratio))
    if n_train >= len(docs):  # ratio too high: keep exactly one doc for val
        n_train = len(docs) - 1
    train_ids = set(idx[:n_train])
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as ft, open(
        val_path, "w", encoding="utf-8"
    ) as fv:
        for i, line in enumerate(docs):
            (ft if i in train_ids else fv).write(line + "\n")
    return SplitResult(
        train_path=train_path,
        val_path=val_path,
        total_docs=len(docs),
        train_docs=n_train,
        val_docs=len(docs) - n_train,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Step 2 — Talos-native byte-level BPE tokenizer (train split only, vocab 1024)
# ---------------------------------------------------------------------------
def train_tokenizer_for_run(
    train_path: str,
    tokenizer_path: str,
    *,
    preset: str = "tiny",
    num_merges: Optional[int] = None,
    minfreq: int = 2,
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
    text_field: str = "text",
) -> ByteLevelBPETokenizer:
    """Train a byte-level BPE on the train split and persist it.

    Uses the canonical preset's tokenizer config
    (:func:`configs.presets.preset_tokenizer_config` — the vocab-1024 contract
    shared by every canonical preset: 256 base bytes + 4 specials + up to 764
    merges), consumed **one document at a time** via the PR-#16 unique-word
    frequency table, so memory stays bounded. Exposes the ``num_merges`` /
    ``minfreq`` / ``max_docs`` / ``max_chars`` knobs. The artifact is written
    next to the checkpoints so eval/generation can load the exact same
    tokenizer later. The trained vocab is checked against the *preset model's*
    vocab size (``ALL_PRESETS[preset]().vocab_size``) so a tokenizer/model
    mismatch fails here with the model's real bound, regardless of which
    canonical preset is being trained.
    """
    config = preset_tokenizer_config(preset)
    model_vocab = ALL_PRESETS[preset]().vocab_size
    texts: Iterable[str] = iter_text_documents(
        [train_path], jsonl_field=text_field, on_invalid="die"
    )
    result = train_tokenizer(
        texts,
        config,
        minfreq=minfreq,
        num_merges=num_merges,
        max_docs=max_docs,
        max_chars=max_chars,
    )
    tokenizer = result.tokenizer
    if tokenizer.vocab_size > model_vocab:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size} exceeds the "
            f"{preset} model's {model_vocab} — config drift in "
            f"configs/presets.{preset}_tokenizer_config"
        )
    tokenizer.save(tokenizer_path)
    log.info(
        "trained tokenizer: vocab=%d (merges=%d, %d docs, %d chars) -> %s",
        tokenizer.vocab_size,
        tokenizer.merge_count,
        result.num_words,
        result.num_bytes,
        tokenizer_path,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Step 3 — canonical preset model with a hard param-count guard
# ---------------------------------------------------------------------------
def check_preset_compat(preset: str, cfg, n_params: int) -> None:
    """Fail fast if the model is not a registered canonical preset.

    Guards the hard requirement for whichever preset is being trained: its
    exact canonical parameter count at its exact canonical vocab size (from
    :data:`configs.canonical.CANONICAL_PRESETS`). Any drift in
    ``configs/presets`` is caught here *before* training starts. Every
    registered canonical preset is enforced through this same path — ``tiny``
    at exactly 254,272 params / vocab 1024, ``tiny_1m`` at exactly 1,000,320,
    ``tiny_10m`` at exactly 9,952,320, ``tiny_100m`` at exactly 96,482,304 —
    with no per-preset code anywhere.
    """
    expected, expected_vocab = CANONICAL_PRESETS[preset]
    problems: List[str] = []
    if cfg.vocab_size != expected_vocab:
        problems.append(
            f"vocab_size={cfg.vocab_size} (expected {expected_vocab})"
        )
    if n_params != expected:
        problems.append(
            f"param count={n_params:,} (expected exactly {expected:,})"
        )
    if problems:
        raise ValueError(
            f"{preset} preset config drift detected — refusing to train: "
            + "; ".join(problems)
            + f". The canonical {preset} preset is exactly {expected:,} params "
            f"at vocab_size {expected_vocab} (configs/presets.{preset}_config). "
            "Fix the preset before training — do not 'fix' this guard."
        )


def check_tiny_compat(cfg, n_params: int) -> None:
    """Fail fast if the model is not the owner-fixed canonical tiny preset.

    Guards the hard requirement: exactly 254,272 parameters at vocab_size 1024
    (hidden 64, 2 layers, 4 heads, 2 KV heads, dense FFN, seq 512). Any drift
    in ``configs/presets.tiny_config`` is caught here *before* training starts.
    (Registry-enforced via :func:`check_preset_compat` — no per-preset logic.)
    """
    check_preset_compat("tiny", cfg, n_params)


def build_preset_model(preset: str) -> TalosGPT:
    """Build a canonical preset model and assert its exact-param contract."""
    cfg = ALL_PRESETS[preset]().derive()
    model = TalosGPT(cfg)
    check_preset_compat(preset, cfg, model.num_parameters())
    return model


def build_tiny_model() -> TalosGPT:
    """Build the canonical tiny model and assert the 254,272-param contract."""
    return build_preset_model("tiny")


# ---------------------------------------------------------------------------
# Step 4 — streaming training with per-epoch validation + checkpoints
# ---------------------------------------------------------------------------
@dataclass
class EpochRow:
    """One epoch's losses and the checkpoint written for it."""

    epoch: int
    steps: int
    global_step: int
    train_loss: float
    val_loss: Optional[float]
    checkpoint: str
    wall_s: float


@dataclass
class TrainingHistory:
    rows: List[EpochRow] = field(default_factory=list)
    tokens_processed: int = 0
    run_wall_s: float = 0.0
    #: token-budget stop state (additive; default = no budget -> False)
    budget_reached: bool = False
    token_budget: Optional[int] = None

    def row(self, epoch: int) -> EpochRow:
        return next(r for r in self.rows if r.epoch == epoch)


def evaluate(
    model: TalosGPT,
    dataset: StreamingTokenizedDataset,
    *,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Optional[float]:
    """Mean causal-LM loss over a streamed (never materialized) dataset.

    Returns ``None`` when the stream contains no full batches (e.g. an empty
    validation split). The model is left in ``train()`` mode afterwards. Each
    batch is token-id-range-checked at assembly (before it reaches the model)
    so a drifted tokenizer fails here with batch context, never as an
    ``IndexError``/CUDA assert inside the model.
    """
    was_training = model.training
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    vocab = model.config.vocab_size
    total, count, steps = 0.0, 0, 0
    try:
        with torch.no_grad():
            for batch in dataset:
                if max_steps is not None and steps >= max_steps:
                    break
                x = batch.to(device).long()
                validate_token_ids(x, vocab, where="validation batch")
                logits, _ = model(x[:, :-1])
                n = x[:, 1:].numel()
                total += float(loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)))
                count += n
                steps += 1
    finally:
        if was_training:
            model.train()
    return (total / count) if count else None


def save_checkpoint(
    path: str,
    model: TalosGPT,
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    tokenizer_path: str,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    rng_state: Optional[dict] = None,
    epoch: Optional[int] = None,
    tokens_consumed: Optional[int] = None,
    run_metadata: Optional[dict] = None,
    tokenizer_fingerprint: Optional[str] = None,
) -> None:
    """One checkpoint artifact: weights + config + step + losses + tokenizer info.

    Format ``talos-training-checkpoint-v1`` stays **backward compatible**: the
    original keys are unchanged and the resume-enabling keys (``optimizer_state_dict``,
    ``rng_state``, ``epoch``, ``tokenizer_fingerprint``) are additive, so
    checkpoints written before this change still load for eval/generation, and
    new checkpoints load in any old reader.
    """
    cfg = model.config
    if tokenizer_fingerprint is None:
        tokenizer_fingerprint = (
            tokenizer_file_sha256(tokenizer_path) if tokenizer_path else None
        )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "model_config": asdict(cfg),
        "n_params": model.num_parameters(),
        "vocab_size": cfg.vocab_size,
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        #: absolute path so the checkpoint is self-describing from anywhere;
        #: the tokenizer artifact itself lives next to the checkpoint in out_dir.
        "tokenizer_path": os.path.abspath(tokenizer_path) if tokenizer_path else None,
        #: sha256 of the serialized tokenizer.json — content identity, so a
        #: swapped/re-trained sidecar is caught on load (audit P0 fix 4). On
        #: the packed-corpus path without a tokenizer sidecar, this is the
        #: manifest-recorded sha256 of the corpus's tokenizer.
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_state_dict": model.state_dict(),
        # Resume-enabling state (additive, v1 format): optimizer + RNG + counters.
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "rng_state": rng_state,
        "epoch": epoch,
        # Token-budget accounting (additive): tokens consumed SINCE RUN START,
        # including every token consumed by earlier resumed sessions.
        "tokens_consumed": None if tokens_consumed is None else int(tokens_consumed),
        # Run-metadata sidecar (additive; the SAME dict as train_run_metadata.json).
        "run_metadata": run_metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (pure dict)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def train_epochs(
    model: TalosGPT,
    train_ds: StreamingTokenizedDataset,
    val_ds: Optional[StreamingTokenizedDataset],
    *,
    out_dir: str,
    tokenizer_path: str,
    lr: float,
    epochs: int,
    device: torch.device,
    seed: int,
    max_steps_per_epoch: Optional[int] = None,
    val_max_steps: Optional[int] = None,
    resume_from: Optional[dict] = None,
    token_budget: Optional[int] = None,
    warmup_tokens: int = 0,
    lr_decay: str = "none",
    run_metadata: Optional[dict] = None,
) -> TrainingHistory:
    """Train the model on the streamed train split, epoch by epoch.

    Per epoch: one pass over the train stream (deterministic order — same
    ``seed`` reproduces the same run), mean train loss, validation loss over a
    *separate* stream (no token-level leakage), and a checkpoint saved with
    weights + config + step + losses (+ optimizer/RNG state, see
    :func:`save_checkpoint`). AdamW + CrossEntropyLoss on
    ``x[:, :-1] -> x[:, 1:]`` — the identical objective examples/tiny_train.py
    uses.

    Resume: when ``resume_from`` (a checkpoint dict from
    :func:`load_checkpoint`) is given, the model/optimizer/RNG state and the
    step/epoch counters are restored *before* the loop, and training continues
    from ``resume_step + 1``. ``--epochs`` is the **target total**: the loop
    runs from ``resume.epoch + 1`` to ``epochs``. Because the pipeline is
    fully deterministic (fixed seed, no RNG in the data path, no dropout),
    a resumed run is bit-identical to an uninterrupted run that never stopped
    — asserted by ``tests/test_training.py::test_resume_bit_exact``.

    Token budget (DELIVERABLE 2): with ``token_budget`` set, training stops as
    soon as tokens consumed SINCE RUN START (restored from the checkpoint on
    resume — ``tokens_consumed`` is recorded in every checkpoint) reaches the
    budget. The partial epoch still gets validation + a checkpoint, then the
    outer loop stops. ``history.tokens_processed`` is the cumulative counter
    (budget accounting); ``history.budget_reached`` reports the stop reason.

    LR schedule (DELIVERABLE 3, additive): ``warmup_tokens`` > 0 ramps ``lr``
    linearly over the first W tokens; ``lr_decay="cosine"`` decays ``lr`` to
    ``MIN_LR_RATIO * lr`` over the budget after warmup. Defaults reproduce the
    fixed-LR trainer exactly (``lr_at`` == ``lr`` at every step). The schedule
    only sets ``param_groups[0]["lr"]`` per step — gradient clipping and
    NaN/Inf detection (wherever present) wrap the optimizer step and are
    untouched by the schedule.
    """
    set_seed(seed)
    schedule = TokenSchedule(lr=lr, warmup_tokens=warmup_tokens, decay=lr_decay,
                             budget=token_budget)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    vocab = model.config.vocab_size
    history = TrainingHistory(token_budget=token_budget)
    start_epoch, global_step = 1, 0
    tokens_consumed = 0
    if resume_from is not None:
        if resume_from.get("optimizer_state_dict") is None:
            raise ValueError(
                "cannot resume: checkpoint has no optimizer_state_dict — it "
                "was written before resume support (old v1 format); retrain "
                "from step 0 or re-save with the current script"
            )
        model.load_state_dict(resume_from["model_state_dict"])
        opt.load_state_dict(resume_from["optimizer_state_dict"])
        restore_rng_state(resume_from["rng_state"])
        # Keep the caller's --lr authoritative (AdamW.state_dict records the
        # lr at save time under the 'lr' group key).
        for group in opt.param_groups:
            group["lr"] = lr
        start_epoch = int(resume_from.get("epoch", 0)) + 1
        global_step = int(resume_from["step"])
        recorded_tokens = resume_from.get("tokens_consumed")
        if recorded_tokens is None:
            if token_budget is not None:
                # Old checkpoint (pre-token-budget): no exact counter. The only
                # deterministic schedule the script has ever produced is
                # batch*(seq-1) tokens per step with constant batch/seq, so the
                # estimate is exact for every legacy run — warn anyway.
                per_step = train_ds.batch_size * (train_ds.seq_len - 1)
                recorded_tokens = global_step * per_step
                log.warning(
                    "resume checkpoint %s has no tokens_consumed counter; "
                    "estimating %d tokens from %d steps at batch*seq-1=%d — "
                    "recorded exactly in the next checkpoint",
                    "step-%d.pt" % global_step, recorded_tokens, global_step,
                    per_step,
                )
            else:
                recorded_tokens = 0
        tokens_consumed = int(recorded_tokens)
        history.tokens_processed = tokens_consumed
        if start_epoch > epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1} "
                f"(>= --epochs {epochs}) — nothing left to train; raise --epochs"
            )
        log.info(
            "resuming from step %d (epoch %d completed, %d tokens consumed) — "
            "continuing to epoch %d",
            global_step, start_epoch - 1, tokens_consumed, epochs,
        )
        if token_budget is not None and tokens_consumed >= token_budget:
            log.warning(
                "token budget %d already consumed at resume (%d) — no further "
                "training needed; the run will finish without new steps",
                token_budget, tokens_consumed,
            )
            history.budget_reached = True
    t_run = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        if history.budget_reached:
            break
        model.train()
        epoch_losses: List[float] = []
        steps = 0
        t_epoch = time.monotonic()
        for batch in train_ds:
            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break
            if history.budget_reached:
                break
            x = batch.to(device).long()
            validate_token_ids(x, vocab, where="train batch")
            # LR schedule: the step's LR is a function of tokens consumed so
            # far (pre-step). Fixed-LR runs (no warmup, decay "none") always
            # resolve to the caller's lr — bit-identical to the old trainer.
            opt.param_groups[0]["lr"] = schedule.lr_at(tokens_consumed)
            logits, _ = model(x[:, :-1])
            loss = loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
            step_tokens = int(x[:, 1:].numel())
            tokens_consumed += step_tokens
            history.tokens_processed = tokens_consumed
            steps += 1
            global_step += 1
            if token_budget is not None and tokens_consumed >= token_budget:
                history.budget_reached = True
                break
        if steps == 0 and history.budget_reached:
            # Budget was already exhausted before this epoch (resume case):
            # nothing new was trained — no val pass, no checkpoint overwrite.
            break
        train_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        )
        val_loss = (
            evaluate(model, val_ds, device=device, max_steps=val_max_steps)
            if val_ds is not None
            else None
        )
        ckpt_path = os.path.join(out_dir, f"step-{global_step}.pt")
        # On the packed path without a tokenizer sidecar file, the recorded
        # identity is the manifest's tokenizer sha256 (from run_metadata).
        ckpt_fp = None
        if tokenizer_path:
            ckpt_fp = tokenizer_file_sha256(tokenizer_path)
        elif run_metadata is not None:
            ckpt_fp = (run_metadata.get("tokenizer") or {}).get("sha256")
        save_checkpoint(
            ckpt_path, model, global_step, train_loss, val_loss, tokenizer_path,
            optimizer=opt, rng_state=capture_rng_state(), epoch=epoch,
            tokens_consumed=tokens_consumed, run_metadata=run_metadata,
            tokenizer_fingerprint=ckpt_fp,
        )
        row = EpochRow(
            epoch=epoch,
            steps=steps,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint=ckpt_path,
            wall_s=time.monotonic() - t_epoch,
        )
        history.rows.append(row)
        log.info(
            "epoch %d/%d: steps=%d train_loss=%.4f val_loss=%s checkpoint=%s",
            epoch, epochs, steps, train_loss,
            "n/a" if val_loss is None else f"{val_loss:.4f}",
            ckpt_path,
        )
    history.run_wall_s = time.monotonic() - t_run
    return history


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------
def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.train_oasst1",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--data", default=None,
                   help="OASST1-shaped JSONL corpus (any path; real or synthetic). "
                        "Exactly one of --data / --packed-dir is required.")
    p.add_argument("--packed-dir", default=None, metavar="DIR",
                   help="train from packed-token .npy shards + manifest.json "
                        "produced by scripts/prepare_corpus.py (rows are used "
                        "verbatim — no re-tokenization). The manifest is "
                        "validated (format, dtype, seq, per-shard row counts, "
                        "tokenizer sha256 vs --tokenizer-json, vocab bounds) "
                        "and val rows come from the val shards. Exactly one of "
                        "--data / --packed-dir is required.")
    p.add_argument("--out-dir", required=True, help="run directory (split, tokenizer, checkpoints, metrics)")
    p.add_argument("--seed", type=int, default=0, help="fixed seed for split + training (default 0)")
    p.add_argument("--preset", default="tiny",
                   help=f"canonical preset to train ({', '.join(sorted(CANONICAL_PRESETS))}; default tiny)")
    p.add_argument("--resume", default=None, metavar="step-<N>.pt | DIR",
                   help="resume training from this checkpoint — a step-<N>.pt "
                        "FILE, or a DIRECTORY of checkpoints (the numerically "
                        "NEWEST valid one wins; corrupt/unloadable candidates "
                        "are skipped with a warning and the next-newest is "
                        "tried). Weights + optimizer + RNG + step/epoch + "
                        "tokens-consumed counters are restored; the tokenizer "
                        "fingerprint must match; --epochs is then the target "
                        "TOTAL. The split + tokenizer are NOT re-trained on "
                        "resume — the checkpoint's sidecar tokenizer is "
                        "verified and reused.")
    # split
    p.add_argument("--split-ratio", type=float, default=0.9, help="train fraction (default 0.9; JSONL path only)")
    p.add_argument("--split-max-docs", type=int, default=None,
                   help="cap how many source docs enter the split (small-subset guard; JSONL path only)")
    # tokenizer (Talos-native BPE trainer, PR #16 knobs)
    p.add_argument("--bpe-num-merges", type=int, default=None,
                   help="exact BPE merges (default: vocab-1024 budget = 764, stopped by minfreq)")
    p.add_argument("--bpe-minfreq", type=int, default=2, help="BPE min pair frequency (default 2)")
    p.add_argument("--bpe-max-docs", type=int, default=None, help="BPE training doc cap")
    p.add_argument("--bpe-max-chars", type=int, default=None, help="BPE training char cap")
    p.add_argument("--tokenizer-json", default=None, metavar="path/to/tokenizer.json",
                   help="load an EXISTING Talos tokenizer.json instead of training "
                        "BPE from --data (the BPE-training path stays the default "
                        "when this is omitted). The file is copied into "
                        "out_dir/tokenizer.json and its sha256 fingerprint is "
                        "recorded in every checkpoint + metrics.json; on --resume "
                        "the provided file's fingerprint must match the "
                        "checkpoint's recorded one. On --packed-dir its sha256 "
                        "must also match the manifest's recorded tokenizer "
                        "sha256 (the rows were packed with that tokenizer). "
                        "When omitted on the packed path, the manifest-recorded "
                        "path is used if it exists on this machine, else the "
                        "run proceeds without a tokenizer sidecar (identity "
                        "still recorded from the manifest).")
    # training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seq", type=int, default=None,
                   help="sequence length per row. Default: 64 for the JSONL "
                        "path; for --packed-dir the manifest's seq_len (an "
                        "explicit --seq that differs from the manifest is an "
                        "error — rows are used verbatim).")
    p.add_argument("--batch", type=int, default=4, help="batch size")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--token-budget", type=int, default=None, metavar="N",
                   help="stop training when the token budget N is consumed "
                        "(tokens since RUN START, including resumed tokens "
                        "restored from checkpoints). The partial epoch still "
                        "gets validation + a checkpoint. Mutually compatible "
                        "with --epochs (whichever limit comes first wins).")
    p.add_argument("--warmup-tokens", type=int, default=0, metavar="W",
                   help="linear LR warmup 0 -> --lr over the first W tokens "
                        "(default 0 = no warmup, fixed LR)")
    p.add_argument("--lr-decay", choices=LR_DECAY_CHOICES, default="none",
                   help="LR decay after warmup: 'none' (default, fixed LR) or "
                        "'cosine' — cosine from --lr down to 10% of it over "
                        "--token-budget (requires --token-budget).")
    p.add_argument("--max-steps-per-epoch", type=int, default=None,
                   help="cap steps per epoch (CI / smoke runs)")
    p.add_argument("--val-max-steps", type=int, default=None,
                   help="cap validation batches (CI / smoke runs)")
    p.add_argument("--device", default=None, help="compute device (default: auto)")
    return p


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _verify_and_load_tokenizer(
    path: str, expected_sha: Optional[str], *, label: str
) -> ByteLevelBPETokenizer:
    """Load a tokenizer.json after a sha256 identity check (loud on mismatch)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = tokenizer_file_sha256(path)
    if expected_sha and actual != expected_sha:
        raise ValueError(
            f"{label} identity mismatch: expected sha256 {expected_sha} "
            f"but {path} hashes to {actual} — refusing to use the wrong "
            "tokenizer"
        )
    return ByteLevelBPETokenizer.from_file(path)


def train_run(args: argparse.Namespace) -> dict:
    """Run the full chain; returns the metrics dict (also saved to metrics.json).

    Two data sources (exactly one required):

    * ``--data <jsonl>`` (the original OASST1 path): deterministic train/val
      split, BPE-trained or ``--tokenizer-json`` tokenizer, streamed batches.
    * ``--packed-dir <dir>`` (corpus-pretraining path, DELIVERABLE 1): rows
      from ``prepare_corpus.py`` ``.npy`` shards used verbatim; the manifest
      is validated (format/dtype/seq/row-counts/tokenizer sha256/vocab) and
      val rows come from the val shards.

    With ``--resume`` (a ``step-<N>.pt`` file, or a directory whose
    numerically-newest VALID checkpoint is used — corrupt candidates skipped
    with a warning) the run **continues** an existing run: the tokenizer is
    loaded from the checkpoint's recorded path and verified by sha256
    fingerprint (P0 fix 4), model/optimizer/RNG + step/epoch + tokens-consumed
    counters are restored inside :func:`train_epochs`, and the packed corpus
    is identity-checked against the checkpoint's recorded provenance.
    Tokenizer training and preset build happen only on a fresh run.
    """
    t0 = time.monotonic()
    device = _resolve_device(args.device)
    set_seed(args.seed)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- data-source sanity + early schedule validation ---------------------
    # New flags are read via getattr so legacy callers (tests, notebooks) that
    # build args Namespaces without them keep working unchanged.
    packed_dir_arg = getattr(args, "packed_dir", None)
    token_budget = getattr(args, "token_budget", None)
    warmup_tokens = getattr(args, "warmup_tokens", 0) or 0
    lr_decay = getattr(args, "lr_decay", "none")
    if packed_dir_arg and args.data:
        raise ValueError("pass exactly one of --data / --packed-dir, not both")
    data_src = "packed" if packed_dir_arg else ("jsonl" if args.data else None)
    if data_src is None:
        raise ValueError("exactly one of --data / --packed-dir is required")
    # Raises early on invalid combos (e.g. cosine without a token budget).
    TokenSchedule(
        lr=args.lr, warmup_tokens=warmup_tokens, decay=lr_decay,
        budget=token_budget,
    )

    # ---- resume: resolve the actual checkpoint (file, or numeric-desc scan) --
    resume_path: Optional[str] = args.resume
    resume_ckpt: Optional[dict] = None
    if args.resume:
        if os.path.isdir(args.resume):
            candidates = list_checkpoint_candidates(args.resume)
            if not candidates:
                raise FileNotFoundError(
                    f"--resume directory has no step-<N>.pt checkpoints: "
                    f"{args.resume}"
                )
            tried: List[str] = []
            for candidate in candidates:
                try:
                    ckpt = load_checkpoint(candidate)
                    validate_resume_checkpoint(ckpt, args.preset)
                    resume_ckpt, resume_path = ckpt, candidate
                    log.info(
                        "resume: selected %s (numeric-newest VALID checkpoint)",
                        os.path.basename(candidate),
                    )
                    break
                except Exception as exc:  # corrupt/unloadable — skip + warn
                    log.warning(
                        "resume candidate %s unusable (%s: %s) — skipping",
                        os.path.basename(candidate), type(exc).__name__, exc,
                    )
                    tried.append(
                        f"{os.path.basename(candidate)} ({type(exc).__name__}: {exc})"
                    )
            if resume_ckpt is None:
                raise FileNotFoundError(
                    f"no VALID resume checkpoint in {args.resume}: all "
                    f"{len(candidates)} candidate(s) failed — " + "; ".join(tried)
                )
        else:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(
                    f"--resume checkpoint not found: {args.resume}"
                )
            resume_ckpt = load_checkpoint(args.resume)
            validate_resume_checkpoint(resume_ckpt, args.preset)

    # ---- canonical preset config, printed before anything else ------------
    preset = args.preset
    if preset not in CANONICAL_PRESETS:
        raise ValueError(
            f"unknown --preset {preset!r}: choose from "
            f"{', '.join(sorted(CANONICAL_PRESETS))}"
        )
    exp_params, exp_vocab = CANONICAL_PRESETS[preset]
    cfg = ALL_PRESETS[preset]().derive()

    # ---- resume runs must continue the SAME data source ---------------------
    ckpt_run_meta = (resume_ckpt or {}).get("run_metadata") or {}
    ckpt_prov = ckpt_run_meta.get("data_provenance") or {}
    ckpt_source = ckpt_prov.get("source")
    if resume_ckpt is not None and ckpt_source:
        if ckpt_source != data_src:
            raise ValueError(
                f"resume checkpoint is from a {ckpt_source!r} run but the CLI "
                f"args select {data_src!r} — resume must continue the same "
                "data source"
            )

    # ---- 0) data: packed manifest (validated) or deterministic JSONL split --
    packed_manifest: Optional[dict] = None
    split: Optional[SplitResult] = None
    if data_src == "packed":
        packed_dir = os.path.abspath(packed_dir_arg)
        provided_tok_sha = (
            tokenizer_file_sha256(args.tokenizer_json)
            if args.tokenizer_json
            else None
        )
        packed_manifest = load_packed_manifest(
            packed_dir,
            expected_seq=None,  # seq resolved below (manifest wins unless --seq matches)
            expected_vocab=cfg.vocab_size,
            expected_tokenizer_sha256=provided_tok_sha,
        )
        m_seq = packed_manifest["seq_len"]
        if args.seq is None or args.seq == m_seq:
            effective_seq = m_seq
        else:
            raise ValueError(
                f"--seq {args.seq} does not match the packed corpus seq_len "
                f"{m_seq} — packed rows are used verbatim; pass --seq {m_seq} "
                "(or omit --seq)"
            )
        counts = packed_manifest["metadata"]["counts"]
        print(f"  packed corpus : {packed_dir} (seq {m_seq}, dtype "
              f"{packed_manifest['dtype']}, train {counts['train_rows']:,} rows "
              f"/ {counts['train_tokens']:,} tokens, val {counts['val_rows']:,} "
              f"rows, {packed_manifest['num_shards']} shards)")
        if resume_ckpt is not None:
            recorded_identity = ckpt_prov.get("manifest_identity")
            if recorded_identity and recorded_identity != manifest_identity(
                packed_manifest
            ):
                raise ValueError(
                    "resume packed corpus does not match the checkpoint's "
                    "recorded corpus identity (seq/dtype/tokenizer/rows differ)"
                    " — refusing to continue training on different data"
                )
    else:
        effective_seq = args.seq if args.seq is not None else 64
        split = split_jsonl(
            args.data,
            os.path.join(out_dir, "data"),
            ratio=args.split_ratio,
            seed=args.seed,
            max_docs=args.split_max_docs,
        )
        print(f"  split: {split.total_docs} docs -> train {split.train_docs} / "
              f"val {split.val_docs} ({os.path.basename(split.train_path)}, "
              f"{os.path.basename(split.val_path)})")

    print("=" * 72)
    print("Talos OASST1-style training run"
          + (f" (RESUMING from {resume_path})" if resume_ckpt else ""))
    print(f"  model preset  : {preset} ({cfg.ffn_type}) — vocab={cfg.vocab_size} "
          f"hidden={cfg.hidden_size} layers={cfg.num_layers} "
          f"heads={cfg.num_attention_heads} kv={cfg.num_kv_heads} "
          f"ffn={cfg.ffn_type} seq={cfg.max_seq_len}")
    print(f"  expected      : EXACTLY {exp_params:,} params, vocab_size {exp_vocab}")
    print(f"  data          : {args.data or packed_dir_arg} ({data_src})")
    print(f"  seed          : {args.seed} | epochs {args.epochs} | seq "
          f"{effective_seq} | batch {args.batch} | lr {args.lr} | device {device}")
    print(f"  budget        : {token_budget if token_budget is not None else 'unbounded'} "
          f"tokens | warmup {warmup_tokens} | lr-decay {lr_decay}")
    print("=" * 72)

    # ---- 1) tokenizer + data provenance record ------------------------------
    manifest_tok_sha: Optional[str] = None
    if packed_manifest is not None:
        manifest_tok_sha = packed_manifest["metadata"]["tokenizer"]["sha256"]
    tokenizer_origin: str
    tokenizer_json = getattr(args, "tokenizer_json", None)
    tokenizer = None
    tokenizer_path: Optional[str] = None
    if resume_ckpt is None:
        tokenizer_path = os.path.join(out_dir, "tokenizer.json")
        if data_src == "packed":
            tok_source = tokenizer_json
            if tok_source is None:
                recorded = packed_manifest["metadata"]["tokenizer"].get("path")
                if recorded and os.path.isfile(recorded):
                    if (
                        manifest_tok_sha is not None
                        and tokenizer_file_sha256(recorded) == manifest_tok_sha
                    ):
                        tok_source = recorded
            if tok_source:
                tokenizer = _verify_and_load_tokenizer(
                    tok_source, manifest_tok_sha,
                    label="--tokenizer-json (must be the manifest's tokenizer)",
                )
                shutil.copyfile(tok_source, tokenizer_path)
                tokenizer_origin = "loaded"
                print(f"  tokenizer     : loaded from {tok_source} (vocab "
                      f"{tokenizer.vocab_size}, {tokenizer.merge_count} merges) "
                      f"— sha256 matches the manifest; copied to {tokenizer_path}")
            else:
                tokenizer_origin = "packed-manifest"
                tokenizer_path = None
                print(f"  tokenizer     : no sidecar tokenizer.json available — "
                      f"identity recorded from the manifest "
                      f"(sha256 {manifest_tok_sha[:12]}…, vocab "
                      f"{packed_manifest['metadata']['tokenizer']['vocab_size']})")
        elif tokenizer_json:
            if not os.path.isfile(tokenizer_json):
                raise FileNotFoundError(
                    f"--tokenizer-json not found: {tokenizer_json}"
                )
            shutil.copyfile(tokenizer_json, tokenizer_path)
            tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
            tokenizer_origin = "loaded"
            print(f"  tokenizer     : loaded from {tokenizer_json} (vocab "
                  f"{tokenizer.vocab_size}, {tokenizer.merge_count} merges) — BPE "
                  f"training skipped; copied to {tokenizer_path}")
        else:
            tokenizer = train_tokenizer_for_run(
                split.train_path,
                tokenizer_path,
                preset=preset,
                num_merges=args.bpe_num_merges,
                minfreq=args.bpe_minfreq,
                max_docs=args.bpe_max_docs,
                max_chars=args.bpe_max_chars,
            )
            tokenizer_origin = "trained"
    else:
        tokenizer_origin = "resumed"
        tokenizer_path = resume_ckpt.get("tokenizer_path")
        recorded_fp = resume_ckpt.get("tokenizer_fingerprint")
        if tokenizer_path and os.path.isfile(tokenizer_path):
            tokenizer = _verify_and_load_tokenizer(
                tokenizer_path, recorded_fp,
                label="resume checkpoint's sidecar tokenizer.json",
            )
            if tokenizer_json:
                provided_fp = tokenizer_file_sha256(tokenizer_json)
                if provided_fp != recorded_fp:
                    raise ValueError(
                        f"--tokenizer-json {tokenizer_json} does not match the "
                        f"resume checkpoint's recorded tokenizer fingerprint "
                        f"(sha256 {provided_fp[:12]}… vs {recorded_fp[:12]}…) — "
                        f"refusing to resume with a different tokenizer"
                    )
                print(f"  tokenizer     : reused from checkpoint — --tokenizer-json "
                      f"matches the recorded fingerprint (sha256 verified)")
            else:
                print(f"  tokenizer     : reused from checkpoint (vocab "
                      f"{tokenizer.vocab_size}, {tokenizer.merge_count} merges) "
                      f"— sha256 verified")
        else:
            # No sidecar file: only valid for a packed run whose identity the
            # manifest carries (verified below against the checkpoint fp).
            if data_src != "packed" or not recorded_fp:
                raise FileNotFoundError(
                    f"resume checkpoint's tokenizer not found ({tokenizer_path!r}) "
                    f"— expected the sidecar tokenizer.json next to the checkpoint"
                )
            if manifest_tok_sha != recorded_fp:
                raise ValueError(
                    f"resume checkpoint records tokenizer sha256 {recorded_fp} "
                    f"but the packed corpus manifest records {manifest_tok_sha} "
                    f"— resuming on a different tokenizer; refusing"
                )
            print(f"  tokenizer     : reused from packed-corpus manifest "
                  f"(sha256 {manifest_tok_sha[:12]}… verified) — no sidecar file")

    # Uniform vocab contract: a realized tokenizer's vocab must fit inside the
    # preset model's embedding rows. Packed rows are additionally range-checked
    # per batch against the model vocab, and the manifest's vocab_size_bound is
    # verified against it by load_packed_manifest.
    model_vocab = ALL_PRESETS[preset]().vocab_size
    if tokenizer is not None and tokenizer.vocab_size > model_vocab:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size} exceeds the {preset} "
            f"model's {model_vocab} — the tokenizer cannot be embedded; use a "
            f"tokenizer built for the canonical vocab-{model_vocab} contract"
        )

    # ---- 2) canonical preset model + hard param-count guard (fail fast) ---
    model = build_preset_model(preset).to(device)
    n_params = model.num_parameters()
    print(f"  model: {n_params:,} params (vocab {model.config.vocab_size}) — config guard OK")

    # ---- 3) streamed token batches -----------------------------------------
    # max_id=cfg.vocab_size: a tokenizer/model vocab mismatch fails at the data
    # source (per-document, with shard/doc context) instead of on the device.
    if packed_manifest is not None:
        train_paths = packed_phase_shard_paths(packed_manifest, packed_dir, "train")
        val_paths = packed_phase_shard_paths(packed_manifest, packed_dir, "val")
        train_ds: Iterable = PackedTokenDataset(
            train_paths, seq_len=effective_seq, batch_size=args.batch,
            expected_dtype=packed_manifest["dtype"], max_id=cfg.vocab_size,
        )
        val_ds: Optional[Iterable] = (
            PackedTokenDataset(
                val_paths, seq_len=effective_seq, batch_size=args.batch,
                expected_dtype=packed_manifest["dtype"], max_id=cfg.vocab_size,
            )
            if val_paths
            else None
        )
    else:
        train_ds = StreamingTokenizedDataset(
            split.train_path, tokenizer, seq_len=effective_seq,
            batch_size=args.batch, mode="pack", eos=True, max_id=cfg.vocab_size,
        )
        val_ds = StreamingTokenizedDataset(
            split.val_path, tokenizer, seq_len=effective_seq,
            batch_size=args.batch, mode="pack", eos=True, max_id=cfg.vocab_size,
        )

    # ---- 4) run-metadata sidecar (DELIVERABLE 4) ---------------------------
    # Build ONCE (fresh or resume-with-provenance-merge); the same dict goes
    # into train_run_metadata.json AND every checkpoint payload.
    training_config = {
        "batch": args.batch,
        "seq": effective_seq,
        "lr": args.lr,
        "lr_decay": lr_decay,
        "warmup_tokens": warmup_tokens,
        "min_lr_ratio": MIN_LR_RATIO,
        "token_budget": token_budget,
        "epochs": args.epochs,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "seed": args.seed,
        #: this trainer has no gradient-clipping/NaN-detection step of its own
        #: (those guard layers live notebook-side and wrap the optimizer step
        #: unchanged); recorded explicitly for the record.
        "gradient_clip_type": None,
        "nan_inf_detection": False,
        "device": str(device),
    }
    tokenizer_meta = {
        "sha256": (
            tokenizer_file_sha256(tokenizer_path)
            if tokenizer_path
            else (manifest_tok_sha if manifest_tok_sha is not None else None)
        ),
        "vocab_size": (
            tokenizer.vocab_size
            if tokenizer is not None
            else packed_manifest["metadata"]["tokenizer"]["vocab_size"]
            if packed_manifest is not None
            else None
        ),
        "merges": (
            tokenizer.merge_count
            if tokenizer is not None
            else packed_manifest["metadata"]["tokenizer"]["merge_count"]
            if packed_manifest is not None
            else None
        ),
        "origin": tokenizer_origin,
    }
    if data_src == "packed":
        data_provenance = {
            "source": "packed",
            "packed_dir": packed_dir,
            "manifest_identity": manifest_identity(packed_manifest),
            "manifest_metadata": packed_manifest["metadata"],
        }
    else:
        data_provenance = {
            "source": "jsonl",
            "data": os.path.abspath(args.data),
            "split": {
                "ratio": args.split_ratio,
                "seed": args.seed,
                "max_docs": args.split_max_docs,
                "train_docs": split.train_docs,
                "val_docs": split.val_docs,
                "total_docs": split.total_docs,
            },
        }
    if resume_ckpt is not None:
        sidecar_path = os.path.join(out_dir, RUN_METADATA_FILENAME)
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as fh:
                run_metadata = json.load(fh)
            run_metadata = merge_run_metadata(
                run_metadata, resumed_from=os.path.abspath(resume_path)
            )
        elif resume_ckpt.get("run_metadata"):
            run_metadata = merge_run_metadata(
                resume_ckpt["run_metadata"],
                resumed_from=os.path.abspath(resume_path),
            )
        else:
            log.warning(
                "resuming an old checkpoint without run metadata — building a "
                "fresh sidecar from the current args"
            )
            run_metadata = build_run_metadata(
                preset=preset, model_cfg=cfg, n_params=n_params,
                training_config=training_config, data_provenance=data_provenance,
                tokenizer=tokenizer_meta, device=str(device),
                resumed_from=os.path.abspath(resume_path),
            )
    else:
        run_metadata = build_run_metadata(
            preset=preset, model_cfg=cfg, n_params=n_params,
            training_config=training_config, data_provenance=data_provenance,
            tokenizer=tokenizer_meta, device=str(device),
        )
    metadata_path = write_run_metadata(out_dir, run_metadata)
    print(f"  run metadata  : {metadata_path}")

    # ---- 5) train with per-epoch validation + checkpoints ----------------
    history = train_epochs(
        model, train_ds, val_ds,
        out_dir=out_dir,
        tokenizer_path=tokenizer_path,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
        max_steps_per_epoch=args.max_steps_per_epoch,
        val_max_steps=args.val_max_steps,
        resume_from=resume_ckpt,
        token_budget=token_budget,
        warmup_tokens=warmup_tokens,
        lr_decay=lr_decay,
        run_metadata=run_metadata,
    )

    # ---- 6) metrics ---------------------------------------------------------
    steps_taken = sum(r.steps for r in history.rows)
    last = history.rows[-1] if history.rows else None
    metrics = {
        "format": "talos-oasst1-training-metrics-v1",
        "params": n_params,
        "vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": tokenizer_meta["vocab_size"],
        "tokenizer_merges": tokenizer_meta["merges"],
        #: content identity of the tokenizer that produced/owns the tokens
        #: (sha256 of out_dir/tokenizer.json, or the packed manifest's record).
        "tokenizer_sha256": tokenizer_meta["sha256"],
        "tokenizer_origin": tokenizer_origin,
        "tokenizer_json_arg": tokenizer_json,
        "data_source": data_src,
        "train_docs": split.train_docs if split is not None else None,
        "val_docs": split.val_docs if split is not None else None,
        "split_seed": split.seed if split is not None else None,
        "train_rows": (
            packed_manifest["metadata"]["counts"]["train_rows"]
            if packed_manifest is not None else None
        ),
        "val_rows": (
            packed_manifest["metadata"]["counts"]["val_rows"]
            if packed_manifest is not None else None
        ),
        "resumed_from": os.path.abspath(resume_path) if args.resume else None,
        "epochs": [asdict(r) for r in history.rows],
        "final_train_loss": last.train_loss if last else None,
        "final_val_loss": last.val_loss if last else None,
        "tokens_processed": history.tokens_processed,
        # token-budget accounting (DELIVERABLE 2) — budget, consumed, steps:
        "token_budget": token_budget,
        "tokens_consumed": history.tokens_processed,
        "steps": steps_taken,
        "budget_reached": history.budget_reached,
        "lr_schedule": TokenSchedule(
            lr=args.lr, warmup_tokens=warmup_tokens, decay=lr_decay,
            budget=token_budget,
        ).to_dict(),
        "wall_s": round(time.monotonic() - t0, 3),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
        ),
        "device": str(device),
        "checkpoint": last.checkpoint if last else None,
        "tokenizer_path": tokenizer_path,
        "run_metadata_file": RUN_METADATA_FILENAME,
    }
    # Stamp the finish time into the sidecar (checkpoints carry the start-time
    # dict — same payload, only "finished" is filled in after training).
    run_metadata.setdefault("timestamps", {})["finished"] = (
        datetime.now(timezone.utc).isoformat()
    )
    write_run_metadata(out_dir, run_metadata)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    print(f"\n  final train loss  : {last.train_loss if last else 'n/a'}")
    print(f"  final val loss    : {last.val_loss if last and last.val_loss is not None else 'n/a'}")
    print(f"  tokens            : {history.tokens_processed:,} consumed"
          + (f" / budget {token_budget:,} (REACHED)" if history.budget_reached else ""))
    print(f"  wall time         : {metrics['wall_s']} s")
    print(f"  peak RSS          : {metrics['peak_rss_mb']} MiB")
    print(f"  checkpoint        : {metrics['checkpoint']}")
    print(f"  metrics           : {os.path.join(out_dir, 'metrics.json')}")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        train_run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())