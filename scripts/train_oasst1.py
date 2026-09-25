"""Reproducible OASST1-style JSONL -> Talos-native tokenizer -> tiny-model training.

This is the owner-scoped reliability pass for the JSONL training path. It wires
**existing** Talos components end-to-end — no new data pipeline, no new training
framework, no new architecture:

    JSONL (OASST1-shaped, any path)
      -> deterministic train/val split        split_jsonl()  (fixed seed, disjoint files)
      -> Talos-native byte-level BPE          tokenizer.train.train_tokenizer()
         (vocab 1024 shared by every canonical  corpus via tokenizer.corpus.iter_text_documents;
          preset — configs.vocab.VOCAB_SIZE,     max_docs/max_chars/num_merges/
          TRAIN SPLIT ONLY,                      knobs exposed as CLI flags)
          minfreq knobs)
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
      tokenizer.json                             # trained on train split only
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
import os
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Sequence

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
) -> None:
    """One checkpoint artifact: weights + config + step + losses + tokenizer info.

    Format ``talos-training-checkpoint-v1`` stays **backward compatible**: the
    original keys are unchanged and the resume-enabling keys (``optimizer_state_dict``,
    ``rng_state``, ``epoch``, ``tokenizer_fingerprint``) are additive, so
    checkpoints written before this change still load for eval/generation, and
    new checkpoints load in any old reader.
    """
    cfg = model.config
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
        "tokenizer_path": os.path.abspath(tokenizer_path),
        #: sha256 of the serialized tokenizer.json — content identity, so a
        #: swapped/re-trained sidecar is caught on load (audit P0 fix 4).
        "tokenizer_fingerprint": tokenizer_file_sha256(tokenizer_path),
        "model_state_dict": model.state_dict(),
        # Resume-enabling state (additive, v1 format): optimizer + RNG + counters.
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "rng_state": rng_state,
        "epoch": epoch,
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
) -> TrainingHistory:
    """Train the tiny model on the streamed train split, epoch by epoch.

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
    """
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    vocab = model.config.vocab_size
    history = TrainingHistory()
    start_epoch, global_step = 1, 0
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
        if start_epoch > epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1} "
                f"(>= --epochs {epochs}) — nothing left to train; raise --epochs"
            )
        log.info(
            "resuming from step %d (epoch %d completed) — continuing to epoch %d",
            global_step, start_epoch - 1, epochs,
        )
    t_run = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        steps = 0
        t_epoch = time.monotonic()
        for batch in train_ds:
            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break
            x = batch.to(device).long()
            validate_token_ids(x, vocab, where="train batch")
            logits, _ = model(x[:, :-1])
            loss = loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
            history.tokens_processed += x[:, 1:].numel()
            steps += 1
            global_step += 1
        train_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        )
        val_loss = (
            evaluate(model, val_ds, device=device, max_steps=val_max_steps)
            if val_ds is not None
            else None
        )
        ckpt_path = os.path.join(out_dir, f"step-{global_step}.pt")
        save_checkpoint(
            ckpt_path, model, global_step, train_loss, val_loss, tokenizer_path,
            optimizer=opt, rng_state=capture_rng_state(), epoch=epoch,
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
    p.add_argument("--data", required=True,
                   help="OASST1-shaped JSONL corpus (any path; real or synthetic)")
    p.add_argument("--out-dir", required=True, help="run directory (split, tokenizer, checkpoints, metrics)")
    p.add_argument("--seed", type=int, default=0, help="fixed seed for split + training (default 0)")
    p.add_argument("--preset", default="tiny",
                   help=f"canonical preset to train ({', '.join(sorted(CANONICAL_PRESETS))}; default tiny)")
    p.add_argument("--resume", default=None, metavar="step-<N>.pt",
                   help="resume training from this checkpoint (weights + optimizer "
                        "+ RNG + step/epoch restored; tokenizer fingerprint must "
                        "match the sidecar; --epochs is then the target TOTAL). "
                        "The split + tokenizer are NOT re-trained on resume — the "
                        "checkpoint's sidecar tokenizer is verified and reused.")
    # split
    p.add_argument("--split-ratio", type=float, default=0.9, help="train fraction (default 0.9)")
    p.add_argument("--split-max-docs", type=int, default=None,
                   help="cap how many source docs enter the split (small-subset guard)")
    # tokenizer (Talos-native BPE trainer, PR #16 knobs)
    p.add_argument("--bpe-num-merges", type=int, default=None,
                   help="exact BPE merges (default: vocab-1024 budget = 764, stopped by minfreq)")
    p.add_argument("--bpe-minfreq", type=int, default=2, help="BPE min pair frequency (default 2)")
    p.add_argument("--bpe-max-docs", type=int, default=None, help="BPE training doc cap")
    p.add_argument("--bpe-max-chars", type=int, default=None, help="BPE training char cap")
    # training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seq", type=int, default=64, help="sequence length per row")
    p.add_argument("--batch", type=int, default=4, help="batch size")
    p.add_argument("--lr", type=float, default=3e-3)
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


def train_run(args: argparse.Namespace) -> dict:
    """Run the full chain; returns the metrics dict (also saved to metrics.json).

    With ``--resume <step-*.pt>`` the run **continues** an existing run: the
    split is re-derived deterministically (identical files, no re-training of
    the split), the tokenizer is loaded from the checkpoint's recorded path and
    verified by sha256 fingerprint (P0 fix 4), and the model/optimizer/RNG +
    counters are restored inside :func:`train_epochs`. Tokenizer training and
    preset build happen only on a fresh run.
    """
    t0 = time.monotonic()
    device = _resolve_device(args.device)
    set_seed(args.seed)
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    # ---- resume: load + pre-validate the checkpoint before anything else ----
    resume_ckpt = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        resume_ckpt = load_checkpoint(args.resume)
        if resume_ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"resume checkpoint has unsupported format "
                f"{resume_ckpt.get('format')!r}: expected {CHECKPOINT_FORMAT!r}"
            )
        ckpt_cfg = ModelConfig(**resume_ckpt["model_config"]).derive()
        ckpt_preset = resolve_preset(ckpt_cfg)
        if ckpt_preset != args.preset:
            raise ValueError(
                f"--resume checkpoint is the {ckpt_preset} preset but --preset "
                f"is {args.preset!r} — the preset must match the resumed run"
            )
        exp_params, exp_vocab = CANONICAL_PRESETS[ckpt_preset]
        if int(resume_ckpt["n_params"]) != exp_params:
            raise ValueError(
                f"resume checkpoint records {resume_ckpt['n_params']:,} params "
                f"but the canonical {ckpt_preset} preset is exactly "
                f"{exp_params:,} — corrupted or tampered checkpoint"
            )
        if resume_ckpt.get("optimizer_state_dict") is None:
            raise ValueError(
                f"resume checkpoint {args.resume} has no optimizer_state_dict — "
                f"it predates resume support (old v1 format); use a checkpoint "
                f"written by this version of the script"
            )

    # ---- canonical preset config, printed before anything else ------------
    preset = args.preset
    if preset not in CANONICAL_PRESETS:
        raise ValueError(
            f"unknown --preset {preset!r}: choose from "
            f"{', '.join(sorted(CANONICAL_PRESETS))}"
        )
    exp_params, exp_vocab = CANONICAL_PRESETS[preset]
    cfg = ALL_PRESETS[preset]().derive()
    print("=" * 72)
    print("Talos OASST1-style training run"
          + (f" (RESUMING from {args.resume})" if resume_ckpt else ""))
    print(f"  model preset  : {preset} ({cfg.ffn_type}) — vocab={cfg.vocab_size} "
          f"hidden={cfg.hidden_size} layers={cfg.num_layers} "
          f"heads={cfg.num_attention_heads} kv={cfg.num_kv_heads} "
          f"ffn={cfg.ffn_type} seq={cfg.max_seq_len}")
    print(f"  expected      : EXACTLY {exp_params:,} params, vocab_size {exp_vocab}")
    print(f"  data          : {args.data}")
    print(f"  seed          : {args.seed} | split ratio {args.split_ratio} | "
          f"epochs {args.epochs} | seq {args.seq} | batch {args.batch} | lr {args.lr} | device {device}")
    print("=" * 72)

    # ---- 1) deterministic train/val split (disjoint files, no leakage) ---
    # Re-derived on resume too: deterministic, so the files are byte-identical
    # to the original run's (no re-tokenization happens — the tokenizer and
    # its fingerprint are taken from the checkpoint below).
    split = split_jsonl(
        args.data,
        os.path.join(out_dir, "data"),
        ratio=args.split_ratio,
        seed=args.seed,
        max_docs=args.split_max_docs,
    )
    print(f"  split: {split.total_docs} docs -> train {split.train_docs} / val {split.val_docs} "
          f"({os.path.basename(split.train_path)}, {os.path.basename(split.val_path)})")

    # ---- 2) tokenizer: train on a fresh run, verify+reuse on resume --------
    if resume_ckpt is None:
        tokenizer_path = os.path.join(out_dir, "tokenizer.json")
        tokenizer = train_tokenizer_for_run(
            split.train_path,
            tokenizer_path,
            preset=preset,
            num_merges=args.bpe_num_merges,
            minfreq=args.bpe_minfreq,
            max_docs=args.bpe_max_docs,
            max_chars=args.bpe_max_chars,
        )
    else:
        tokenizer_path = resume_ckpt["tokenizer_path"]
        if not tokenizer_path or not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(
                f"resume checkpoint's tokenizer not found ({tokenizer_path!r}) "
                f"— expected the sidecar tokenizer.json next to the checkpoint"
            )
        tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
        recorded_fp = resume_ckpt.get("tokenizer_fingerprint")
        if not recorded_fp:
            raise ValueError(
                f"resume checkpoint {args.resume} has no tokenizer_fingerprint "
                f"— written before identity fingerprints existed; use a "
                f"checkpoint written by this version of the script"
            )
        actual_fp = tokenizer_file_sha256(tokenizer_path)
        if actual_fp != recorded_fp:
            raise ValueError(
                f"tokenizer identity mismatch on resume: checkpoint records "
                f"sha256 {recorded_fp} but the sidecar tokenizer.json at "
                f"{tokenizer_path} hashes to {actual_fp} — the tokenizer was "
                f"swapped/re-trained since the run; refusing to continue with "
                f"the wrong tokenizer"
            )
        print(f"  tokenizer     : reused from checkpoint (vocab {tokenizer.vocab_size}, "
              f"{tokenizer.merge_count} merges) — sha256 verified")

    # ---- 3) canonical preset model + hard param-count guard (fail fast) ---
    model = build_preset_model(preset).to(device)
    n_params = model.num_parameters()
    print(f"  model: {n_params:,} params (vocab {model.config.vocab_size}) — config guard OK")

    # ---- 4) streamed token batches (existing data pipeline) --------------
    # max_id=cfg.vocab_size: a tokenizer/model vocab mismatch fails at the data
    # source (per-document, with shard/doc context) instead of on the device.
    train_ds = StreamingTokenizedDataset(
        split.train_path, tokenizer, seq_len=args.seq, batch_size=args.batch,
        mode="pack", eos=True, max_id=cfg.vocab_size,
    )
    val_ds = StreamingTokenizedDataset(
        split.val_path, tokenizer, seq_len=args.seq, batch_size=args.batch,
        mode="pack", eos=True, max_id=cfg.vocab_size,
    )

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
    )

    last = history.rows[-1]
    metrics = {
        "format": "talos-oasst1-training-metrics-v1",
        "params": n_params,
        "vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_merges": tokenizer.merge_count,
        #: content identity of the trained tokenizer (sha256 of tokenizer.json)
        #: — a swapped/re-trained sidecar is detectable against this (P0 fix 4).
        "tokenizer_sha256": tokenizer_file_sha256(tokenizer_path),
        "train_docs": split.train_docs,
        "val_docs": split.val_docs,
        "split_seed": split.seed,
        "resumed_from": os.path.abspath(args.resume) if args.resume else None,
        "epochs": [asdict(r) for r in history.rows],
        "final_train_loss": last.train_loss,
        "final_val_loss": last.val_loss,
        "tokens_processed": history.tokens_processed,
        "wall_s": round(time.monotonic() - t0, 3),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
        ),
        "device": str(device),
        "checkpoint": last.checkpoint,
        "tokenizer_path": tokenizer_path,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    print(f"\n  final train loss  : {last.train_loss:.4f}")
    print(f"  final val loss    : {last.val_loss if last.val_loss is None else f'{last.val_loss:.4f}'}")
    print(f"  tokens            : {history.tokens_processed:,} ({history.tokens_processed / max(metrics['wall_s'], 1e-9):,.0f} tok/s)")
    print(f"  wall time         : {metrics['wall_s']} s")
    print(f"  peak RSS          : {metrics['peak_rss_mb']} MiB")
    print(f"  checkpoint        : {last.checkpoint}")
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