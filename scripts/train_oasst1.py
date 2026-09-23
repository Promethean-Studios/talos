"""Reproducible OASST1-style JSONL -> Talos-native tokenizer -> tiny-model training.

This is the owner-scoped reliability pass for the JSONL training path. It wires
**existing** Talos components end-to-end — no new data pipeline, no new training
framework, no new architecture:

    JSONL (OASST1-shaped, any path)
      -> deterministic train/val split        split_jsonl()  (fixed seed, disjoint files)
      -> Talos-native byte-level BPE          tokenizer.train.train_tokenizer()
         (vocab_size 1024, TRAIN SPLIT ONLY,   corpus via tokenizer.corpus.iter_text_documents;
          max_docs/max_chars/num_merges/       knobs exposed as CLI flags)
          minfreq knobs)
      -> canonical tiny preset                configs.presets.tiny_config()
         (254,272 params, vocab 1024)          fails fast on any config drift
      -> streaming training                   data.tokenized.StreamingTokenizedDataset
         (same objective/components as         + AdamW + CrossEntropyLoss, pack mode,
          examples/tiny_train.py)              x[:, :-1] -> x[:, 1:]
      -> per-epoch checkpoint artifact        weights + model config + step + train/val
         + tokenizer.json next to it           loss + tokenizer path in one .pt file
      -> train + validation loss saved        out_dir/metrics.json (losses, wall time,
                                              peak RSS, params, tokenizer vocab)

Why the loop lives here instead of examples/tiny_train.py: the existing
``train_stream`` helper is step-based (no epoch boundaries, no evaluation, no
checkpointing). Everything it uses is reused verbatim; this module only adds the
epoch/eval/checkpoint orchestration the task requires. The artifact layout is
flat so a checkpoint and its tokenizer always sit side by side::

    out_dir/
      data/train.jsonl        data/val.jsonl      # disjoint, deterministic split
      tokenizer.json                             # trained on train split only
      step-<N>.pt                                # checkpoint (weights+config+losses)
      metrics.json                               # run report

Usage::

    python -m tools.make_synthetic_oasst1 --docs 300 --seed 0 --output /tmp/oasst1.jsonl
    python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir runs/oasst1-tiny \\
        --epochs 3 --seq 64 --batch 4 --seed 0
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

import torch

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the script
# also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.canonical import CANONICAL_PRESETS  # noqa: E402
from configs.presets import ALL_PRESETS, tiny_tokenizer_config  # noqa: E402
from data.tokenized import StreamingTokenizedDataset  # noqa: E402
from model import TalosGPT  # noqa: E402
from model.utils import get_logger, set_seed  # noqa: E402
from tokenizer.corpus import iter_text_documents  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402
from tokenizer.train import train_tokenizer  # noqa: E402

log = get_logger("scripts.train_oasst1")

#: Owner-fixed canonical tiny prototype: EXACTLY 254,272 params at vocab 1024.
EXPECTED_TINY_PARAMS = 254_272
EXPECTED_TINY_VOCAB = 1024

CHECKPOINT_FORMAT = "talos-training-checkpoint-v1"


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
    num_merges: Optional[int] = None,
    minfreq: int = 2,
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
    text_field: str = "text",
) -> ByteLevelBPETokenizer:
    """Train a vocab-1024 byte-level BPE on the train split and persist it.

    Uses the canonical :func:`configs.presets.tiny_tokenizer_config` (1024 =
    256 base bytes + 4 specials + up to 764 merges), consumed **one document at
    a time** via the PR-#16 unique-word frequency table, so memory stays
    bounded. Exposes the new ``num_merges`` / ``minfreq`` / ``max_docs`` /
    ``max_chars`` knobs. The artifact is written next to the checkpoints so
    eval/generation can load the exact same tokenizer later.
    """
    config = tiny_tokenizer_config()
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
    if tokenizer.vocab_size > EXPECTED_TINY_VOCAB:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size} exceeds the tiny "
            f"model's {EXPECTED_TINY_VOCAB} — config drift in "
            f"configs/presets.tiny_tokenizer_config"
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
    ``configs/presets`` is caught here *before* training starts. ``tiny`` is
    enforced at exactly 254,272 params / vocab 1024; ``tiny_1m`` at exactly
    1,000,320 params / vocab 1024; ``tiny_10m`` at exactly 9,952,320 params /
    vocab 1024.
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
    validation split). The model is left in ``train()`` mode afterwards.
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
) -> None:
    """One checkpoint artifact: weights + full config + step + losses + tokenizer path."""
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
        "model_state_dict": model.state_dict(),
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
) -> TrainingHistory:
    """Train the tiny model on the streamed train split, epoch by epoch.

    Per epoch: one pass over the train stream (deterministic order — same
    ``seed`` reproduces the same run), mean train loss, validation loss over a
    *separate* stream (no token-level leakage), and a checkpoint saved with
    weights + config + step + losses. AdamW + CrossEntropyLoss on
    ``x[:, :-1] -> x[:, 1:]`` — the identical objective examples/tiny_train.py
    uses.
    """
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    vocab = model.config.vocab_size
    history = TrainingHistory()
    global_step = 0
    t_run = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        steps = 0
        t_epoch = time.monotonic()
        for batch in train_ds:
            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break
            x = batch.to(device).long()
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
            ckpt_path, model, global_step, train_loss, val_loss, tokenizer_path
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
    """Run the full chain; returns the metrics dict (also saved to metrics.json)."""
    t0 = time.monotonic()
    device = _resolve_device(args.device)
    set_seed(args.seed)
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

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
    print("Talos OASST1-style training run")
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
    split = split_jsonl(
        args.data,
        os.path.join(out_dir, "data"),
        ratio=args.split_ratio,
        seed=args.seed,
        max_docs=args.split_max_docs,
    )
    print(f"  split: {split.total_docs} docs -> train {split.train_docs} / val {split.val_docs} "
          f"({os.path.basename(split.train_path)}, {os.path.basename(split.val_path)})")

    # ---- 2) Talos-native BPE tokenizer on the TRAIN split only -----------
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer = train_tokenizer_for_run(
        split.train_path,
        tokenizer_path,
        num_merges=args.bpe_num_merges,
        minfreq=args.bpe_minfreq,
        max_docs=args.bpe_max_docs,
        max_chars=args.bpe_max_chars,
    )

    # ---- 3) canonical preset model + hard param-count guard (fail fast) ---
    model = build_preset_model(preset).to(device)
    n_params = model.num_parameters()
    print(f"  model: {n_params:,} params (vocab {model.config.vocab_size}) — config guard OK")

    # ---- 4) streamed token batches (existing data pipeline) --------------
    train_ds = StreamingTokenizedDataset(
        split.train_path, tokenizer, seq_len=args.seq, batch_size=args.batch,
        mode="pack", eos=True,
    )
    val_ds = StreamingTokenizedDataset(
        split.val_path, tokenizer, seq_len=args.seq, batch_size=args.batch,
        mode="pack", eos=True,
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
    )

    last = history.rows[-1]
    metrics = {
        "format": "talos-oasst1-training-metrics-v1",
        "params": n_params,
        "vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_merges": tokenizer.merge_count,
        "train_docs": split.train_docs,
        "val_docs": split.val_docs,
        "split_seed": split.seed,
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