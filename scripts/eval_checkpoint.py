"""CLI: reproducible evaluation of a Talos training checkpoint.

Loads a ``talos-training-checkpoint-v1`` artifact (a ``step-<N>.pt`` file, or
a directory containing one), rebuilds the model from the checkpoint's own
``model_config``, reloads the sidecar ``tokenizer.json`` and computes the full
metric report against a JSONL validation split (default: the checkpoint's own
``data/val.jsonl`` when it sits next to the tokenizer):

    validation loss            (natural-log mean cross-entropy, same objective
                                and batch layout as training)
    perplexity                 exp(loss), natural-log base
    next-token accuracy        argmax(logits) == next token, same batches
    parameter count            must equal the checkpoint's recorded n_params
    tokens processed           over the evaluated batches
    throughput                 tokens/sec (eval wall time)
    peak RSS                   process metric (ru_maxrss)

Usage::

    python -m scripts.train_oasst1 --data runs/data.jsonl --out-dir runs/oasst1-tiny \\
        --epochs 3 --seq 64 --batch 4 --seed 0
    python -m scripts.eval_checkpoint --checkpoint runs/oasst1-tiny \\
        --data runs/oasst1-tiny/data/val.jsonl --out-metrics runs/oasst1-tiny/eval-metrics.json

Results are printed and saved as a metrics JSON file (same conventions as the
training run's ``metrics.json``) so runs are comparable. Same checkpoint +
split + seed -> identical numbers (fixed seed, deterministic batch order;
asserted by the test suite).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the
# script also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.harness import EvalResult, run_eval  # noqa: E402


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.eval_checkpoint",
        description=__doc__.splitlines()[0],
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="step-<N>.pt file, or a directory containing one (newest is used)",
    )
    p.add_argument(
        "--data", default=None,
        help="validation JSONL split (default: the checkpoint's own "
             "<out_dir>/data/val.jsonl when it exists)",
    )
    p.add_argument(
        "--train-data", default=None,
        help="train JSONL split; when given, train loss is also reported",
    )
    # eval batch shape — defaults mirror the training loop (train_oasst1.py)
    p.add_argument("--seq", type=int, default=64, help="sequence length per row")
    p.add_argument("--batch", type=int, default=4, help="batch size")
    p.add_argument(
        "--keep-partial", action="store_true",
        help="also score the final partial batch (default: drop it, exactly "
             "like the training loop, so loss matches the recorded val_loss)",
    )
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="cap the number of eval batches (CI / smoke runs)",
    )
    p.add_argument("--seed", type=int, default=0, help="fixed seed (default 0)")
    p.add_argument("--device", default=None, help="compute device (default: auto)")
    p.add_argument(
        "--out-metrics", default=None,
        help="metrics JSON path (default: <checkpoint dir>/eval-metrics.json)",
    )
    return p


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def print_report(r: EvalResult) -> None:
    print("=" * 72)
    print("Talos checkpoint evaluation report")
    print(f"  checkpoint    : {r.checkpoint_path} (step {r.checkpoint_step}, "
          f"{r.checkpoint_format})")
    print(f"  val split     : {r.data}")
    print(f"  train split   : {r.train_data or 'n/a'}")
    print(f"  params        : {r.params:,} (vocab {r.vocab_size}, tokenizer "
          f"{r.tokenizer_vocab_size}, {r.tokenizer_merges} merges) — "
          f"n_params guard OK")
    print(f"  val loss      : {r.val_loss:.4f} "
          f"(checkpoint recorded: {_fmt(r.checkpoint_val_loss)})")
    print(f"  val perplexity: {r.val_perplexity:.4f} (exp of natural-log loss)")
    print(f"  val accuracy  : {r.val_accuracy:.6f} (next-token argmax)")
    print(f"  train loss    : {_fmt(r.train_loss)}")
    print(f"  tokens        : {r.tokens_processed:,}")
    print(f"  eval wall     : {r.eval_wall_s} s")
    print(f"  throughput    : {r.throughput_tok_per_s:,.0f} tok/s")
    print(f"  peak RSS      : {r.peak_rss_mb} MiB (process metric, ru_maxrss)")
    print(f"  seed          : {r.seed} | seq {r.seq_len} | batch {r.batch_size} "
          f"| drop_last {r.drop_last} | device {r.device}")
    print(f"  metrics file  : {r.metrics_path}")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        result = run_eval(
            args.checkpoint,
            data=args.data,
            train_data=args.train_data,
            seq_len=args.seq,
            batch_size=args.batch,
            drop_last=not args.keep_partial,
            max_steps=args.max_steps,
            seed=args.seed,
            device=args.device,
            out_metrics=args.out_metrics,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())