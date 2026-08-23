"""CLI: build a tiny Talos model and run a forward + backward pass.

Usage::

    python -m scripts.cli [--steps 3] [--seq 64] [--seed 0] [--moe]

By default this builds the dense ``tiny`` config. Pass ``--moe`` to build a
tiny MoE variant exercising the same code path.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import torch

from model import TalosGPT, ModelConfig
from model.utils import get_logger, set_seed

log = get_logger("cli")


def make_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a tiny Talos model forward/backward.")
    p.add_argument("--steps", type=int, default=3, help="optimiser steps to run")
    p.add_argument("--seq", type=int, default=64, help="sequence length per step")
    p.add_argument("--batch", type=int, default=2, help="batch size")
    p.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--moe", action="store_true", help="use a tiny MoE config")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "plain", "flash"],
        help="attention backend (flash falls back to plain if unavailable)",
    )
    return p.parse_args()


def tiny_config(moe: bool) -> ModelConfig:
    if not moe:
        return ModelConfig(
            vocab_size=1024, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_kv_heads=2, head_dim=16,
            ffn_type="dense", intermediate_size=256, max_seq_len=512,
        ).derive()
    return ModelConfig(
        vocab_size=1024, hidden_size=64, num_layers=2,
        num_attention_heads=4, num_kv_heads=2, head_dim=16,
        ffn_type="moe", num_experts=8, num_experts_per_tok=2,
        num_shared_experts=1, moe_intermediate_size=128,
        load_balance_coef=0.01, max_seq_len=512,
    ).derive()


def main(argv: Optional[list] = None) -> None:
    args = make_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    cfg = tiny_config(args.moe)
    model = TalosGPT(cfg)
    model.to(device)
    log.info("built %s with %d params", type(model).__name__, model.num_parameters())
    if args.moe:
        from configs.compute import estimate

        est = estimate(cfg)
        log.info(
            "total=%d active/token=%d (%.2f%%)",
            est.total_params, est.active_params,
            100 * est.active_params / est.total_params,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    vocab = cfg.vocab_size
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(args.steps):
        x = torch.randint(0, vocab, (args.batch, args.seq), device=device)
        t0 = time.perf_counter()
        logits, _ = model(x)
        targets = x  # next-token prediction baseline
        loss = loss_fn(logits.reshape(-1, vocab), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        log.info("step %d loss=%.4f time=%.3fs", step, float(loss), dt)

    print(f"\nOK: tiny model ran {args.steps} forward/backward steps, final "
          f"loss={float(loss):.4f} (backend={type(model.backend).__name__})")


if __name__ == "__main__":
    sys.exit(main())
