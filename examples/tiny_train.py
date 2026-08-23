"""Train the tiny Talos model for a few steps (forward + backward + optimizer).

Usage::

    python examples/tiny_train.py --steps 50 --seq 128 --moe
"""
from __future__ import annotations

import argparse

import torch

from model import TalosGPT, ModelConfig
from model.utils import get_logger, set_seed

log = get_logger("examples.tiny_train")


def build_config(moe: bool) -> ModelConfig:
    if not moe:
        return ModelConfig(
            vocab_size=2048, hidden_size=128, num_layers=3,
            num_attention_heads=8, num_kv_heads=4, head_dim=16,
            ffn_type="dense", intermediate_size=512, max_seq_len=512,
        ).derive()
    return ModelConfig(
        vocab_size=2048, hidden_size=128, num_layers=3,
        num_attention_heads=8, num_kv_heads=4, head_dim=16,
        ffn_type="moe", num_experts=16, num_experts_per_tok=3,
        num_shared_experts=1, moe_intermediate_size=256,
        load_balance_coef=0.01, max_seq_len=512,
    ).derive()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--moe", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = build_config(args.moe)
    model = TalosGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    log.info("built %s with %d params (moe=%s)", cfg.ffn_type,
             model.num_parameters(), args.moe)
    for step in range(args.steps):
        x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq))
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            log.info("step %4d loss=%.4f", step, float(loss))
    print(f"OK: final loss={float(loss):.4f}")


if __name__ == "__main__":
    main()
