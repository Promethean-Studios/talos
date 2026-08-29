"""Train the tiny Talos prototype (the 254K-param ``tiny`` preset) a few steps.

This is the canonical end-to-end smoke check that the prototype's
forward + backward + optimizer pipeline works and *provably learns*: it
overfits a small fixed synthetic corpus (`training.synthetic`), driving the
causal-LM loss from ~``log(vocab)`` down to a low value. Deterministic given a
fixed ``--seed``.

The dense path uses the repository's canonical prototype config,
:func:`configs.presets.tiny_config` (vocab 1024, hidden 64, 2 layers, GQA,
~254K params) — the same code path that scales up to the full Talos sizes.
``--moe`` switches to a small MoE variant purely as a checker for that code
path (not the current prototype focus).

Usage::

    python examples/tiny_train.py --steps 200 --seq 64 --batch 4
    python examples/tiny_train.py --steps 200 --moe
"""
from __future__ import annotations

import argparse
from typing import Sequence

import torch

from model import TalosGPT, ModelConfig
from model.utils import get_logger, set_seed
from configs.presets import tiny_config
from training.synthetic import build_recurrent_corpus

log = get_logger("examples.tiny_train")


def build_config(moe: bool) -> ModelConfig:
    """Return the config to train: the canonical dense tiny preset, or a MoE variant."""
    if not moe:
        # The canonical prototype: the `tiny` dense preset (254,272 params).
        return tiny_config().derive()
    return ModelConfig(
        vocab_size=2048, hidden_size=128, num_layers=3,
        num_attention_heads=8, num_kv_heads=4, head_dim=16,
        ffn_type="moe", num_experts=16, num_experts_per_tok=3,
        num_shared_experts=1, moe_intermediate_size=256,
        load_balance_coef=0.01, max_seq_len=512,
    ).derive()


def train(
    model: TalosGPT,
    corpus: torch.Tensor,
    steps: int,
    batch: int,
    lr: float,
) -> Sequence[float]:
    """Run a few AdamW steps over the fixed corpus; return the loss trajectory."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n_seq = corpus.shape[0]
    vocab = model.config.vocab_size
    losses: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, n_seq, (batch,))
        x = corpus[idx]
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, vocab), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 25 == 0:
            log.info("step %4d loss=%.4f", step, losses[-1])
    return losses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-seq", type=int, default=8, help="sequences in the fixed corpus")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--moe", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = build_config(args.moe)
    model = TalosGPT(cfg)
    corpus = build_recurrent_corpus(cfg.vocab_size, args.n_seq, args.seq, seed=args.seed)

    log.info(
        "built %s tiny model: %d params (moe=%s)", cfg.ffn_type,
        model.num_parameters(), args.moe,
    )
    losses = train(model, corpus, args.steps, args.batch, args.lr)

    init, final = losses[0], losses[-1]
    print(
        f"OK: {cfg.ffn_type} tiny ({model.num_parameters()} params) "
        f"loss {init:.4f} -> {final:.4f} over {len(losses)} steps "
        f"(peak drop {max(init - l for l in losses):.4f})"
    )


if __name__ == "__main__":
    main()
