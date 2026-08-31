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
    python examples/tiny_train.py --device cuda --steps 300 --seq 64 --batch 8
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import torch

# Make the repo-root packages (`model`, `configs`, `training`, `data`)
# importable when this file is run directly as a script — e.g. in Colab:
#   cd talos && python examples/tiny_train.py --steps 300 --seq 64 --batch 8
# Python only puts the script's own directory (`examples/`) on ``sys.path``
# for direct execution, so we add the repo root explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import TalosGPT, ModelConfig
from model.utils import get_logger, set_seed
from configs.presets import tiny_config
from data.tokenized import StreamingTokenizedDataset
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.vocab import TokenizerConfig
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


def choose_device(device: str | None) -> torch.device:
    """Resolve the compute device: explicit ``--device`` wins, otherwise auto-detect.

    Prefers a CUDA GPU when one is available (e.g. a Google Colab GPU runtime)
    and transparently falls back to CPU otherwise, so the exact same command
    works on a laptop, a Colab GPU, or a CPU-only box.
    """
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # Sample batch indices on the same device as the corpus so inputs stay
        # on the accelerator when one is present — no host/device transfers.
        idx = torch.randint(0, n_seq, (batch,), device=corpus.device)
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


def train_stream(
    model: TalosGPT,
    dataset: StreamingTokenizedDataset,
    steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Train on *streamed* token batches (real data; never materializes the corpus).

    ``dataset`` yields ``(batch, seq_len)`` int32 token blocks; memory is
    O(batch) regardless of corpus size, so this path trains directly on
    sharded JSONL that is far larger than RAM.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    vocab = model.config.vocab_size
    losses: list[float] = []
    batches = iter(dataset)
    for step in range(steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(dataset)  # next epoch over the stream
            try:
                batch = next(batches)
            except StopIteration:
                log.warning("data stream exhausted after %d steps", step)
                break
        x = batch.to(device).long()
        # Causal-LM objective: position t predicts token t+1.
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
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
    ap.add_argument(
        "--data", type=str, default=None,
        help="train on real data instead of the synthetic corpus: path to a "
        "pipeline manifest.json, a shard directory, or a .jsonl file "
        "(streamed document-by-document via data.tokenized — O(batch) RAM)",
    )
    ap.add_argument(
        "--tokenizer", type=str, default=None,
        help="tokenizer JSON saved by ByteLevelBPETokenizer.save; default is a "
        "byte-fallback tokenizer with the model's vocab size",
    )
    ap.add_argument(
        "--data-mode", choices=("pack", "padded"), default="pack",
        help="packed contiguous blocks (default) or one-padded-doc-per-row",
    )
    ap.add_argument("--data-limit-docs", type=int, default=None, help="cap docs per epoch")
    ap.add_argument(
        "--device", type=str, default=None,
        help="compute device (e.g. 'cuda', 'cpu'). Defaults to auto-detect: "
        "CUDA when available, else CPU. In Colab choose a GPU runtime and this "
        "uses the GPU automatically.",
    )
    args = ap.parse_args()

    device = choose_device(args.device)
    set_seed(args.seed)
    cfg = build_config(args.moe)
    model = TalosGPT(cfg).to(device)

    if args.data:
        tokenizer = (
            ByteLevelBPETokenizer.from_file(args.tokenizer)
            if args.tokenizer
            else ByteLevelBPETokenizer(TokenizerConfig(vocab_size=cfg.vocab_size))
        )
        dataset = StreamingTokenizedDataset(
            args.data,
            tokenizer,
            seq_len=args.seq,
            batch_size=args.batch,
            mode=args.data_mode,
            eos=True,
            limit_docs=args.data_limit_docs,
        )
        log.info(
            "built %s tiny model: %d params (moe=%s) on device=%s; streaming data from %s",
            cfg.ffn_type, model.num_parameters(), args.moe, device, args.data,
        )
        losses = train_stream(model, dataset, args.steps, args.lr, device)
        source = f"streamed {args.data}"
    else:
        log.info(
            "built %s tiny model: %d params (moe=%s) on device=%s",
            cfg.ffn_type, model.num_parameters(), args.moe, device,
        )
        corpus = build_recurrent_corpus(
            cfg.vocab_size, args.n_seq, args.seq, seed=args.seed, device=device,
        )
        losses = list(train(model, corpus, args.steps, args.batch, args.lr))
        source = "synthetic recurrent corpus"

    init, final = losses[0], losses[-1]
    print(
        f"OK: {cfg.ffn_type} tiny ({model.num_parameters()} params, device={device}, "
        f"data={source}) loss {init:.4f} -> {final:.4f} over {len(losses)} steps "
        f"(peak drop {max(init - l for l in losses):.4f})"
    )


if __name__ == "__main__":
    main()
