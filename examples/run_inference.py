"""End-to-end inference on the tiny prototype: prefill + KV-cache decode.

Demonstrates and *verifies* the KV-cache inference path on the canonical tiny
preset (the 254K-param prototype):

1. Optionally train the tiny model briefly on the deterministic recurrent
   synthetic corpus (``training.synthetic``) so its output is coherent — with
   ``--steps 0`` an untrained (random-init) model is used and the run only
   exercises the mechanics.
2. Prefill a prompt in one pass, then greedy-decode new tokens one at a time
   through the incremental KV cache.
3. With ``--verify``, run the core inference-correctness check: feeding the
   full prompt through prefill in one shot vs. decoding token-by-token through
   the KV cache must produce the same logits at every position (report the max
   absolute difference). When trained, the decoded continuation is also
   checked against the corpus recurrence ``x[t] = (x[t-2] + x[t-1]) mod vocab``.

Usage::

    python examples/run_inference.py --prompt_len 64 --decode 8
    python examples/run_inference.py --steps 300 --verify --decode 24
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# Make the repo-root packages (`model`, `configs`, `inference`, `training`)
# importable when this file is run directly as a script (same pattern as
# examples/tiny_train.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.presets import tiny_config
from inference.generate import generate, prefill_decode_max_abs_diff
from model import TalosGPT
from model.utils import get_logger, set_seed
from training.synthetic import build_recurrent_corpus

log = get_logger("examples.inference")


def train_briefly(model: TalosGPT, steps: int, seed: int) -> list[float]:
    """Overfit the fixed recurrent corpus so decoding shows learned structure."""
    cfg = model.config
    corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=64, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, corpus.shape[0], (4,))
        x = corpus[idx]
        # Causal-LM objective: position t predicts token t+1.
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == steps - 1:
            log.info("train step %4d loss=%.4f", step, losses[-1])
    model.eval()
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--steps", type=int, default=0,
        help="train this many steps on the recurrent synthetic corpus before "
        "decoding (0 = random-init model, mechanics only)",
    )
    ap.add_argument(
        "--verify", action="store_true",
        help="also verify prefill vs KV-cache decode produce identical logits "
        "at every position, and (when trained) that the decoded continuation "
        "follows the corpus recurrence",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = tiny_config().derive()
    model = TalosGPT(cfg).eval()
    log.info("tiny prototype: %d params, backend=%s", model.num_parameters(), model.backend)

    if args.steps > 0:
        losses = train_briefly(model, args.steps, args.seed)
        print(f"trained: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    set_seed(args.seed + 1)
    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len))
    with torch.no_grad():
        generated = generate(model, prompt, args.decode, greedy=True)
    print(f"OK: generated {args.decode} tokens from a {args.prompt_len}-token prompt: {generated}")

    if args.verify:
        report = prefill_decode_max_abs_diff(model, prompt)
        print(
            f"VERIFY prefill vs KV-decode logits over {report.seq_len} positions: "
            f"max_abs_diff={report.max_abs_diff:.3e} (fp32 tolerance 1e-4), "
            f"greedy choices match={report.argmax_match}"
        )
        assert report.max_abs_diff < 1e-4, f"prefill/decode mismatch: {report}"
        assert report.argmax_match, "greedy choices differ between prefill and decode"

        if args.steps > 0:
            # The corpus follows x[t] = (x[t-2] + x[t-1]) mod vocab; a model
            # trained on it must continue the pattern from any prompt prefix.
            ref = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=64, seed=args.seed)
            row = ref[0, : args.prompt_len].unsqueeze(0)
            with torch.no_grad():
                cont = generate(model, row, args.decode, greedy=True)
            expected = ref[0, args.prompt_len : args.prompt_len + args.decode].tolist()
            matches = sum(a == b for a, b in zip(cont, expected))
            # Does every generated token satisfy the recurrence given the two
            # tokens before it (crossing the prompt boundary)? This is the
            # corpus's learnable structure, so it is the coherence criterion.
            full = row[0].tolist() + cont
            rec_ok = all(
                full[i] == (full[i - 1] + full[i - 2]) % cfg.vocab_size
                for i in range(args.prompt_len, len(full))
            )
            print(
                f"VERIFY coherent decode: {matches}/{args.decode} tokens match the corpus "
                f"continuation; continuation follows recurrence={rec_ok}"
            )
            assert rec_ok, "decoded continuation violates the recurrent pattern"


if __name__ == "__main__":
    main()
