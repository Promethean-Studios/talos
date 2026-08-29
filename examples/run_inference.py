"""Buffered (prefill + decode) inference over random tokens in the tiny model.

Demonstrates the KV cache end-to-end: prefill a long prompt in one pass, then
generate tokens one at a time using the incremental cache.

Usage::

    python examples/run_inference.py --prompt_len 64 --decode 8
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# Make the repo-root packages (`model`, `configs`) importable when this file is
# run directly as a script (same pattern as examples/tiny_train.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import TalosGPT
from model.utils import get_logger, set_seed
from configs.presets import tiny_config

log = get_logger("examples.inference")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = tiny_config().derive()
    model = TalosGPT(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len))
    with torch.no_grad():
        # Prefill: one forward over the whole prompt, caching K/V.
        cache = model.new_cache(1, prompt.device, torch.float32)
        _, cache = model(prompt, use_cache=True, cache=cache)
        log.info("prefilled %d tokens (cache len=%d)", args.prompt_len, cache.length)

        # Decode: one token at a time, extending the cache.
        tok = prompt[:, -1:]
        generated = []
        for i in range(args.decode):
            pos = torch.tensor([[args.prompt_len - 1 + i]])
            logits, cache = model(tok, position_ids=pos, use_cache=True, cache=cache)
            tok = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated.append(int(tok[0, 0]))
            log.info("decode step %d -> token %d", i, generated[-1])
    print(f"OK: generated {args.decode} tokens: {generated}")


if __name__ == "__main__":
    main()
