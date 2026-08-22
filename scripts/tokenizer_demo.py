"""Train a small demo byte-level BPE tokenizer and run a round-trip demo.

Usage::

    python -m scripts.tokenizer_demo [--vocab-size 512] [--output tokenizer_demo.json]
        [--save]

Builds a synthetic multilingual + code + math corpus (no copyrighted data),
trains a byte-level BPE tokenizer on it to the requested vocab size, prints a
vocab report and a Unicode round-trip demo, and — with ``--save`` — writes the
resulting ``tokenizer.json`` model file for later reuse / resume.

See ``docs/tokenizer.md`` for design notes and ``tokenizer/train.py`` for the
full production CLI (point it at your own corpus with ``--corpus``).
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import Optional

from tokenizer.train import train_tokenizer
from tokenizer.vocab import TokenizerConfig

_WORDS = (
    "the quick brown fox model attention transformer neural network tokenizer "
    "byte merge training inference scaling efficient reasoning capability"
).split()
_MATH = ["∑", "∫", "≈", "≤", "≥", "±", "√", "∞", "dx", "x", "2", "f(", ")"]
_CODE = [
    "def foo(x):",
    "    return x + 1",
    "class Tokenizer:",
    "        self._rank = 0",
    "for i in range(10):",
    "    print(i)",
]


def make_corpus(seed: int = 0) -> list:
    """Deterministic synthetic corpus (English + CJK + Arabic + code + math)."""
    rng = random.Random(seed)
    docs = []
    for _ in range(60):
        docs.append(" ".join(rng.choice(_WORDS) for _ in range(15)))
    for _ in range(30):
        docs.append(" ".join(rng.choice(_MATH) for _ in range(8)))
    for _ in range(20):
        docs.append(rng.choice(_CODE) * 5)
    docs.append("你好世界 語言模型 深度学习 العربية हिन्दी Русский")
    docs.append("∫ x² dx ≈ 42 ∑ᵢ aᵢ ≤ 10 🚀")
    return docs


DEMO_TEXTS = [
    "Hello, world! 你好世界 العربية हिन्दी",
    "def foo(x):\n    return x + 1",
    "math ∑ f(x) = ∫ x² dx ≈ 42 🚀",
]


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Train a small demo tokenizer + round-trip demo.")
    p.add_argument("--vocab-size", type=int, default=512, help="total vocab size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true", help="write tokenizer.json to --output")
    p.add_argument("--output", default="tokenizer_demo.json", help="output JSON model file")
    args = p.parse_args(argv)

    corpus = make_corpus(args.seed)
    result = train_tokenizer(corpus, TokenizerConfig(vocab_size=args.vocab_size))
    tok = result.tokenizer

    print(f"\nVocab report: {result.summary()}\n")
    ok = True
    for text in DEMO_TEXTS:
        ids = tok.encode(text)
        back = tok.decode(ids)
        good = back == text
        ok = ok and good
        print(f"[{'OK' if good else 'FAIL'}] {len(ids):>3} tokens | {text[:44]!r}")
    if args.save:
        tok.save(args.output)
        print(f"\nSaved model to {args.output} (load with ByteLevelBPETokenizer.from_file).")
    print(f"\nRound-trip demo: {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    sys.exit(main())
