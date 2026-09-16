"""Synthetic OASST1-style JSONL generator for tokenizer acceptance runs.

The real OpenAssistant dataset is large and licensed; this generator produces a
deterministic, self-contained corpus with the same *shape* (``{"text": ...}``
JSONL records of human/assistant conversation turns) so the trainer's
streaming / memory-bounded path can be exercised at 1,800 documents without
downloading anything. No copyrighted data: all text is drawn from the small
public-domain-style word pools below.

Usage::

    python -m tools.make_synthetic_oasst1 --docs 1800 --seed 0 \
        --output /tmp/oasst_synth.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Iterable, List

_HUMAN_INTROS = [
    "Can you explain", "What is the difference between", "How do I",
    "Why does", "Please summarize", "Give me an example of", "Help me fix",
    "Which of these is better", "Translate this into simpler words",
    "What are the main ideas in", "Compare and contrast",
    "Write a short tutorial about",
]

_TOPICS = [
    "neural networks", "gradient descent", "attention mechanisms",
    "the tiny language model", "byte pair encoding", "regular expressions",
    "unit testing", "memory management", "distributed training",
    "quantization", "tokenizers", "data pipelines", "inference servers",
    "open source licensing", "benchmarking", "reproducibility",
]

_ASSN_VERBS = [
    "explain", "describe", "walk through", "outline", "illustrate",
    "compare", "clarify", "detail", "introduce", "review",
]

_ASSN_AUX = [
    "The key idea is", "A useful analogy is", "Start from first principles",
    "There are three main steps", "In practice", "For example",
    "Note that", "A common pitfall is", "To summarize", "Importantly",
]

_DETAIL = [
    "the model maps tokens to vectors", "loss decreases monotonically",
    "merges are learned greedily", "the gradient points uphill",
    "weights are updated by a small step", "context grows linearly",
    "the vocabulary is fixed up front", "inference is memory bound",
    "checkpoints enable resuming", "validation guards against overfitting",
    "throughput is measured in tokens per second", "determinism aids debugging",
    "the loader streams documents", "batch size trades speed for memory",
]


def doc_for(seed: int) -> str:
    """One OASST1-shaped human/assistant exchange (deterministic per seed)."""
    rng = random.Random(seed)
    turns: List[str] = []
    for _turn in range(rng.randint(1, 3)):
        context = rng.choice(["in practice", "using PyTorch", "at small scale", ""])
        ask = rng.choice(["Please be concrete.", "Give examples.", "Be brief."])
        question = (
            f"{rng.choice(_HUMAN_INTROS)} {rng.choice(_TOPICS)} {context}, "
            f"and what are the tradeoffs? {ask}"
        )
        parts = [
            rng.choice(_ASSN_VERBS),
            rng.choice(_TOPICS),
            ": ",
            rng.choice(_ASSN_AUX),
            ": ",
        ]
        for _ in range(rng.randint(2, 5)):
            parts.append(rng.choice(_DETAIL))
            parts.append(". ")
        answer = "".join(parts).capitalize()
        turns.append(f"### Human: {question.strip()}\n### Assistant: {answer}")
    return "\n\n".join(turns)


def generate(n_docs: int, seed: int = 0) -> Iterable[str]:
    """Yield ``n_docs`` deterministic OASST1-shaped documents."""
    for i in range(n_docs):
        yield doc_for(seed * 100_000 + i)


def main(argv: list | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", type=int, default=1800, help="documents to emit")
    p.add_argument("--seed", type=int, default=0, help="deterministic seed")
    p.add_argument("--output", required=True, help="output JSONL path")
    args = p.parse_args(argv)

    total_chars = 0
    with open(args.output, "w", encoding="utf-8") as fh:
        for i, doc in enumerate(generate(args.docs, args.seed)):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            total_chars += len(doc)
            if (i + 1) % 500 == 0:
                print(f"wrote {i + 1} docs", flush=True)
    print(f"done: {args.docs} docs, {total_chars / 1e6:.1f} MB chars -> {args.output}")


if __name__ == "__main__":
    main()