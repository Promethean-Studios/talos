"""Small, deterministic synthetic corpora for smoke-testing the training loop.

The Talos prototype (the 254K-param ``tiny`` preset) must be validated end to
end on CPU before any scaling. Training on *fresh random* input can't prove the
optimizer learns anything — predicting i.i.d. uniform tokens is an unsolved
problem with a loss floor of ``log(vocab)``. The helper here instead builds a
small **fixed** corpus whose tokens follow a low-order recurrence, so a causal
LM has real, learnable structure: after enough steps the model overfits the
corpus and the loss collapses from ``log(vocab)`` toward a low value. This is a
legitimate supervised memorization task that exercises forward, backward, the
optimizer and (via the recurrence) multi-token context through attention.

The recurrence ``x[t] = (x[t-2] + x[t-1]) mod vocab`` (a Fibonacci sequence mod
``vocab``) is barrier-free to implement and deterministic given a seed — the
next token depends on the two previous tokens, so the model must use context,
not just a per-position constant.
"""
from __future__ import annotations

import torch

__all__ = ["build_recurrent_corpus"]


def build_recurrent_corpus(
    vocab_size: int,
    n_sequences: int = 8,
    seq_len: int = 64,
    seed: int = 0,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a fixed, deterministic corpus with learnable structure.

    Each of ``n_sequences`` rows is a length-``seq_len`` token sequence where
    ``x[t] = (x[t-2] + x[t-1]) mod vocab``. The first two tokens are drawn
    uniformly at random (seeded). The whole corpus is fixed for a given
    ``seed``, so training on it is reproducible and an overfitted model drives
    the causal-LM loss well below ``log(vocab)``.

    Args:
        vocab_size: Size of the token vocabulary (must be > 0).
        n_sequences: Number of independent sequences in the corpus.
        seq_len: Tokens per sequence.
        seed: Random seed for the two initial tokens of each sequence.
        device: Where to place the returned tensor (default CPU).

    Returns:
        ``torch.LongTensor`` of shape ``(n_sequences, seq_len)``.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty((n_sequences, seq_len), dtype=torch.long, device=device)
    prev2 = torch.randint(0, vocab_size, (n_sequences,), generator=generator)
    prev1 = torch.randint(0, vocab_size, (n_sequences,), generator=generator)
    for t in range(seq_len):
        out[:, t] = prev1
        prev2, prev1 = prev1, (prev2 + prev1) % vocab_size
    out = out.to(device)
    return out
