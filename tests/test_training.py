"""Training smoke/integration tests: a tiny model must demonstrably learn.

These assert the core owner-priority property of the prototype — that the
forward/backward/optimizer pipeline actually *works*, i.e. loss decreases on a
genuine tiny model. We train the canonical ``tiny`` preset (254,272 params) on
a small fixed synthetic corpus (`training.synthetic`) and require the
causal-LM loss to fall well below its initial value. Deterministic via a fixed
seed; small enough (a few hundred steps, ~1s on CPU) to run in the normal
suite.
"""
from __future__ import annotations

import torch

from configs.presets import tiny_config
from model import TalosGPT
from model.utils import set_seed
from training.synthetic import build_recurrent_corpus

# Loss must drop by at least this many nats from its step-0 value. The tiny
# model overfits the fixed corpus from ~log(1024) (~6.9) to well under 1, so a
# margin of 1.5 is comfortably met while remaining robust to small param drift.
DROP_MARGIN = 1.5


def test_tiny_training_decreases_loss() -> None:
    set_seed(0)
    cfg = tiny_config().derive()
    model = TalosGPT(cfg).train()
    corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=32, seed=0)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    n_seq = corpus.shape[0]

    initial: float | None = None
    final: float | None = None
    batch, steps = 4, 120
    for step in range(steps):
        idx = torch.randint(0, n_seq, (batch,))
        x = corpus[idx]
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss = float(loss.detach())
        if step == 0:
            initial = loss
        final = loss

    assert initial is not None and final is not None
    assert torch.isfinite(torch.tensor(initial)) and torch.isfinite(torch.tensor(final))
    # Loss must start high (~log vocab) and fall substantially on the fixed corpus.
    assert initial > 5.0, f"expected a high initial loss near log(vocab), got {initial:.3f}"
    assert final < initial - DROP_MARGIN, (
        f"loss did not decrease enough: {initial:.3f} -> {final:.3f} "
        f"(need a drop > {DROP_MARGIN})"
    )


def test_tiny_training_is_deterministic() -> None:
    def run_once() -> float:
        set_seed(0)
        cfg = tiny_config().derive()
        model = TalosGPT(cfg).train()
        corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=16, seed=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        final = None
        for _ in range(30):
            x = corpus[torch.randint(0, corpus.shape[0], (2,))]
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            final = float(loss.detach())
        return float(final)

    a, b = run_once(), run_once()
    assert a == b, f"training not deterministic with fixed seed: {a} vs {b}"
