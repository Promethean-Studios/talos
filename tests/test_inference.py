"""Inference regression tests: prefill == KV-cache decode, and coherent decode.

These lock in the two properties the prototype's inference path must have:

1. **Logit equivalence** — feeding a sequence through the model in one shot
   (prefill) vs. decoding it token-by-token through the incremental KV cache
   must produce the same logits at *every* position (within fp32 tolerance).
   This is the correctness proof for the buffered inference path.
2. **Coherent generation** — a tiny model trained on the deterministic
   recurrent corpus (`training.synthetic`) must greedy-decode a continuation
   that reproduces the corpus pattern ``x[t] = (x[t-2] + x[t-1]) mod vocab``.

All tests are deterministic (fixed seeds) and CPU-friendly (the 254,272-param
canonical ``tiny`` preset, short sequences).
"""
from __future__ import annotations

import torch

from configs.presets import tiny_config
from inference.generate import generate, prefill_decode_max_abs_diff, prefill
from model import TalosGPT
from model.utils import set_seed
from training.synthetic import build_recurrent_corpus

# fp32 tolerance for prefill-vs-decode logits. Both paths run the same math in
# fp32; only the reduction order differs, so the observed max abs diff is at
# the 1e-7 level (measured), far below this guard band.
LOGIT_ATOL = 1e-4

# Training budget for the coherent-decode test (~1-3 s on CPU).
TRAIN_STEPS = 300
PROMPT_LEN = 8
DECODE_LEN = 24


def _tiny_model(seed: int = 0) -> TalosGPT:
    set_seed(seed)
    return TalosGPT(tiny_config().derive()).eval()


def _train_on_recurrence(model: TalosGPT, seed: int = 0) -> tuple[torch.Tensor, float]:
    """Overfit the fixed recurrent corpus; return (corpus, final_loss)."""
    cfg = model.config
    corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=32, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    final = float("inf")
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, corpus.shape[0], (4,))
        x = corpus[idx]
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        final = float(loss.detach())
    model.eval()
    return corpus, final


def test_prefill_matches_kv_decode_all_positions() -> None:
    """One-shot prefill and token-by-token KV decode agree at every position."""
    model = _tiny_model(seed=0)
    set_seed(1)
    ids = torch.randint(0, model.config.vocab_size, (1, 48))

    report = prefill_decode_max_abs_diff(model, ids)

    assert report.seq_len == 48
    assert report.max_abs_diff < LOGIT_ATOL, f"prefill/decode mismatch: {report}"
    assert report.argmax_match, f"greedy choices diverge: {report}"


def test_prefill_matches_kv_decode_chunked() -> None:
    """Equivalence also holds when prefill takes the chunked attention path."""
    set_seed(0)
    cfg = tiny_config().derive()
    cfg.attention_chunk_size = 4  # force the chunked path for seq > 4
    model = TalosGPT(cfg).eval()
    set_seed(2)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    report = prefill_decode_max_abs_diff(model, ids)

    assert report.max_abs_diff < LOGIT_ATOL, f"chunked prefill/decode mismatch: {report}"
    assert report.argmax_match


def test_prefill_cache_length_and_logits_shape() -> None:
    """Prefill caches the whole prompt and returns next-token logits."""
    model = _tiny_model(seed=3)
    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    with torch.no_grad():
        logits, cache = prefill(model, ids)
    assert logits.shape == (1, 12, model.config.vocab_size)
    assert cache.length == 12


def test_greedy_decode_reproduces_recurrent_corpus() -> None:
    """A trained tiny model greedy-decodes the exact corpus continuation.

    The corpus follows ``x[t] = (x[t-2] + x[t-1]) mod vocab``. After overfit-
    ting it, the model must (a) reproduce the held continuation of a training
    row token-for-token via KV-cache decode and (b) emit tokens that satisfy
    the recurrence across the prompt boundary. Also re-verifies prefill/decode
    logit equivalence on the *trained* weights (sharper logits are the harder
    case) and asserts decode is deterministic.
    """
    model = _tiny_model(seed=0)
    corpus, final_loss = _train_on_recurrence(model, seed=0)
    # The model must have genuinely learned the pattern, not just moved a bit.
    assert final_loss < 0.5, f"expected near-overfit loss, got {final_loss:.4f}"

    cfg = model.config
    row = corpus[0]
    prompt = row[:PROMPT_LEN].unsqueeze(0)
    with torch.no_grad():
        cont = generate(model, prompt, DECODE_LEN, greedy=True)

    expected = row[PROMPT_LEN : PROMPT_LEN + DECODE_LEN].tolist()
    assert cont == expected, (
        f"decoded continuation diverges from the corpus pattern:\n  got      {cont}\n"
        f"  expected {expected}"
    )
    # Recurrence holds across the prompt boundary for every generated token.
    full = row[:PROMPT_LEN].tolist() + cont
    for i in range(PROMPT_LEN, len(full)):
        assert full[i] == (full[i - 1] + full[i - 2]) % cfg.vocab_size, (
            f"recurrence violated at generated index {i - PROMPT_LEN}"
        )
    # Deterministic: identical prompt -> identical tokens on a second run.
    with torch.no_grad():
        again = generate(model, prompt, DECODE_LEN, greedy=True)
    assert again == cont

    # Trained-weight logit equivalence (the harder, sharper-logits case).
    report = prefill_decode_max_abs_diff(model, row.unsqueeze(0))
    assert report.max_abs_diff < LOGIT_ATOL, f"trained prefill/decode mismatch: {report}"
    assert report.argmax_match


def test_prefill_accepts_optional_empty_cache() -> None:
    """Design §6.4: ``prefill`` takes an optional pre-built empty cache.

    Passing a caller-created cache must (a) use *that* object (the returned
    cache is the same instance), (b) produce logits identical to the default
    path, and (c) leave the default path (``cache=None``) unchanged.
    """
    model = _tiny_model(seed=0)
    set_seed(1)
    prompt = torch.randint(
        0, model.config.vocab_size, (1, 12), generator=torch.Generator().manual_seed(3)
    )
    with torch.no_grad():
        logits_default, cache_default = prefill(model, prompt)  # unchanged default
        provided = model.new_cache(1, prompt.device, next(model.parameters()).dtype)
        logits_passed, cache_returned = prefill(model, prompt, cache=provided)
    assert cache_returned is provided  # the caller's cache object is reused
    assert cache_returned.length == prompt.shape[1]
    assert float((logits_default - logits_passed).abs().max()) <= LOGIT_ATOL
