"""Buffered (prefill + KV-cache decode) generation for Talos models.

This is the canonical, tested inference path for the prototype: prefill a
prompt in one forward pass (writing K/V into a :class:`~model.cache.KVCache`),
then decode new tokens one at a time against the cache. The same functions are
used by ``examples/run_inference.py`` and the regression tests in
``tests/test_inference.py`` so there is exactly one prefill/decode code path to
verify and benchmark.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from model import TalosGPT
from model.cache import KVCache

__all__ = [
    "prefill",
    "decode_step",
    "generate",
    "prefill_decode_max_abs_diff",
    "EquivalenceReport",
]


def prefill(
    model: TalosGPT,
    prompt: torch.Tensor,
) -> Tuple[torch.Tensor, KVCache]:
    """Run the whole prompt through the model in one pass, caching K/V.

    Args:
        model: A ``TalosGPT`` (should be in ``eval()`` mode for inference).
        prompt: ``(batch, prompt_len)`` token ids.

    Returns:
        ``(logits, cache)`` where ``logits`` is ``(batch, prompt_len, vocab)``
        and ``cache`` holds the prompt's K/V states (``cache.length ==
        prompt_len``). ``logits[:, t]`` is the distribution over the token at
        position ``t + 1``.
    """
    cache = model.new_cache(
        prompt.shape[0], prompt.device, next(model.parameters()).dtype
    )
    logits, cache = model(prompt, use_cache=True, cache=cache)
    return logits, cache


def decode_step(
    model: TalosGPT,
    cache: KVCache,
    token: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Feed a single token at absolute ``position`` through the cached model.

    Args:
        model: The same ``TalosGPT`` that produced ``cache``.
        cache: Running KV cache (from :func:`prefill` or the previous step).
        token: ``(batch, 1)`` token ids.
        position: Absolute position of ``token`` in the sequence (the cache's
            current length when generation continues from the cache).

    Returns:
        Logits ``(batch, 1, vocab)`` for the token that follows ``token``.
    """
    pos = torch.full((token.shape[0], 1), position, dtype=torch.long, device=token.device)
    logits, cache = model(token, position_ids=pos, use_cache=True, cache=cache)
    return logits


def generate(
    model: TalosGPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    greedy: bool = True,
    temperature: float = 1.0,
) -> List[int]:
    """Greedy- (or temperature-sampled-) decode ``max_new_tokens`` tokens.

    Prefills ``prompt`` once, then decodes incrementally. The first generated
    token is the argmax of the prefill's final position.

    Args:
        model: A ``TalosGPT`` in eval mode.
        prompt: ``(1, prompt_len)`` token ids (batch 1).
        max_new_tokens: How many tokens to generate.
        greedy: If True pick argmax; otherwise sample from the softmax at
            ``temperature``.
        temperature: Sampling temperature (ignored when ``greedy``).

    Returns:
        The list of generated token ids (length ``max_new_tokens``).
    """
    if prompt.shape[0] != 1:
        raise ValueError("generate() supports batch size 1 (got batch %d)" % prompt.shape[0])
    with torch.no_grad():
        logits, cache = prefill(model, prompt)
        tok = logits[:, -1:, :]  # (1, 1, vocab): last position's distribution
        generated: List[int] = []
        for step in range(max_new_tokens):
            if greedy:
                nxt = tok.argmax(dim=-1)  # (1, 1) token ids
            else:
                probs = torch.softmax(tok[:, -1] / temperature, dim=-1)  # (1, vocab)
                nxt = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated.append(int(nxt[0, 0]))
            if step + 1 < max_new_tokens:
                tok = decode_step(model, cache, nxt, cache.length)
    return generated


def prefill_decode_max_abs_diff(
    model: TalosGPT,
    ids: torch.Tensor,
) -> "EquivalenceReport":
    """The core inference-correctness check: prefill == KV-cache decode.

    Feeds the full sequence through the model in one shot (plain forward, no
    cache) and again token-by-token through an incrementally-built KV cache,
    then compares the logits at **every** position.

    Args:
        model: A ``TalosGPT`` in eval mode.
        ids: ``(1, seq)`` token ids to compare on.

    Returns:
        An :class:`EquivalenceReport` with the maximum absolute logit
        difference over all positions and whether per-position greedy choices
        agree exactly.
    """
    if ids.shape[0] != 1:
        raise ValueError("prefill_decode_max_abs_diff() supports batch size 1")
    seq = ids.shape[1]
    with torch.no_grad():
        full_logits, _ = model(ids)  # one-shot prefill (no cache)

        cache = model.new_cache(1, ids.device, next(model.parameters()).dtype)
        max_diff = 0.0
        argmax_match = True
        for t in range(seq):
            tok = ids[:, t : t + 1]
            step_logits = decode_step(model, cache, tok, t)
            ref = full_logits[0, t]
            got = step_logits[0, 0]
            max_diff = max(max_diff, float((ref - got).abs().max()))
            if int(ref.argmax()) != int(got.argmax()):
                argmax_match = False
    return EquivalenceReport(max_abs_diff=max_diff, argmax_match=argmax_match, seq_len=seq)


class EquivalenceReport:
    """Result of :func:`prefill_decode_max_abs_diff`."""

    def __init__(self, max_abs_diff: float, argmax_match: bool, seq_len: int) -> None:
        self.max_abs_diff = max_abs_diff
        self.argmax_match = argmax_match
        self.seq_len = seq_len

    def __repr__(self) -> str:
        return (
            f"EquivalenceReport(seq={self.seq_len}, max_abs_diff={self.max_abs_diff:.3e}, "
            f"argmax_match={self.argmax_match})"
        )
