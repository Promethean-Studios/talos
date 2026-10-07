"""Buffered (prefill + KV-cache decode) generation for Talos models.

This is the canonical, tested inference path for the prototype: prefill a
prompt in one forward pass (writing K/V into a :class:`~model.cache.KVCache`),
then decode new tokens one at a time against the cache. The same functions are
used by ``examples/run_inference.py`` and the regression tests in
``tests/test_inference.py`` so there is exactly one prefill/decode code path to
verify and benchmark.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from model import TalosGPT
from model.cache import KVCache

__all__ = [
    "prefill",
    "decode_step",
    "generate",
    "apply_repetition_penalty",
    "top_k_filter",
    "top_p_filter",
    "prefill_decode_max_abs_diff",
    "EquivalenceReport",
]


def prefill(
    model: TalosGPT,
    prompt: torch.Tensor,
    cache: Optional[KVCache] = None,
) -> Tuple[torch.Tensor, KVCache]:
    """Run the whole prompt through the model in one pass, caching K/V.

    Args:
        model: A ``TalosGPT`` (should be in ``eval()`` mode for inference).
        prompt: ``(batch, prompt_len)`` token ids.
        cache: Optional cache object implementing the KV-cache protocol
            (``length`` / ``last_len()`` / ``update(layer, key, value,
            start_pos)``). Pass a fresh/empty cache to reuse the canonical
            prefill path with an alternative cache implementation (e.g. the
            DDM experiment's disk-tiered cache); when ``None`` (the default)
            a new :class:`~model.cache.KVCache` is created, exactly as before.

    Returns:
        ``(logits, cache)`` where ``logits`` is ``(batch, prompt_len, vocab)``
        and ``cache`` holds the prompt's K/V states (``cache.length ==
        prompt_len``). ``logits[:, t]`` is the distribution over the token at
        position ``t + 1``.
    """
    if cache is None:
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


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: List[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    """CTRL-style repetition penalty over ``token_ids`` (non-mutating).

    For every distinct id in ``token_ids`` its logit is pushed away from
    sampling: positive logits are divided by ``penalty`` (demoted), negative
    logits are multiplied by ``penalty`` (pushed further negative), so a
    repeatedly-emitted token is less likely to be chosen again. ``penalty ==
    1.0`` (or an empty ``token_ids``) is the identity.

    Args:
        logits: ``(batch, vocab)`` raw logits.
        token_ids: token ids seen so far in this sample (may include prompt
            ids when prompt-penalization is wanted).
        penalty: ``> 0``; ``1.0`` disables. Values ``> 1`` penalize repetition,
            values ``< 1`` encourage it.

    Returns:
        A new tensor with the penalized logits (input is never modified).
    """
    if penalty == 1.0 or not token_ids:
        return logits
    out = logits.clone()
    vocab = out.size(-1)
    for t in set(token_ids):
        if 0 <= t < vocab:
            lt = out[:, t]
            out[:, t] = torch.where(lt > 0, lt / penalty, lt * penalty)
    return out


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Truncate to the ``k`` highest-scoring tokens; set the rest to ``-inf``.

    Non-mutating. ``k <= 0`` or ``k >= vocab`` is the identity (top-k off).

    Args:
        logits: ``(batch, vocab)`` logits.
        k: how many tokens to keep (by score, ties broken by index).

    Returns:
        A new tensor where only the top-``k`` positions keep their logits
        (all others ``-inf``, i.e. zero probability after softmax).
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    topk = torch.topk(logits, k, dim=-1)
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, topk.indices, True)
    return logits.masked_fill(~keep, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) truncation: keep the smallest set with mass ``>= p``.

    Called on raw logits: the softmax distribution is computed internally to
    find the cutoff, then every token outside the nucleus is set to ``-inf``
    (zero probability after the caller's softmax, which renormalizes the
    nucleus to mass 1). Non-mutating. ``p >= 1.0`` is the identity (top-p off).

    Args:
        logits: ``(batch, vocab)`` logits.
        p: cumulative-probability cutoff in ``(0, 1]``; the single highest-mass
            token is always kept, so the filtered set is never empty.

    Returns:
        A new tensor with all non-nucleus logits set to ``-inf``.
    """
    if p >= 1.0 or p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()  # keep the token that crosses p
    remove[..., 0] = False  # ... and the single most probable token
    remove = remove.scatter(-1, sorted_indices, remove)
    return logits.masked_fill(remove, float("-inf"))


def generate(
    model: TalosGPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    penalize_prompt: bool = False,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """Greedy- (or temperature-sampled-) decode ``max_new_tokens`` tokens.

    Prefills ``prompt`` once, then decodes incrementally. The first generated
    token is the argmax of the prefill's final position.

    Decoding controls compose in a fixed order, applied per step: repetition
    penalty on the raw logits first, then temperature, then top-k, then top-p,
    then softmax/sampling (greedy = argmax after penalty/top-k/top-p
    filtering). ``greedy`` + ``repetition_penalty`` is valid and useful: the
    penalty can demote an already-emitted token below the runner-up, so greedy
    decoding escapes byte-loops while staying deterministic and RNG-free.
    (top-k/top-p never remove the argmax, so in greedy mode they cannot change
    the *choice* — they only matter for sampling.) When all controls are at
    their defaults the decode is bit-identical to the pre-upgrade greedy /
    temperature path (regression-tested).

    ``eos_token_id`` enables early stopping: when the sampled token is EOS the
    loop stops **before** emitting it, so the returned list never contains EOS
    and may be shorter than ``max_new_tokens`` (``len < max_new_tokens`` ==
    EOS stop; ``max_new_tokens`` tokens == length stop).

    Sequence-length policy (mirrors the CLI in ``scripts/generate.py``): the
    model's ``max_seq_len`` bounds absolute positions, so ``prompt_len +
    max_new_tokens`` must fit. When a prompt is too long, it is truncated
    **on the left** — the most recent ``max_seq_len - max_new_tokens`` tokens
    are kept, since a causal LM's nearest context is what conditions the
    continuation. ``max_new_tokens >= max_seq_len`` is a clear ``ValueError``
    (there must be room for a non-empty prompt). With the truncation in place
    the decode loop can never drive the RoPE position table (or the KV cache)
    past ``max_seq_len``; callers that bypass ``generate()`` (e.g. a raw
    ``decode_step`` loop) are protected by the named position check in
    ``model/rotary.py``.

    Args:
        model: A ``TalosGPT`` in eval mode.
        prompt: ``(1, prompt_len)`` token ids (batch 1).
        max_new_tokens: How many tokens to generate.
        greedy: If True pick argmax; otherwise sample from the softmax at
            ``temperature``.
        temperature: Sampling temperature (ignored when ``greedy``).
        top_k: Keep only the ``top_k`` highest-scoring tokens before softmax/
            argmax; ``0`` (default) disables.
        top_p: Nucleus cutoff in ``(0, 1]`` — keep the smallest set with
            probability mass ``>= top_p``; ``1.0`` (default) disables.
        repetition_penalty: CTRL-style penalty ``> 0`` applied to every id
            emitted so far in this sample (``> 1`` discourages repetition,
            ``1.0`` (default) disables, ``< 1`` encourages it).
        penalize_prompt: Also apply the repetition penalty to the prompt's
            token ids (default False — only tokens generated in this sample
            are penalized).
        eos_token_id: Stop (without emitting) when this id is sampled;
            ``None`` (default) never stops early.

    Returns:
        The list of generated token ids (length ``max_new_tokens``, or shorter
        when ``eos_token_id`` is set and EOS was sampled).
    """
    if prompt.shape[0] != 1:
        raise ValueError("generate() supports batch size 1 (got batch %d)" % prompt.shape[0])
    max_seq_len = model.config.max_seq_len
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if max_new_tokens >= max_seq_len:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} leaves no room for a prompt "
            f"within max_seq_len={max_seq_len} (need max_new_tokens < max_seq_len)"
        )
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0 (0 = off), got {top_k}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1] (1.0 = off), got {top_p}")
    if repetition_penalty <= 0:
        raise ValueError(
            f"repetition_penalty must be > 0 (1.0 = off), got {repetition_penalty}"
        )
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError(
            f"eos_token_id={eos_token_id} out of range for vocab "
            f"{model.config.vocab_size}"
        )
    prompt_len = prompt.shape[1]
    if prompt_len + max_new_tokens > max_seq_len:
        keep = max_seq_len - max_new_tokens
        prompt = prompt[:, -keep:]  # left-truncate: keep the most recent tokens
    with torch.no_grad():
        logits, cache = prefill(model, prompt)
        tok = logits[:, -1:, :]  # (1, 1, vocab): last position's distribution
        generated: List[int] = []
        penalized: List[int] = prompt[0].tolist() if penalize_prompt else []
        for step in range(max_new_tokens):
            logits_v = tok[:, -1, :]  # (1, vocab)
            if repetition_penalty != 1.0:
                logits_v = apply_repetition_penalty(
                    logits_v, penalized + generated, repetition_penalty
                )
            if not greedy:
                logits_v = logits_v / temperature
            if top_k:
                logits_v = top_k_filter(logits_v, top_k)
            if top_p < 1.0:
                logits_v = top_p_filter(logits_v, top_p)
            if greedy:
                nxt = logits_v.argmax(dim=-1, keepdim=True)  # (1, 1) token ids
            else:
                probs = torch.softmax(logits_v, dim=-1)  # (1, vocab)
                nxt = torch.multinomial(probs, num_samples=1)  # (1, 1)
            token_id = int(nxt[0, 0])
            if eos_token_id is not None and token_id == eos_token_id:
                break  # stop before emitting EOS (caller sees len < max_new_tokens)
            generated.append(token_id)
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
