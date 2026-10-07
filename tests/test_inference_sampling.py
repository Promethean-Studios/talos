"""Decoding-control tests for ``inference.generate`` (top-k / top-p /
repetition penalty / EOS early stopping) and the CLI's EOS resolution.

Remedy R1 of the Styx generation diagnosis: the owner's greedy byte-loop
repetition needs anti-repetition decoding levers. These tests lock in

1. the **greedy-identity regression** — with all decoding controls at their
   defaults the new ``generate()`` is bit-identical to the pre-upgrade plain
   argmax decode (verified against a hand-rolled reference loop);
2. the **composition contract** — repetition penalty on raw logits, then
   temperature, then top-k, then top-p, then softmax/sample (greedy = argmax
   after penalty/top-k/top-p filtering);
3. per-control correctness (top-k truncation, top-p mass cutoff, penalty
   ranking change, EOS stops the loop) and input validation.

Models here are tiny *scripted* stubs (logits fixed per absolute position) so
the decode loop mechanics are tested without any training, and a short
real-model identity check covers the full path. All tests are CPU-only,
deterministic, and fast.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence

import pytest
import torch

from configs.presets import tiny_config
from inference.generate import (
    apply_repetition_penalty,
    decode_step,
    generate,
    prefill,
    top_k_filter,
    top_p_filter,
)
from model import TalosGPT
from model.utils import set_seed


# ---------------------------------------------------------------------------
# Scripted stub model: logits are fixed per absolute position, so the token
# stream a generate() call produces is exactly ``argmax/sample(script[p])``
# for p = prompt_len-1, prompt_len, ... — no weights, no training.
# ---------------------------------------------------------------------------
class _ScriptedCache:
    def __init__(self, length: int) -> None:
        self.length = length


class _ScriptedModel:
    """Minimal TalosGPT stand-in implementing the KV-cache inference protocol."""

    def __init__(
        self,
        vocab_size: int,
        script: Sequence[torch.Tensor],
        max_seq_len: int = 512,
    ) -> None:
        self.config = SimpleNamespace(vocab_size=vocab_size, max_seq_len=max_seq_len)
        self._script = [s if s.ndim == 2 else s.unsqueeze(0) for s in script]
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # dtype/device source

    def parameters(self) -> List[torch.nn.Parameter]:
        return iter([self._dummy])

    def new_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> _ScriptedCache:
        del batch, device, dtype
        return _ScriptedCache(0)

    def __call__(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[_ScriptedCache] = None,
    ) -> tuple[torch.Tensor, _ScriptedCache]:
        del use_cache
        if cache is None:
            cache = _ScriptedCache(0)
        if position_ids is None:
            # prefill: logits for every prompt position
            logits = torch.stack([self._row(i) for i in range(input_ids.shape[1])], dim=1)
            cache.length = input_ids.shape[1]
        else:
            logits = self._row(int(position_ids[0, 0])).unsqueeze(0)  # (1, 1, V)
            cache.length = int(position_ids[0, 0]) + 1
        return logits, cache

    def _row(self, position: int) -> torch.Tensor:
        if position >= len(self._script):
            return self._script[-1]
        return self._script[position]


def _row(pairs: Sequence[tuple[int, float]], vocab: int, fill: float = -10.0) -> torch.Tensor:
    """A (1, vocab) logits row: ``pairs`` (token_id, logit) at their ids, rest ``fill``."""
    v = torch.full((1, vocab), fill)
    for token_id, logit in pairs:
        v[0, token_id] = logit
    return v


def _rand_script(vocab: int, length: int, seed: int) -> List[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(1, vocab, generator=gen) for _ in range(length)]


# ---------------------------------------------------------------------------
# Greedy identity regression: defaults must reproduce the pre-upgrade decode.
# ---------------------------------------------------------------------------
def test_greedy_default_bit_identical_to_plain_argmax() -> None:
    """With all controls at defaults, generate() == the old argmax loop, exactly."""
    vocab, prompt_len, n_new = 32, 4, 8
    script = _rand_script(vocab, prompt_len + n_new + 2, seed=0)
    model = _ScriptedModel(vocab, script)
    prompt = torch.tensor([[3, 7, 1, 9]])

    with torch.no_grad():
        got = generate(model, prompt, n_new, greedy=True)
        # hand-rolled pre-upgrade loop: prefill -> argmax/decode_step per step
        logits, cache = prefill(model, prompt)
        tok = logits[:, -1:, :]
        ref: List[int] = []
        for _ in range(n_new):
            nxt = tok.argmax(dim=-1)
            ref.append(int(nxt[0, 0]))
            tok = decode_step(model, cache, nxt, cache.length)

    assert got == ref
    # Explicit default-valued new kwargs take the same path.
    again = generate(
        model, prompt, n_new,
        greedy=True, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, penalize_prompt=False, eos_token_id=None,
    )
    assert again == ref


def test_real_model_defaults_are_stable() -> None:
    """The real tiny model: defaults vs explicit defaults agree (full path)."""
    set_seed(0)
    model = TalosGPT(tiny_config().derive()).eval()
    prompt = torch.randint(
        0, model.config.vocab_size, (1, 12), generator=torch.Generator().manual_seed(2)
    )
    with torch.no_grad():
        out = generate(model, prompt, 10, greedy=True)
        out_explicit = generate(
            model, prompt, 10,
            greedy=True, top_k=0, top_p=1.0, repetition_penalty=1.0,
        )
    assert out == out_explicit
    assert len(out) == 10


# ---------------------------------------------------------------------------
# Repetition penalty.
# ---------------------------------------------------------------------------
def test_repetition_penalty_demotes_seen_tokens() -> None:
    """CTRL-style: positive logits / R, negative logits * R; identity at R=1."""
    logits = torch.tensor([[1.0, 0.9, -0.5, 2.0, 0.0, 0.3, -1.0, 0.7]])
    penalized = apply_repetition_penalty(logits, [0, 2, 7], 2.0)
    assert penalized[0, 0] == pytest.approx(0.5)   # 1.0 / 2
    assert penalized[0, 2] == pytest.approx(-1.0)  # -0.5 * 2
    assert penalized[0, 7] == pytest.approx(0.35)  # 0.7 / 2
    assert penalized[0, 3] == pytest.approx(2.0)   # untouched ids unchanged
    # Identity when disabled or when no valid ids are given.
    assert torch.equal(apply_repetition_penalty(logits, [0], 1.0), logits)
    assert torch.equal(apply_repetition_penalty(logits, [999, -3], 2.0), logits)


def test_repetition_penalty_changes_repeated_token_ranking() -> None:
    """Penalty + greedy works: demoting a seen token lets an unseen one win.

    Also documents the one-per-step semantics: the penalty is applied exactly
    once per step from the raw logits, so among equally-penalized tokens the
    original ranking is preserved (a pure [5,5,5,5] loop becomes [5,12,5,5] —
    the alternation breaks the monotone loop).
    """
    vocab = 16
    # First and every later position: token 5 leads (1.0), unseen token 12
    # trails (0.95), everything else is deeply negative.
    script = [_row([(5, 1.0), (12, 0.95)], vocab)] * 6
    model = _ScriptedModel(vocab, script)
    prompt = torch.tensor([[0, 1]])

    loop = generate(model, prompt, 4, greedy=True)
    assert loop == [5, 5, 5, 5]

    broken = generate(model, prompt, 4, greedy=True, repetition_penalty=1.5)
    # step1: 5 (1.0 > 0.95). step2: 5 penalized to 0.667 < 0.95 -> 12. step3:
    # both penalized (5 -> 0.667, 12 -> 0.633) -> 5. step4: same -> 5.
    assert broken == [5, 12, 5, 5]
    assert broken != loop


# ---------------------------------------------------------------------------
# Top-k.
# ---------------------------------------------------------------------------
def test_top_k_truncation() -> None:
    """Only the top-k logits survive; the rest are -inf (0 probability)."""
    logits = torch.tensor([[0.3, 1.7, -0.2, 2.5, 0.1, -1.0, 0.9, 0.4]])
    kept = top_k_filter(logits, 3)
    assert kept[0, 3] == pytest.approx(2.5)
    assert kept[0, 1] == pytest.approx(1.7)
    assert kept[0, 6] == pytest.approx(0.9)
    assert kept[0, 0] == float("-inf")
    assert kept[0, 5] == float("-inf")
    # k <= 0 and k >= vocab are the identity.
    assert torch.equal(top_k_filter(logits, 0), logits)
    assert torch.equal(top_k_filter(logits, logits.size(-1)), logits)


def test_top_k_one_forces_single_token_when_sampling() -> None:
    """top_k=1 makes temperature sampling deterministic (one survivor)."""
    vocab, prompt_len, n_new = 8, 2, 6
    model = _ScriptedModel(vocab, _rand_script(vocab, prompt_len + n_new, seed=3))
    prompt = torch.tensor([[2, 5]])
    set_seed(0)
    a = generate(model, prompt, n_new, greedy=False, temperature=1.0, top_k=1)
    set_seed(1)
    b = generate(model, prompt, n_new, greedy=False, temperature=1.0, top_k=1)
    assert a == b
    assert len(a) == n_new


# ---------------------------------------------------------------------------
# Top-p.
# ---------------------------------------------------------------------------
def test_top_p_mass_cutoff() -> None:
    """The nucleus is the smallest set with cumulative mass >= p (renormalized)."""
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -0.5, -1.0, -2.0, -3.0]])
    # softmax masses (approx): [.621, .228, .084, .031, .019, .011, .004, .002]
    for p, n_keep in ((0.7, 2), (0.95, 4)):
        filtered = top_p_filter(logits, p)
        probs = torch.softmax(filtered, dim=-1)
        assert probs.sum().item() == pytest.approx(1.0), "nucleus must renormalize"
        kept = (probs > 0).nonzero()[:, 1].tolist()
        assert kept == list(range(n_keep)), (
            f"p={p}: expected the top {n_keep} tokens to survive, got {kept}"
        )
    assert torch.equal(top_p_filter(logits, 1.0), logits)


# ---------------------------------------------------------------------------
# EOS early stopping + validation.
# ---------------------------------------------------------------------------
def test_eos_stops_the_loop_without_emitting_eos() -> None:
    """EOS stops generation; the EOS id itself never appears in the output."""
    vocab = 16
    script = [
        _row([(0, 2.0)], vocab),   # prompt position 0 (unused for decoding)
        _row([(3, 2.0)], vocab),   # first generated token -> 3
        _row([(7, 2.0)], vocab),   # second -> 7 (the EOS id): stop here
        _row([(4, 2.0)], vocab),
        _row([(5, 2.0)], vocab),
    ]
    model = _ScriptedModel(vocab, script)
    prompt = torch.tensor([[1, 1]])

    out = generate(model, prompt, 8, greedy=True, eos_token_id=7)
    assert out == [3]               # len < max_new_tokens == EOS stop
    assert 7 not in out

    full = generate(model, prompt, 8, greedy=True)
    assert full == [3, 7, 4, 5, 5, 5, 5, 5]  # without EOS: always emits max_new

    # EOS id out of range is a clear error, never a crash.
    with pytest.raises(ValueError, match="eos_token_id"):
        generate(model, prompt, 4, eos_token_id=vocab + 5)


def test_generate_validates_decoding_controls() -> None:
    """Invalid control values raise ValueError before any decoding."""
    model = _ScriptedModel(16, _rand_script(16, 4, seed=4))
    prompt = torch.tensor([[0, 1]])
    bad = [
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.0},
        {"eos_token_id": 99},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            generate(model, prompt, 4, **kwargs)


def test_greedy_unaffected_by_top_k_top_p() -> None:
    """top-k/top-p never remove the argmax, so greedy choice is unchanged."""
    vocab, prompt_len, n_new = 16, 2, 6
    model = _ScriptedModel(vocab, _rand_script(vocab, prompt_len + n_new, seed=5))
    prompt = torch.tensor([[1, 9]])
    plain = generate(model, prompt, n_new, greedy=True)
    filtered = generate(model, prompt, n_new, greedy=True, top_k=3, top_p=0.6)
    assert plain == filtered


def test_sampling_reproducible_with_fixed_seed() -> None:
    """Sampling with modifiers + a fixed seed is reproducible; seeds differ."""
    vocab, prompt_len, n_new = 16, 2, 12
    model = _ScriptedModel(vocab, _rand_script(vocab, prompt_len + n_new, seed=7))
    prompt = torch.tensor([[0, 4]])
    set_seed(11)
    a = generate(model, prompt, n_new, greedy=False, temperature=0.8, top_p=0.9)
    set_seed(11)
    b = generate(model, prompt, n_new, greedy=False, temperature=0.8, top_p=0.9)
    assert a == b
    set_seed(12)
    c = generate(model, prompt, n_new, greedy=False, temperature=0.8, top_p=0.9)
    assert a != c  # fixed seeds -> deterministic; the draw differs


# ---------------------------------------------------------------------------
# CLI-side EOS resolution (scripts/generate.py).
# ---------------------------------------------------------------------------
class _TokWithEos:
    eos_id = 1023


class _TokNoEos:
    pass


def test_resolve_eos_token_id() -> None:
    """Explicit id wins; else the tokenizer's EOS; absent EOS -> no-op (None)."""
    from scripts.generate import resolve_eos_token_id

    assert resolve_eos_token_id(None, _TokWithEos()) == (1023, True)
    assert resolve_eos_token_id(5, _TokWithEos()) == (5, True)
    assert resolve_eos_token_id(5, _TokNoEos()) == (5, True)
    assert resolve_eos_token_id(None, _TokNoEos()) == (None, False)