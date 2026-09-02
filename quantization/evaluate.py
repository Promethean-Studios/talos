"""Correctness metrics for the prototype quantization experiment.

Pure measurement helpers: logit similarity (mean/max absolute difference,
cosine similarity, greedy agreement), greedy-generation agreement between two
models, and a shared teacher-forced eval loss. All operate on the canonical
model interface (:class:`model.gpt.TalosGPT` forward / ``inference.generate.generate``)
so the numbers describe the *real* prefill + KV-decode path, not a proxy.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["compare_logits", "compare_generations", "eval_loss", "logit_similarity_by_prompt"]


def compare_logits(ref_logits: torch.Tensor, got_logits: torch.Tensor) -> Dict[str, Any]:
    """Per-position logit-closeness metrics between two ``(batch, seq, vocab)`` tensors.

    Returns mean/max absolute difference, mean/min cosine similarity over
    positions, greedy (argmax) agreement rate and the absolute mismatch count.
    Casts to fp32 before comparing so the metric itself never runs in bf16.
    """
    if ref_logits.shape != got_logits.shape:
        raise ValueError(f"shape mismatch: {ref_logits.shape} vs {got_logits.shape}")
    ref = ref_logits.detach().float()
    got = got_logits.detach().float()
    diff = (ref - got).abs()
    cos = F.cosine_similarity(ref, got, dim=-1)  # (batch, seq)
    agree = (ref.argmax(dim=-1) == got.argmax(dim=-1)).float()
    return {
        "max_abs_diff": round(float(diff.max()), 6),
        "mean_abs_diff": round(float(diff.mean()), 6),
        "cosine_mean": round(float(cos.mean()), 6),
        "cosine_min": round(float(cos.min()), 6),
        "greedy_agreement": round(float(agree.mean()), 6),
        "greedy_mismatches": int((agree == 0).sum()),
        "num_positions": int(agree.numel()),
    }


def compare_generations(ref_ids: List[int], got_ids: List[int]) -> Dict[str, Any]:
    """Token-level agreement between two greedy generations of the same prompt.

    ``first_divergence_step`` is the index of the first differing token (or the
    full length when identical).
    """
    if len(ref_ids) != len(got_ids):
        raise ValueError(f"length mismatch: {len(ref_ids)} vs {len(got_ids)}")
    n = len(ref_ids)
    first_div = n
    for i, (a, b) in enumerate(zip(ref_ids, got_ids)):
        if a != b:
            first_div = i
            break
    matches = sum(1 for a, b in zip(ref_ids, got_ids) if a == b)
    return {
        "num_tokens": n,
        "agreement": round(matches / max(n, 1), 6),
        "first_divergence_step": first_div,
        "identical": first_div == n,
    }


def eval_loss(model: nn.Module, corpus: torch.Tensor, vocab_size: int) -> float:
    """Aligned causal-LM loss of ``model`` over the whole ``corpus`` (fp32 CE).

    Uses the properly-aligned objective (input ``x[:, :-1]`` predicts
    ``x[:, 1:]``) — the same objective as training and ``tools/benchmark.py``.
    Logits are cast to fp32 before the cross-entropy so every precision is
    scored by the *same* measure.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _ = model(corpus[:, :-1])
        loss = F.cross_entropy(
            logits.float().reshape(-1, vocab_size), corpus[:, 1:].reshape(-1)
        )
    if was_training:
        model.train()
    return float(loss)


@torch.no_grad()
def logit_similarity_by_prompt(
    ref_model: nn.Module,
    got_model: nn.Module,
    prompts: List[torch.Tensor],
) -> Dict[str, Dict[str, Any]]:
    """Compare one-shot forward logits (fp32 reference vs quantized model).

    ``prompts`` are ``(1, L)`` token-id tensors keyed in the result by their
    sequence length. The one-shot (non-cached) forward isolates weight-precision
    error from KV-cache interactions; the generation comparison in
    ``tools/quantize_experiment.py`` covers the full KV-decode path on top.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for prompt in prompts:
        ref_logits, _ = ref_model(prompt)
        got_logits, _ = got_model(prompt)
        out[str(prompt.shape[1])] = compare_logits(ref_logits, got_logits)
    return out
