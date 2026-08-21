"""Attention mask construction for causal, sliding-window and hybrid schemes.

Two kinds of mask are produced:

* **dense additive masks** (``(..., T, S)``) of 0 / ``-inf`` for use with the
  plain non-chunked attention backend and with FlashAttention's mask argument;
* **per-row key ranges** describing each query's visible span, which let the
  chunked functional backend attend a sliding window without ever materialising
  a ``T x S`` matrix (near-linear memory at 128K).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

NEG_INF: float = -3.3895e38  # near -max float32, safe under exp


def causal_mask(
    seq_len: int,
    window_size: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Lower-triangular (causal) additive mask, optionally banded.

    Args:
        seq_len: number of query/key positions.
        window_size: if > 0, only allow keys within the last ``window_size``
            positions (sliding-window attention); 0 means full causal.
    Returns:
        ``(seq_len, seq_len)`` tensor of 0 / ``NEG_INF``.
    """
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
    k_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    allowed = k_idx <= q_idx  # causal
    if window_size > 0:
        allowed = allowed & (q_idx - k_idx < window_size)  # within window
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, NEG_INF)
    return mask


def key_ranges(
    seq_len: int,
    start_pos: int,
    window_size: int = 0,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-query [low, high) key ranges for the chunked backend.

    For causal attention the high bound of query ``i`` is ``i+1``. For sliding
    window the low bound is ``max(0, i - window_size)``; for full causal it is
    0. ``start_pos`` offsets the query indices (decode caching).

    Returns:
        ``(lo, hi)`` LongTensors of shape ``(seq_len,)`` giving, per query, the
        inclusive-exclusive key index range within the *global* key space.
    """
    query_global = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long)
    hi = query_global + 1
    if window_size > 0:
        # key range [row - (window_size-1), row] => exactly `window_size` keys.
        lo = torch.clamp(query_global - window_size + 1, min=0)
    else:
        lo = torch.zeros_like(query_global)
    return lo, hi


def pad_range_mask(scores: torch.Tensor) -> None:
    """No-op kept for API symmetry; the chunked backend builds masks internally."""
    return None
