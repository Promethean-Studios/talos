"""Attention backends behind a clean :class:`AttentionInterface`.

Forge decouples *what* attention computes from *how* it is executed:

* :class:`PlainAttentionBackend` — a fully functional, auto-differentiable
  implementation that runs on any CPU/GPU. It supports two execution modes:
  a direct path for short sequences and a **chunked** path for long sequences
  that bounds peak memory to ``O(chunk * seq)`` (or ``O(chunk * (chunk + w))``
  for sliding window) instead of ``O(seq^2)``. This is what keeps 128K contexts
  feasible on the functional backend.
* :class:`FlashAttentionBackend` — an optimized GPU kernel. Its import is
  guarded so Forge runs fine without ``flash-attn`` installed; when the package
  is present it is used automatically, otherwise we fall back to plain.

Both implement the same :class:`AttentionInterface`, so a config can swap them
with zero model changes.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model.masking import NEG_INF, causal_mask

logger = logging.getLogger("forge.attention")


class AttentionInterface(ABC):
    """Uniform attention backend contract.

    Inputs are already projected and GQA-expanded: q/k/v all have shape
    ``(batch, num_heads, seq, head_dim)``. Returns the attended output of shape
    ``(batch, num_heads, seq, head_dim)``.
    """

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_size: int = 0,
        causal: bool = True,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run attention.

        Args:
            query/key/value: ``(B, H, T, D)`` tensors.
            mask: optional additive ``0/-inf`` mask broadcastable to
                ``(B, H, T, S)``. When None, a causal/sliding mask is built.
            window_size: > 0 enables sliding-window masking (ignored if an
                explicit ``mask`` is supplied).
            causal: whether to apply causal masking.
            scale: attention scale; defaults to ``1/sqrt(head_dim)``.
        Returns:
            Attended ``(B, H, T, D)`` output.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # allow nn.Module-style call
        return self.forward(*args, **kwargs)


# ------------------------------------------------------------------------------
# Plain (functional) backend
# ------------------------------------------------------------------------------

class PlainAttentionBackend(AttentionInterface):
    """Fully functional attention, correct on CPU and any GPU.

    Args:
        chunk_size: when > 0 and smaller than the sequence, the query sequence
            is processed in blocks of this size. This keeps peak memory
            near-linear. Set to 0 for the straightforward O(seq^2) path.
    """

    def __init__(self, chunk_size: int = 0) -> None:
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_size: int = 0,
        causal: bool = True,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if query.dim() != 4:
            raise ValueError(f"query must be 4D (B,H,T,D), got {query.shape}")
        batch, heads, seq, head_dim = query.shape
        scale = scale if scale is not None else head_dim ** -0.5

        use_chunked = (
            self.chunk_size > 0
            and self.chunk_size < seq
            and mask is None
            and causal
        )
        if use_chunked:
            return self._chunked(query, key, value, window_size, scale)
        return self._direct(query, key, value, mask, window_size, causal, scale)

    # -- direct (non-chunked) ---------------------------------------------------
    def _direct(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        window_size: int,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale  # (B,H,T,S)
        if mask is None:
            if not causal:
                raise ValueError("causal=False requires an explicit mask")
            # The queries are a suffix of the keys (prefill: equal length;
            # decode: the single new token at the end of the cache). Build a
            # rectangular causal/banded mask sized (num_queries, num_keys).
            nq = scores.shape[-2]
            ns = scores.shape[-1]
            q_global = torch.arange(ns - nq, ns, device=scores.device).unsqueeze(1)
            k_global = torch.arange(ns, device=scores.device).unsqueeze(0)
            allowed = k_global <= q_global  # causal
            if window_size > 0:
                allowed = allowed & (q_global - k_global < window_size)
            mask = torch.zeros(nq, ns, dtype=scores.dtype, device=scores.device)
            mask = mask.masked_fill(~allowed, NEG_INF)
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, value)

    # -- chunked (bounded-memory) path -------------------------------------------
    def _chunked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        window_size: int,
        scale: float,
    ) -> torch.Tensor:
        batch, heads, seq, head_dim = query.shape
        chunk = self.chunk_size
        out = torch.empty_like(query)
        device = query.device

        for c0 in range(0, seq, chunk):
            c1 = min(c0 + chunk, seq)
            qc = query[:, :, c0:c1, :]  # (B,H,C,D)
            rows = qc.shape[-2]

            if window_size > 0:
                k_start = max(0, c0 - window_size)
                key_slice = key[:, :, k_start:c1, :]
                value_slice = value[:, :, k_start:c1, :]
            else:
                k_start = 0
                key_slice = key[:, :, :c1, :]
                value_slice = value[:, :, :c1, :]

            scores = torch.matmul(qc, key_slice.transpose(-1, -2)) * scale
            # Build per-row allowed-key mask for this chunk.
            rows_idx = torch.arange(c0, c1, device=device, dtype=torch.long)
            cols_idx = torch.arange(
                k_start, k_start + key_slice.shape[-2], device=device, dtype=torch.long
            )
            allowed = cols_idx.unsqueeze(0) <= rows_idx.unsqueeze(1)  # causal
            if window_size > 0:
                # Look back exactly `window_size` keys: [row - (w-1), row].
                allowed = allowed & (
                    cols_idx.unsqueeze(0) >= rows_idx.unsqueeze(1) - window_size + 1
                )
            mask = torch.zeros_like(scores)
            mask = mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), NEG_INF)
            probs = F.softmax(scores + mask, dim=-1)
            out[:, :, c0:c1, :] = torch.matmul(probs, value_slice)
        return out

    def __repr__(self) -> str:
        return f"PlainAttentionBackend(chunk_size={self.chunk_size})"


# ------------------------------------------------------------------------------
# FlashAttention backend (guarded import)
# ------------------------------------------------------------------------------

class FlashAttentionBackend(AttentionInterface):
    """Optimized FlashAttention-2/3 GPU backend.

    Requires the optional ``flash-attn`` package. If it cannot be imported,
    :meth:`available` returns False and the caller should fall back to
    :class:`PlainAttentionBackend` (the model factory does this automatically).

    flash-attn supports GQA natively (different numbers of q/kv heads), but for
    uniformity we receive already-expanded tensors and pass ``num_heads_q`` /
    ``num_heads_k`` accordingly.
    """

    _flash = None  # lazy module cache

    @classmethod
    def available(cls) -> bool:
        """Whether the flash-attn kernel may be imported on this host."""
        try:
            import flash_attn  # noqa: F401

            cls._flash = flash_attn
            return True
        except Exception:  # pragma: no cover - dep must not be installed
            return False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_size: int = 0,
        causal: bool = True,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if not self.available():  # pragma: no cover - only reachable if FA missing
            raise RuntimeError("flash-attn not installed; use PlainAttentionBackend")
        # flash_attn_func takes (B, T, H, D) layout.
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if causal and window_size == 0:
            win = (-1, -1)  # no sliding window
        else:
            left = window_size if window_size > 0 else -1
            win = (left, left)
        out = self._flash.flash_attn_func(
            q,
            k,
            v,
            None if causal else ...,
            causal=causal,
            window_size=win,
        )
        return out.transpose(1, 2)


def build_attention_backend(backend: str = "auto", chunk_size: int = 0) -> AttentionInterface:
    """Return an attention backend per strategy.

    ``backend`` is one of:
      * ``"auto"`` — prefer FlashAttention when installed, else plain;
      * ``"plain"`` — always the functional backend;
      * ``"flash"`` — the FlashAttention backend (raises if unavailable).
    """
    if backend == "plain":
        return PlainAttentionBackend(chunk_size=chunk_size)
    if backend == "flash":
        if FlashAttentionBackend.available():
            return FlashAttentionBackend()
        raise RuntimeError("flash backend requested but flash-attn is not installed")
    if backend == "auto":
        if FlashAttentionBackend.available():
            logger.info("Using FlashAttention backend (flash-attn installed).")
            return FlashAttentionBackend()
        logger.info("flash-attn not installed; using plain functional backend.")
        return PlainAttentionBackend(chunk_size=chunk_size)
    raise ValueError(f"unknown attention backend: {backend!r}")
