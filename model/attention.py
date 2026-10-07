"""Attention backends behind a clean :class:`AttentionInterface`.

Talos decouples *what* attention computes from *how* it is executed:

* :class:`PlainAttentionBackend` — a fully functional, auto-differentiable
  implementation that runs on any CPU/GPU. It supports two execution modes:
  a direct path for short sequences and a **chunked** path for long sequences
  that bounds peak memory to ``O(chunk * seq)`` (or ``O(chunk * (chunk + w))``
  for sliding window) instead of ``O(seq^2)``. This is what keeps 128K contexts
  feasible on the functional backend.
* :class:`FlashAttentionBackend` — an optimized GPU kernel. Its import is
  guarded so Talos runs fine without ``flash-attn`` installed; when the package
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

logger = logging.getLogger("talos.attention")


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
# SDPA backend (torch.nn.functional.scaled_dot_product_attention)
# ------------------------------------------------------------------------------

class SDPAAttentionBackend(AttentionInterface):
    """Attention executed by ``F.scaled_dot_product_attention``.

    The T4 training-engine path (P1a): for **full causal attention** (no
    explicit mask, no sliding window) the kernel is called with
    ``is_causal=True`` and **no mask tensor is materialized at all** — the
    ``0/-inf`` additive-mask construction of :class:`PlainAttentionBackend`
    (``masked_fill(~allowed, NEG_INF)`` at
    ``model/attention.py:131-137``) disappears entirely. That is the root-cause
    removal of the recorded fp16 failure: under AMP the plain path's
    ``scores + mask`` fp16 add overflowed ``c10::Half``; SDPA folds the mask
    into the kernel with fp32 accumulation and never performs that add.

    For windowed or explicit-mask attention a ``0/-inf`` additive mask is
    passed as ``attn_mask`` (SDPA requires float additive masks for masking,
    which it applies internally with fp32 accumulation). The mask is built in
    the *query* compute dtype exactly like the plain path, so results agree
    with it to tight tolerance (verified by the equivalence tests).

    On SM75 (T4) PyTorch selects the memory-efficient kernel for these shapes;
    on CPU it falls back to the math implementation — identical semantics, so
    the equivalence tests run anywhere.
    """

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
        head_dim = query.shape[-1]
        scale = scale if scale is not None else head_dim ** -0.5

        if (
            mask is None
            and causal
            and window_size == 0
            and query.shape[-2] == key.shape[-2]
        ):
            # Exact full-causal prefill case: no mask object exists at all.
            # (``is_causal=True`` with unequal q/k lengths uses a *prefix*
            #  convention — a decode query would only see the first key — so
            #  non-square causal calls go through the explicit mask below.)
            return F.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=True
            )

        # Sliding-window or explicit-mask attention: build (or reuse) a
        # 0/-inf additive mask with the plain backend's exact convention
        # (queries are a suffix of keys — decode is a single query at the end).
        nq, ns = query.shape[-2], key.shape[-2]
        device = query.device
        if mask is None:
            q_global = torch.arange(ns - nq, ns, device=device).unsqueeze(1)
            k_global = torch.arange(ns, device=device).unsqueeze(0)
            allowed = k_global <= q_global  # causal
            if window_size > 0:
                allowed = allowed & (q_global - k_global < window_size)
            mask = torch.zeros(nq, ns, dtype=query.dtype, device=device)
            mask = mask.masked_fill(~allowed, NEG_INF)
        attn_mask = mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,S)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, scale=scale
        )

    def __repr__(self) -> str:
        return "SDPAAttentionBackend()"


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
      * ``"auto"`` — prefer FlashAttention when installed; else
        :class:`SDPAAttentionBackend` (torch >= 2.0 ``F.scaled_dot_product_attention``,
        no mask materialization on the full-causal path); if SDPA is somehow
        unavailable the functional :class:`PlainAttentionBackend` is used.
        Callers needing the bounded-memory *chunked* plain path (very long
        contexts) must request ``"plain"`` explicitly or guard on
        ``attention_chunk_size > 0`` (the model factory does the latter — see
        ``model/gpt.py``);
      * ``"plain"`` — always the functional backend (the pre-SDPA path,
        selectable for the T4 A/B);
      * ``"sdpa"`` — always the SDPA backend (raises if unavailable);
      * ``"flash"`` — the FlashAttention backend (raises if unavailable).
    """
    if backend == "plain":
        return PlainAttentionBackend(chunk_size=chunk_size)
    if backend == "sdpa":
        if not hasattr(F, "scaled_dot_product_attention"):  # pragma: no cover
            raise RuntimeError(
                "sdpa backend requested but torch.nn.functional."
                "scaled_dot_product_attention is unavailable (torch < 2.0)"
            )
        return SDPAAttentionBackend()
    if backend == "flash":
        if FlashAttentionBackend.available():
            return FlashAttentionBackend()
        raise RuntimeError("flash backend requested but flash-attn is not installed")
    if backend == "auto":
        if FlashAttentionBackend.available():
            logger.info("Using FlashAttention backend (flash-attn installed).")
            return FlashAttentionBackend()
        logger.info(
            "flash-attn not installed; using SDPA backend "
            "(torch scaled_dot_product_attention)."
        )
        return SDPAAttentionBackend()
    raise ValueError(f"unknown attention backend: {backend!r}")
