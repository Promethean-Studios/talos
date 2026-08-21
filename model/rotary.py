"""Rotary Position Embeddings (RoPE) with optional YaRN long-context scaling.

RoPE rotates query/key vectors by an angle proportional to absolute position,
giving the transformer a *relative* position prior that generalises beyond the
training length. YaRN ("YaRN: Efficient context window extension") rescales the
base frequencies in a frequency-dependent way so that a model trained at one
length can be applied at a much longer context with minimal degradation.

Caching convention
------------------
``precompute_freqs_cis`` returns ``cos``/``sin`` of shape ``(max_seq, head_dim//2)``
— one rotation angle per *pair* of channels. :func:`apply_rotary_pos_emb`
treats the head dimension as ``head_dim//2`` adjacent pairs and rotates each
pair by its angle, which is exactly an orthonormal rotation and therefore
preserves vector norms.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    scaling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute per-pair cos/sin for every position up to ``max_seq_len``.

    Args:
        head_dim: dimension of the query/key vectors (must be even).
        max_seq_len: maximum number of positions to cache.
        base: RoPE base frequency.
        scaling: optional YaRN spec, e.g.
            ``{"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}``.

    Returns:
        ``(cos, sin)`` each of shape ``(max_seq_len, head_dim // 2)``.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    # Frequencies for each of the head_dim//2 rotation pairs.
    arange = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / torch.pow(base, arange / head_dim)  # (head_dim//2,)

    if scaling is not None:
        scaling = dict(scaling)
        s_type = scaling.get("type", "yarn")
        if s_type == "yarn":
            inv_freq = _apply_yarn(inv_freq, head_dim, scaling)
        elif s_type in (None, "none"):
            pass
        else:
            raise ValueError(f"Unsupported rope_scaling type: {s_type!r}")

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim//2)
    return angles.cos(), angles.sin()


def _apply_yarn(
    inv_freq: torch.Tensor,
    head_dim: int,
    scaling: dict,
) -> torch.Tensor:
    """Apply the YaRN frequency rescaling (paper: https://arxiv.org/abs/2309.00071).

    Blends an interpolated frequency (scaled base) with the original using a
    linear ramp over the low-frequency rotation planes. Low frequencies keep
    the original spacing (allowing extrapolation to unseen long ranges) while
    high frequencies are interpolated.
    """
    factor = float(scaling.get("factor", 1.0))
    original_max = int(scaling.get("original_max_position_embeddings", 4096))
    base = float(scaling.get("base", 10000.0))
    beta_fast = float(scaling.get("beta_fast", 32.0))
    beta_slow = float(scaling.get("beta_slow", 1.0))
    if factor <= 0:
        raise ValueError("YaRN factor must be > 0")

    n = inv_freq.numel()  # head_dim // 2
    arange = torch.arange(0, head_dim, 2).float()

    # Frequencies if the base were scaled by `factor` (pure interpolation).
    freq_inter = 1.0 / torch.pow(base * factor, arange / head_dim)
    # Which rotation planes to ramp (wavelengths in the troublesome region).
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, head_dim, base, original_max
    )
    ramp = _yarn_linear_ramp_mask(low, high, n)  # (head_dim//2,) in [0, 1]
    freqs = freq_inter * ramp + inv_freq * (1.0 - ramp)
    extrapolation_factor = float(scaling.get("extrapolation_factor", 1.0))
    interpolation_factor = float(scaling.get("interpolation_factor", 1.0))
    return (
        extrapolation_factor * freqs
        + ((1.0 - extrapolation_factor) * inv_freq * interpolation_factor)
    )


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """Return the rotation-pair index corresponding to ``num_rotations``."""
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def _yarn_find_correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[int, int]:
    """Return the (low, high) rotation-pair indices to ramp, as in the paper."""
    low = math.floor(
        _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    )
    low = max(low, 0)
    high = min(high, dim - 1)
    return low, high


def _yarn_linear_ramp_mask(low: int, high: int, dim: int) -> torch.Tensor:
    """A linear ramp from 0 (below ``low``) to 1 (above ``high``)."""
    if low == high:
        high += 2  # ensure a usable ramp window
    ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(ramp, 0.0, 1.0)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply RoPE to ``x`` (..., T, head_dim) using per-pair cos/sin (T, D//2).

    When ``positions`` is given they gather the corresponding cos/sin caches
    (used for incremental decode). The head dimension is split into adjacent
    pairs; each pair ``(u, v)`` is rotated by angle ``theta``:
    ``(u', v') = (u*cos - v*sin, v*cos + u*sin)`` — an orthonormal rotation.
    """
    if positions is not None:
        cos = cos.index_select(0, positions)
        sin = sin.index_select(0, positions)
    num_pairs = cos.shape[-1]
    xr = x.reshape(*x.shape[:-1], num_pairs, 2)
    u = xr[..., 0]  # (..., T, num_pairs)
    v = xr[..., 1]
    # cos/sin are (T, num_pairs); they broadcast against (..., T, num_pairs).
    out_u = u * cos - v * sin
    out_v = v * cos + u * sin
    return torch.stack([out_u, out_v], dim=-1).reshape_as(x)


class RotaryEmbedding(nn.Module):
    """Module holding the RoPE cos/sin cache for the full model."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling = scaling
        cos, sin = precompute_freqs_cis(head_dim, max_seq_len, base, scaling)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the cos/sin needed for ``x``'s sequence length."""
        if positions is None:
            seq_len = x.shape[-2]
            return (
                self.cos_cached[:seq_len].to(x.dtype),
                self.sin_cached[:seq_len].to(x.dtype),
            )
        return (
            self.cos_cached.index_select(0, positions).to(x.dtype),
            self.sin_cached.index_select(0, positions).to(x.dtype),
        )
