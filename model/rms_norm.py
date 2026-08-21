"""Root-mean-square layer normalization (RMSNorm).

RMSNorm pre-norm is used everywhere in Forge (attention pre-norm and the final
head norm). Unlike LayerNorm it has no mean-centering or bias term — only a
per-channel learnable gain, which makes it cheap and stable at scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension with a learned per-dimension gain.

    ``x  ->  x / sqrt(mean(x^2) + eps) * weight``
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden_size)
        dtype = x.dtype
        # Compute variance in float32 for numerical stability, then cast back.
        x_f = x.to(torch.float32)
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.to(torch.float32)).to(dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


def param_count(hidden_size: int) -> int:
    """Number of trainable parameters in one RMSNorm layer (the gain)."""
    return hidden_size
