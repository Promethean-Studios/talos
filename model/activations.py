"""Activation functions, including the SwiGLU gated feed-forward block."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU) = x * sigmoid(x)."""
    return F.silu(x)


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward: ``(x @ gate) * silu(x @ up) @ down``.

    Args:
        hidden_size: input/output width.
        intermediate_size: width of the ``up``/``gate`` projections.
        bias: whether linear layers have biases (default False for LLMs).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
