"""Feed-forward network interface and the dense SwiGLU block.

The key design decision: **dense and MoE feed-forwards share one interface** so
a configuration can pick either per size with zero code change. Both return
``(output, aux_loss)`` where ``aux_loss`` is 0 for the dense block.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from model.activations import SwiGLU


class FFNInterface(ABC):
    """Uniform feed-forward contract.

    ``forward(x)`` returns ``(output, aux_loss)``. For a dense FFN the auxiliary
    loss is a zero tensor; for MoE it is the (weighted) load-balancing loss.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the feed-forward to ``x`` (..., hidden_size)."""
        raise NotImplementedError


class DenseFFNBlock(nn.Module, FFNInterface):
    """A standard dense SwiGLU feed-forward."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.swiglu = SwiGLU(hidden_size, intermediate_size, dropout=dropout)
        # Aux-loss term for a dense block is always zero (kept as a buffer so it
        # has the right device/dtype without being a trainable parameter).
        self.register_buffer("zero_aux", torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.swiglu(x), self.zero_aux

    def __repr__(self) -> str:
        return f"DenseFFNBlock(hidden={self.swiglu.hidden_size}, " \
               f"intermediate={self.swiglu.intermediate_size})"
