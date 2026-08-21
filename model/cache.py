"""Incremental KV (key/value) cache for prefill + decode inference.

The cache stores every layer's projected K/V embeddings so that during
autoregressive decoding we only compute the current token's attention rather
than re-running the whole prefix. Shapes follow GQA: K/V have ``num_kv_heads``
(not ``num_attention_heads``) rows, giving the memory savings GQA is designed
for. Sliding-window decode is handled by truncating the returned slice to the
last ``window_size`` positions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class KVCache(nn.Module):
    """Flat, pre-allocated KV cache for all layers.

    Layout: ``(num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)``.

    The cache is a set of non-persistent buffers. It is *not* part of the
    model's trainable state; ``reset()`` must be called before a new sequence.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self._dtype = dtype
        self._device = device

        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        # Current used length (0 = empty). Updated on every write.
        self.register_buffer("_length", torch.zeros(1, dtype=torch.long), persistent=False)

    @property
    def length(self) -> int:
        """Number of cached tokens (0 if the cache is empty)."""
        return int(self._length[0])

    def reset(self) -> None:
        """Zero the cache (safe to call between sequences)."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._length.zero_()

    def last_len(self) -> int:
        """Alias for :attr:`length` (used by the model for decode positions)."""
        return self.length

    def update(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write a new key/value block and return the full cached prefix.

        Args:
            layer: layer index to write into.
            key: ``(batch, num_kv_heads, T, head_dim)`` new keys.
            value: ``(batch, num_kv_heads, T, head_dim)`` new values.
            start_pos: absolute starting position of the new block along the
                sequence dim (0 during prefill, previous length during decode).

        Returns:
            ``(keys, values)`` of shape ``(batch, num_kv_heads, cur_len, head_dim)``
            being the whole cached prefix up to ``start_pos + T``.
        """
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(f"layer {layer} out of range [0, {self.num_layers})")
        batch, kv, t, dim = key.shape
        if kv != self.num_kv_heads:
            raise ValueError(
                f"expected {self.num_kv_heads} kv heads, got {kv}"
            )
        if dim != self.head_dim:
            raise ValueError(f"expected head_dim {self.head_dim}, got {dim}")
        if batch != self.batch_size:
            raise ValueError(f"cache holds batch {self.batch_size}, got batch {batch}")
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"sequence end {end} exceeds max_seq_len {self.max_seq_len}; "
                "call cache.reset() or increase max_seq_len"
            )
        self.k_cache[layer, :, :, start_pos:end, :] = key
        self.v_cache[layer, :, :, start_pos:end, :] = value
        if end > self._length[0]:
            self._length[0] = end
        return (
            self.k_cache[layer, :, :, :end, :],
            self.v_cache[layer, :, :, :end, :],
        )

    def get(self, layer: int, cur_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached keys/values for ``layer`` up to ``cur_len``."""
        return (
            self.k_cache[layer, :, :, :cur_len, :],
            self.v_cache[layer, :, :, :cur_len, :],
        )

    def get_window(
        self, layer: int, cur_len: int, window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the last ``window_size`` cached keys/values for decode."""
        lo = max(0, cur_len - window_size)
        return (
            self.k_cache[layer, :, :, lo:cur_len, :],
            self.v_cache[layer, :, :, lo:cur_len, :],
        )

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"max_seq={self.max_seq_len}, batch={self.batch_size}, "
            f"head_dim={self.head_dim})"
        )
