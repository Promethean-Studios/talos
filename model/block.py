"""Decoder layer: pre-norm attention + pre-norm feed-forward, dense or MoE."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.activations import SwiGLU
from model.attention import AttentionInterface
from model.cache import KVCache
from model.config import ModelConfig
from model.ffn import DenseFFNBlock, FFNInterface
from model.moe import MoEFFNBlock
from model.rms_norm import RMSNorm
from model.rotary import RotaryEmbedding, apply_rotary_pos_emb


class SelfAttention(nn.Module):
    """Grouped-query attention with RoPE and a pluggable execution backend.

    K/V are projected with ``num_kv_heads`` (GQA) and cached as such; they are
    expanded to the query head count immediately before the attention kernel.
    """

    def __init__(self, config: ModelConfig, backend: AttentionInterface) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.gqa_ratio = self.num_heads // self.num_kv_heads
        self.backend = backend
        self.dropout = nn.Dropout(config.dropout)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            base=config.rope_theta,
            scaling=config.rope_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        window_size: int = 0,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        head_dim = self.head_dim

        q = self.q_proj(x).view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, head_dim).transpose(1, 2)

        # RotaryEmbedding already gathers the cache by `positions` (decode) or
        # slices it to the current length (prefill), so cos/sin are aligned to
        # `q`/`k`'s sequence dim — no further gather here.
        cos, sin = self.rotary(q, positions)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if cache is not None:
            start_pos = int(positions[0]) if positions is not None else 0
            k, v = cache.update(layer_idx, k, v, start_pos)

        # Expand GQA K/V rows up to the query head count.
        if self.gqa_ratio > 1:
            k = k.repeat_interleave(self.gqa_ratio, dim=1)
            v = v.repeat_interleave(self.gqa_ratio, dim=1)

        out = self.backend(
            q, k, v,
            window_size=window_size,
            causal=True,
            scale=head_dim ** -0.5,
        )
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * head_dim)
        return self.o_proj(self.dropout(out))


def build_ffn(config: ModelConfig) -> FFNInterface:
    """Build the correct feed-forward block for the config (dense or MoE)."""
    if config.ffn_type == "moe":
        return MoEFFNBlock(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            num_shared_experts=config.num_shared_experts,
            grouped_experts=config.grouped_experts,
            load_balance_coef=config.load_balance_coef,
            router_jitter_noise=config.router_jitter_noise,
            dropout=config.dropout,
        )
    return DenseFFNBlock(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
    )


class DecoderLayer(nn.Module):
    """One transformer decoder layer: pre-norm attention + pre-norm FFN.

    Implements the hybrid sliding-window / periodic full-attention scheme: each
    layer knows whether it is a *full* attention layer (``window_size == 0``)
    or a *sliding-window* layer (``window_size > 0``).
    """

    def __init__(self, config: ModelConfig, backend: AttentionInterface) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attention = SelfAttention(config, backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ffn = build_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        window_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(hidden, aux_loss)`` where ``aux_loss`` is the MoE loss."""
        residual = x
        h = self.input_layernorm(x)
        h = self.attention(h, positions=positions, cache=cache,
                           layer_idx=layer_idx, window_size=window_size)
        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        ffn_out, aux_loss = self.ffn(h)
        x = residual + ffn_out
        return x, aux_loss
