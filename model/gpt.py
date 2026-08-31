"""The TalosGPT model: a decoder-only MoE/gated transformer.

One code path builds a model from a :class:`ModelConfig` whether it is the tiny
dev model or the ~400B MoE. The model supports both a plain training forward
(all tokens, full/sliding attention) and incremental inference via a
:class:`KVCache`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.attention import AttentionInterface, build_attention_backend
from model.block import DecoderLayer
from model.cache import KVCache
from model.config import ModelConfig
from model.rms_norm import RMSNorm


class TalosGPT(nn.Module):
    """Decoder-only transformer with GQA attention, RoPE, and dense-or-MoE FFN."""

    def __init__(self, config: ModelConfig, attention_backend: Optional[AttentionInterface] = None) -> None:
        super().__init__()
        self.config = config
        if config.head_dim is None:
            config.derive()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        backend = attention_backend or build_attention_backend(
            backend="auto", chunk_size=config.attention_chunk_size
        )
        self.backend = backend
        self.layers = nn.ModuleList(
            [DecoderLayer(config, backend) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)

        self.apply(self._init_weights)
        # Scale embedding output to reduce early training instability.
        self.embed_scale = math.sqrt(config.hidden_size)

        if self.lm_head is not None:
            # Include the lm_head in weight init for consistency.
            self._init_weights(self.lm_head)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            if hasattr(module, "weight"):
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Forward pass.

        Args:
            input_ids: ``(batch, seq)`` token ids.
            position_ids: optional ``(batch, seq)`` positions (recommended for
                decode). Defaults to ``[past_len, past_len + seq)``.
            use_cache: whether to update ``cache`` with K/V states.
            cache: an existing :class:`KVCache` (prefetched for prefill, or a
                running cache reused across decode steps).

        Returns:
            ``(logits, cache)`` — logits ``(batch, seq, vocab)`` and the
            (possibly newly created) KV cache.
        """
        batch, seq = input_ids.shape
        self._validate_seq(seq)

        if position_ids is None:
            if cache is not None and use_cache:
                start = cache.last_len()
            else:
                start = 0
            position_ids = torch.arange(start, start + seq, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if use_cache and cache is None:
            # The cache stores projected K/V activations, so its dtype must be
            # the model's compute dtype — NOT input_ids' dtype (long token ids).
            cache = self.new_cache(
                batch, input_ids.device, self.embed_tokens.weight.dtype
            )

        hidden = self.embed_tokens(input_ids) * self.embed_scale

        aux_loss = hidden.new_zeros(())
        for idx, layer in enumerate(self.layers):
            window_size = self.config.layer_window_size(idx)
            hidden, layer_aux = layer(
                hidden,
                positions=position_ids[0],
                cache=cache if use_cache else None,
                layer_idx=idx,
                window_size=window_size,
            )
            aux_loss = aux_loss + layer_aux

        hidden = self.final_norm(hidden)
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = torch.matmul(hidden, self.embed_tokens.weight.t())
        return logits, cache

    def new_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        """Create a fresh KV cache shaped for this model."""
        return KVCache(
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

    def _validate_seq(self, seq: int) -> None:
        if seq > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Number of parameters in this model."""
        return sum(
            p.numel()
            for p in self.parameters()
            if not trainable_only or p.requires_grad
        )

    def __repr__(self) -> str:
        return (
            f"TalosGPT(type={self.config.ffn_type}, layers={self.config.num_layers}, "
            f"hidden={self.config.hidden_size}, heads={self.config.num_attention_heads}, "
            f"kv_heads={self.config.num_kv_heads}, params={self.num_parameters():,}, "
            f"backend={type(self.backend).__name__})"
        )
