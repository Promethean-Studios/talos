"""Model configuration for Talos.

A single :class:`ModelConfig` describes every Talos model from the tiny dev
config up to the ~400B MoE. The same code path in :mod:`model` is instantiated
from any of these — only the numbers change.

This module deliberately has *no* torch dependency so that config tools
(e.g. ``scripts/print_configs.py``) can run without importing the model stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Hyper-parameters for a Talos decoder-only MoE transformer.

    All dimensions are derived from a small set of knobs. :meth:`derive`
    fills in any fields that can be computed from the rest, so callers can
    either set every field explicitly or pass the minimal set and call
    ``derive()``.

    Field groups:
      * vocabulary / sequence: ``vocab_size``, ``max_seq_len``
      * model shape: ``hidden_size``, ``num_layers``, ``head_dim``
      * GQA attention: ``num_attention_heads``, ``num_kv_heads``
      * feed-forward: ``ffn_type`` (``"dense"`` | ``"moe"``), ``intermediate_size``
      * MoE: ``num_experts``, ``num_experts_per_tok``, ``num_shared_experts``,
              ``grouped_experts``, ``load_balance_coef``
      * RoPE: ``rope_theta``, ``rope_scaling`` (YaRN spec)
      * attention scheme: ``attention_type`` (``"full"``|``"swa"``|``"hybrid"``),
              ``sliding_window_size``, ``periodic_full_every``,
              ``full_attention_layers``, ``attention_chunk_size``
      * regularization / misc: ``dropout``, ``layer_norm_eps``,
              ``tie_word_embeddings``, ``initializer_range``
    """

    # --- vocabulary / sequence -------------------------------------------------
    vocab_size: int = 32_000
    max_seq_len: int = 2048

    # --- model shape -----------------------------------------------------------
    hidden_size: int = 4096
    num_layers: int = 32

    # --- GQA attention ----------------------------------------------------------
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: Optional[int] = None  # derived: hidden_size // num_attention_heads

    # --- feed-forward / MoE -----------------------------------------------------
    ffn_type: str = "dense"  # "dense" | "moe"
    intermediate_size: Optional[int] = None  # dense FFN width; derived 4*hidden

    # MoE (used only when ffn_type == "moe")
    num_experts: int = 0
    num_experts_per_tok: int = 0  # top-k routed experts
    num_shared_experts: int = 0
    moe_intermediate_size: Optional[int] = None  # per-expert FFN width
    grouped_experts: Optional[int] = None  # number of expert groups, if grouping
    load_balance_coef: float = 0.01  # auxiliary load-balance loss weight
    router_jitter_noise: float = 0.0  # >0 enables training-time routing noise

    # --- RoPE -------------------------------------------------------------------
    rope_theta: float = 10_000.0
    rope_scaling: Optional[dict] = None  # e.g. {"type": "yarn", "factor": 8.0,
    #                                     #  "original_max_position_embeddings": 4096}

    # --- attention scheme ---------------------------------------------------------
    # "full"   -> every layer is full causal attention (short-context default)
    # "swa"    -> every layer is sliding-window attention
    # "hybrid" -> sliding-window layers plus periodic full-attention layers
    attention_type: str = "full"
    sliding_window_size: int = 0  # window width when swa/hybrid
    periodic_full_every: int = 0  # every K-th layer uses full attention (0 = none)
    full_attention_layers: Optional[List[int]] = None  # explicit override list
    # Chunk size for the functional (non-FlashAttention) backend. When > 0 the
    # plain backend processes queries in blocks of this size, bounding peak
    # memory to O(chunk * seq) instead of O(seq^2). 0 selects clamping for short
    # sequences automatically at runtime.
    attention_chunk_size: int = 0

    # --- regularization / misc ----------------------------------------------------
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # --- helpers ------------------------------------------------------------------
    def derive(self) -> "ModelConfig":
        """Fill derived fields and validate the config.

        Mutates and returns ``self``. Call once before use.
        """
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0, (
                "hidden_size must be divisible by num_attention_heads"
            )
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.ffn_type == "dense" and self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.ffn_type == "moe" and self.moe_intermediate_size is None:
            # Per-expert FFN width; total MoE width (num_experts * this) is far
            # larger than a dense FFN (4*hidden), which is what makes MoE
            # "wider" while keeping only top-k experts active per token.
            self.moe_intermediate_size = self.hidden_size
        self._validate()
        return self

    def _validate(self) -> None:
        if self.ffn_type not in ("dense", "moe"):
            raise ValueError(f"ffn_type must be 'dense' or 'moe', got {self.ffn_type!r}")
        if self.attention_type not in ("full", "swa", "hybrid"):
            raise ValueError(
                f"attention_type must be full|swa|hybrid, got {self.attention_type!r}"
            )
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                "GQA requires num_attention_heads % num_kv_heads == 0 "
                f"(got {self.num_attention_heads} / {self.num_kv_heads})"
            )
        if self.ffn_type == "moe":
            if self.num_experts <= 0:
                raise ValueError("MoE config requires num_experts > 0")
            if self.num_experts_per_tok <= 0:
                raise ValueError("MoE config requires num_experts_per_tok > 0 (top-k)")
            if self.num_experts_per_tok > self.num_experts:
                raise ValueError("top-k (num_experts_per_tok) cannot exceed num_experts")
            if self.num_shared_experts < 0:
                raise ValueError("num_shared_experts must be >= 0")
            if self.grouped_experts is not None and self.grouped_experts <= 0:
                raise ValueError("grouped_experts must be > 0 when set")
        if self.attention_type == "hybrid" and self.sliding_window_size <= 0:
            raise ValueError("hybrid attention requires sliding_window_size > 0")
        if self.attention_type == "swa" and self.sliding_window_size <= 0:
            raise ValueError("swa attention requires sliding_window_size > 0")
        if self.max_seq_len <= 0 or self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("max_seq_len, vocab_size and hidden_size must be positive")

    # ------------------------------------------------------------------
    def full_layer_indices(self) -> Tuple[int, ...]:
        """Return layer indices (0-based) that use full (non-sliding) attention.

        Works for ``full``, ``swa`` and ``hybrid`` attention types.
        """
        if self.attention_type == "full":
            return tuple(range(self.num_layers))
        if self.attention_type == "swa":
            return ()
        # hybrid: periodic full-attention layers, unless explicitly overridden.
        if self.full_attention_layers is not None:
            return tuple(self.full_attention_layers)
        if self.periodic_full_every <= 0:
            return ()
        return tuple(i for i in range(self.num_layers) if i % self.periodic_full_every == 0)

    def layer_window_size(self, layer_idx: int) -> int:
        """Return the sliding-window width used by ``layer_idx``.

        Full layers and full-attention models return 0 (meaning: no banding).
        """
        if layer_idx in self.full_layer_indices():
            return 0
        return self.sliding_window_size

    # ------------------------------------------------------------------
    def param_count_breakdown(self) -> dict:
        """Return a rough analytic total-parameter breakdown (integers).

        This is a lightweight, dependency-free estimate used by config tooling;
        ``configs.compute`` provides the full estimate (weights memory, active
        params, per-token FLOPs). Numbers assume un-tied embeddings.
        """
        from model.rms_norm import param_count  # local import to avoid torch at module load

        qkv, _ = self._attention_dims()
        d = self.derive() if self.head_dim is None else self
        head_dim = d.head_dim
        attn_per_layer = (
            hidden(self.hidden_size, self.num_attention_heads * head_dim)
            + hidden(self.hidden_size, self.num_kv_heads * head_dim)
            + hidden(self.hidden_size, self.num_kv_heads * head_dim)
            + hidden(self.num_attention_heads * head_dim, self.hidden_size)
        )
        norms_per_layer = 2 * self.hidden_size
        ffn_per_layer = self._ffn_param_count()
        embedding = self.vocab_size * self.hidden_size
        lm_head = 0 if self.tie_word_embeddings else embedding
        total = (
            embedding
            + lm_head
            + self.num_layers * (attn_per_layer + norms_per_layer + ffn_per_layer)
            + self.hidden_size  # final norm
        )
        return {
            "embedding": embedding,
            "lm_head": lm_head,
            "per_layer_attention": attn_per_layer,
            "per_layer_ffn": ffn_per_layer,
            "total": total,
        }

    def _ffn_param_count(self) -> int:
        if self.ffn_type == "dense":
            return 3 * self.hidden_size * self.intermediate_size
        # MoE: every expert contributes an up+gate (2*H*I) and down (I*H).
        per_expert = 3 * self.hidden_size * self.moe_intermediate_size
        return per_expert * (self.num_experts + self.num_shared_experts)

    def active_ffn_param_count(self) -> int:
        """Params in the feed-forward actually executed per token."""
        if self.ffn_type == "dense":
            return 3 * self.hidden_size * self.intermediate_size
        per_expert = 3 * self.hidden_size * self.moe_intermediate_size
        active_experts = self.num_experts_per_tok + self.num_shared_experts
        return per_expert * active_experts

    def _attention_dims(self) -> Tuple[int, int]:
        d = self.derive() if self.head_dim is None else self
        return d.num_attention_heads * d.head_dim, d.num_kv_heads * d.head_dim

    def __str__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ModelConfig(type={self.ffn_type}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, heads={self.num_attention_heads}, "
            f"kv_heads={self.num_kv_heads}, vocab={self.vocab_size}, "
            f"seq={self.max_seq_len})"
        )


def hidden(in_features: int, out_features: int) -> int:
    """Number of parameters in a ``bias=False`` linear layer."""
    return in_features * out_features


def configs_from_iterable(iterable: Iterable[ModelConfig]) -> List[ModelConfig]:
    """Derive every config in an iterable and return the list."""
    return [c.derive() for c in iterable]
