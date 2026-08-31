"""Pre-canned model configs for every scale, sharing one code path.

These are the six configs that drive development, training and deployment.
They differ only in *numbers*, never in architecture. All are
:class:`~model.config.ModelConfig` dataclasses; the compute helper
(``configs.compute``) turns them into parameter/FLOP estimates.

For long-context scaling the configs opt into the **hybrid** attention scheme
(sliding window + periodic full attention) and YaRN RoPE scaling. The ``tiny``
config is the dev-sized model that runs on one consumer GPU.
"""
from __future__ import annotations

from model.config import ModelConfig
from tokenizer.vocab import TokenizerConfig


def tiny_config() -> ModelConfig:
    """Tiny dev model — fits on a single consumer GPU, CPU-instantiable."""
    return ModelConfig(
        vocab_size=1024,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_type="dense",
        intermediate_size=256,
        max_seq_len=512,
        attention_type="full",
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
    )


def tiny_tokenizer_config() -> TokenizerConfig:
    """Tokenizer sized to the tiny prototype model.

    The model↔tokenizer contract (``tokenizer/model_compat.py``) requires
    ``tokenizer_vocab_size <= model_vocab_size``: the tokenizer's ids map 1:1
    onto embedding rows and the model keeps the rest as padding. The tiny model
    holds **1024** rows, while ``TokenizerConfig`` defaults to 32768 — a
    tokenizer trained with the default overflows the tiny model (ids ≥ 1024
    cannot be embedded; specials would sit at 32764+).

    This config fixes ``vocab_size=1024``: 256 base byte tokens + 4 special
    tokens (ids 1020..1023, LLaMA-style at the top) + **764 merge slots**
    (``1024 - 256 - 4``). Training learns at most 764 merges, so every token id
    stays inside the model. Byte-level encoding needs no merges to be lossless,
    so English round-trips exactly even before/at any merge budget.
    """
    return TokenizerConfig(vocab_size=1024)


def small_config() -> ModelConfig:
    """Small research model (single-GPU / small-cluster fit)."""
    return ModelConfig(
        vocab_size=4096,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        num_kv_heads=4,
        head_dim=64,
        ffn_type="dense",
        intermediate_size=2048,
        max_seq_len=4096,
        attention_type="full",
        layer_norm_eps=1e-5,
    )


def medium_config() -> ModelConfig:
    """Medium model — multi-GPU training target for ablations."""
    return ModelConfig(
        vocab_size=16384,
        hidden_size=1024,
        num_layers=16,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=64,
        ffn_type="dense",
        intermediate_size=4096,
        max_seq_len=8192,
        attention_type="full",
        layer_norm_eps=1e-5,
    )


def large_config() -> ModelConfig:
    """Large dense model — long-context (hybrid attention) demonstration."""
    return ModelConfig(
        vocab_size=32768,
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="dense",
        intermediate_size=8192,
        max_seq_len=32768,
        attention_type="hybrid",
        sliding_window_size=8192,
        periodic_full_every=8,  # one full layer per 8
        attention_chunk_size=1024,
        rope_theta=500000.0,
        layer_norm_eps=1e-5,
    )


def scale_100b_config() -> ModelConfig:
    """~100B-total-parameter MoE (mid-scale training / inference target)."""
    return ModelConfig(
        vocab_size=65536,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="moe",
        num_experts=128,
        num_experts_per_tok=8,
        num_shared_experts=2,
        moe_intermediate_size=2048,
        grouped_experts=None,
        load_balance_coef=0.01,
        router_jitter_noise=0.0,
        max_seq_len=32768,
        attention_type="hybrid",
        sliding_window_size=8192,
        periodic_full_every=8,
        attention_chunk_size=1024,
        rope_theta=500000.0,
        layer_norm_eps=1e-5,
    )


def scale_400b_config() -> ModelConfig:
    """~400B-total / ~30B-active MoE with 128K context.

    Design (see docs/architecture.md): 46 layers, hidden 6144, GQA with 48
    query / 8 KV heads, 128 routed experts + 2 shared, top-6 routing. Tuning
    notes: the brief's "~256 experts, top-k ~6" and "~400B total / ~30B active"
    are mutually inconsistent — with 256 experts and top-6, active params would
    be only ~2.3% of total (~9B), not 30B. We keep top-k ~6 (as requested) and
    reduce the expert count to 128 so that 6 routed + 2 shared of 130 give the
    ~6% active fraction that yields ~30B active from ~400B total.
    """
    return ModelConfig(
        vocab_size=131072,          # 128K tokens
        hidden_size=6144,
        num_layers=46,
        num_attention_heads=48,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="moe",
        num_experts=128,
        num_experts_per_tok=6,
        num_shared_experts=2,
        moe_intermediate_size=3584,
        grouped_experts=None,
        load_balance_coef=0.01,
        router_jitter_noise=0.0,
        max_seq_len=131072,         # 128K context
        attention_type="hybrid",
        sliding_window_size=16384,  # 16K sliding window
        periodic_full_every=8,      # every 8th layer is full attention
        attention_chunk_size=1024,
        rope_theta=500000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "base": 500000.0,
        },
        layer_norm_eps=1e-5,
    )


ALL_PRESETS = {
    "tiny": tiny_config,
    "small": small_config,
    "medium": medium_config,
    "large": large_config,
    "100b": scale_100b_config,
    "400b": scale_400b_config,
}
