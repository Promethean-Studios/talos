"""Forge model stack: config, layers, attention backends, MoE and the GPT model."""
from model.config import ModelConfig
from model.gpt import ForgeGPT
from model.cache import KVCache
from model.rms_norm import RMSNorm
from model.rotary import RotaryEmbedding, apply_rotary_pos_emb
from model.activations import SwiGLU
from model.ffn import DenseFFNBlock, FFNInterface
from model.moe import MoEFFNBlock, load_balancing_loss
from model.attention import (
    AttentionInterface,
    PlainAttentionBackend,
    FlashAttentionBackend,
    build_attention_backend,
)
from model.block import DecoderLayer

__all__ = [
    "ModelConfig",
    "ForgeGPT",
    "KVCache",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "SwiGLU",
    "DenseFFNBlock",
    "FFNInterface",
    "MoEFFNBlock",
    "load_balancing_loss",
    "AttentionInterface",
    "PlainAttentionBackend",
    "FlashAttentionBackend",
    "build_attention_backend",
    "DecoderLayer",
]

__version__ = "0.1.0"
