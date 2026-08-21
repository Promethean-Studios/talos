"""Tests for the dense-vs-MoE shared FFN interface and the decoder layer."""
import torch

from model.block import DecoderLayer, build_ffn
from model.config import ModelConfig
from model.ffn import DenseFFNBlock
from model.moe import MoEFFNBlock


def _ffn_cfg(**kw) -> ModelConfig:
    base = dict(hidden_size=16, num_attention_heads=4, num_kv_heads=4, head_dim=4,
                max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base).derive()


def test_build_ffn_dense():
    cfg = _ffn_cfg(ffn_type="dense", intermediate_size=64)
    ffn = build_ffn(cfg)
    assert isinstance(ffn, DenseFFNBlock)
    x = torch.randn(2, 16)
    out, aux = ffn(x)
    assert out.shape == (2, 16)
    assert aux.item() == 0.0  # dense has no aux loss


def test_build_ffn_moe():
    cfg = _ffn_cfg(ffn_type="moe", num_experts=8, num_experts_per_tok=2,
                   num_shared_experts=1, moe_intermediate_size=32)
    ffn = build_ffn(cfg)
    assert isinstance(ffn, MoEFFNBlock)
    x = torch.randn(2, 16)
    out, aux = ffn(x)
    assert out.shape == (2, 16)
    assert aux.item() > 0.0


def test_dense_and_moe_share_interface():
    # Both must expose forward(x) -> (out, aux)
    for ffn in (DenseFFNBlock(16, 64), MoEFFNBlock(16, 8, 2, 32, 1)):
        out, aux = ffn(torch.randn(3, 16))
        assert out.shape == (3, 16)
        assert torch.isfinite(aux)


def test_decoder_layer_full_attention():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_attention_heads=4,
                      num_kv_heads=2, head_dim=8, ffn_type="dense",
                      intermediate_size=128, max_seq_len=32,
                      attention_type="full").derive()
    layer = DecoderLayer(cfg, __import__("model.attention", fromlist=["*"]).PlainAttentionBackend())
    x = torch.randn(1, 16, 32)
    out, aux = layer(x, layer_idx=0, window_size=0)
    assert out.shape == x.shape
    assert aux.item() == 0.0


def test_decoder_layer_sliding_window():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_attention_heads=4,
                      num_kv_heads=2, head_dim=8, ffn_type="moe",
                      num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
                      moe_intermediate_size=64, max_seq_len=32,
                      attention_type="hybrid", sliding_window_size=8,
                      periodic_full_every=2).derive()
    from model.attention import PlainAttentionBackend

    layer = DecoderLayer(cfg, PlainAttentionBackend())
    x = torch.randn(1, 20, 32)
    out, aux = layer(x, layer_idx=1, window_size=8)  # sliding layer
    assert out.shape == x.shape
    assert aux.item() > 0.0  # MoE aux loss present
