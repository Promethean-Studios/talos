"""Tests for the attention backends: correctness, chunking, sliding window."""
import torch
import torch.nn.functional as F

from model.attention import PlainAttentionBackend, build_attention_backend
from model.block import SelfAttention
from model.config import ModelConfig


def _naive(q, k, v, mask=None, window_size=0, scale=None):
    """Reference attention for correctness checks."""
    scale = scale or (q.shape[-1] ** -0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is None:
        mask = torch.finfo(scores.dtype).min if False else None
        from model.masking import causal_mask

        mask = causal_mask(scores.shape[-1], window_size=window_size,
                           device=scores.device, dtype=scores.dtype)
    scores = scores + mask
    p = F.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def _qkv(batch=2, seq=8, heads=3, kv_heads=3, d=16):
    torch.manual_seed(3)
    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, kv_heads, seq, d)
    v = torch.randn(batch, kv_heads, seq, d)
    return q, k, v


def test_plain_matches_naive_causal():
    q, k, v = _qkv()
    backend = PlainAttentionBackend()
    out = backend(q, k, v, causal=True, scale=q.shape[-1] ** -0.5)
    ref = _naive(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)


def test_plain_matches_naive_sliding():
    q, k, v = _qkv(seq=16)
    backend = PlainAttentionBackend()
    out = backend(q, k, v, window_size=4, causal=True, scale=q.shape[-1] ** -0.5)
    ref = _naive(q, k, v, window_size=4)
    assert torch.allclose(out, ref, atol=1e-5)


def test_chunked_matches_nonchunked_causal():
    q, k, v = _qkv(seq=16)
    direct = PlainAttentionBackend()(q, k, v, causal=True, scale=q.shape[-1] ** -0.5)
    chunked = PlainAttentionBackend(chunk_size=4)(q, k, v, causal=True,
                                                  scale=q.shape[-1] ** -0.5)
    assert torch.allclose(chunked, direct, atol=1e-5)


def test_chunked_matches_nonchunked_sliding():
    q, k, v = _qkv(seq=16)
    direct = PlainAttentionBackend()(q, k, v, window_size=5, causal=True,
                                     scale=q.shape[-1] ** -0.5)
    chunked = PlainAttentionBackend(chunk_size=4)(q, k, v, window_size=5, causal=True,
                                                  scale=q.shape[-1] ** -0.5)
    assert torch.allclose(chunked, direct, atol=1e-5)


def test_chunked_bounded_memory_long_seq():
    # Long sequence with chunked attention: must run and be finite.
    q, k, v = _qkv(batch=1, seq=8192, heads=2, kv_heads=2, d=16)
    backend = PlainAttentionBackend(chunk_size=512)
    out = backend(q, k, v, window_size=256, causal=True)
    assert out.shape == (1, 2, 8192, 16)
    assert torch.isfinite(out).all()


def test_gqa_expansion():
    # SelfAttention must expand kv heads to query heads via repeat_interleave.
    cfg = ModelConfig(hidden_size=16, num_attention_heads=4, num_kv_heads=2,
                      head_dim=8, max_seq_len=32).derive()
    attn = SelfAttention(cfg, PlainAttentionBackend())
    x = torch.randn(1, 16, 16)
    out = attn(x, window_size=0)
    assert out.shape == (1, 16, 16)
    # Ratio of heads = 4/2 = 2 -> each kv row duplicated twice.
    assert attn.gqa_ratio == 2


def test_plain_requires_4d():
    backend = PlainAttentionBackend()
    q = torch.randn(2, 4, 8)
    try:
        backend(q, q, q, causal=True)
        assert False, "expected ValueError for non-4D input"
    except ValueError:
        pass


def test_build_attention_backend():
    assert isinstance(build_attention_backend("plain"), PlainAttentionBackend)
    # 'auto' should fall back to plain when flash-attn is absent.
    assert isinstance(build_attention_backend("auto"), PlainAttentionBackend)
