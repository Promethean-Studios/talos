"""Tests for the incremental KV cache (prefill + decode)."""
import torch

from model.cache import KVCache


def make_cache(**kw):
    base = dict(num_layers=2, num_kv_heads=4, head_dim=8, max_seq_len=16, batch_size=1)
    base.update(kw)
    return KVCache(**base)


def test_prefill_then_decode_read():
    c = make_cache()
    k = torch.randn(1, 4, 5, 8)
    v = torch.randn(1, 4, 5, 8)
    ck, cv = c.update(0, k, v, start_pos=0)
    assert ck.shape == v.shape == (1, 4, 5, 8)
    assert c.length == 5
    # decode one more token
    k1 = torch.randn(1, 4, 1, 8)
    v1 = torch.randn(1, 4, 1, 8)
    ck, cv = c.update(0, k1, v1, start_pos=5)
    assert ck.shape == (1, 4, 6, 8)
    assert c.length == 6
    # the token at position 5 equals the newly written one
    assert torch.allclose(ck[:, :, 5:6, :], k1)


def test_sliding_window_get():
    c = make_cache(max_seq_len=32)
    # write 20 tokens
    k = torch.randn(1, 4, 20, 8)
    c.update(0, k, torch.randn(1, 4, 20, 8), start_pos=0)
    ck, cv = c.get_window(0, cur_len=20, window_size=8)
    assert ck.shape == (1, 4, 8, 8)
    # and the last window equals the last 8 written
    assert torch.allclose(ck, k[:, :, -8:, :])


def test_cache_reset():
    c = make_cache()
    c.update(0, torch.randn(1, 4, 3, 8), torch.randn(1, 4, 3, 8), 0)
    assert c.length == 3
    c.reset()
    assert c.length == 0
    assert c.k_cache.abs().sum().item() == 0.0


def test_cache_layer_independence():
    c = make_cache()
    c.update(0, torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8), 0)
    c.update(1, torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8), 0)
    # layer 0 length and layer 1 length both tracked
    assert c.length == 2
    assert (c.k_cache[0].abs() + c.k_cache[1].abs()).sum() > 0


def test_cache_overflow_raises():
    c = make_cache(max_seq_len=5)
    try:
        c.update(0, torch.randn(1, 4, 6, 8), torch.randn(1, 4, 6, 8), 0)
        assert False, "expected ValueError on overflow"
    except ValueError:
        pass
