"""Tests for RoPE and YaRN scaling."""
import torch

from model.rotary import apply_rotary_pos_emb, precompute_freqs_cis


def test_freqs_shapes():
    cos, sin = precompute_freqs_cis(8, 64, base=10000.0)
    assert cos.shape == (64, 4)
    assert sin.shape == (64, 4)


def test_rotary_preserves_norm():
    # The per-pair rotation is orthonormal, so it must preserve norms exactly.
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32, 16)  # (B, H, T, D)
    cos, sin = precompute_freqs_cis(16, 32, base=10000.0)
    out = apply_rotary_pos_emb(x, cos, sin)
    assert torch.allclose(x.pow(2).sum(-1), out.pow(2).sum(-1), atol=1e-5)


def test_rotary_preserves_dot_products_on_diagonal():
    # Translation invariance: rotating the same vector at positions i and j,
    # the pairwise dot product should depend only on (i - j), so it is constant
    # along diagonals of the affinity matrix.
    torch.manual_seed(0)
    vec = torch.randn(16)
    t = 32
    x = vec.view(1, 1, 1, 16).expand(1, 1, t, 16)  # same vec at every pos
    cos, sin = precompute_freqs_cis(16, t, base=10000.0)
    rot = apply_rotary_pos_emb(x, cos, sin)[0, 0]  # (t, 16)
    aff = rot @ rot.t()  # (t, t)
    for i in range(t - 1):
        for j in range(t - 1):
            assert abs(aff[i, j] - aff[i + 1, j + 1]).item() < 1e-4, (i, j)


def test_positions_lookup_decode():
    # Gather-by-positions should match slicing-by-range for the same positions.
    x = torch.randn(2, 2, 8, 16)
    cos, sin = precompute_freqs_cis(16, 32, base=10000.0)
    positions = torch.arange(3, 11)
    out_gather = apply_rotary_pos_emb(x, cos, sin, positions)
    out_slice = apply_rotary_pos_emb(x, cos[3:11], sin[3:11])
    assert torch.allclose(out_gather, out_slice, atol=1e-6)


def test_yarn_scale_changes_frequencies():
    cos_plain, _ = precompute_freqs_cis(16, 128, base=10000.0)
    cos_yarn, _ = precompute_freqs_cis(
        16, 128, base=10000.0,
        scaling={"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 128},
    )
    # With YaRN the long-range spacing is wider, so caches differ materially.
    assert not torch.allclose(cos_plain, cos_yarn)
    assert torch.isfinite(cos_yarn).all()


def test_yarn_invalid_type():
    import pytest

    with pytest.raises(ValueError):
        precompute_freqs_cis(16, 32, scaling={"type": "bogus", "factor": 2.0})


def test_odd_head_dim_raises():
    import pytest

    with pytest.raises(ValueError):
        precompute_freqs_cis(7, 32)
