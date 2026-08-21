"""Tests for attention mask construction (causal, sliding-window, ranges)."""
import torch

from model.masking import NEG_INF, causal_mask, key_ranges


def test_causal_mask_lower_triangular():
    m = causal_mask(5)
    assert m.shape == (5, 5)
    assert (m == 0).diagonal().all()
    # Above diagonal -> -inf, diagonal and below -> 0.
    upper = torch.triu(torch.ones_like(m, dtype=torch.bool), diagonal=1)
    assert (m[upper] < -1e30).all()
    assert (m[~upper] == 0).all()


def test_sliding_window_mask_band():
    m = causal_mask(8, window_size=3)
    # row i, col j allowed iff j <= i and i - j < 3
    for i in range(8):
        for j in range(8):
            allowed = j <= i and (i - j) < 3
            if allowed:
                assert m[i, j].item() == 0.0, (i, j)
            else:
                assert m[i, j].item() < -1e30, (i, j)


def test_band_is_narrower_than_full():
    m_full = causal_mask(16)
    m_band = causal_mask(16, window_size=4)
    # band has fewer non-suppressed entries
    assert (m_band == NEG_INF).sum() > (m_full == NEG_INF).sum()


def test_key_ranges_causal():
    lo, hi = key_ranges(6, start_pos=0, window_size=0, chunk_size=2)
    assert lo.tolist() == [0, 0, 0, 0, 0, 0]
    assert hi.tolist() == [1, 2, 3, 4, 5, 6]


def test_key_ranges_sliding():
    lo, hi = key_ranges(6, start_pos=0, window_size=3, chunk_size=2)
    assert lo.tolist() == [0, 0, 0, 1, 2, 3]
    assert hi.tolist() == [1, 2, 3, 4, 5, 6]


def test_key_ranges_decode_offset():
    lo, hi = key_ranges(1, start_pos=10, window_size=3, chunk_size=2)
    assert lo.tolist() == [8]
    assert hi.tolist() == [11]
