"""Tests for the notebook's torch glue (``talos_torch_utils``).

Tiny CPU tensors only: masked/chunked loss, precision helpers, gradient clipping
and batching. Skipped when torch is absent (CI installs CPU torch, so it runs).
"""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import talos_torch_utils as tu  # noqa: E402


def _logits(batch=2, seq=5, vocab=11, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=generator)


def _targets(batch=2, seq=5, vocab=11, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=generator)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def test_unmasked_loss_matches_plain_cross_entropy():
    logits, targets = _logits(), _targets()
    expected = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    got = tu.masked_cross_entropy(logits, targets)
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_all_ones_mask_matches_plain_cross_entropy():
    logits, targets = _logits(), _targets()
    mask = torch.ones_like(targets)
    expected = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    assert float(tu.masked_cross_entropy(logits, targets, mask)) == pytest.approx(
        float(expected), rel=1e-6)


def test_mask_is_a_mean_over_supervised_positions_only():
    logits, targets = _logits(), _targets()
    mask = torch.zeros_like(targets)
    mask[:, :2] = 1                     # only the first two positions count
    per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1), reduction="none").reshape(targets.shape)
    expected = per_token[mask.bool()].mean()
    assert float(tu.masked_cross_entropy(logits, targets, mask)) == pytest.approx(
        float(expected), rel=1e-6)


def test_chunking_does_not_change_the_loss():
    logits, targets = _logits(batch=3, seq=7), _targets(batch=3, seq=7)
    mask = torch.ones_like(targets)
    mask[:, :3] = 0
    whole = tu.masked_cross_entropy(logits, targets, mask, chunk_tokens=10_000)
    chunked = tu.masked_cross_entropy(logits, targets, mask, chunk_tokens=4)
    assert float(whole) == pytest.approx(float(chunked), rel=1e-6)


def test_an_all_zero_mask_is_an_error_not_a_nan():
    logits, targets = _logits(), _targets()
    with pytest.raises(ValueError, match="no supervised tokens"):
        tu.masked_cross_entropy(logits, targets, torch.zeros_like(targets))


def test_shape_mismatches_are_rejected():
    logits, targets = _logits(), _targets()
    with pytest.raises(ValueError):
        tu.masked_cross_entropy(logits, targets[:, :-1])
    with pytest.raises(ValueError):
        tu.masked_cross_entropy(logits, targets, torch.ones(2, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        tu.masked_cross_entropy(logits.reshape(-1, logits.size(-1)), targets)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def test_autocast_kwargs_per_precision():
    assert tu.autocast_kwargs("cuda", "bf16")["dtype"] is torch.bfloat16
    assert tu.autocast_kwargs("cuda", "fp16")["dtype"] is torch.float16
    assert tu.autocast_kwargs("cuda", "fp32")["enabled"] is False
    with pytest.raises(ValueError):
        tu.autocast_kwargs("cuda", "fp8")


def test_grad_scaler_only_for_fp16():
    assert tu.make_grad_scaler(enabled=False) is None
    scaler = tu.make_grad_scaler(enabled=True)
    assert scaler is not None


# ---------------------------------------------------------------------------
# Optimizer and clipping
# ---------------------------------------------------------------------------
def test_build_optimizer_falls_back_to_plain_adamw():
    model = torch.nn.Linear(3, 2)
    optimizer, note = tu.build_optimizer(model.parameters(), lr=1e-3, use_8bit_adam=False)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert "plain AdamW" in note
    with pytest.raises(ValueError):
        tu.build_optimizer([], lr=1e-3)


def test_clip_and_norm_returns_the_pre_clip_norm():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    norm = tu.clip_and_norm([parameter], max_norm=1.0)
    assert norm == pytest.approx(5.0)                    # pre-clip norm
    assert float(parameter.grad.norm()) == pytest.approx(1.0)   # clipped
    assert tu.clip_and_norm([torch.nn.Parameter(torch.zeros(2))], 1.0) == 0.0
    with pytest.raises(ValueError):
        tu.clip_and_norm([parameter], 0.0)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
class _Row:
    def __init__(self, ids, mask=None):
        self.ids = ids
        self.mask = mask


def test_rows_to_tensor_without_masks():
    ids, mask = tu.rows_to_tensor([_Row([1, 2, 3]), _Row([4, 5, 6])], device="cpu")
    assert tuple(ids.shape) == (2, 3) and ids.dtype == torch.long
    assert mask is None


def test_rows_to_tensor_carries_the_loss_mask():
    ids, mask = tu.rows_to_tensor(
        [_Row([1, 2, 3], [0, 1, 1]), _Row([4, 5, 6])],   # a None mask = fully supervised
        device="cpu", with_mask=True)
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert ids.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_rows_to_tensor_rejects_ragged_batches():
    with pytest.raises(ValueError):
        tu.rows_to_tensor([_Row([1, 2]), _Row([1, 2, 3])], device="cpu")
    with pytest.raises(ValueError):
        tu.rows_to_tensor([], device="cpu")
