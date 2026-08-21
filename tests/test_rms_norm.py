"""Tests for RMSNorm."""
import torch

from model.rms_norm import RMSNorm


def test_rms_norm_shape_and_dtype():
    x = torch.randn(3, 7, 16)
    norm = RMSNorm(16, eps=1e-5)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_rms_norm_unit_variance():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32) * 5.0  # mean-free, large scale
    norm = RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        # variance of normalized-but-unscaled (gain=1) should be ~1
        out = norm(x)
        var = out.float().pow(2).mean(dim=-1)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_rms_norm_no_mean_shift():
    # RMSNorm does not subtract the mean; verify against a manual computation.
    torch.manual_seed(1)
    x = torch.randn(2, 6, 8)
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        out = norm(x)
    manual = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, manual, atol=1e-5)


def test_rms_norm_gain_is_learned():
    norm = RMSNorm(8)
    assert norm.weight.requires_grad
    x = torch.randn(2, 8, 8)
    out = norm(x)
    out.sum().backward()
    assert norm.weight.grad is not None
    assert norm.weight.grad.shape == (8,)
