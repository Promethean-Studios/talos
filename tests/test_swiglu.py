"""Tests for the SwiGLU activation block."""
import torch

from model.activations import SwiGLU, silu


def test_swiglu_shape():
    swiglu = SwiGLU(16, 48)
    x = torch.randn(2, 5, 16)
    out = swiglu(x)
    assert out.shape == (2, 5, 16)


def test_swiglu_equals_manual():
    torch.manual_seed(0)
    swiglu = SwiGLU(8, 24)
    x = torch.randn(3, 8)
    with torch.no_grad():
        out = swiglu(x)
    gate = silu(x @ swiglu.gate_proj.weight.t())
    up = x @ swiglu.up_proj.weight.t()
    down = (gate * up) @ swiglu.down_proj.weight.t()
    assert torch.allclose(out, down, atol=1e-5)


def test_swiglu_gradients():
    swiglu = SwiGLU(8, 16)
    x = torch.randn(4, 8, requires_grad=True)
    out = swiglu(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    for p in swiglu.parameters():
        assert p.grad is not None


def test_silu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = silu(x)
    # silu(x) = x * sigmoid(x)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(out, expected)
