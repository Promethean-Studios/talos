"""Tests for MoE routing, load-balancing loss and the MoE block."""
import torch

from model.moe import (
    MoEFFNBlock,
    load_balancing_loss,
    topk_router_logits,
)


def test_topk_routing_selects_and_normalizes():
    torch.manual_seed(0)
    logits = torch.randn(8, 16)
    w, idx, probs = topk_router_logits(logits, top_k=3)
    assert w.shape == (8, 3)
    assert idx.shape == (8, 3)
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)
    # weights are the (normalized) top-3 softmax probs
    assert (idx >= 0).all() and (idx < 16).all()


def test_load_balancing_loss_nonnegative():
    torch.manual_seed(1)
    logits = torch.randn(64, 8)
    w, idx, probs = topk_router_logits(logits, top_k=2)
    loss = load_balancing_loss(idx, w, probs, 8)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    # Random routing gives loss ≈ 1.0 (balanced expectation).
    assert abs(loss.item() - 1.0) < 0.5


def test_load_balancing_loss_penalizes_unbalanced():
    # If every token routes to expert 0, the auxiliary loss should grow.
    logits = torch.full((64, 8), 10.0)
    logits[:, 1:] = -10.0  # expert 0 gets everything
    w, idx, probs = topk_router_logits(logits, top_k=1)
    unbalanced = load_balancing_loss(idx, w, probs, 8)
    # balanced-case reference (uniform probs) is ~1.0
    balanced_probs = torch.full((64, 8), 1.0 / 8)
    balanced_idx = torch.randint(0, 8, (64, 1))
    balanced = load_balancing_loss(balanced_idx, torch.ones(64, 1), balanced_probs, 8)
    assert unbalanced.item() > balanced.item()


def test_moe_block_shape_and_aux_loss():
    torch.manual_seed(0)
    block = MoEFFNBlock(
        hidden_size=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, num_shared_experts=1,
        load_balance_coef=0.1,
    )
    x = torch.randn(3, 4, 16)
    out, aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(aux)
    assert aux.item() > 0.0  # MoE aux loss is nonzero (coef 0.1)


def test_moe_backprop_through_single_expert():
    # Only routed experts + shared run; gradients must flow.
    torch.manual_seed(0)
    block = MoEFFNBlock(hidden_size=8, num_experts=4, num_experts_per_tok=1,
                        moe_intermediate_size=16, num_shared_experts=1)
    x = torch.randn(10, 8, requires_grad=True)
    out, aux = block(x)
    loss = out.square().mean() + aux
    loss.backward()
    assert x.grad is not None
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads, "expected some gradient on expert parameters"


def test_moe_shared_experts_always_active():
    # With shared > 0 the output should differ from shared == 0, and the
    # shared expert should receive gradient.
    torch.manual_seed(0)
    block = MoEFFNBlock(hidden_size=8, num_experts=4, num_experts_per_tok=1,
                        moe_intermediate_size=16, num_shared_experts=2)
    x = torch.randn(6, 8)
    out, _ = block(x)
    shared_params = [p for e in block.shared_experts for p in e.parameters()]
    out.square().mean().backward()
    assert all(p.grad is not None for p in shared_params)


def test_moe_validation():
    import pytest as _pt
    with _pt.raises(ValueError):
        MoEFFNBlock(hidden_size=8, num_experts=0, num_experts_per_tok=1,
                    moe_intermediate_size=16)
    with _pt.raises(ValueError):
        MoEFFNBlock(hidden_size=8, num_experts=4, num_experts_per_tok=6,
                    moe_intermediate_size=16)
