"""Integration tests for the full ForgeGPT model (forward/backward, cache)."""
import torch

from model import ForgeGPT, ModelConfig


def dense_tiny(**overrides) -> ModelConfig:
    kw = dict(
        vocab_size=256, hidden_size=64, num_layers=2, num_attention_heads=4,
        num_kv_heads=2, head_dim=16, ffn_type="dense", intermediate_size=256,
        max_seq_len=64,
    )
    kw.update(overrides)
    return ModelConfig(**kw).derive()


def moe_tiny(**overrides) -> ModelConfig:
    kw = dict(
        vocab_size=256, hidden_size=64, num_layers=2, num_attention_heads=4,
        num_kv_heads=2, head_dim=16, ffn_type="moe", num_experts=8,
        num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=128,
        load_balance_coef=0.01, max_seq_len=64,
    )
    kw.update(overrides)
    return ModelConfig(**kw).derive()


def test_forward_shape_tiny_dense():
    torch.manual_seed(0)
    model = ForgeGPT(dense_tiny())
    x = torch.randint(0, 256, (2, 16))
    logits, cache = model(x)
    assert logits.shape == (2, 16, 256)
    assert cache is None  # no cache by default


def test_forward_backward_tiny_dense():
    torch.manual_seed(0)
    model = ForgeGPT(dense_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (2, 32))
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), x.reshape(-1)
    )
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    # all trainable params received a gradient
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"missing grads: {missing[:5]}"


def test_forward_backward_tiny_moe():
    torch.manual_seed(0)
    model = ForgeGPT(moe_tiny())
    x = torch.randint(0, 256, (2, 32))
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), x.reshape(-1)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        "router" in n and p.grad is not None for n, p in model.named_parameters()
    )


def test_prefill_decode_match_dense():
    torch.manual_seed(0)
    model = ForgeGPT(dense_tiny(max_seq_len=64)).eval()
    seq = 40
    ids = torch.randint(0, 256, (1, seq))
    with torch.no_grad():
        full_logits, _ = model(ids)

    cache = model.new_cache(1, ids.device, torch.float32)
    split = 20
    with torch.no_grad():
        _, cache = model(ids[:, :split], use_cache=True, cache=cache)
        for t in range(split, seq):
            tok = ids[:, t : t + 1]
            pos = torch.tensor([[t]])
            logits_i, cache = model(tok, position_ids=pos, use_cache=True, cache=cache)
            assert torch.allclose(
                logits_i[0, 0], full_logits[0, t], atol=1e-4
            ), f"decode mismatch at t={t}"
    assert cache.length == seq


def test_prefill_decode_match_sliding():
    torch.manual_seed(0)
    cfg = dense_tiny(max_seq_len=128, attention_type="hybrid",
                     sliding_window_size=8, periodic_full_every=4)
    model = ForgeGPT(cfg).eval()
    seq = 96
    ids = torch.randint(0, 256, (1, seq))
    with torch.no_grad():
        full_logits, _ = model(ids)
    cache = model.new_cache(1, ids.device, torch.float32)
    split = 32
    with torch.no_grad():
        _, cache = model(ids[:, :split], use_cache=True, cache=cache)
        for t in range(split, seq):
            tok = ids[:, t : t + 1]
            pos = torch.tensor([[t]])
            logits_i, cache = model(tok, position_ids=pos, use_cache=True, cache=cache)
            assert torch.allclose(
                logits_i[0, 0], full_logits[0, t], atol=1e-4
            ), f"sliding decode mismatch at t={t}"


def test_sequence_length_validation():
    model = ForgeGPT(dense_tiny(max_seq_len=64))
    x = torch.randint(0, 256, (1, 65))
    try:
        model(x)
        assert False, "expected ValueError for over-long sequence"
    except ValueError:
        pass


def test_param_count_reasonable():
    model = ForgeGPT(dense_tiny())
    # hidden=64: params should be in the ~200-400K range (analytic estimate).
    n = model.num_parameters()
    assert 100_000 < n < 500_000


def test_tied_embeddings():
    cfg = dense_tiny(tie_word_embeddings=True)
    model = ForgeGPT(cfg)
    assert model.lm_head is None
    x = torch.randint(0, 256, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 8, 256)
