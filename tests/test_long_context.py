"""Long-context integration test.

Builds a small MoE/hybrid model whose ``max_seq_len`` is very large (16K) and
verifies inference runs on a sequence at that full length using the functional
(chunked) backend — i.e. no catastrophic O(seq^2) memory blow-up because the
hybrid sliding-window + periodic-full scheme with chunked execution keeps peak
memory near-linear.

This runs on CPU with only torch + pytest.
"""
import torch

from model import TalosGPT, ModelConfig, PlainAttentionBackend


def build_long_config(seq_len: int) -> ModelConfig:
    """A tiny model configured for a very long context with hybrid attention."""
    # Layers 0 and 2 are *full-attention* layers; layers 1 and 3 are
    # sliding-window (window 256). Full layers run through the chunked path.
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=4,
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=16,
        ffn_type="moe",
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=64,
        max_seq_len=seq_len,
        attention_type="hybrid",
        sliding_window_size=256,
        full_attention_layers=[0, 2],  # explicit: layers 0 & 2 are full
        attention_chunk_size=512,      # chunked functional backend
        layer_norm_eps=1e-5,
    ).derive()


def test_long_context_hybrid_forward_at_max_len():
    seq_len = 16384
    torch.manual_seed(0)
    cfg = build_long_config(seq_len)
    # Force the plain backend explicitly (no flash-attn here anyway).
    backend = PlainAttentionBackend(chunk_size=cfg.attention_chunk_size)
    model = TalosGPT(cfg, attention_backend=backend).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with torch.no_grad():
        logits, _ = model(ids)

    assert logits.shape == (1, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    # sanity: dense scores should not be degenerate (nonzero variance)
    assert logits.float().std().item() > 0.0


def test_long_context_prefill_then_decode():
    # Prefill the cache at a long sequence, then decode a few tokens, verifying
    # buffered inference works end-to-end without re-materialising the prefix.
    seq_len = 8192
    torch.manual_seed(0)
    cfg = build_long_config(seq_len)
    model = TalosGPT(cfg, attention_backend=PlainAttentionBackend(512)).eval()

    # Prefill slightly less than max so decoded positions stay in range.
    prefill_len = seq_len - 3
    ids = torch.randint(0, cfg.vocab_size, (1, prefill_len))
    with torch.no_grad():
        full_logits, _ = model(ids)

        cache = model.new_cache(1, ids.device, torch.float32)
        _, cache = model(ids, use_cache=True, cache=cache)
        assert cache.length == prefill_len

        # Decode three new tokens (within max_seq_len) and check shapes are sane.
        for t in range(prefill_len, seq_len):
            tok = torch.randint(0, cfg.vocab_size, (1, 1))
            pos = torch.tensor([[t]])
            logits_i, cache = model(tok, position_ids=pos, use_cache=True, cache=cache)
            assert logits_i.shape == (1, 1, cfg.vocab_size)
            assert torch.isfinite(logits_i).all()
            assert cache.length == t + 1
