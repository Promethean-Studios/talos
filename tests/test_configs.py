"""Tests for the config presets and the compute estimate helper."""
from configs.compute import estimate, format_report
from configs.presets import ALL_PRESETS


def test_all_presets_derive_and_estimate():
    for name, builder in ALL_PRESETS.items():
        cfg = builder().derive()
        rep = estimate(cfg)
        assert rep.total_params > 0
        assert rep.active_params > 0
        assert rep.active_params <= rep.total_params
        assert rep.weights_mem_bf16 == rep.total_params * 2
        assert rep.flops_per_token > 0


def test_tiny_is_small():
    rep = estimate(ALL_PRESETS["tiny"]())
    assert rep.total_params < 1_000_000


def test_100b_scale_between_50b_and_150b():
    rep = estimate(ALL_PRESETS["100b"]())
    assert 50e9 < rep.total_params < 150e9


def test_400b_total_active_targets():
    rep = estimate(ALL_PRESETS["400b"]())
    # ~400B total
    assert 3.0e11 < rep.total_params < 5.0e11  # 300B..500B
    # ~30B active per token
    assert 2.0e10 < rep.active_params < 4.0e10  # 20B..40B
    # sparsity: active << total
    assert rep.active_params / rep.total_params < 0.15


def test_400b_short_context_is_128k_plus():
    cfg = ALL_PRESETS["400b"]().derive()
    assert cfg.max_seq_len == 131072
    assert cfg.vocab_size == 131072
    assert cfg.attention_type == "hybrid"


def test_large_uses_hybrid_attention():
    cfg = ALL_PRESETS["large"]().derive()
    assert cfg.attention_type == "hybrid"
    assert cfg.periodic_full_every > 0
    full = cfg.full_layer_indices()
    assert len(full) > 0  # at least one full layer
    assert len(full) < cfg.num_layers  # but not all


def test_format_report_contains_numbers():
    text = format_report(estimate(ALL_PRESETS["tiny"]()))
    assert "total params" in text
    assert "FLOPs" in text


def test_compute_cli_runs():
    from configs.compute import main
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["tiny", "--summary"])
    assert "tiny" in buf.getvalue()
