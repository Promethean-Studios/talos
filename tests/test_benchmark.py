"""Smoke tests for the tiny-prototype performance-benchmark harness.

These assert the harness runs end-to-end and produces well-formed output with
sane bounds. They are deliberately **not** a performance gate — tokens/sec and
latency vary by machine and thread count, so only structural properties, the
exact parameter counts, and determinism are pinned. The committed CPU baseline
lives in ``benchmarks/baseline-tiny-cpu.json``; regenerate it on any machine
with ``python tools/benchmark.py`` (see ``benchmarks/README.md``).

Everything here runs in-process with small parameters so the suite stays fast
and CPU-friendly; the committed baseline itself uses the harness's defaults.
"""
from __future__ import annotations

import json

from tools.benchmark import (  # type: ignore[import]
    SCHEMA_VERSION,
    render_markdown,
    run_all,
    run_inference_section,
    run_model_section,
)

TINY_PARAMS = 254_272  # the canonical tiny preset's parameter count


def test_model_section_param_counts_and_sizes() -> None:
    """Param counts are exact and the size estimates are consistent."""
    m = run_model_section(device="cpu", seed=0)
    assert m["params_total"] == TINY_PARAMS
    assert m["params_trainable"] == TINY_PARAMS
    # Dense preset: every parameter participates in a forward pass, so the
    # hook-measured active count equals the total (this is the MoE-generic path).
    assert m["params_active"] == TINY_PARAMS
    assert 0.9 < m["size_fp32_mb"] < 1.1
    assert 0.4 < m["size_bf16_mb"] < 0.6
    assert m["size_bf16_mb"] * 2 == m["size_fp32_mb"]
    assert m["optimizer_states_fp32_mb_adamw"] == m["size_fp32_mb"] * 2


def test_run_all_end_to_end_sane_bounds() -> None:
    """Full harness run produces well-formed, JSON-serializable, sane output."""
    results = run_all(
        seed=0,
        train_steps=30,
        train_warmup=1,
        prompt_len=32,
        decode_tokens=8,
        prefill_repeats=3,
        infer_warmup=1,
        in_process=True,
    )
    assert results["schema_version"] == SCHEMA_VERSION
    assert results["seed"] == 0
    assert results["env"]["device"] == "cpu"
    assert results["env"]["torch"]  # environment provenance is recorded
    assert results["git"]["commit"]

    model = results["model"]
    assert model["params_total"] == TINY_PARAMS

    train = results["train"]
    assert train["config"]["batch"] == 4
    assert train["config"]["steps"] == 30
    # Aligned causal-LM objective: one position per sequence is dropped.
    assert train["tokens_per_step"] == 4 * 63
    assert train["tokens_per_sec"] > 0
    assert train["wall_time_s"] > 0
    # The baseline is tied to a *learned* model: the loss must drop below its
    # untrained start (untrained CE on 1024 tokens is ~ln(1024) ~= 6.93).
    assert 0 < train["loss_end"] < train["loss_start"] < 7.0
    assert len(train["losses"]) == 30
    assert train["peak_rss_mb"] > 0

    infer = results["inference"]
    assert infer["config"]["prompt_len"] == 32
    assert infer["config"]["decode_tokens"] == 8
    assert infer["prefill_ms_mean"] > 0
    assert infer["prefill_ms_p50"] > 0
    assert infer["prefill_tokens_per_sec"] > 0
    assert infer["decode_ms_mean"] > 0
    assert infer["decode_ms_p50"] > 0
    assert infer["decode_tokens_per_sec"] > 0
    assert infer["generate_e2e_ms"] > 0
    assert len(infer["generated"]) == 8
    assert all(0 <= t < 1024 for t in infer["generated"])
    assert infer["peak_rss_mb"] > 0

    # The committed-artifact form: JSON round-trips and the markdown renders.
    json.loads(json.dumps(results))
    md = render_markdown(results)
    assert "254,272" in md
    assert "tokens/s" in md
    assert "CPU baseline" in md


def test_measured_quantities_deterministic() -> None:
    """Same seed → identical generated tokens (throughput may vary; outputs may not)."""
    a = run_inference_section(
        device="cpu", seed=0, prompt_len=32, decode_tokens=12,
        prefill_repeats=2, warmup=1,
    )
    b = run_inference_section(
        device="cpu", seed=0, prompt_len=32, decode_tokens=12,
        prefill_repeats=2, warmup=1,
    )
    assert a["generated"] == b["generated"]
