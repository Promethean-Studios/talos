"""Tests for the prototype quantization experiment (quantization/ package).

Covers:
* tensor-level symmetric int8 round-trip error bound (``|W - s*q| <= s``),
* per-channel and per-tensor granularity semantics,
* whole-model quantization for both precisions (logit similarity + greedy
  agreement vs the fp32 reference on a briefly-trained tiny model),
* non-mutating vs in-place entry points, ``skip`` support,
* payload-size accounting (fp32 > bf16 > int8),
* the KV-cache/generate path on a quantized model (the ``.weight`` shim),
* the experiment runner's output shape (schema, verdicts, compression).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import TalosGPT  # noqa: E402
from model.utils import set_seed  # noqa: E402
from quantization import (  # noqa: E402
    QuantizedEmbedding,
    QuantizedLinear,
    dequantize_int8,
    payload_bytes,
    quantization_error,
    quantize_int8_symmetric,
    quantize_model,
    quantize_model_,
)
from quantization.evaluate import compare_generations, compare_logits  # noqa: E402
from configs.presets import tiny_config  # noqa: E402

_VOCAB = 128  # small standalone models for tensor-level tests


def _tiny_trained(steps: int = 5) -> TalosGPT:
    """Tiny preset with a few real AdamW steps so logits are non-degenerate."""
    from training.synthetic import build_recurrent_corpus

    set_seed(0)
    cfg = tiny_config().derive()
    model = TalosGPT(cfg)
    corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=4, seq_len=32, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for i in range(steps):
        x = corpus[i % 4 : i % 4 + 2]
        logits, _ = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _ref_and_prompt(model: TalosGPT, length: int = 16):
    prompt = torch.arange(length).unsqueeze(0) % model.config.vocab_size
    with torch.no_grad():
        ref_logits, _ = model(prompt)
    return prompt, ref_logits


# ---------------------------------------------------------------------------
# Tensor-level symmetric int8
# ---------------------------------------------------------------------------
class TestInt8RoundTrip:
    def test_per_channel_bound(self) -> None:
        set_seed(0)
        w = torch.randn(64, 32) * 0.05
        q, scale = quantize_int8_symmetric(w, "per_channel")
        assert q.dtype == torch.int8 and scale.dtype == torch.float32
        assert scale.shape == (64, 1)
        assert q.abs().max() <= 127
        err = (dequantize_int8(q, scale, torch.float32) - w).abs()
        # Round-to-nearest on a symmetric grid: |W - s*q| <= s, per row.
        assert torch.all(err <= scale + 1e-7)

    def test_per_tensor_scalar_scale(self) -> None:
        set_seed(1)
        w = torch.randn(16, 8)
        q, scale = quantize_int8_symmetric(w, "per_tensor")
        assert scale.dim() == 0
        err = (dequantize_int8(q, scale, torch.float32) - w).abs()
        assert float(err.max()) <= float(scale) + 1e-7

    def test_int8_range_and_roundtrip_quality(self) -> None:
        set_seed(2)
        w = torch.randn(128, 64) * 0.1
        stats = quantization_error(w, "per_channel")
        # int8 symmetric round-trip is ~1/127 relative on well-scaled weights.
        assert stats["max_relative"] <= 2.0 / 127
        assert stats["mean_abs"] <= stats["max_abs"]
        assert stats["max_abs"] > 0.0

    def test_zero_weight_does_not_collapse(self) -> None:
        w = torch.zeros(4, 4)
        q, scale = quantize_int8_symmetric(w, "per_channel")
        assert float(scale.min()) > 0.0
        assert torch.equal(dequantize_int8(q, scale), torch.zeros_like(w))

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            quantize_int8_symmetric(torch.randn(4, 4), "per_row")
        with pytest.raises(ValueError):
            quantize_int8_symmetric(torch.randn(4, 4), num_bits=4)
        with pytest.raises(TypeError):
            quantize_int8_symmetric(torch.randint(0, 8, (4, 4)))


# ---------------------------------------------------------------------------
# Whole-model quantization: correctness vs fp32
# ---------------------------------------------------------------------------
class TestQuantizeModelInt8:
    @pytest.mark.parametrize("granularity", ["per_channel", "per_tensor"])
    def test_logits_close_and_greedy_mostly_identical(self, granularity: str) -> None:
        model = _tiny_trained()
        prompt, ref_logits = _ref_and_prompt(model)
        quantized, report = quantize_model(model, precision="int8", granularity=granularity)
        quantized.eval()
        # Exactly the Linear/Embedding leaves are replaced: embed + lm_head +
        # 2 layers x (q, k, v, o, gate, up, down) = 16 modules.
        assert report.precision == "int8"
        assert len(report.replaced) == 16
        assert report.params_quantized == 253952  # everything except 5 RMSNorm gains
        with torch.no_grad():
            got_logits, _ = quantized(prompt)
        metrics = compare_logits(ref_logits, got_logits)
        assert metrics["cosine_mean"] >= 0.995
        assert metrics["cosine_min"] >= 0.99
        assert metrics["greedy_agreement"] >= 0.85
        assert metrics["max_abs_diff"] <= 0.05

    def test_generation_via_kv_cache(self) -> None:
        from inference.generate import generate

        model = _tiny_trained(steps=3)
        prompt = torch.zeros((1, 8), dtype=torch.long)
        with torch.no_grad():
            ref = generate(model, prompt, max_new_tokens=8, greedy=True)
        quantized, _ = quantize_model(model, precision="int8", granularity="per_channel")
        quantized.eval()
        # Exercises prefill + KV decode on the int8 model (needs the .weight
        # shim for the cache-dtype lookup in TalosGPT.forward).
        with torch.no_grad():
            got = generate(quantized, prompt, max_new_tokens=8, greedy=True)
        assert len(got) == 8 and all(isinstance(t, int) for t in got)
        assert compare_generations(ref, got)["agreement"] >= 0.75

    def test_state_dict_stores_int8_payload(self) -> None:
        model = _tiny_trained(steps=2)
        quantized, _ = quantize_model(model, precision="int8")
        sd = quantized.state_dict()
        assert sd["embed_tokens.weight_int8"].dtype == torch.int8
        assert sd["embed_tokens.weight_scale"].dtype == torch.float32
        # 4x smaller main payload + tiny fp32 scales.
        int8_bytes = sum(
            t.numel() * t.element_size() for t in sd.values() if t.dtype == torch.int8
        )
        fp32_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        assert int8_bytes < fp32_bytes / 3.0

    def test_skip_leaves_module_fp32(self) -> None:
        model = _tiny_trained(steps=2)
        # quantize_model_ mutates in place and returns only the report; the
        # (model, report) tuple comes from the non-mutating quantize_model.
        report = quantize_model_(model, precision="int8", skip=("lm_head",))
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert len(report.replaced) == 15
        # In-place variant mutated the caller's model.
        assert isinstance(model.embed_tokens, QuantizedEmbedding)

    def test_in_place_and_nonmutating_agree(self) -> None:
        model = _tiny_trained(steps=2)
        prompt, ref_logits = _ref_and_prompt(model)
        copy_q, _ = quantize_model(model, precision="int8")  # non-mutating
        assert isinstance(model.embed_tokens, torch.nn.Embedding)  # original intact
        assert next(model.parameters()).dtype == torch.float32
        quantize_model_(model, precision="int8")  # in place
        with torch.no_grad():
            a, _ = copy_q(prompt)
            b, _ = model(prompt)
        assert torch.equal(a, b)


class TestQuantizeModelBf16:
    def test_bf16_compute_stays_close_to_fp32(self) -> None:
        model = _tiny_trained()
        prompt, ref_logits = _ref_and_prompt(model)
        quantized, report = quantize_model(model, precision="bf16")
        assert report.precision == "bf16"
        assert report.params_quantized == 254272
        assert all(p.dtype == torch.bfloat16 for p in quantized.parameters())
        with torch.no_grad():
            got_logits, _ = quantized(prompt)
        metrics = compare_logits(ref_logits, got_logits)
        assert metrics["cosine_mean"] >= 0.999
        assert metrics["greedy_agreement"] >= 0.8
        assert metrics["max_abs_diff"] <= 0.05

    def test_rejects_granularity_for_bf16(self) -> None:
        model = _tiny_trained(steps=1)
        with pytest.raises(ValueError):
            quantize_model(model, precision="bf16", granularity="per_tensor")
        with pytest.raises(ValueError):
            quantize_model(model, precision="fp16")


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------
class TestPayloadBytes:
    def test_precision_ordering_and_ratio(self) -> None:
        model = _tiny_trained(steps=2)
        fp32 = payload_bytes(model)
        bf16 = payload_bytes(quantize_model(model, precision="bf16")[0])
        int8 = payload_bytes(quantize_model(model, precision="int8")[0])
        assert fp32["total_bytes"] == 254272 * 4  # no extra buffers in fp32
        assert fp32["total_bytes"] > bf16["total_bytes"] > int8["total_bytes"]
        assert bf16["total_bytes"] == 254272 * 2
        # int8 payload + fp32 scales lands near 1 B/param (3.5-4x vs fp32).
        assert 3.0 <= fp32["total_bytes"] / int8["total_bytes"] <= 4.5
        assert int8["bytes_per_param"] < 2.0

    def test_quantized_linear_forward_is_dequantized_matmul(self) -> None:
        set_seed(3)
        lin = torch.nn.Linear(32, 16)
        ql = QuantizedLinear.from_linear(lin, "per_channel")
        assert ql.weight_int8.shape == (16, 32)
        assert ql.bias is not None
        x = torch.randn(2, 32)
        got = ql(x)
        manual = x @ ql.dequantize_weight().t() + ql.bias
        assert torch.allclose(got, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# Experiment runner output contract
# ---------------------------------------------------------------------------
class TestExperimentRunner:
    def test_quick_run_schema_and_verdicts(self, tmp_path) -> None:
        from tools.quantize_experiment import run_experiment

        results = run_experiment(
            seed=0, device="cpu", steps=3, batch=2, seq=32, n_seq=2,
            prompt_lengths=(8, 16), decode_tokens=4, warmup=1, prefill_repeats=2,
        )
        # Top-level schema.
        for key in ("name", "schema_version", "preset", "device", "seed", "env",
                    "train", "config", "precisions", "bottom_line", "peak_rss_mb"):
            assert key in results
        assert results["name"] == "talos-tiny-quant"
        assert set(results["precisions"]) == {"fp32", "bf16", "int8_per_channel", "int8_per_tensor"}

        for name, r in results["precisions"].items():
            assert r["payload"]["total_mb"] > 0
            assert 0.0 <= r["eval_loss"] < 20.0
            assert r["logit_similarity"], "expected per-prompt-length metrics"
            for length, m in r["logit_similarity"].items():
                assert set(m) >= {"max_abs_diff", "mean_abs_diff", "cosine_mean",
                                  "cosine_min", "greedy_agreement", "num_positions"}
                assert 0.0 <= m["greedy_agreement"] <= 1.0
                assert 0.0 < m["cosine_mean"] <= 1.0
            assert set(r["generation"]) >= {"reference", "quantized", "identical"}
            assert "decode_tokens_per_sec" in r["throughput"]

        # fp32 control row: model vs itself is exact.
        for m in results["precisions"]["fp32"]["logit_similarity"].values():
            assert m["max_abs_diff"] == 0.0 and m["greedy_agreement"] == 1.0
        # Verdicts: quantized variants compress and stay sane.
        bl = results["bottom_line"]
        assert bl["fp32"]["compression_vs_fp32"] == 1.0
        assert bl["bf16"]["compression_vs_fp32"] == pytest.approx(2.0, abs=0.01)
        assert bl["int8_per_channel"]["compression_vs_fp32"] > 3.0
        assert bl["int8_per_tensor"]["compression_vs_fp32"] > 3.0
        for name in ("bf16", "int8_per_channel", "int8_per_tensor"):
            assert bl[name]["greedy_agreement_min"] >= 0.8
            assert bl[name]["cosine_mean_min"] >= 0.99
            assert abs(bl[name]["eval_loss_delta"]) < 0.05

    def test_markdown_renders_from_results(self, tmp_path) -> None:
        from tools.quantize_experiment import main

        out = str(tmp_path / "quant.json")
        assert main(["--quick", "--out", out, "--no-md",
                     "--steps", "2", "--decode-tokens", "4"]) == 0
        assert os.path.exists(out)
        import json

        with open(out, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["schema_version"] == 1
        # --no-md suppresses the markdown file but the JSON is complete.
        assert not os.path.exists(str(tmp_path / "quant.md"))
        md = tmp_path / "quant2.json"
        assert main(["--quick", "--out", str(md), "--steps", "2",
                     "--decode-tokens", "4"]) == 0
        assert (tmp_path / "quant2.md").exists()
