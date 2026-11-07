# Talos — Phase 1 MVP Summary

Status: **FULLY COMPLETE.** All Phase 1 modules implemented, `pytest` passes on
CPU, the tiny config runs forward+backward end to end, and the long-context
test passes.

## Structure created

This is a historical record of the Phase 1 build (2026-08/09). The code now
lives in the public repository `Promethean-Studios/talos`; file paths below
refer to the repository root.

| Path | Contents |
|---|---|
| `model/` | `config.py` (ModelConfig), `rms_norm.py`, `activations.py` (SwiGLU), `rotary.py` (RoPE+YaRN), `attention.py` (AttentionInterface, Plain + FlashAttention backends, chunked), `moe.py` (top-k routing + load-balance + MoE block), `ffn.py` (dense + shared FFNInterface), `block.py` (DecoderLayer, hybrid attention), `gpt.py` (TalosGPT), `cache.py` (KV cache), `masking.py`, `utils.py` (seeds/logging) |
| `configs/` | `presets.py` (6 configs), `compute.py` (params/memory/FLOPs helper + CLI) |
| `tests/` | unit tests for every module + integration + long-context (see below) |
| `docs/` | `architecture.md` — all design decisions + where hardware-specific code is abstracted |
| `scripts/` | `cli.py` (tiny fwd/bwd), `run_tests.sh` |
| `examples/` | `tiny_train.py`, `run_inference.py` |
| `training/ data/ tokenizer/ inference/ distributed/ evaluation/` | Phase 2+ placeholders (`__init__.py` + `PLAN.md`) |
| top-level | `pyproject.toml`, `README.md`, `.gitignore`, `PHASE1.md`, `conftest.py` |

Dependencies installed in a venv at `/opt/forge-venv` (torch 2.13 +cpu, numpy,
pytest). `flash-attn` is optional and intentionally NOT installed.

## Config parameter counts (from `configs.compute`)

| name | ffn | total params | active/token | BF16 weights | FLOPs/tok |
|---|---|---|---|---|---|
| tiny | dense | 254 K | 254 K (100%) | ~0.5 MB | ~0.3 M |
| small | dense | 35.7 M | 35.7 M | ~68 MiB | ~67 M |
| medium | dense | 285 M | 285 M | ~544 MiB | ~537 M |
| large | dense | 1.64 B | 1.64 B | ~3.06 GiB | ~3.15 B |
| 100b | MoE | 107 B | 9.93 B (9.3%) | ~198 GiB | ~19.3 B |
| 400b | MoE | **400.7 B** | **30.0 B (7.5%)** | ~746 GiB | ~58 B |

(The `small`/`medium`/`large` rows are full counts printed by the helper;
numbers above are the headline figures — run `python -m configs.compute` for
exact values.)

The ~400B config is `hidden=6144, layers=46, heads=48 (head_dim 128),
kv_heads=8, vocab=131072, 128 routed experts, top-k=6, 2 shared,
per-expert=3584, hybrid attention (16K window, full every 8), YaRN, 128K ctx`.

## Test results

Run from repo root with only `torch` + `pytest`:

```
python -m pytest
```

All tests pass on CPU, including:

- RMSNorm, SwiGLU, RoPE/YaRN, GQA attention (functional fallback), MoE routing +
  load-balancing loss, KV cache (prefill + decode), sliding/full masking,
  dense-vs-MoE interface.
- **Integration**: tiny config forward + backward (dense and MoE), and a
  prefill→decode equivalence check (incremental decode logits match a single
  full forward, for both full and sliding-window attention).
- **Long-context**: a small hybrid model (sliding 256 + full layers) runs a
  16K-token forward on CPU through the functional *chunked* backend (bounded,
  near-linear memory), plus a long prefill→decode check at 8K. Verified no
  catastrophic O(L²) blow-up.

## Faster/verify yourself

```
PYTHON=python3 bash scripts/run_tests.sh       # run tests
python -m scripts.cli --steps 3 --seq 64        # tiny fwd/backward
python -m configs.compute --summary             # param counts
```
(The shared team venv is `/opt/forge-venv/bin/python`.)

## Deviations from the brief (documented in `docs/architecture.md`)

1. **~400B config expert count = 128, not ~256.** `"~256 experts, top-k ~6"`
   and `"~400B total / ~30B active"` are mathematically inconsistent: 256
   experts with top-6 gives only ~2.3% active (~9–13B), not 30B. To have the
   helper *genuinely* report ~400B/~30B (the brief's explicit "tune the numbers
   so the helper genuinely reports" requirement) with top-k ≈ 6, the expert
   count is 128 so the active fraction is ~6.2%.
2. **"MoE FFN dimension larger than dense"** is interpreted as *total* MoE width
   (128 × 3584 = 458,752) vs a dense FFN (4 × 6144 = 24,576) — per-expert width
   is typically smaller than hidden in real MoE LLMs (e.g. DeepSeek-V3).
3. **Chunked functional attention** added as the mechanism that makes the
   functional fallback genuinely near-linear memory at long context (beyond
   just relying on FlashAttention).

## Not in Phase 1 (deferred to later phases, per brief)

Distributed training (DP/TP/PP/EP, BF16/FP8, checkpoints), data pipeline,
tokenizer training, inference server, quantization, evaluation harness. The
directories exist with `PLAN.md` scoping notes.

**Status since (2026-11-06):** the tokenizer, data pipeline, training
pipeline with checkpoint/resume, inference (prefill + KV-cache decode) and
the evaluation harness have all since been implemented and tested;
quantization has an int8 experiment at 254K. **Distributed training is
still not implemented.** See `docs/STATUS.md` for the current picture.
