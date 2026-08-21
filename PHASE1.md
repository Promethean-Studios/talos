# Forge — Phase 1 MVP Summary

Status: **FULLY COMPLETE.** All Phase 1 modules implemented, `pytest` passes on
CPU, the tiny config runs forward+backward end to end, and the long-context
test passes.

## Structure created

Repository at `/home/team/shared/forge/`.

| Path | Contents |
|---|---|
| `model/` | `config.py` (ModelConfig), `rms_norm.py`, `activations.py` (SwiGLU), `rotary.py` (RoPE+YaRN), `attention.py` (AttentionInterface, Plain + FlashAttention backends, chunked), `moe.py` (top-k routing + load-balance + MoE block), `ffn.py` (dense + shared FFNInterface), `block.py` (DecoderLayer, hybrid attention), `gpt.py` (ForgeGPT), `cache.py` (KV cache), `masking.py`, `utils.py` (seeds/logging) |
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
| small | dense | ~11 M | ~11 M | — | — |
| medium | dense | ~82 M | ~82 M | — | — |
| large | dense | ~0.8 B | ~0.8 B | — | — |
| 100b | MoE | 106.6 B | 9.9 B (9.3%) | ~198 GiB | — |
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
