# Talos tiny prototype — performance baseline (cpu)

Recorded 2026-08-31T23:39:25Z · seed 0 · commit `12afe99f60c9df018458227a4b11f17b691f89b2` (branch `bench/perf-baseline`)

| environment | value |
|---|---|
| torch | 2.13.0+cpu |
| python | 3.12.3 |
| cpu | Intel(R) Xeon(R) Processor @ 2.90GHz |
| cores / torch threads | 2 / 2 |
| device | cpu (cuda_available=False) |

**This is a CPU baseline** — numbers are specific to this machine and thread count.

## Model — `tiny` preset (dense)

| metric | value |
|---|---|
| total parameters | 254,272 |
| trainable parameters | 254,272 |
| active parameters (measured via forward hooks) | 254,272 |
| fp32 model size | 0.97 MB |
| bf16 model size | 0.485 MB |
| AdamW optimizer states (fp32, estimate) | 1.94 MB |

## Training throughput (synthetic recurrent corpus)

| metric | value |
|---|---|
| config | batch=4, seq=64, steps=100 (warmup 3), lr=0.001, AdamW |
| **train throughput** | **27,946.5 tokens/s** (252 tokens/step × 100 steps in 0.9017 s) |
| loss trajectory | 6.9463 (untrained eval) → 6.7473 → **1.8025** (min 1.8025) |
| peak RSS | 308.9 MB |

## Inference (prefill + KV-cache decode, batch 1, greedy)

| metric | value |
|---|---|
| config | prompt_len=64, decode_tokens=64, 20 prefill repeats |
| prefill latency | **1.75 ms** mean (p50 1.747, max 1.837) → 36,575.5 tokens/s |
| decode latency (KV cache) | **0.993 ms/token** mean (p50 0.967, max 1.475) → **1,006.6 tokens/s** |
| generate() end-to-end | 64.97 ms for 64 tokens |
| peak RSS | 225.2 MB |

> Regenerate on any machine with: `python tools/benchmark.py` (writes `benchmarks/baseline-tiny-cpu.json` + this `.md`).
