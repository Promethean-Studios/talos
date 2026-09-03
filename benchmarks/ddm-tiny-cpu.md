# DDM KV-tier experiment — talos-tiny-ddm-kv-tier (schema v1)

* Branch `ddm/kv-tier-experiment` @ `d8e5eecc61a0` · torch 2.13.0+cpu · Intel(R) Xeon(R) Processor @ 2.90GHz · 2 threads · 15 timed R/T pairs

## Verdict

**MEMORY-ONLY TRADE: mechanism is correct, but no acceptable benefit at the tiny CPU scale.**

| Quantity | Value |
|---|---:|
| Persistent-KV RAM reduction (tiered vs resident) | **87.7%** (32,256 B vs 262,152 B peak) |
| Paired median transaction throughput (T/R) | **0.8152** (95% CI [0.8041, 0.8440]) |
| Logit max abs diff (prefill / 128 decode steps) | 0.000e+00 / 0.000e+00 |
| Greedy agreement (positions) | 384/384 prefill, 128/128 decode; continuation match: True |
| Teacher-forced loss delta | 0.000e+00 (max abs over 2 held-out sequences) |
| Transient KV staging (tiered peak) | 131,072 B (not a memory saving — required by full attention) |
| Cold-tier bytes moved | read 27,328,512 B / written 262,144 B |
| Cold-page eviction (POSIX_FADV_DONTNEED) | requested=16×, failed=0×

## Timings (mean over timed pairs, tiered vs resident)

| Metric | Resident | Tiered | Δ (T vs R) |
|---|---:|---:|---:|
| Prefill (ms) | 14.90 | 12.96 | -13.00% |
| Transaction (ms) | 163.62 | 189.78 | +15.99% |
| Transaction tokens/s | 3,177.35 | 2,700.32 | -15.01% |
| Decode tokens/s | 872.39 | 724.68 | -16.93% |
| Decode p50 / p95 (ms) | 1.069 / 1.287 | 1.343 / 1.565 | — |

## Interpretation rules (design §5, verbatim)

- ✅ PASS — Rule 1 (quality): Any violation of the logit, loss, or greedy-agreement guardrail means the tiered implementation is incorrect.
- ❌ FAIL — Rule 2 (benefit gates): Support the hypothesis only when all three benefit gates pass: >=75% persistent-KV reduction, paired-median transaction throughput >=90% of resident with bootstrap CI lower bound >=0.90, and all quality guards pass.
- ✅ PASS — Rule 3 (memory-only trade): If persistent RAM falls by >=75% but throughput is <90% of resident, record the exact slowdown and conclude: "mechanism is correct, but no acceptable benefit at the tiny CPU scale."
- ✅ PASS — Rule 4 (valid negative): No benefit at this scale is a successful experimental outcome.

## Memory inventory (design §3D, exact logical bytes)

| Quantity | Resident | Tiered |
|---|---:|---:|
| model parameter bytes | 1,017,088 | 1,017,088 |
| persistent KV RAM (peak) | 262,152 | 32,256 |
| transient KV staging (peak) | 0 | 131,072 |
| cold file payload (peak) | 0 | 262,144 |
| bytes read total | 0 | 27,328,512 |
| bytes written total | 0 | 262,144 |

Informational process peak RSS (quality child): 320,472 kB — dominated by interpreter/Torch/page cache; the logical persistent-KV measure above is the acceptance metric (§5.6).

## Fixed configuration

```json
{
  "batch": 1,
  "block_size": 64,
  "cpu_threads": 2,
  "decode_policy": "greedy",
  "decode_tokens": 128,
  "device": "cpu",
  "dtype": "float32",
  "heldout_seed": 1,
  "logit_atol": 0.0001,
  "loss_delta_atol": 1e-05,
  "max_seq_len": 512,
  "pair_orders": [
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 0
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 1
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 2
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 3
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 4
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 5
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 6
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 7
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 8
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 9
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 10
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 11
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 12
    },
    {
      "order": [
        "tiered",
        "resident"
      ],
      "pair_id": 13
    },
    {
      "order": [
        "resident",
        "tiered"
      ],
      "pair_id": 14
    }
  ],
  "parameters": 254272,
  "preset": "configs.presets.tiny_config().derive()",
  "prompt_tokens": 384,
  "repeats": 15,
  "train_final_loss": null,
  "train_seed": 0,
  "train_stats": {
    "batch": 4,
    "loss_final_batch": 3.2361,
    "loss_initial": 6.9411,
    "lr": 0.001,
    "seq_len": 64,
    "steps": 100,
    "train_seconds": 1.0,
    "train_tokens": 25200,
    "train_tokens_per_sec": 25195.6
  },
  "train_steps": 100,
  "transaction_tokens": 512,
  "warmup_per_condition": 2
}
```
