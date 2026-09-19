# Talos long-budget A/B — 254K vs 1M at the published baseline's step shape (7,680 × 32 × 64 ≈ 15.5M tokens)

*2026-09-19 · Phase B follow-up · branch `bench/phase-b-15m` · main tip `48bba04` when branched.*
*Every number is traceable to the committed artifacts in this directory (`metrics-*-15m.json`,
`eval-*-15m.json`, `probe-batch32.json`, `generation_samples-15m.md`) and run logs under
`/tmp/phaseb-15m/`. No estimates are placed in result slots.*

## 1. What this run is

Owner directive: re-run the 254K-vs-1M A/B at the **published baseline's exact token budget**
(7,680 steps × batch 32 × seq 64 ≈ **15.48M tokens** per model) to test whether `tiny_1m`
(1,000,320 params) **overtakes** `tiny` (254,272 params) once given ~5.3× more tokens than the
Phase-B 3-epoch budget (2.91M). Both models train on the **identical corpus, split, tokenizer,
seed and step shape**; the 254K row doubles as a CPU-vs-T4 anchor against the owner's published
`PrometheanStudio/talos-mini-254k-oasst1` checkpoint (same 7,680 steps / batch 32 / seq 64 on a
T4 GPU).

**No code changes.** The pipeline's existing `--epochs` / `--max-steps-per-epoch` / `--batch` knobs
(`scripts/train_oasst1.py`) already express the required shape: at batch 32 the packed corpus
yields **exactly 480 batches per pass** (measured: 480 full batches, one partial dropped), so
`--epochs 16 --max-steps-per-epoch 480 --batch 32` = **16 full corpus passes = exactly 7,680
optimizer steps**, with a train loss + full-val eval + checkpoint every 480 steps (final
checkpoint `step-7680.pt`, matching the published checkpoint name). Defaults unchanged; test suite
untouched (265 passed / 2 skipped on main).

## 2. Verification of assets (task step 1)

- **Corpus:** `/tmp/phaseb/oasst1_2000.jsonl` (2,000 docs = rows 0–1999 of `OpenAssistant/oasst1`
  train) — sha256 `e75b10aa…69a69` **== recorded in `benchmarks/phase-b/data-provenance.json`** ✓
  ($/tmp survived, so no re-fetch needed).
- **Tokenizer:** cached Phase-B `tokenizer.json` **byte-identical to the published
  `/tmp/hf_tokenizer.json`** (sha256 `58e4ad40…549e48`, 764 merges, vocab 1024) ✓. Both training
  runs retrained the tokenizer on their own train split anyway (pipeline behavior; deterministic,
  ~106 s, byte-identical output, re-verified per run, §10).
- **Canonical guards:** passed for both presets before training (tiny 254,272 / vocab 1024;
  tiny_1m 1,000,320 / vocab 1024 — enforced by `configs.canonical` inside `scripts/train_oasst1.py`).
- **Published baseline:** `/tmp/hf_step-7680.pt` + `/tmp/hf_metrics.json` present (254,272 params,
  7,680 steps, batch 32, seq 64, lr 3e-3, T4, train 1.7859 / val 1.9177). `hf_metrics.json`
  records **no LR schedule** (single `learning_rate: 0.003`), so plain AdamW lr 3e-3 is kept as
  instructed — **no schedule deviation to document**.

## 3. Memory probe (task step 2 — batch-32 gate, before any full run)

`tiny_1m` at batch 32, one full 480-step pass, mirroring the training loop exactly (AdamW, CE
`x[:, :-1] -> x[:, 1:]`): **peak RSS 504.3 MiB** — well under the ~1.5 GB gate; train throughput
**19,607 tok/s** (~2.1× the Phase-B batch-4 rate); steps-per-epoch at batch 32 = **exactly 480**
(this fixes the 7,680-step shape). Full numbers in `probe-batch32.json`.

## 4. Method (identical for both models)

| knob | value |
|---|---:|
| data | 2,000 docs → 1,800 train / 200 val, seed 0, ratio 0.9 (split byte-identical to Phase B) |
| tokenizer | Talos-native byte-level BPE, train split only, vocab 1024 (764 merges), byte-identical to published |
| seq / batch / lr / opt | 64 / 32 / 3e-3 / AdamW (published baseline's exact shape; lr as published) |
| epochs / steps | 16 epochs × 480 steps = **7,680 steps** (15,482,880 tokens ≈ 15.5M) per model |
| seed | 0 |
| guards | canonical param-count + vocab guard before training (254,272 / 1,000,320) |
| eval protocol | full val stream (51 batches / 102,816 tokens) after **every 480 steps** (16 val measurements); final 7,680-step checkpoint re-evaluated with `scripts/eval_checkpoint` (batch 32, drop_last) — bit-exact vs recorded |
| generation | `scripts/generate`, greedy, 5 fixed prompts, 32 new tokens, **fresh CLI process per row** (clean per-process timing) |

## 5. Owner's comparison table — LOCAL A/B at 15.5M tokens

| Metric | tiny 254,272 | tiny_1m 1,000,320 |
|---|---:|---:|
| Parameters | 254,272 | 1,000,320 (3.93×) |
| Steps / tokens | 7,680 / 15,482,880 | 7,680 / 15,482,880 |
| Train loss (final, step 7,680) | **1.7937** | __TBD__ |
| Validation loss (final, step 7,680) | **__TBD__** | __TBD__ |
| Tokens per sec (train phase, mean of epoch walls) | **48.3K** | __TBD__ |
| Training time | 320.8 s train phase · 427.3 s run total¹ | __TBD__ |
| Peak memory (CPU peak RSS — no GPU on this box) | 441.5 MiB train · 254.0 MiB eval | __TBD__ MiB train · __TBD__ eval |
| Checkpoint size (final `step-7680.pt`) | __TBD__ B | __TBD__ B |
| Eval throughput (val 102,816 tok) | 142,400 tok/s | __TBD__ |
| Val perplexity / next-token accuracy | 6.8724 / 0.4428 | __TBD__ / __TBD__ |
| Decode latency (greedy KV-cache, clean CLI) | __TBD__ ms/token | __TBD__ ms/token |

¹ `wall_s` in `metrics.json` includes the ~106 s train-split BPE + 16 val evals + 17 checkpoints
(same for both runs). Train phase = tokens / Σ(epoch walls).

Per-480-step detail (train / val), tiny:

| epoch | step | train | val | epoch wall |
|---|---:|---:|---:|---:|
| 1 | 480 | 3.0169 | 2.8544 | 20.2 s |
| 2 | 960 | 2.6040 | 2.7782 | 20.0 s |
| 3 | 1440 | 2.4361 | 2.6218 | 19.7 s |
| 4 | 1920 | 2.3245 | 2.4632 | 19.9 s |
| 5 | 2400 | 2.2411 | 2.3281 | 19.8 s |
| 6 | 2880 | 2.1662 | 2.2360 | 20.1 s |
| 7 | 3360 | 2.0993 | 2.1692 | 20.1 s |
| 8 | 3840 | 2.0425 | 2.1278 | 19.9 s |
| 9 | 4320 | 1.9939 | 2.0709 | 20.1 s |
| 10 | 4800 | 1.9520 | 2.0356 | 20.1 s |
| 11 | 5280 | 1.9141 | 2.0088 | 20.2 s |
| 12 | 5760 | 1.8814 | 1.9950 | 20.4 s |
| 13 | 6240 | 1.8535 | 1.9656 | 20.4 s |
| 14 | 6720 | 1.8278 | 1.9620 | 20.3 s |
| 15 | 7200 | 1.8052 | 1.9485 | 20.1 s |
| 16 | 7680 | 1.7937 | **1.9275** | 19.9 s |

tiny_1m per-480-step detail: __TBD__ (from `metrics-tiny-1m-15m.json`).

## 6. Three-way loss view (the question this run answers)

| config | tokens | tiny train / val | tiny_1m train / val | notes |
|---|---:|---|---:|---|
| 3-epoch local A/B (Phase B, PR #22) | 2,908,332 | 2.1610 / 2.3704 | 2.1826 / 2.3950 | tiny wins (1M underfits); batch 4 |
| **15.5M local A/B (this run)** | 15,482,880 | **1.7937 / 1.9275** | __TBD__ / __TBD__ | batch 32; **__TBD__** |
| published T4 (owner reference — anchor for tiny only) | 15,482,880 | **1.7859 / 1.9177** | — | T4 GPU, unknown data order / eval protocol |

Gain from the larger budget (3-epoch → 15.5M tokens), same local pipeline:
tiny val 2.3704 → 1.9275 (**−18.7%**, Δ −0.443); tiny_1m __TBD__ → __TBD__ (Δ −__TBD__).

**Answer to the A/B question (does 1M overtake 254K with enough tokens?): __TBD__** — detailed in
§10.

**CPU-vs-T4 caveat on the anchor row:** the published row ran on a T4 GPU with an unreported
data order and eval protocol; our CPU row is batch-32 with the stated seed-0 split and per-480-step
full-val protocol. The fact that CPU tiny lands within Δ+0.008 train / Δ+0.010 val of the T4 row
at the identical step shape is therefore a strong pipeline-fidelity signal, not an exact hardware
comparison.

## 7. Evaluation of final checkpoints (`scripts/eval_checkpoint`, seed 0, seq 64, batch 32)

- **tiny**: val loss **1.9275096343604716 == recorded 1.9275096343604716** (bit-exact) · ppl
  6.8724 · acc 0.4428 · 142,400 tok/s · peak RSS 254.0 MiB — canonical guard OK
  (254,272 / vocab 1024).
- **tiny_1m**: __TBD__

## 8. Generation

All 5 prompts × 3 checkpoints (local tiny, local tiny_1m, **published HF baseline**) in
`generation_samples-15m.md`: greedy, deterministic, 32 tokens, **fresh CLI process per row**
(clean per-process timing — avoids the Phase-B in-process driver inflation). Decode ms/token from
the CLI's own wall-token accounting. __TBD__ (tiny_1m + published rows after training completes).

## 9. Environment & provenance

- CPU: Intel(R) Xeon(R) @ 2.90GHz, 2 cores, no HT · **no GPU** (peak memory = CPU peak RSS,
  `ru_maxrss`, per `docs/SCALING.md` §5).
- RAM: 3,932 MB total; load ≤ 0.8 during runs (probe/training ran alone).
- torch 2.13.0+cpu (`get_num_threads()=2`); python 3.12; venv `/opt/forge-venv`.
- git: branch `bench/phase-b-15m` off `48bba04`, **no code changes** (knobs already existed).
- seeds: 0 everywhere; deterministic (same command → identical metrics).
- Disk: checkpoints + logs live in `/tmp/phaseb-15m/` (weights not committed; JSON/MD only, per
  benchmarks convention).

## 10. Honest footnotes / deviations / verdict reasoning

- No LR schedule in the published `metrics.json` → plain AdamW 3e-3 kept, as instructed.
- Batch 32 changes optimizer dynamics vs Phase B's batch 4 (8× fewer updates per corpus pass at
  the same lr — visible as higher epoch-1 losses at batch 32, e.g. tiny 3.0169 vs 2.6297);
  all comparisons in this report are same-batch-size, so the A/B is clean; the 3-epoch rows exist
  for the budget-scaling view, not as same-dynamics comparators.
- __TBD__