# Talos Scaling Experiments

This document records Talos's parameter-scaling progression — what has actually
been **measured** at each scale, and what is planned but **not yet run**. It
exists to keep every scaling claim traceable to a real run (or explicitly
labeled as a plan/estimate). Nothing here is extrapolated from theory; rows
below are either backed by merged-PR artifacts or marked *not yet run*.

---

## 1. Scope statement (read this first)

- The long-term research target is a **~400B-total / ~30B-active MoE
  decoder-only base LLM with a 128K-token context window** (see
  `docs/architecture.md` and `configs/presets.py`). That is a **future
  research target, not a demonstrated result**.
- **No model larger than 254,272 parameters had been trained in this
  repository until Phase B (2026-09-19), when the `tiny_1m` preset was trained
  end-to-end on a real OASST1 subset — see the ~1M row in §2 and
  `benchmarks/report-tiny-1m.md`.** The codebase contains untrained *configs*
  for larger sizes (`small` 35.7M, `medium` 285M, `large` 1.64B, `100b`,
  `400b` — verifiable via `python -m configs.compute --summary`), but a config
  is an estimate, not a result. Configs become results only when a training
  run is executed, checkpointed, and evaluated through the harness.
- This document claims nothing beyond measured or clearly-labeled-estimated
  numbers. The progression table below is the authoritative scoreboard.

## 2. The progression table

Status legend: ✅ **measured** (real run, artifacts in this repo) · ⬜ *not yet
run* (placeholder — planned, no results).

| Scale (dense) | Parameters | Train loss* | Val loss* | Throughput | Peak memory | Convergence notes | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 254K (`tiny`) | 254,272 | **0.2521** (final, epoch 3) | **0.2473** (final, epoch 3) | **~12.5K tok/s** train · **~73–77K tok/s** eval · **~1.1 ms/token** decode | **~314 MiB** train · **~231 MiB** eval (peak RSS) | Loss strictly decreased on train *and* val across all 3 epochs; deterministic under seed 0; eval recompute bit-exact vs. recorded val loss. Output text is incoherent babble (expected at this scale). | ✅ PRs #17–#19 |
| ~1M (`tiny_1m`) | 1,000,320 | **2.1826** (final, epoch 3) | **2.3950** (final, epoch 3) | **~9.2K tok/s** train phase (6,884 tok/s whole-run) · **31.2K tok/s** eval · **2.0 ms/token** decode | **378.2 MiB** train · **237.8 MiB** eval (peak RSS) | Real OASST1 2,000-doc subset (1,800/200, seed 0, BPE 764 merges, seq 64, batch 4, lr 3e-3, 3 epochs = 11,541 steps). Train+val loss strictly decreased across all 3 epochs; eval recompute bit-exact vs recorded; canonical 1,000,320 guard enforced. **Honest A/B at this budget: final val 2.3950 vs tiny's 2.3704 on the identical data/steps — the 1M model did NOT beat 254K here** (underfits; see `benchmarks/phase-b/report-tiny-1m.md`). | ✅ Phase B, PR #22 |
| ~10M | *not yet run* | — | — | — | — | — | ⬜ |
| ~100M | *not yet run* | — | — | — | — | — | ⬜ |

\* Natural-log mean cross-entropy (nats). **Row 254K** was measured on the fixed
synthetic-OASST JSONL corpus (400 docs, 360/40 train/val split, BPE vocab 1024,
seq 64, batch 4, lr 3e-3, AdamW). **Row ~1M (Phase B)** was measured on a real
OASST1 2,000-doc subset (rows 0–1999 of `OpenAssistant/oasst1` train split,
1,800/200 seed-0 split, BPE vocab 1024 = 764 merges, seq 64, batch 4, lr 3e-3,
3 epochs). Losses are comparable only across runs sharing corpus/config; the
two rows are **not** cross-comparable — the 254K-vs-1M A/B on the *same*
corpus lives in `benchmarks/phase-b/report-tiny-1m.md`.

### 2.1 The 254K row — the real measured run

Verification run from the maintenance cycle (2026-09-15, CPU-only box), fully
reproducible with the commands in §4:

- **Config** — `tiny` preset: 254,272 params / vocab 1024 / hidden 64 / 2 layers
  / 4 heads / 2 KV heads / dense FFN / max_seq_len 512. Exact-parameter guard
  enforced before training (fails fast on drift).
- **Data** — `tools.make_synthetic_oasst1 --docs 400 --seed 0` → deterministic
  fixed-seed split into 360 train / 40 val JSONL files (disjoint, no leakage);
  tokenizer (vocab 1024 = 256 byte tokens + 4 specials + 764 merges) trained on
  the **train split only**.
- **Training** (PR #17 / #18) — 3 epochs × 737 steps = 2,211 steps, 557,172
  tokens processed:
  - epoch 1: train **1.0431** · val **0.3447**
  - epoch 2: train **0.2930** · val **0.2563**
  - epoch 3: train **0.2521** · val **0.2473**
  - wall time **43.5–44.5 s** (~8.2 s/epoch) · throughput **12,508 tok/s** ·
    peak RSS **313.1–314.2 MiB** (in-process `ru_maxrss`)
  - checkpoint `step-2211.pt` + sidecar `tokenizer.json` + `metrics.json`
    (`talos-training-checkpoint-v1` format)
- **Evaluation** (PR #18) — on `step-2211.pt` against the same val split, with
  the checkpoint-recorded batch defaults (seq 64 / batch 4 / drop_last):
  - val loss **0.2472547096607489** — bit-for-bit identical to the loss recorded
    in the checkpoint; a second independent CLI run reproduced it exactly
  - val perplexity **1.2805** · val next-token accuracy **0.918110**
  - train loss (same checkpoint) **0.2342**
  - eval throughput **72,819–77,191 tok/s** (wall-clock, varies with load) ·
    eval peak RSS **230.7–231.3 MiB** · eval wall ~2.7 s
- **Generation** (PR #19) — greedy KV-cache decode at **~1.1 ms/token** on CPU.
  Output is expected low-quality babble at this scale, e.g.:
  - `"The capital of France is"` → `" beproducibility: the key idea i"`
  - `"Once upon a time"` → `"n prate three main steps: the vo"`
  - `"How do I bake a cake?"` → `" Please be concrete.\n### Assista"`
  Reported honestly: the point of this scale is a reliable, deterministic
  train → eval → generate pipeline, **not** model quality.
- **Tokenizer trainer** (PR #16) — the streaming BPE trainer that backs this
  run handled the ~1,800-doc / 764-merge workload (which previously killed a
  VM) at **peak RSS 62.4 MiB**, with knobs `max_docs` / `max_chars` /
  `num_merges` / `minfreq` and atomic saves.

Verdict at this scale: the full loop config → model → tokenizer → data →
training → checkpoint → eval → generation is exercised and regression-tested
(255 passed / 2 skipped), with committed artifacts in `benchmarks/` (baseline,
quantization, DDM, and the consolidated `benchmarks/report-tiny.md`).

### 2.1b The ~1M row — the real measured run (Phase B, 2026-09-19)

Full detail: `benchmarks/phase-b/report-tiny-1m.md` (committed with this PR,
including the owner's comparison table). Summary (CPU: Intel Xeon @ 2.90 GHz,
2 cores, torch 2.13.0+cpu, **no GPU**):

- **Data:** 2,000-doc real OASST1 subset (rows 0–1999, provenance recorded) →
  fixed-seed 1,800/200 split → train-split-only BPE, **764 merges / vocab 1024**
  (~105 s; byte-identical to the owner's published tokenizer).
- **Training:** 3 epochs × 3,847 steps = **11,541 steps**, 2,908,332 tokens,
  lr 3e-3, batch 4, seq 64. Train 2.6426→2.3516→**2.1826**; val
  2.6759→2.5017→**2.3950**. Wall 422.5 s (train phase 317.5 s), peak RSS
  **378.2 MiB**, checkpoint **4,014,773 B**.
- **Eval:** val loss **bit-exact** vs recorded (2.3949950500474033), ppl
  10.9681, acc 0.3309, 31,220 tok/s, peak RSS 237.8 MiB; canonical 1,000,320
  guard enforced.
- **A/B verdict (identical data/tokenizer/steps):** tiny final val 2.3704 vs
  tiny_1m 2.3950 — **no val-loss improvement from 4× params at this budget**
  (1M model underfits; 2.8× slower per step). Reported honestly; no
  superiority claim.
- **Generation:** 5 prompts × {local tiny, local tiny_1m, published HF T4
  baseline} greedy; decode 1.3 / 2.0 / 1.1 ms/token; all guards pass; output
  is babble everywhere (expected at this scale).

### 2.1c The 15.5M-token A/B (Phase B follow-up, 2026-09-19) — long-budget 254K-vs-1M
Full detail: `benchmarks/phase-b/report-tiny-1m-15M.md`. Owner directive: re-run the A/B at the
**published baseline's exact step shape** (7,680 steps × batch 32 × seq 64 ≈ **15.48M tokens**
per model) to test whether `tiny_1m` overtakes `tiny` given ~5.3× more tokens than the Phase-B
3-epoch budget (2.91M). No code changes — the existing `--epochs / --max-steps-per-epoch /
--batch` knobs express the shape (at batch 32 the packed corpus is exactly 480 batches/pass →
`--epochs 16 --max-steps-per-epoch 480 --batch 32` = exactly 7,680 steps; final checkpoint
`step-7680.pt`, matching the published checkpoint name).
- **Memory gate (batch 32):** tiny_1m one-pass probe → peak RSS **504.3 MiB** (gate < 1.5 GB) ✓;
  19,607 tok/s train phase; 480 steps/epoch fixed. (`probe-batch32.json`.)
- **tiny (254,272):** 7,680 steps / 15,482,880 tokens, lr 3e-3, batch 32. Train 3.0169→**1.7937**;
  val 2.8544→**1.9275** (16 per-480-step measurements). Wall 427.3 s run total (train phase
  320.8 s), peak RSS **441.5 MiB**, checkpoint 1,026,842 B. Eval: val loss **bit-exact** vs
  recorded (1.9275096343604716), ppl 6.8724, acc 0.4428, 142,400 tok/s. **CPU-vs-T4 anchor:
  published T4 row is train 1.7859 / val 1.9177 — CPU lands within +0.008 train / +0.010 val at
  the identical step shape** (T4 ran an unrecorded data order / eval protocol; see report §6).
- **tiny_1m (1,000,320):** same shape — train 3.0015→**1.7345**; val 2.8738→**1.8697** (16
  per-480-step measurements; val strictly decreased every epoch). Wall 1,172.0 s run total
  (train phase 1,065.2 s → **14,535 tok/s**), peak RSS **546.0 MiB** train / **269.2 MiB** eval,
  checkpoint 4,014,737 B. Eval: val loss **bit-exact** vs recorded (1.8697498154128187), ppl
  6.4867, acc 0.4583, val-only throughput 43,479 tok/s; `--train-data` pass reports train loss
  1.8148 under the eval protocol (see report §7/§10 for why it differs from the loop's recorded
  epoch mean). Canonical 1,000,320 guard enforced.
- **A/B question** (does 1M overtake 254K with enough tokens?): **YES at 15.5M tokens** — local
  tiny_1m final val **1.8697 < tiny's 1.9275** (Δ −0.058, −3.0%) and train 1.7345 < 1.7937,
  acc 0.4583 > 0.4428. Cross-over at epoch 14 / step 6,720 (~13.5M tokens). At the 3-epoch
  budget 1M lost (2.3950 vs 2.3704); with ~5.3× more tokens it converts the budget into a
  *larger* val gain (Δ −0.525 vs tiny's −0.443) and overtakes. Cost: 3.3× training wall
  (1,065.2 s vs 321.2 s train phase), 14.5K vs 48.2K tok/s. Single-budget point; both curves
  still falling at step 7,680 — see the three-way view in
  `benchmarks/phase-b/report-tiny-1m-15M.md` §6.
## 3. Not-yet-run scales (placeholders)

The ~1M / ~10M / ~100M rows are the scaling ladder; only the ~1M row has been
run so far (Phase B, 2026-09-19 — real OASST1 data; see §2.1b below). The
~10M / ~100M rows are **planned, not executed**. When each scale is
run, it must fill the table with the **same metric conventions** (§5) so rows
are comparable. Per-scale expectations below are **honest engineering
estimates, not results** — they exist only to scope the run.

### What would be recorded per scale (same for all sizes)

Same harness as the 254K row: `--data <train.jsonl>` / train-split-only BPE /
deterministic split / per-epoch train+val loss / checkpoints per epoch / eval
via `scripts.eval_checkpoint` / generation via `scripts.generate`. Recorded
per row: parameter count (exact), final train loss, final val loss, val
perplexity, val next-token accuracy, tokens processed, train throughput (tok/s),
eval throughput (tok/s), decode latency (ms/token), peak RSS (train and eval),
wall time, corpus + config fingerprint, machine/environment provenance (torch
version, CPU/GPU, thread count, git commit, seed). Loss curves are expected to
be published alongside (available in the per-epoch checkpoints already).

### ~1M parameters (estimated hardware)

- **Rough config** (now precisely defined as the `tiny_1m` preset — see
  "Config: `tiny_1m`" below): dense, the `tiny` architecture scaled up on the
  same code path (`configs/`), keeping `vocab_size=1024` and `max_seq_len=512`
  unchanged.
- **Honest estimate:** fits on a single modern CPU (a few hundred MiB–1 GiB
  RSS) or trivially on any GPU. Expect training wall-times of minutes to ~1 h
  on the same 400-doc / 3-epoch workload, CPU throughput plausibly in the same
  order of magnitude as tiny (measured, not guaranteed, lower per-token FLOPs
  aside — *estimate only*).
- **Purpose:** first check that the loss/perplexity improvements of the 254K
  row persist at ~4× parameters on the same corpus, and a first curve point
  for scaling documentation.

#### Config: `tiny_1m` (implemented; Phase B training run complete — see §2.1b)

The ~1M scaling step is **defined and has one complete training run** (Phase B,
2026-09-19, real OASST1 2,000-doc subset). This block documents the config;
the measured numbers live in the §2 progression table and in
`benchmarks/report-tiny-1m.md`.

**Architecture — `tiny` scaled up, no redesign.** `tiny_1m`
(`configs/presets.py`, canonical registry `configs/canonical.py`) keeps every
architectural ratio of the 254K `tiny` preset and changes only the model-size
numbers:

| knob | `tiny` (254K) | `tiny_1m` (~1M) | rationale |
|---|---:|---:|---|
| `hidden_size` | 64 | 128 | 2× width |
| `num_layers` | 2 | 3 | 1.5× depth |
| `num_attention_heads` | 4 | 8 | scaled with hidden (2:1 GQA kept) |
| `num_kv_heads` | 2 | 4 | 2:1 GQA ratio unchanged |
| `head_dim` | 16 | 16 | unchanged |
| `intermediate_size` | 256 | 512 | FFN stays at exactly 4× hidden |
| `ffn_type` | dense | dense | unchanged |
| `vocab_size` | 1024 | 1024 | unchanged (tokenizer contract intact) |
| `max_seq_len` | 512 | 512 | unchanged |
| `attention_type` / RoPE | full / theta 1e4 | full / theta 1e4 | unchanged |
| `tie_word_embeddings` | False | False | un-tied, unchanged |
| biases | none | none | unchanged |

**Exact parameter count: 1,000,320** (verified programmatically —
`TalosGPT(tiny_1m_config().derive()).num_parameters() == 1_000_320`, asserted
by `tests/test_tiny_1m.py` and enforced by the canonical registry
`configs/canonical.py`).

**Parameter arithmetic** (dense, un-tied embeddings, no biases — the same
formula as `ModelConfig.param_count_breakdown`):

- embedding = `V · H = 1024 · 128 = 131,072`
- lm_head (un-tied) = `V · H = 131,072`
- per-layer attention = `H·(Hq·d) + H·(Hkv·d) + H·(Hkv·d) + (Hq·d)·H`
  = `128·128 + 128·64 + 128·64 + 128·128` = `16,384 + 8,192 + 8,192 + 16,384`
  = **49,152**
- per-layer norms = `2 · H = 256`
- per-layer FFN = `3 · H · I = 3 · 128 · 512` = **196,608**
- per layer total = `49,152 + 256 + 196,608` = `246,016`; × 3 layers = `738,048`
- final norm = `H = 128`
- **total = `131,072 + 131,072 + 738,048 + 128` = 1,000,320**

**Design notes vs the 254K comparison.** Total parameters scale ~3.93×
(1,000,320 / 254,272 ≈ 3.93), landing in the 0.9–1.1M target. Depth is 3
layers rather than the 4 first sketched during design: keeping the tiny FFN
ratio at exactly 4× hidden, 4 layers would give 1,246,336 params (~25% over
the 1.1M ceiling), while 3 layers at the *same* ratios give 1,000,320. This
preserves every architectural ratio of the 254K model (2:1 GQA, head_dim 16,
FFN 4× hidden, full attention, un-tied embeddings, no biases) instead of
re-tuning the FFN width, so the parameter scaling is attributable purely to
`hidden 64→128` and `layers 2→3` — the cleanest apples-to-apples scaling
comparison. `vocab_size` and `max_seq_len` are unchanged, so the existing
vocab-1024 tokenizer (`tiny_tokenizer_config`) is used verbatim with no
retraining, and the `tokenizer_vocab ≤ model_vocab` contract is untouched.

### ~10M parameters (estimated hardware)
- **Rough config** (now precisely defined as the `tiny_10m` preset — see
  "Config: `tiny_10m`" below): dense, the `tiny_1m` architecture *widened* on
  the same code path (`configs/`), keeping `vocab_size=1024` and
  `max_seq_len=512` unchanged.
- **Honest estimate:** comfortably fits one mid-range GPU (8–16 GiB) or a
  many-core CPU. fp32 weights are ~40 MiB and AdamW optimizer state ~80 MiB,
  so main memory is dominated by activations (vanishingly small at
  batch 32 × seq 64; a few GiB at batch 32 × seq 512 on a T4). Training times
  on the order of tens of minutes to a few hours for a 2,000-doc OASST1
  subset on a T4 — the first *meaningful* quality datapoint beyond 1M params.
- **Purpose:** the next point on the scaling ladder — re-check whether the
  1M-vs-254K advantage persists at ~10× parameters on the same corpus and
  step shape, before any MoE-shaped work.

#### Config: `tiny_10m` (implemented; validation complete, training pending)
The ~10M scaling step is **defined, registered as a canonical preset, and
validated end-to-end** (construction, forward/backward, checkpoint round-trip,
generation, full test suite); a real OASST1 training run is the pending next
experiment. This block documents the config; measured numbers will live in the
§2 progression table once the run completes.

**Architecture — `tiny_1m` widened, no redesign.** `tiny_10m`
(`configs/presets.py`, canonical registry `configs/canonical.py`) keeps every
architectural ratio of the 254K `tiny` / 1M `tiny_1m` presets and changes only
the width numbers:
| knob | `tiny_1m` (~1M) | `tiny_10m` (~10M) | rationale |
|---|---:|---:|---|
| `hidden_size` | 128 | 448 | 3.5× width |
| `num_layers` | 3 | 3 | unchanged (pure width scaling) |
| `num_attention_heads` | 8 | 28 | scaled with hidden (2:1 GQA kept) |
| `num_kv_heads` | 4 | 14 | 2:1 GQA ratio unchanged |
| `head_dim` | 16 | 16 | unchanged |
| `intermediate_size` | 512 | 1792 | FFN stays at exactly 4× hidden |
| `ffn_type` | dense | dense | unchanged |
| `vocab_size` | 1024 | 1024 | unchanged (tokenizer contract intact) |
| `max_seq_len` | 512 | 512 | unchanged |
| `attention_type` / RoPE | full / theta 1e4 | full / theta 1e4 | unchanged |
| `tie_word_embeddings` | False | False | un-tied, unchanged |
| biases | none | none | unchanged |
**Exact parameter count: 9,952,320** (verified programmatically —
`TalosGPT(tiny_10m_config().derive()).num_parameters() == 9_952_320`, asserted
by `tests/test_tiny_10m.py` and enforced by the canonical registry
`configs/canonical.py`).
**Parameter arithmetic** (dense, un-tied embeddings, no biases — the same
formula as `ModelConfig.param_count_breakdown`):
- embedding = `V · H = 1024 · 448 = 458,752`
- lm_head (un-tied) = `V · H = 458,752`
- per-layer attention = `H·(Hq·d) + H·(Hkv·d) + H·(Hkv·d) + (Hq·d)·H`
  = `448·448 + 448·224 + 448·224 + 448·448` = `200,704 + 100,352 + 100,352 + 200,704`
  = **602,112**
- per-layer norms = `2 · H = 896`
- per-layer FFN = `3 · H · I = 3 · 448 · 1792` = **2,408,448**
- per layer total = `602,112 + 896 + 2,408,448` = `3,011,456`; × 3 layers = `9,034,368`
- final norm = `H = 448`
- **total = `458,752 + 458,752 + 9,034,368 + 448` = 9,952,320**
**Design notes vs the ~1M comparison.** Total parameters scale ~9.95×
(9,952,320 / 1,000,320 ≈ 9.95), landing in the 9–11M target (the ~10M analog
of the 0.9–1.1M band used for `tiny_1m`). Width was chosen over depth: keeping
the same 3 layers as `tiny_1m` and widening `hidden 128→448` (with heads
8→28, FFN 512→1792 — every ratio preserved) makes the ~10× parameter jump
attributable purely to width, the cleanest apples-to-apples scaling
comparison, mirroring how `tiny_1m` scaled from `tiny` (hidden 64→128,
layers 2→3). `vocab_size` and `max_seq_len` are unchanged, so the existing
vocab-1024 tokenizer (`tiny_tokenizer_config`) is used verbatim with no
retraining, and the `tokenizer_vocab ≤ model_vocab` contract is untouched.

### ~100M parameters (estimated hardware)

#### Config: `tiny_100m` (implemented; validation complete, training pending)

- **Config:** dense, the canonical family's **vocab 1024** (per the audit §14
  recommendation — single source of truth `configs.vocab.VOCAB_SIZE`), hidden
  **1024**, **6 layers**, 64/32 GQA heads (2:1), head_dim 16, dense SwiGLU FFN
  at 4× hidden, un-tied embeddings, no biases, seq 512. Exact, registry-pinned
  parameter count: **96,482,304** (`hidden-1024 × 6` was chosen over
  `hidden-896 × 8` — see the arithmetic in `configs/presets.py`
  `tiny_100m_config` and the audit §13 memory table, which was computed for
  exactly this shape).
- **Honest estimate:** fits a single 16 GB T4 in fp32 (audit §13: ~2.2–2.4 GiB
  at batch 32 × seq 64, ~7.7–9.2 GiB at batch 32 × seq 512, incl. CUDA
  overhead); weights + grads + AdamW = 16×P ≈ 1.44 GiB regardless of batch.
  Recommended T4 starting point: batch 32 × seq 64 (same step shape as the
  10M runs) — see the PR validation ladder for the per-shape memory table.
- **Purpose:** pre-announced stopping point — the largest single-GPU dense
  run the team plans before any MoE work; the last scale where a claimed
  "scaling trend" can be honestly supported by measured points. The preset is
  implemented and fully validated (exact count, forward/backward, checkpoint
  round-trip, generation smoke, suite green); a training run is the next
  owner decision (see the business plan's in-flight section).

> **Note on existing presets:** `configs/presets.py` already contains
> `small` (35.7M), `medium` (285M), `large` (1.64B) and the `100b`/`400b` MoE
> configs. These are **design artifacts only** — parameter/FLOP estimates via
> `configs.compute`, none have been trained. The progression grid above uses
> dedicated dense configs sized to 254K → 1M → 10M → 100M; the 254K/1M/10M/100M
> rows have canonical presets (`tiny`/`tiny_1m`/`tiny_10m`/`tiny_100m`), and
> having a registered preset is **not** evidence of a trained model — the 100M
> preset is validated but untrained as of this writing.

## 4. Reproducing the 254K row (exact commands, CPU-only)

```bash
# (1) synthetic OASST1-style corpus (no network pulls; real OASST1 JSONL also works)
python -m tools.make_synthetic_oasst1 --docs 400 --seed 0 --output /tmp/oasst1.jsonl

# (2) split -> train-split-only BPE -> 254,272-param guard -> 3 epochs -> checkpoints + metrics
python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir /tmp/talos_run \
    --epochs 3 --seq 64 --batch 4 --seed 0

# (3) eval (val split resolved from the checkpoint; recomputed loss should be bit-exact)
python -m scripts.eval_checkpoint --checkpoint /tmp/talos_run --seed 0

# (4) generation (expect babble at this scale — that is the honest expectation)
python -m scripts.generate --checkpoint /tmp/talos_run \
    --prompt "The capital of France is" --max-new-tokens 32

# (5) whole test suite
python -m pytest -q
```

## 5. Metric conventions

- **Loss** = natural-log mean cross-entropy (`-log p`, nats); **perplexity** =
  `exp(loss)` on the same base; **next-token accuracy** = argmax agreement over
  the same batches. All defined and computed in `evaluation/harness.py`.
- **Throughput** = wall-clock tokens/s (hardware-specific; environment
  provenance is recorded in every artifact, see `benchmarks/*.json`).
- **Peak RSS** = in-process `resource.getrusage().ru_maxrss`, labeled honestly
  as a *process* metric.
- **Determinism** = fixed seed + file-order pack batches (no RNG in the data
  pipeline); a claim of determinism requires two identical outputs.
- **Comparability** = rows are comparable only if corpus, tokenizer config,
  seq, batch, lr, and optimizer match; this is recorded per row.

## 6. Known limitations (be explicit)

- All measurements to date are **single-machine CPU or the owner's published
  single-GPU (T4) reference**; no multi-node numbers exist anywhere in this
  repo.
- The original 254K benchmark row uses a **synthetic** corpus; the ~1M Phase B
  row uses a **real OASST1 2,000-doc subset** (corpus differs per row — see
  each row's own data section; rows are not cross-comparable).
- **No long-context** data or runs exist; the 128K-context target is
  architecture-only (implemented + unit-tested) until a training run at that
  context length exists.
- The 400B/30B-active MoE target stays deferred until the progression above
  has produced measured evidence at ≥2 scales beyond 254K and the owner
  explicitly green-lights scaling work.