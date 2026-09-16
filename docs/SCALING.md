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
- **No model larger than 254,272 parameters has ever been trained in this
  repository.** The codebase contains untrained *configs* for larger sizes
  (`small` 35.7M, `medium` 285M, `large` 1.64B, `100b`, `400b` — verifiable via
  `python -m configs.compute --summary`), but a config is an estimate, not a
  result. Configs become results only when a training run is executed,
  checkpointed, and evaluated through the harness.
- This document claims nothing beyond measured or clearly-labeled-estimated
  numbers. The progression table below is the authoritative scoreboard.

## 2. The progression table

Status legend: ✅ **measured** (real run, artifacts in this repo) · ⬜ *not yet
run* (placeholder — planned, no results).

| Scale (dense) | Parameters | Train loss* | Val loss* | Throughput | Peak memory | Convergence notes | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 254K (`tiny`) | 254,272 | **0.2521** (final, epoch 3) | **0.2473** (final, epoch 3) | **~12.5K tok/s** train · **~73–77K tok/s** eval · **~1.1 ms/token** decode | **~314 MiB** train · **~231 MiB** eval (peak RSS) | Loss strictly decreased on train *and* val across all 3 epochs; deterministic under seed 0; eval recompute bit-exact vs. recorded val loss. Output text is incoherent babble (expected at this scale). | ✅ PRs #17–#19 |
| ~1M | *not yet run* | — | — | — | — | — | ⬜ |
| ~10M | *not yet run* | — | — | — | — | — | ⬜ |
| ~100M | *not yet run* | — | — | — | — | — | ⬜ |

\* Natural-log mean cross-entropy (nats), measured on the fixed synthetic-OASST
JSONL corpus (400 docs, 360/40 train/val split, BPE vocab 1024, seq 64, batch 4,
lr 3e-3, AdamW). Loss values are only comparable across runs that share this
corpus/configuration — they are **not** comparable to public benchmarks.

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

## 3. Not-yet-run scales (placeholders)

The ~1M / ~10M / ~100M rows are **planned, not executed**. When each scale is
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

- **Rough config** (to be defined precisely): dense, vocab ~2048, hidden ~256,
  4–6 layers — a scaled-up `tiny` on the same code path (`configs/`), with a
  `model_vocab_size >= tokenizer_vocab_size` tokenizer (e.g. 2048).
- **Honest estimate:** fits on a single modern CPU (a few hundred MiB–1 GiB
  RSS) or trivially on any GPU. Expect training wall-times of minutes to ~1 h
  on the same 400-doc / 3-epoch workload, CPU throughput plausibly in the same
  order of magnitude as tiny (measured, not guaranteed, lower per-token FLOPs
  aside — *estimate only*).
- **Purpose:** first check that the loss/perplexity improvements of the 254K
  row persist at ~4× parameters on the same corpus, and a first curve point
  for scaling documentation.

### ~10M parameters (estimated hardware)

- **Rough config:** dense, vocab ~4096, hidden ~512, 8 layers (small-cluster
  fit, per `configs/presets.py` design).
- **Honest estimate:** comfortably fits one mid-range GPU (8–16 GiB) or a
  many-core CPU; expect multi-GiB RSS and training times on the order of tens
  of minutes to a few hours for a 400-doc synthetic corpus. A larger corpus
  (real OASST1 subset, no network pulls at build time) would make this the
  first *meaningful* quality datapoint — still a toy by LLM standards.
- **Purpose:** first scale where tokenizer richness (vocab > 1024) and
  context length (seq 256–512) become non-trivial; first checkpoint large
  enough that quantization (already shown viable at tiny scale) matters for
  storage.

### ~100M parameters (estimated hardware)

- **Rough config:** dense, vocab ~8192, hidden ~1024, 16–24 layers (single-
  high-end-GPU or small-multi-GPU fit).
- **Honest estimate:** needs a high-end GPU (24+ GiB) or a small multi-GPU
  node; expect tens of GiB RSS; training from hours to ~a day on a modest
  corpus. BF16 weights ~200 MiB and AdamW optimizer state still fit one large
  GPU. This is the first scale where hybrid/sliding-window attention (already
  implemented and unit-tested) would be exercised in a real training run.
- **Purpose:** pre-announced stopping point — the largest single-GPU dense
  run the team plans before any MoE work; the last scale where a claimed
  "scaling trend" can be honestly supported by measured points.

> **Note on existing presets:** `configs/presets.py` already contains
> `small` (35.7M), `medium` (285M), `large` (1.64B) and the `100b`/`400b` MoE
> configs. These are **design artifacts only** — parameter/FLOP estimates via
> `configs.compute`, none have been trained. The progression grid above uses
> dedicated dense configs sized to 254K → 1M → 10M → 100M; adding those configs
> is part of the follow-up work, not evidence of a trained model.

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

- All measurements to date are **single-machine CPU**; no GPU and no
  multi-node numbers exist anywhere in this repo.
- The corpus is **synthetic**; real-data runs are planned but not yet measured.
- **No long-context** data or runs exist; the 128K-context target is
  architecture-only (implemented + unit-tested) until a training run at that
  context length exists.
- The 400B/30B-active MoE target stays deferred until the progression above
  has produced measured evidence at ≥2 scales beyond 254K and the owner
  explicitly green-lights scaling work.