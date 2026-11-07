# Talos project status

**This is the single project-status overview.** It is the one table that says
what exists, what has actually been run, and how strong the evidence is for
each claim. Every other document links here instead of keeping its own
project-level status table. Last verified against `main` on **2026-11-07**.

Evidence levels used below (kept deliberately narrow):

| Level | Means |
|---|---|
| **measured** | A real run happened and its artifact (checkpoint, metrics, benchmark JSON, held-out score) is in this repo. Numbers may be quoted. |
| **verified in CI** | Code exists, is exercised by the test suite, and the tests pass in CI. No training/benchmark numbers are implied. |
| **historical** | A run happened outside this repo (owner-reported). No numbers are asserted here unless a repo artifact backs them. |
| **estimate** | Arithmetic or a memory model computed from a config — not a measurement. |
| **design only** | A config/plan exists; nothing is built or run. |

---

## Status overview

| Area | Status | Evidence level | Where |
|---|---|---|---|
| **Canonical ladder** — `tiny` 254,272 / `tiny_1m` 1,000,320 / `tiny_10m` 9,952,320 / `tiny_100m` 96,482,304 params (vocab 1,024, dense decoder-only, 2:1 GQA, RoPE, head_dim 16, SwiGLU, seq 512, un-tied, no biases) | Implemented; configs **frozen** and registry-pinned to their exact counts | verified in CI (`tests/test_configs.py`, `tests/test_tiny_1m.py`, `tests/test_tiny_10m.py`, `tests/test_tiny_100m.py`) | `configs/presets.py`, `configs/canonical.py` |
| **`tiny` (254K) training + eval + generation** | Trained end-to-end on a deterministic synthetic OASST1-style corpus; loss decreases on train and val; output is babble (expected at this scale) | measured | `benchmarks/report-tiny.md`, `docs/SCALING.md` §2.1 |
| **`tiny_1m` (1M) training** | Trained end-to-end on a real OASST1 2,000-doc subset (Phase B), plus a 15.5M-token A/B against `tiny` | measured | `benchmarks/phase-b/report-tiny-1m.md`, `benchmarks/phase-b/report-tiny-1m-15M.md` |
| **`tiny_10m` (10M) training** | Config implemented and validated (construction, fwd/bwd, checkpoint round-trip, generation). **No 10M training artifacts (loss, metrics, checkpoint) exist in this repo.** The owner has described a historical 10M run; nothing here reproduces or verifies it, so no numbers are stated — *owner input needed* | config: verified in CI; run: historical / unverified | `docs/SCALING.md` §3 |
| **`tiny_100m` (100M) training** | Config implemented and validated; **not trained** | config: verified in CI; run: none | `docs/SCALING.md` §3, `docs/trainer.md` |
| **Talos 1B presets** — `talos_1b` **1,036,060,672** (16 L, FFN 7,808, vocab **49,152**, **primary**) · `talos_1b_21l` 995,973,120 · `talos_1b_spec16` 774,440,960 (both vocab 32,000; all: seq 2,048, hidden 2,048, 16Q/4KV GQA, head_dim 128, RMSNorm pre-norm, RoPE, tied embeddings, no biases, SwiGLU) | **Architecture implemented and tested — NOT trained.** No 1B checkpoint exists anywhere; no 1B-scale measurement of any kind exists | verified in CI (`tests/test_talos_1b.py`, `tests/test_vocab_seam.py`) — on the **meta device**, because the team CPU box cannot hold a 1B fp32 model | `configs/presets.py`, `configs/canonical.py`, `docs/SCALING.md` §7 |
| **1B training engine** — BF16/FP16 AMP (opt-in), gradient checkpointing, gradient accumulation, SDPA with plain fallback, grad clipping, always-on NaN/Inf detection with safety checkpoints, checkpoint/resume protections | Implemented and tested; all flags are opt-in and the no-flag run is bit-identical to the legacy engine | verified in CI (`tests/test_trainer_engine_w2.py`, `tests/test_trainer_stability.py`, `tests/test_trainer_token_budget.py`) | `scripts/train_oasst1.py`, `docs/trainer.md` §31 |
| **`talos_1b_32k` tokenizer (v2)** | Shipped asset: vocab 32,000 (256 bytes + 4 specials + 31,740 merges), exact-BPE encoder, sha256-pinned, trained on a documented 16,000-row FineWeb-Edu sample that is **not** committed. Paired with `talos_1b_21l` / `talos_1b_spec16` | measured (held-out compression) + verified in CI (`tests/test_tokenizer_32k.py`, `tests/test_tokenizer_regression.py`) | `assets/tokenizers/`, `docs/tokenizer.md` |
| **`talos_1b_49k` tokenizer** | Shipped asset, **paired with `talos_1b`** (`"presets": ["talos_1b"]`) — the owner decided the 1B vocabulary on **2026-11-07**: vocab 49,152, exact-BPE encoder, sha256-pinned, same documented 16,000-row sample | measured (held-out compression) + verified in CI (`tests/test_tokenizer_49k.py`) | `assets/tokenizers/`, `docs/tokenizer.md` |
| **Canonical (1,024-vocab) tokenizer path** | Each canonical run trains its own vocab-1024 BPE inside the run; the legacy heap encoder is retained and remains the default for these presets | verified in CI | `docs/tokenizer.md`, `scripts/train_oasst1.py` |
| **Packed corpus preparation** (`scripts/prepare_corpus.py`) | Implemented: streaming HF corpus → packed `.npy` shards + manifest, disjoint val region, atomic writes | verified in CI (`tests/test_corpus_prep.py`) | `docs/trainer.md` §8 |
| **Training corpus** | Owner-prepared: **205,002,065 train + 15,001,394 val tokens**, 46 packed seq-512 shards. Lives on the **owner's Drive, not in this repo** — no corpus numbers are reproducible from a clone | owner-reported; not in repo | `docs/trainer.md` §7 |
| **CPU benchmarks (254K, 1M)** | Throughput, peak RSS, decode latency, loss trajectories recorded with the reproducible harness | measured (CPU, single machine) | `benchmarks/`, `benchmarks/phase-b/` |
| **1B benchmark tooling** (`scripts/benchmark_1b.py`) | Exists for the **owner** to run on their RTX 3060 12 GB (OOM microbatch ladder, activation matrix, save→kill→resume check, inference timing). **It has not been run; no RTX 3060 or any GPU result exists in this repo** | tool: verified in CI (`tests/test_benchmark_1b.py`); results: none | `scripts/benchmark_1b.py`, `docs/SCALING.md` §7 |
| **Inference** | Canonical prefill/decode path with KV cache; prefill == KV-cache decode exactly; greedy + sampling CLI | measured (CPU latency) + verified in CI | `inference/generate.py`, `scripts/generate.py` |
| **Quantization** | int8 viable at 254K (measured experiment); nothing quantized beyond that | measured (CPU, 254K only) | `benchmarks/quant-tiny-cpu.md` |
| **Styx-100M** (separate, earlier model) | Previously trained; **generation quality is unresolved** — it is not a usable general-purpose model. The anti-repetition decoding controls (remedy R1) are merged into this repo's generation CLI. No Styx checkpoint or metrics are in this repo | historical; R1 knobs verified in CI | `scripts/generate.py`, `docs/trainer.md` §21.4 |
| **~400B total / ~30B active MoE, 128K context** | Long-term research target. A **design artifact only**: a config plus a parameter/FLOP estimate. Owner-gated — not built, trained, or simulated, and not current work | design only / estimate | `configs/presets.py`, `docs/architecture.md` §9 |

---

## What this means in one paragraph

Four small presets (254K → 100M) are implemented and registry-pinned; the 254K
and 1M presets have real training runs with committed metrics, the 10M and 100M
presets do not. The Talos 1B architecture is complete, tested, and deliberately
untrained: no checkpoint, no 1B-scale loss, and no GPU benchmark exists. The
1B track has two shipped, sha256-verified tokenizer assets — the 49K asset now
paired with the primary `talos_1b` (owner decision 2026-11-07, which raises it
to 1,036,060,672 params) and the 32K v2 asset paired with `talos_1b_21l` /
`talos_1b_spec16`; both have measured held-out compression figures. Everything
larger than 100M except the 1B presets is a config, not a result.

## Related documents

- `docs/SCALING.md` — the per-scale progression table, memory models, and the
  1B architecture arithmetic.
- `docs/trainer.md` — the training execution reference (flags, guards,
  checkpoint format, resume).
- `docs/tokenizer.md` — tokenizer design, the shipped assets, and the pairing
  guards.
- `docs/architecture.md` — design decisions for the model code.
- `benchmarks/` — the committed measurement artifacts.
