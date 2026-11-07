<img width="432" height="124" alt="Talos" src="https://github.com/user-attachments/assets/b4cbb3f5-875f-419d-a74f-1eb073459a78" />

# Talos

**Talos** is an open-source research project by **Promethean Studios** building a
family of small, reproducible decoder-only language models on one code path, and
documenting exactly what has and has not been measured at each scale.

## Status at a glance (2026-11-06)

- **Architecture and tooling: complete** for four canonical presets (254K, 1M,
  10M, 100M parameters) and three 1B presets. All are CI-tested at their exact
  parameter counts.
- **Trained:** the 254K and 1M presets, on small public corpora. Both produce
  incoherent text — they are pipeline validations, not useful models.
- **Talos 1B: implemented, not trained.** No 1B checkpoint exists, and no
  1B-scale measurement (loss, throughput, VRAM) exists. The architecture is
  tested on the meta device.
- **Not trained:** the 10M and 100M presets (configs only), and every larger
  design config.

The full evidence table — one row per area, each labeled *measured* / *verified
in CI* / *historical* / *estimate* / *design only* — is in
[`docs/STATUS.md`](docs/STATUS.md). Read that before quoting any number from
this repository.

## Model presets

### Canonical ladder (configs frozen, exact counts registry-pinned)

All four are dense decoder-only, vocab 1,024, 2:1 GQA, RoPE (theta 1e4),
head_dim 16, SwiGLU FFN at 4× hidden, max sequence 512, un-tied embeddings, no
biases, FP32.

| preset | parameters | hidden | layers | Q / KV heads | FFN |
|---|---:|---:|---:|---:|---:|
| `tiny` | **254,272** | 64 | 2 | 4 / 2 | 256 |
| `tiny_1m` | **1,000,320** | 128 | 3 | 8 / 4 | 512 |
| `tiny_10m` | **9,952,320** | 448 | 3 | 28 / 14 | 1,792 |
| `tiny_100m` | **96,482,304** | 1,024 | 6 | 64 / 32 | 4,096 |

The counts are enforced by `configs/canonical.py` and asserted by the tests, so
a drift fails CI rather than quietly changing the model.

### Talos 1B (tied embeddings; vocab 49,152 primary / 32,000 siblings)

Common architecture: max sequence 2,048 · hidden 2,048 ·
16 query / 4 KV heads (4:1 GQA) · head_dim 128 · RMSNorm pre-norm · RoPE ·
SwiGLU dense FFN · no biases · **tied** input/output embeddings · SDPA with a
plain functional fallback · BF16/FP16 AMP, gradient checkpointing and gradient
accumulation available as opt-in trainer flags. Vocabulary: **49,152** for the
primary `talos_1b` (owner decision 2026-11-07), **32,000** for `talos_1b_21l`
and `talos_1b_spec16`.

| preset | layers | FFN | parameters | note |
|---|---:|---:|---:|---|
| `talos_1b` | 16 | 7,808 | **1,036,060,672** | **primary** — the owner's FFN-adjusted 1B (2026-11-04) at vocab 49,152 (2026-11-07) |
| `talos_1b_21l` | 21 | 5,504 | **995,973,120** | earlier depth-adjusted variant, preserved (vocab 32,000) |
| `talos_1b_spec16` | 16 | 5,504 | **774,440,960** | the literal 16-layer spec (vocab 32,000) |

These are **architecture, not results**: the presets are validated on the meta
device (the dev box cannot hold a 1B fp32 model), and no training run has been
started. See [`docs/SCALING.md`](docs/SCALING.md) §7 for the parameter and
memory arithmetic.

## Tokenizers

| asset | vocab | encoder | paired with | status |
|---|---:|---|---|---|
| `talos_1b_49k` | 49,152 | exact BPE | `talos_1b` | **primary** 1B tokenizer (owner decision 2026-11-07) |
| `talos_1b_32k` (**v2**) | 32,000 | exact BPE | `talos_1b_21l`, `talos_1b_spec16` | paired with the two 32K siblings |
| vocab-1024 BPE | 1,024 | legacy heap encoder (retained) | trained inside each canonical run | unchanged workflow for 254K/1M/10M/100M |

`talos_1b_32k` v2 is trained on a documented 16,000-row FineWeb-Edu sample
(ODC-By 1.0, sha256-pinned, not committed) and measures **4.5039 chars/token on
a 200-row held-out slice and 4.5195 on a 1,000-row held-out slice**, versus
4.4216 / 4.4273 for v1 on the identical text — i.e. **1.83% / 2.04% fewer
tokens**. Higher chars/token is better. `talos_1b_49k` measures 4.6411 / 4.6525
(−4.73% / −4.84% tokens vs v1, ~3% fewer tokens than the 32K v2).

**The owner decided the 1B vocabulary on 2026-11-07:** the primary `talos_1b`
moves to vocab 49,152 with `talos_1b_49k` (**1,036,060,672 params**), while
`talos_1b_21l` (995,973,120) and `talos_1b_spec16` (774,440,960) keep vocab
32,000 and `talos_1b_32k`. Both assets are sha256-pinned and unchanged.

Two rules that matter in practice:

1. **Assets are sha256-verified at load time**, and a tokenizer must pair with
   its model exactly (the 1B assets refuse every canonical preset; a
   1,024-vocab tokenizer refuses every 1B preset; `talos_1b_49k` and
   `talos_1b_32k` refuse each other's presets). Mismatches fail loudly.
2. **Any dataset tokenized with a different tokenizer — including the old
   1,024-vocab one, 32K v1, and the 32K asset itself once you train `talos_1b`
   — must be regenerated before use.** The pairing guard refuses mismatched IDs
   rather than silently training on them.

Details: [`docs/tokenizer.md`](docs/tokenizer.md).

## Training

The trainer (`scripts/train_oasst1.py`) supports two data paths — a JSONL path
that trains its BPE on the fly, and a packed-token path
(`scripts/prepare_corpus.py` → `.npy` shards + manifest). Implemented and
tested features:

- **AMP:** `--amp fp16` (autocast + GradScaler) or `--amp bf16` (autocast, no
  scaler; requires SM80+, so not on a T4). Default is FP32 and bit-identical to
  the pre-AMP engine.
- **Attention:** `--attention-backend sdpa` (default `auto` prefers it) with the
  plain functional fallback available and used by the canonical tests.
- **Memory:** `--grad-checkpoint` (activation checkpointing) and `--grad-accum N`
  (gradient accumulation; `--batch` is then the micro-batch).
- **Stability:** NaN/Inf loss and gradient detection is always on — a bad step
  skips the optimizer update, logs a structured event, writes a safety
  checkpoint, and aborts after `--max-consecutive-bad-steps` (default 3).
  `--grad-clip` bounds the norm and records the pre-clip value.
- **Checkpoints:** atomic writes, `--save-every-tokens N` intra-epoch cadence,
  `--token-budget` stop, step-cap and no-overwrite guards, completed-optimizer-
  step accounting, and resume validation on preset, parameter count, tokenizer
  fingerprint and corpus identity.

The training corpus is **owner-prepared and not in this repository**: 205,002,065
train + 15,001,394 validation tokens in 46 packed seq-512 shards, held on the
owner's Drive. Nothing in the repo can reproduce those exact tokens from a
clone.

Reference: [`docs/trainer.md`](docs/trainer.md).

## Quick start

```bash
# parameter counts for every preset (no download, no GPU)
python -m configs.compute --summary

# tiny forward/backward smoke test on CPU
python -m scripts.cli --steps 3 --seq 64

# train the 254K preset on a deterministic synthetic corpus (CPU, ~45 s)
python -m tools.make_synthetic_oasst1 --docs 400 --seed 0 --output /tmp/oasst1.jsonl
python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir /tmp/talos_run \
    --epochs 3 --seq 64 --batch 4 --seed 0

# eval + generation on the resulting checkpoint
python -m scripts.eval_checkpoint --checkpoint /tmp/talos_run --seed 0
python -m scripts.generate --checkpoint /tmp/talos_run \
    --prompt "The capital of France is" --max-new-tokens 32

# tokenizer compression on a held-out JSONL slice
python -m scripts.measure_tokenizer_compression --asset talos_1b_32k \
    --slice heldout=/tmp/fw_heldout_1k.jsonl

# test suite
python -m pytest -q
```

Expect incoherent text from the 254K model — the point of that run is a
deterministic config → data → train → checkpoint → eval → generate loop, not
language quality.

## Repository layout

```text
talos/
├── model/          # Transformer: attention backends, RoPE, RMSNorm, SwiGLU, MoE, KV cache
├── configs/        # Presets, canonical registry, parameter/FLOP helper
├── tokenizer/      # Byte-level BPE (train/encode/decode), committed-asset registry + guards
├── assets/tokenizers/  # Committed tokenizer assets + MANIFEST.json (sha256 pins)
├── data/           # Streaming data pipeline + packed-token dataset
├── training/       # Training-side utilities (schedules, synthetic corpora)
├── scripts/        # CLIs: corpus prep, training, eval, generation, tokenizer tools, benchmarks
├── inference/      # Canonical prefill/decode + generation
├── evaluation/     # Reproducible checkpoint evaluation harness
├── quantization/   # int8 experiments
├── distributed/    # Distributed-training scaffolding (not implemented)
├── benchmarks/     # Committed measurement artifacts
├── docs/           # Architecture, scaling, tokenizer, trainer, status
└── tests/          # Test suite
```

## Documentation

| Document | Covers |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | **Start here.** The single project-status overview, with an evidence level per row. |
| [`docs/architecture.md`](docs/architecture.md) | Every model design decision: GQA, RoPE/YaRN, RMSNorm, SwiGLU, MoE routing, attention abstraction, KV cache. |
| [`docs/SCALING.md`](docs/SCALING.md) | Measured scaling progression, metric conventions, the 1B parameter/memory arithmetic. |
| [`docs/tokenizer.md`](docs/tokenizer.md) | Tokenizer design, the shipped 32K/49K assets, pairing guards, regeneration rules. |
| [`docs/trainer.md`](docs/trainer.md) | Training execution reference: flags, guards, checkpoint format, resume semantics. |
| [`benchmarks/`](benchmarks/) | Committed measurement artifacts and their provenance. |
| [`data/README.md`](data/README.md) | The streaming data pipeline. |

## Scope

The long-term research direction is a much larger sparse (MoE) model; the
relevant configs exist as **design artifacts only** and are owner-gated — see
[`docs/architecture.md`](docs/architecture.md) §9. Nothing in this repository
trains, benchmarks, or simulates that model today.

## License

See the repository license for the current licensing terms.
