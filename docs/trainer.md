# Talos `tiny_100m` — T4/Colab Training Reference (authoritative, repo-state `d8bfbdd`)

**Target reader:** an agent or the owner executing the full `tiny_100m` pretraining run on a Google
Colab T4 with **minimal ambiguity**. Every component below is labeled **CURRENT** (implemented and
verified in this repo), **PARTIAL** (implemented with gaps), **MISSING** (needs implementation), or
**PROPOSED** (a plan or recommendation, not code). Nothing is documented that was not verified by
reading the code or running it on 2026-09-27; things that exist only in a plan or in the owner's
reported records are labeled as such.

Verification basis: repository `main` at `d8bfbdd` (merge of PR #28; corpus prep #27 and token-budget
trainer #28 merged). Live checks ran under `/opt/forge-venv/bin/python` (torch 2.13) with the repo on
`PATH`. **Honesty rule followed throughout: the trainer does NOT (yet) contain gradient clipping,
NaN/Inf detection, super-save checkpoints, a benchmark mode, GPU-memory/ETA logging, or a preflight
script. Those are stated as MISSING and specified, not implied to exist.**

---

## Status table (read this first)

| # | Component | Status | Source (file:line) |
|---|---|---|---|
| 1 | `tiny_100m` preset config = exactly **96,482,304 params** | CURRENT (verified live) | `configs/presets.py:96-126` (`tiny_100m_config`) |
| 2 | Canonical preset registry (`CANONICAL_PRESETS`, exact (params, vocab) tuples) | CURRENT | `configs/canonical.py:28-35` |
| 3 | Single vocab source of truth `VOCAB_SIZE = 1024` | CURRENT | `configs/vocab.py:44` |
| 4 | Model constructor `TalosGPT(config)` with GQA/RoPE/dense-SwiGLU, un-tied, bias-free | CURRENT | `model/gpt.py:24-66`, `model/block.py:26-40`, `model/ffn.py:33-48` |
| 5 | `ModelConfig` + `derive()` (head_dim / FFN width derived, validated) | CURRENT | `model/config.py:17-159` |
| 6 | `num_parameters()` exact count | CURRENT | `model/gpt.py:168-170` |
| 7 | Hard parameter-count guard in trainer (fails fast on drift) | CURRENT | `scripts/train_oasst1.py:503-533` (`check_preset_compat`), wired at `:1212-1215` |
| 8 | Token-ID bounds guards (batch assembly + embedding seam, 3 layers) | CURRENT | `model/gpt.py:100-104`, `model/utils.py:57`, `scripts/train_oasst1.py:717/448`, `data/packed.py:58-86` |
| 9 | Established tokenizer loaded via `--tokenizer-json`, sha256 fingerprint machinery | CURRENT | `scripts/train_oasst1.py:927-941, 977-991, 1140-1208` |
| 10 | Tokenizer identity (sha256) recorded in every checkpoint + `metrics.json`; verified on resume | CURRENT | `scripts/train_oasst1.py:626-683, 1308-1317` |
| 11 | Packed-corpus preparation CLI (`scripts/prepare_corpus.py`, HF streaming → `.npy` shards + manifest) | CURRENT | `scripts/prepare_corpus.py:557-591` (CLI), `:333-548` (core) |
| 12 | Packed dataset + manifest validation (`data/packed.py`) | CURRENT | `data/packed.py:87-268` (`load_packed_manifest`, `manifest_identity`, `packed_phase_shard_paths`), `:270-` (`PackedTokenDataset`) |
| 13 | Token-budget stop (`--token-budget`), warmup, cosine decay (min 10 %) | CURRENT | `scripts/train_oasst1.py:127-198` (`TokenSchedule`), `:950-962` (flags), `:828-840` (stop) |
| 14 | Resume: explicit `step-<N>.pt` OR directory with numeric-newest + corrupt-fallback | CURRENT | `scripts/train_oasst1.py:206-221, 993-1015` |
| 15 | Resume identity rejections (preset/params, tokenizer fp, data source, packed corpus identity) | CURRENT | `scripts/train_oasst1.py:222-252, 1060-1105, 1178-1208` |
| 16 | Checkpoint v1 format (weights+optimizer+RNG+counters+run metadata embedded) | CURRENT | `scripts/train_oasst1.py:626-683` |
| 17 | `train_run_metadata.json` sidecar (atomic write, same dict as in-checkpoint) | CURRENT | `scripts/train_oasst1.py:282-340` |
| 18 | `metrics.json` run report (losses, tokens, wall, peak RSS, tokenizer identity…) | CURRENT | `scripts/train_oasst1.py:1426-1494` |
| 19 | Validation per epoch (held-out val loss; mean CE → perplexity via eval harness) | CURRENT | `scripts/train_oasst1.py:588-625`, `evaluation/harness.py:80-99` |
| 20 | Eval CLI (`scripts/eval_checkpoint.py`) + generation CLI (`scripts/generate.py`) | CURRENT | `scripts/eval_checkpoint.py:47-85`, `scripts/generate.py:265-291` |
| 21 | Safetensors release export for the 100M preset | CURRENT | `scripts/export_safetensors.py:309-312` (+ tests `tests/test_safetensors_io.py`) |
| 22 | Grad clipping / NaN-Inf detection in the trainer loop | **MISSING** — trainer metadata explicitly records `gradient_clip_type: None`, `nan_inf_detection: False`; a code comment claims the guards "live notebook-side", but the current shared Colab generator (`shared/colab/gen_notebook.py`) contains neither | `scripts/train_oasst1.py:1330-1334`, `:738-739` |
| 23 | Super-save / intra-epoch checkpointing | **MISSING** — checkpoints are written once per epoch end only; no `--super-save-every` flag anywhere in the repo or notebook generator | `scripts/train_oasst1.py:842-867` |
| 24 | Benchmark mode (s/step, tok/s, peak VRAM) | **MISSING** — no trainer-side benchmark flag (spec in §17) | n/a |
| 25 | Preflight gate script (`scripts/preflight.py`) | **MISSING** — proposed in §16 | n/a |
| 26 | GPU-memory / ETA logging in the trainer | **MISSING** — only host peak RSS and wall time are recorded | `scripts/train_oasst1.py:1478-1479` |
| 27 | FP16/AMP training | **MISSING + FORBIDDEN** — owner-reported fp16 failure (see §11, §28); BF16 unsupported on T4; FP32 is the only mode | n/a |
| 28 | Google Drive layout / Drive storage preflight | **PROPOSED** — a deployment convention, not code (§14) | n/a |
| 29 | Colab T4 tool / path (platform) | PARTIAL — lead-side Colab tool registered but connection times out (re-tested 2026-09-27); manual notebooks at `shared/colab/` are the working fallback | `shared/colab/README.md` |
| 30 | OASST1 data preserved for later SFT | CURRENT (untouched in repo; see §24) | plan §24 |

Legend: **CURRENT** = verified in this repo at `d8bfbdd` · **PARTIAL** = exists with gaps ·
**MISSING** = no code, must be implemented · **PROPOSED** = a plan/recommendation, not code.

---

## 1. Purpose

This document is the **execution reference for pretraining the `tiny_100m` preset
(96,482,304 parameters) on a Google Colab Tesla T4**, using the current repository state
(`main` @ `d8bfbdd`, the merge of PRs #26/#27/#28). It covers the exact pipeline:

```
FineWeb-Edu (sample-10BT, streaming, ODC-By) ──prepare_corpus.py──▶ packed .npy shards + manifest.json
        │
        ▼
scripts/train_oasst1.py --preset tiny_100m --packed-dir <corpus> --token-budget N  ──▶  step-<N>.pt
        │                                                                                │
        ├── resume (--resume <ckpt> or <dir>) across Colab sessions                      │
        └── per-epoch val loss + metrics.json + train_run_metadata.json                  │
                                                                                        ▼
                                                              scripts/eval_checkpoint.py + scripts/generate.py
```

The plan (owner directive 2026-09-27) is: **large general-purpose corpus → tiny_100m pretraining →
OASST1 SFT (later) → evaluation**. OASST1 is preserved untouched for the SFT stage and **must never
be mixed into the base corpus** (§24).

Two hard constraints from the plan:

1. **FP16/AMP stays OFF** unless the attention mask path is proven safe under AMP (it is not — see
   §11 and §28 for the owner-reported overflow). **BF16 does not exist on T4 (SM7.5). FP32 is the
   safe and only sanctioned mode.**
2. **Before any long run**: smoke tests + a short benchmark on the new dataset, all config recorded
   in metadata, and a **gate**: the proposed training report (dataset, token budget, batch, LR
   config, checkpoint schedule, expected T4 runtime, dataset passes) must be approved by the owner
   **before** launching the full run. This document is the reference for writing that report; the
   report itself (numbers that only a T4 can produce) is out of scope here.

---

## 2. Repository Audit

### 2.1 Files that matter for this run

| File | Role |
|---|---|
| `configs/vocab.py` | Single `VOCAB_SIZE = 1024` (leaf module, no imports) |
| `configs/presets.py` | All preset configs incl. `tiny_100m_config()` (`:96`) and `preset_tokenizer_config()` (`:216`) |
| `configs/canonical.py` | `CANONICAL_PRESETS` exact (params, vocab); `resolve_preset`, `expected_params` |
| `model/config.py` | `ModelConfig` dataclass + `derive()` + validation |
| `model/gpt.py` | `TalosGPT` (embed, layers, lm_head, embed_scale, init, token-id guard, `num_parameters`) |
| `model/attention.py` | Plain (chunked, masked-softmax) + optional FlashAttention backends — **the fp16-bug site** `:182-184` |
| `model/block.py`, `model/ffn.py`, `model/rms_norm.py`, `model/rotary.py` | Decoder layer, dense SwiGLU FFN, RMSNorm, RoPE |
| `model/utils.py` | `set_seed`, `validate_token_ids` |
| `data/packed.py` | Packed shard reader + manifest validation (`PackedTokenDataset`, `load_packed_manifest`, `manifest_identity`, `packed_phase_shard_paths`) |
| `data/tokenized.py` | `StreamingTokenizedDataset` (JSONL path only) |
| `data/readers.py` | `HuggingFaceReader` / `ParquetReader` / `JSONLReader` (used by prepare_corpus) |
| `scripts/prepare_corpus.py` | Corpus prep CLI (streaming HF → packed `.npy` + manifest) |
| `scripts/train_oasst1.py` | **The trainer** (both data paths, all guards, resume, token budget, metadata) |
| `scripts/eval_checkpoint.py` | Eval CLI (loss/ppl/acc/throughput/RSS) |
| `scripts/generate.py` | Generation CLI (greedy or temperature sampling) |
| `scripts/export_safetensors.py` | Safetensors release export |
| `tokenizer/tokenizer.py` | `ByteLevelBPETokenizer` (incl. `from_file`), `tokenizer_file_sha256` |
| `tokenizer/vocab.py` | Vocab layout: 256 base bytes + 4 specials at top (1020-1023) + up to 764 merges |
| `tests/test_tiny_100m.py` | Preset validation ladder (exact count, fwd/bwd, ckpt round-trip, prefill==decode, …) |
| `tests/test_trainer_token_budget.py` | Token-budget/resume/metadata tests |

### 2.2 Recent commits (the state this doc reflects)

| Commit | Summary |
|---|---|
| `d8bfbdd` (HEAD, main) | Merge PR #28 `feature/trainer-token-budget-100m` — **the token-budget trainer**: `--token-budget` stop, packed-corpus training, `train_run_metadata.json`, robust resume (numeric scan + corrupt fallback). Inner commit `0684366`. |
| `85e96b1` | `feat(corpus-prep): packed-token corpus preparation + trainer --tokenizer-json` (PR #27) — `scripts/prepare_corpus.py`, `data/packed.py`, tokenizer fingerprint wiring. |
| `d34c54e` | `feat(100m): tiny_100m preset (96,482,304 params), VOCAB_SIZE source of truth, safetensors release format` (PR #26) — the preset, `configs/vocab.py`, `scripts/export_safetensors.py`, `tests/test_tiny_100m.py`, `tests/test_safetensors_io.py`. |
| `1b265bf` | `fix(p0): P0 pipeline guards` (PR #25) — token-ID bounds, generate length guard, resume, tokenizer fingerprint. |
| `0b5f1a9` / `b39d010` / `48bba04` / `47dde8f` | tiny_10m preset (#24); 254K-vs-1M A/B at 15.5M-token budget (#23); tiny_1m real-OASST1 run (#22); tiny_1m preset (#21). |

### 2.3 Honest gaps in the audit trail (known, not fixed)

- **No measured T4 GPU numbers exist in the repo or `/home/team/shared`.** Every recorded throughput
  is CPU. The owner's published T4 baseline (`benchmarks/phase-b/report-tiny-1m.md` §5) recorded
  config + losses (train 1.7859 / val 1.9177) but **no wall-clock/tok/s**. Batch benchmarks quoted in
  §11 are owner-reported records, cited as such, not repo artifacts.
- **No trained `tokenizer.json` is shipped in the repo** (only the legacy 512-vocab test fixture
  `tests/data/tokenizer_vocab512_legacy.json`); the established tokenizer must be downloaded from
  HF (§6). Its identity sha256 `58e4ad40…` is the canonical check, and `benchmarks/phase-b/
  data-provenance.json` records the published tokenizer as byte-identical to the team-trained one.
- **The owner's 32-point preflight checklist is not present verbatim in the repo or `/home/team/
  shared`** (searched on 2026-09-27). §16 provides a 32-point gate reconstructed from the repo's
  real guards and the plan, each point mapped to code or marked NEEDS IMPLEMENTATION.

---

## 3. Current Implementation Status

Everything an uninterrupted (or resumed) pretraining run needs end-to-end is **CURRENT**:

- preset → model → parameter-count guard (fail-fast) → packed data → token-budget schedule →
  per-epoch train/val → checkpoint (v1, resume-capable) → metadata → metrics → resume; plus
  eval and generation entrypoints for the finished run.

Everything that protects a **long unattended Colab campaign** is **MISSING and specified** here:

- gradient clipping, NaN/Inf loss+grad detection and abort-with-checkpoint semantics (§12);
- super-save / intra-epoch checkpoints (§13);
- a benchmark mode and GPU-memory/ETA logging (§15, §17);
- a preflight gate script (§16).

These are honest gaps: the trainer's own metadata names them (`gradient_clip_type: None`,
`nan_inf_detection: False`, `scripts/train_oasst1.py:1330-1334`), and no wrapper in the repo or the
shared Colab notebooks provides them either (verified by grep on 2026-09-27).

---

## 4. Hardware / Runtime

| Item | Value | Source / status |
|---|---|---|
| GPU target | NVIDIA Tesla T4, 16 GB VRAM (SM 7.5) | Colab free tier; notebook asserts T4 |
| Compute mode | **FP32 only.** No AMP autocast anywhere in the repo. BF16 unavailable on SM7.5; fp16 documented broken for this attention mask path (§11, §28) | verified: no `autocast`/`half()` in `scripts/train_oasst1.py`, `model/` |
| Peak-VRAM model (computed, audit §13) | B=32/S=64 ≈ **2.2–2.4 GiB**; B=32/S=512 ≈ **7.7–9.2 GiB** (fp32 incl. CUDA overhead); weights+grads+AdamW = 16×P ≈ **1.44 GiB** regardless of batch | `shared/talos-100m-audit.md` §13 (computed, not measured) |
| Recommended T4 starting shape | **batch 32 × seq 64** (2,048 tokens/step) | `docs/SCALING.md:327` (repo recommendation) |
| CPU box (this repo's dev box) | no GPU; all repo-recorded benchmarks are CPU | `benchmarks/phase-b/report-tiny-1m.md` §4 |
| Colab tooling | Platform-blocked 2026-09-27: lead-side Colab tool registered but connection times out; member sessions see zero tools. Manual notebooks `shared/colab/talos_{1m,10m}_colab.ipynb` (tiny_10m/tiny_1m presets) are the working fallback; a 100M notebook does NOT exist yet | plan + `shared/colab/README.md` |

Notes:
- VRAM is a **non-issue** at preset scale; the binding constraint on a T4 is **compute throughput**
  (see §11). The audit's memory table is a *computed model*, not a measurement.
- Colab free-tier practicalities: runtime resets, ~12.7 GB system RAM, ~78 GB scratch disk,
  session timeouts — the reason `--resume` and per-epoch checkpoints exist and Drive persistence
  (§14) matters.

---

## 5. Model Specification (tiny_100m — VERIFIED)

Import path, verified live on 2026-09-27:

```python
from configs.presets import tiny_100m_config      # → ModelConfig (un-derived ok)
from model import TalosGPT
cfg = tiny_100m_config().derive()                  # fills head_dim etc.; validates
model = TalosGPT(cfg)                              # constructor
model.num_parameters()                             # 96,482,304  ← VERIFIED
```

Verified values (construct + forward on CPU; `logits.shape == (B, S, 1024)`):

| Field | Value | Where defined |
|---|---|---|
| `vocab_size` | 1024 | `configs/vocab.py:44` → `presets.py:97` |
| `hidden_size` | 1024 | `presets.py:98` |
| `num_layers` | 6 | `presets.py:99` |
| `num_attention_heads` (Q) | 64 | `presets.py:100` |
| `num_kv_heads` (KV) | 32 (2:1 GQA) | `presets.py:101` |
| `head_dim` | 16 | `presets.py:102` |
| `ffn_type` | `"dense"` (SwiGLU) | `presets.py:103` |
| `intermediate_size` | 4096 (4× hidden) | `presets.py:104` |
| `max_seq_len` | 512 | `presets.py:105` |
| `attention_type` | `"full"` (causal) | `presets.py:106` |
| `rope_theta` | 10000.0 | `presets.py:107` |
| `layer_norm_eps` | 1e-5 | `presets.py:108` |
| `tie_word_embeddings` | False (un-tied; `lm_head` exists `gpt.py:37`) | config default `config.py:105` |
| biases | **None** — every `nn.Linear` is `bias=False` | `gpt.py:37`, `block.py:36-39` |
| `dropout` | 0.0 | config default |
| init | `normal_(0, 0.02)` on Linear/Embedding, RMSNorm weight=1, `embed_scale = sqrt(hidden)` | `gpt.py:48-66` |
| Param count | **96,482,304** (`2·V·H + L·(15H²+2H) + H` = 2·1024·1024 + 6·15,730,688 + 1024) | `presets.py:110-113` docstring; **verified live** |

### 5.1 Hard preflight parameter-count requirement (CURRENT)

The trainer **refuses to start** if the built model's parameter count or vocab does not exactly
match the registry entry for `--preset tiny_100m`:

- `check_preset_compat(preset, cfg, n_params)` (`scripts/train_oasst1.py:503-533`) raises
  `ValueError` on any mismatch and prints the expected count before training. Wired at
  `train_run()` step 2 (`:1212-1215`: `build_preset_model` → guard).
- The same exact-count check runs again on **resume** (`validate_resume_checkpoint`,
  `:222-252`, checks `n_params` and resolves the checkpoint's preset).
- The canonical registry itself is `configs/canonical.py:28-35`; `expected_params(cfg)` and
  `resolve_preset(cfg)` are the programmatic entrypoints.

For an external preflight script (§16) this check should be **re-run explicitly** before launching:
`CANONICAL_PRESETS["tiny_100m"] == (model.num_parameters(), model.config.vocab_size)`.

### 5.2 Save / load

- **Save**: `save_checkpoint(...)` (`train_oasst1.py:626-683`) writes `torch.save(payload, path)` —
  format `talos-training-checkpoint-v1`, fields in §13.
- **Load**: `load_checkpoint(path)` (`:684-687`) → `torch.load(map_location="cpu",
  weights_only=False)`. Loading a checkpoint **does not** by itself rebuild the model — the trainer
  builds the preset model first and `load_state_dict` at resume (`:759`).
- Eval/generation load via `evaluation/harness.py` / `inference/generate.py` + CLI wrappers.

---

## 6. Tokenizer

**The established tokenizer is the ONLY sanctioned one for this run.** It is NOT trained by the
trainer on this path and NOT regenerated from the corpus.

| Property | Value | Source / status |
|---|---|---|
| Established artifact | `tokenizer.json` published at `https://huggingface.co/PrometheanStudio/talos-mini-254k-oasst1/resolve/main/tokenizer.json` | public, no token; per research memo §1 |
| sha256 (canonical identity) | `58e4ad40b174c9fde3cec8e86188115bcdee173e7fcd188e7ba139a04b549e48` (26,108 bytes) | research memo §1; byte-identical to team-trained tokenizer (`benchmarks/phase-b/data-provenance.json`) |
| vocab size | 1024 | verified via `ByteLevelBPETokenizer` |
| merges | 764 (vocab 1024 = 256 base byte tokens + 4 specials + 764 BPE merges) | `configs/vocab.py:44` docstring; `tokenizer/vocab.py:93` |
| Load | `tokenizer.tokenizer.ByteLevelBPETokenizer.from_file(path)` | `tokenizer/tokenizer.py:284` |
| Special-token ids | BOS `<|beginoftext|>`=1020, **EOS `<|endoftext|>`=1021**, PAD `<|pad|>`=1022, UNK `<|unk|>`=1023 (specials at the top, `vocab_size-n_special..vocab_size-1`) | `tokenizer/vocab.py:130-135` (+ `:26-31`); audit §14 |
| EOS use in packing | every doc is stored as `encode(text) + [eos_id]` (incl. the last); `pad_id` fills the final partial row | `scripts/prepare_corpus.py:25-31` docstring; `data/tokenized.py` pack mode |

### 6.1 How the trainer consumes it (CURRENT)

- **Flag**: `--tokenizer-json <path>` (default `None` = BPE-train-from-`--data`, which is the JSONL
  path only). On the **packed path**, when `--tokenizer-json` is omitted, the trainer uses the
  manifest-recorded tokenizer path if it exists on disk, else runs without a sidecar file (identity
  still verified from the manifest sha256) — `scripts/train_oasst1.py:927-941` (help text)
  and `:1140-1208` (resolution logic).
- **Fingerprint machinery**: `tokenizer_file_sha256(path)` (`tokenizer/tokenizer.py:35`) hashes the
  serialized file. The trainer records it in every checkpoint (`tokenizer_fingerprint`, `:655-657`)
  and in `metrics.json` (`tokenizer_sha256`, `:1437`); the preparation manifest records it too
  (`prepare_corpus.py:345-350`, `:378-383`).
- **Mismatch detection (all CURRENT, all loud)**:
  - fresh run: `_verify_and_load_tokenizer(path, expected_sha=None)` (`:977-991`) just loads; the
    vocab-fit contract `tokenizer.vocab_size <= model_vocab` is enforced at `:1200-1207`;
  - resume: the sidecar's sha256 must equal the checkpoint's recorded fingerprint (`:1178-1196`);
    `--tokenizer-json` on resume must match the checkpoint fingerprint too (`:1186-1192`);
  - packed path: the manifest's recorded tokenizer sha256 must equal the provided file's, and on
    resume the checkpoint's fingerprint must equal the manifest's (`:1198-1208`).
- **Proposed Colab location (PROPOSED, configurable — pick any path, pass it to both scripts)**:
  `/content/drive/MyDrive/Talos/Styx_100M/tokenizer/tokenizer.json`. Nothing in the repo hard-codes
  this path; it is a deployment convention from the plan. **Verify `sha256sum` after download.**

---

## 7. Dataset / Corpus

| Property | Value | Source / status |
|---|---|---|
| Dataset | `HuggingFaceFW/fineweb-edu`, config **`sample-10BT`**, split **`train`** | research memo §3.1 (measured); default of `prepare_corpus.py:562-566` |
| Access | **Streaming via `datasets`, ungated, NO auth token** (verified 2026-09-27 on this box; HF API `gated:false`) | research memo §1-2, §5 |
| Format | 14 parquet files, 28.5 GB on disk; 9,672,101 docs; fields include `text`, `token_count`, `score`/`int_score` | memo §3.1 |
| License | **ODC-By v1.0** (+ Common Crawl ToU) — attribution-only; redistributing checkpoints trained on it is fine | memo §3.1 |
| Measured tokenizer ratio | **≈1.0 tokens/char** (measured 1.000893 on 3.0 M chars) → tokens ≈ text GB | memo §1, §3.1 |
| Total size | ≈ **41–46 B Talos tokens** (extrapolated from measured samples, ±10 %) | memo §3.1 |
| Genuine scale fit | Chinchilla-anchored 1.93 B tokens ≈ 4.5 % of the subset (~430 K docs, ~2 GB text) | memo §4 |
| Runner-up | `HuggingFaceFW/fineweb` `sample-10BT` (~50 B tokens, ODC-By) — use if broader web over edu-density is wanted | memo §3.2 |

Caveats (stated by the research memo, keep when quoting numbers):
- Corpus totals are **extrapolations** from 3–4 M-char samples + card/HF totals (±10 %).
- Every token ≈ 1 char with this tokenizer, so seq 512 ≈ ~512 chars ≈ ~90 words — document-packing
  quality matters (§8).
- `datasets` 5.0.1 printed a cosmetic `PyGILState_Release` error at interpreter shutdown after
  streaming on this box (harmless; see §28).

---

## 8. Data Preparation (`scripts/prepare_corpus.py`)

Run once (per desired corpus slice), streamed end-to-end; output is the **`--packed-dir`** the
trainer consumes. **BPE is NOT trained here** — `--tokenizer-json` is required.

### 8.1 Exact CLI (from `make_arg_parser`, `prepare_corpus.py:557-591`)

| Flag | Type | Default | Required | Meaning |
|---|---|---|---|---|
| `--dataset` | str | `"HuggingFaceFW/fineweb-edu"` | no | HF dataset id (streaming, ungated) |
| `--config` | str | `"sample-10BT"` | no | dataset config/subset |
| `--split` | str | `"train"` | no | dataset split |
| `--tokenizer-json` | path | — | **yes** | established Talos `tokenizer.json` to LOAD (sha256/vocab/merges recorded) |
| `--target-tokens` | int | — | **yes** | TOTAL budget — stop once `val_tokens + train_tokens >= target` (**includes the val slice**) |
| `--val-tokens` | int | `15_000_000` | no | tokens carved from the DISJOINT val region |
| `--val-skip-docs` | int | `10_000` | no | docs skipped at the stream head before the val region (disjoint-region gap) — val region is *after* the gap, train docs start *after* the val region |
| `--seq` | int | `512` | no | packed row width |
| `--dtype` | `int32`\|`uint16` | `"int32"` | no | on-disk token dtype (uint16 safe: vocab 1024 < 65536) |
| `--out-dir` | path | — | **yes** | output directory (shards + manifests) |
| `--stream-buffer-docs` | int | `1_000` | no | max encoded docs held in RAM between shard flushes |
| `--rows-per-shard` | int | `16_384` | no | max rows per `.npy` shard |
| `--text-field` | str | `"text"` | no | record field holding document text |
| `--no-hf-metadata` | flag | off | no | skip the best-effort HF dataset revision/sha lookup |

Example:
```bash
python -m scripts.prepare_corpus \
  --tokenizer-json /content/drive/MyDrive/Talos/Styx_100M/tokenizer/tokenizer.json \
  --target-tokens 2000000000 --val-tokens 15000000 \
  --out-dir /content/drive/MyDrive/Talos/Styx_100M/data/fw-edu-2b
```

### 8.2 Semantics that matter (verified from code/docstring)

- **Packing convention**: every document is `encode(text) + [eos]` (EOS appended, including the
  last doc); token ids validated against `VOCAB_SIZE` at encode time via
  `validate_token_ids` (`prepare_corpus.py:23-31`, `:333-`); the final partial row is padded with
  `pad_id`.
- **Disjoint val carve**: skip `val-skip-docs` docs, collect val docs until `val_tokens` reached,
  then collect train docs from *after* the val region. The val region's position is **independent
  of `--target-tokens`** (stable across budgets) and the two regions never share a source doc.
- **`target_tokens` INCLUDES val**: stopping condition is `val_tokens + train_tokens >=
  target_tokens`; if the stream is exhausted first, metadata records `truncated: true`.
- **Real vs incl-padding tokens** (recorded separately — a downstream trainer consumes the *real*
  counts; `rows*seq` overstates because of the padded tail row of each region):
  `val_tokens`/`train_tokens` (real, incl. EOS separators) vs `val_tokens_incl_padding`/
  `train_tokens_incl_padding` (`rows*seq`) and `val_pad_tokens`/`train_pad_tokens`.
- **Atomicity**: all `.npy` shards are written via `NpyShardWriter` using temp file +
  `os.replace`; both JSON outputs (`manifest.json`, `run_metadata.json`) are written via
  `_atomic_write_json` (tmp + `os.replace`) — `prepare_corpus.py:99-186`, `:549-555`.
- **Manifest** (`manifest.json`, format `talos-packed-tokens-v1`): top-level `format`, `dtype`,
  `seq_len`, `eos_id`, `pad_id`, `num_shards`, `shards` (val shards listed first, then train;
  each entry = `{name, phase, rows, ...}` from the writer), and a `metadata` block = the same dict
  as `run_metadata.json`: `schema`, `script_version`, `timestamp`, `args`, `dataset`
  (dataset/config/split/revision/sha), `tokenizer` (path/sha256/vocab_size/merge_count), `packing`
  (seq/eos_id/pad_id/dtype/vocab_size_bound/rows_per_shard/stream_buffer_docs), `val_region`,
  `train_region`, `counts` (full token/row/char/pad counters incl. `total_tokens`, `truncated`),
  `wall_s` — `prepare_corpus.py:344-548`.
- **Disk/RAM**: bounded — at most `stream_buffer_docs` docs' tokens + one shard in RAM; a
  16 384-row × seq-512 int32 shard is ~32 MiB (`data/packed.py:25-36`). 2 B real tokens ≈
  7.7 GB int32 / 3.9 GB uint16 on disk (research memo §3.1).
- **Dataset-order determinism**: prepare reads the HF stream front-to-back; the manifest records
  the best-effort HF revision/sha (skip with `--no-hf-metadata`). If a later re-run with the same
  flags must be bit-identical, the HF dataset revision must be pinned the same way (recorded, but
  no auto-pin); the val-region placement is independent of `--target-tokens` by construction.