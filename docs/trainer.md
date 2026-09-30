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
| 22 | Grad clipping / NaN-Inf detection in the trainer loop | **CURRENT** (hardening pass) — loss finiteness checked before `backward`, gradient finiteness after, on EVERY step; a bad step skips the optimizer update, logs a structured event into `metrics.json`, writes a safety checkpoint, and aborts after `--max-consecutive-bad-steps` (default 3). `--grad-clip` bounds the total grad norm and records the pre-clip norm | `scripts/train_oasst1.py:979-1018` (loop guards), `:895-950` (`handle_bad_step`), `:606-620` (`BadStepsAbort`), metadata `:1573-1582` |
| 23 | Super-save / intra-epoch checkpointing | **CURRENT** — `--save-every-tokens N` writes an atomic v1 checkpoint at every absolute multiple of N tokens since run start (restored on resume), through the same single checkpoint writer; epoch-end checkpoints unchanged | `scripts/train_oasst1.py:1015-1039` (periodic), `:661-734` (single writer + atomic `os.replace`), flag `:1162-1170` |
| 24 | Benchmark mode (s/step, tok/s, peak VRAM) | **MISSING** — no trainer-side benchmark flag (spec in §17) | n/a |
| 25 | Preflight gate script (`scripts/preflight.py`) | **MISSING** — proposed in §16 | n/a |
| 26 | GPU-memory / ETA logging in the trainer | **MISSING (partially improved)** — still no VRAM/ETA sampling, ONLY host peak RSS + wall time; the hardening pass DID add grad-norm / bad-step / abort records to `metrics.json` (see §15.2) | `scripts/train_oasst1.py:1704-1739` (stability metrics) — GPU/ETA itself remains n/a |
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
- preset -> model -> parameter-count guard (fail-fast) -> packed data -> token-budget schedule ->
  per-epoch train/val -> checkpoint (v1, resume-capable, atomic, three cadences: epoch-end /
  `--save-every-tokens` / NaN-safety) -> metadata -> metrics -> resume; plus eval and generation
  entrypoints for the finished run.
- **Numerical stability is CURRENT (hardening pass 2026-09-27):** NaN/Inf loss+grad detection
  always-on in the loop; bad steps skipped + safety-checkpointed + logged to `metrics.json`;
  abort after 3 consecutive bad steps with a recoverable checkpoint; `--grad-clip` with pre-clip
  norm recording; atomic `.pt` writes (§12, §13).
Still **MISSING and specified** here (unchanged by the hardening pass):
- a benchmark mode and GPU-memory/ETA logging (§15, §17);
- a preflight gate script (§16);
- the 100M Colab notebook and packed-aware eval path (§29).
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
---

## 9. Training Objective (VERIFIED in the loop)

The packed path and the JSONL path use the **identical objective** (`scripts/train_oasst1.py:717-724`):

```python
logits, _ = model(x[:, :-1])                     # input  = tokens 0..S-2  (shape B, S-1, V)
loss = loss_fn(logits.reshape(-1, vocab),        #        → (B*(S-1), V)
               x[:, 1:].reshape(-1))             # target = tokens 1..S-1  (B*(S-1),)
opt.zero_grad(); loss.backward(); opt.step()
```

- `loss_fn = torch.nn.CrossEntropyLoss()` (`:751`) — **mean reduction** over `B*(S-1)` tokens.
  (Note the eval path uses `CrossEntropyLoss(reduction="sum")` and normalizes externally — §9.2.)
- Every batch is token-id-range-checked before the model (`validate_token_ids`, `:717`), and the
  model re-checks at the embedding seam (`model/gpt.py:100-104`).
- One optimizer step = one batch (no gradient accumulation anywhere — see §10).
- Per-epoch mean train loss: `sum(epoch_losses)/len(epoch_losses)` (`:841-843`), recorded per epoch.

### 9.1 Shapes

| Quantity | Value |
|---|---|
| micro-batch | `(B, S)` token rows |
| logits | `(B, S-1, V)` with V = 1024 (verified live: `(2, 64, 1024)` for a 64-token probe) |
| tokens per step (micro) | `B × (S-1)` — **this is the number the token budget counts** |
| loss reduction | mean (train); sum-then-/count (val) |

### 9.2 Validation loss / perplexity

`evaluate()` (`:588-625`) computes mean NLL over the val stream: `sum(loss)/count` over all full
batches; `val_max_steps` caps the number of val batches (CI/smoke). It returns `None` for an empty
val stream. Perplexity is **not computed by the trainer**; the eval harness reports
`val_perplexity` (`evaluation/harness.py:90`) — `exp(val_loss)` on the same mean-NLL basis. Use
`scripts/eval_checkpoint.py` for the official per-checkpoint loss/ppl/acc numbers (§27).

---

## 10. Hyperparameters

### 10.1 Optimizer — from code, `scripts/train_oasst1.py:749-751`

```python
opt = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
```

Only `lr` is configurable. **Everything else is torch AdamW default**:

| Parameter | Value | Configurable? |
|---|---|---|
| optimizer | AdamW | fixed |
| `lr` | `--lr` default **3e-3** | `---lr` |
| `betas` | `(0.9, 0.999)` | **NEEDS IMPLEMENTATION** (not exposed) |
| `eps` | `1e-8` | **NEEDS IMPLEMENTATION** (not exposed) |
| `weight_decay` | `0.01` (AdamW default) | **NEEDS IMPLEMENTATION** (not exposed) |
| gradient clipping | `--grad-clip N` default **0 = off** (§12) | **CURRENT** — `torch.nn.utils.clip_grad_norm_` after backward; pre-clip norm recorded |
| gradient accumulation | none (batch == micro-batch == optimizer step) | **MISSING** if ever needed |

### 10.2 LR schedule (CURRENT; `TokenSchedule`, `:127-198`)

| Flag | Default | Meaning |
|---|---|---|
| `--lr` | `3e-3` | peak LR; linear warmup 0→lr over `--warmup-tokens`, then decay per `--lr-decay`; step LR is a pure function of tokens consumed so far (pre-step count) |
| `--warmup-tokens` | `0` | linear warmup span; 0 = no warmup (fixed LR). Must be `< token budget` when budget set |
| `--lr-decay` | `"none"` | `none` (fixed LR) or `cosine` — cosine from `lr` down to `min_lr_ratio × lr = 10 %` over `budget - warmup`; **cosine requires `--token-budget`** |
| `MIN_LR_RATIO` | `0.1` (code constant `:120`) | floor for cosine decay, not a flag |

### 10.3 Batch / sequence / tokens

| Quantity | Value | Notes |
|---|---|---|
| `--batch` | default 4 (`:948-949`) | for `tiny_100m` on a T4 use 32 at S=64 (plan/§11); batch 4 is the CLI default, mirror of tiny A/B runs |
| `--seq` | JSONL default 64; **packed: manifest `seq_len` wins** (`:1066-1079`); explicit `--seq` must equal the manifest or it errors | `--packed-dir` rows are used verbatim |
| tokens/step | `B × (S-1)` (e.g. 32×63 = 2,016 at B=32/S=64; 4×511 = 2,044 at B=4/S=512) | counts toward `--token-budget` |

### 10.4 Frequencies (what exists TODAY)

| Event | Frequency | Where |
|---|---|---|
| validation pass | **once per epoch end** | no intra-epoch val; an epoch on the packed path = one pass over the whole packed train stream |
| checkpoint (`step-<N>.pt`) | **epoch end** (always) + **`--save-every-tokens N`** (intra-epoch, at every absolute multiple of N tokens since run start — restored on resume) + **safety checkpoints** on every skipped NaN/Inf step (§12) | one shared atomic writer (`save_checkpoint`) |
| `metrics.json` / metadata | written at run end — **and on a NaN-abort** (`BadStepsAbort` still writes metrics + finish-stamped sidecar) | shared `_write_final_artifacts` |
| budget stop | mid-epoch allowed: the partial epoch still gets val + a checkpoint | budget break does NOT skip the epoch-end checkpoint |

---

## 11. Batch / Throughput (T4)

### 11.1 Owner-reported T4 benchmarks — CITED AS OWNER-REPORTED, not repo artifacts

These were reported by the owner; they are **not present in the repo or `/home/team/shared`** (all
repo throughput records are CPU; the audit found no measured T4 tok/s anywhere). Treat magnitudes as
truth, exact digits as unverifiable:

| Shape | Owner-reported value |
|---|---|
| B=1 (next-token) | ~0.465 s/step, ~2.153 steps/s, ~1.82 GB peak VRAM |
| B=4, seq 512 (next-token) | ~0.553 s/step, ~1.808 steps/s, ~4.3 GB peak VRAM, ~2,044 tokens/step |

Implications (if transferable to tiny_100m at S=512): roughly **2 steps/s × 2,044 tokens ≈ ~3.7K
tok/s** at B=4/S=512 → a 200M-token milestone ≈ ~15 h of pure GPU time; a 1.93 B-token Chinchilla
budget ≈ ~145 h. These are **estimates from owner-reported numbers**, not measurements — see §26.

### 11.2 Computed VRAM model (audit §13, computed not measured)

| Shape | fp32 peak VRAM (computed) |
|---|---|
| B=32, S=64 | ~2.2–2.4 GiB |
| B=32, S=512 | ~7.7–9.2 GiB |
| weights+grads+AdamW (any batch) | ~1.44 GiB |

Repo recommendation: **batch 32 × seq 64 to start** (`docs/SCALING.md:327`). The audit concludes
VRAM is not the binding constraint on a 16 GB T4; throughput is.

### 11.3 Precision — the hard rule

- **FP32 is the safe and only sanctioned mode** (no AMP code exists in the repo at all).
- **BF16 is unsupported on T4 (SM 7.5)** — do not attempt.
- **FP16/AMP FAILED (owner-reported, do NOT enable):**
  `RuntimeError: value cannot be converted to type c10::Half without overflow`
  in the attention masked-fill path: `model/attention.py:182-184` builds
  `mask = torch.zeros_like(scores); mask = mask.masked_fill(~allowed…, NEG_INF)` — under fp16
  autocast the `-inf` fill in fp16 overflows. The plan forbids AMP until this path is proven safe.

### 11.4 Required preflight batch benchmark (before the real run)

Because no measured T4 number exists for `tiny_100m` specifically: run a short benchmark on the T4
first (spec in §17 — currently NEEDS IMPLEMENTATION as a mode; a manual 20-step run with
`--max-steps-per-epoch` is the working substitute). Record s/step, tok/s and peak VRAM
(`torch.cuda.max_memory_allocated()` + `nvidia-smi`) for your chosen (B, S); then convert your token
budget to wall-clock: `budget / tok_s`.

---

## 12. Numerical Stability (IMPLEMENTED — hardening pass 2026-09-27)

The guards below live **inside the trainer loop** (`train_epochs`,
`scripts/train_oasst1.py:742-1079`) and are **always on** — no flag disables the
detection. Verified by reading the loop + the tests in
`tests/test_trainer_stability.py` (all green, synthetic tiny-preset runs).

| Guard | Status | Where |
|---|---|---|
| NaN/Inf **loss** detection (BEFORE `backward`) | **CURRENT** — `if not torch.isfinite(loss)` → bad step | `:978-985` |
| NaN/Inf **gradient** detection (AFTER `backward`) | **CURRENT** — first param with a non-finite `.grad` is identified by name | `:986-1001` |
| bad-step handling | skip the optimizer step **entirely** (no `backward` for a bad loss, no `opt.step()` for a bad grad — weights keep the last-good state) | `:895-950` (`handle_bad_step`) |
| loud structured events | one dict per bad step in `history.bad_step_events` → `metrics.json["bad_steps"]`: `{step, tokens_consumed, tensor, stat (nan/inf), loss, consecutive_bad_steps, checkpoint}` + an `ERROR` log line | `:919-938` |
| safety checkpoint | written IMMEDIATELY on every bad step, same v1 format/atomic writer, named `step-<N>.pt` (N = the attempted step), holding the **last-good** weights/optimizer/RNG + the pre-step `tokens_consumed` | `:907-918` |
| abort after K consecutive | `--max-consecutive-bad-steps` (default 3; `0` = never abort): after K consecutive bad steps `train_epochs` raises `BadStepsAbort`; `train_run` still writes `metrics.json` + finish-stamps the sidecar, then exits non-zero (code 3). The last safety checkpoint is the recoverable state and passes every resume-validation rule | `:939-950`, `:1742-1776`, `main` `:1819-1831` |
| consecutive counter | reset to 0 by every good step | `:1008` |
| gradient clipping | `--grad-clip N` → `torch.nn.utils.clip_grad_norm_` after the finiteness check, BEFORE `opt.step()`; the **PRE-clip** total norm is recorded in `history.last_grad_norm`, every subsequent checkpoint (`last_grad_norm` payload key) and `metrics.json`; `0` (default) = off, step untouched | `:1002-1006`, payload `:727-730` |
| metadata self-report | `training_config.gradient_clip_type` = `"max_grad_norm"` (or `None` when off), `grad_clip_max_norm` = the numeric value, `nan_inf_detection: True`, `max_consecutive_bad_steps`, `save_every_tokens` — the OLD `None`/`False` literal is gone | `:1573-1582` |

**Behavior on a bad step** (identical for a NaN/Inf loss and a non-finite
gradient): the step is *attempted* (it consumes a step number, so checkpoint
filenames stay unique and monotonic) but **no weight update and no
tokens-consumed increment happen**. The pre-step state is persisted as the
safety checkpoint; the event is logged; training continues. Only after
`--max-consecutive-bad-steps` *consecutive* failures does the run abort —
never silently continue into poisoned weights, and never with an unrecorded
loss of state (the last-good checkpoint is on disk before the abort
propagates).

**Not implemented (unchanged):** GPU-memory/ETA logging (§15.3) and the
preflight script (§16) remain MISSING; the FP16/AMP caveat of §11/§28 still
holds (this pass adds no AMP — FP32 only).

---

## 13. Checkpointing

### 13.1 v1 format (`talos-training-checkpoint-v1`) — fields, from `save_checkpoint` (`:661-734`)

| Field | Content |
|---|---|
| `format` | `"talos-training-checkpoint-v1"` |
| `step` | global optimizer step (monotonic across resume; bad steps still consume a number) |
| `model_config` | `asdict(cfg)` (already derived) |
| `n_params` | `model.num_parameters()` (checked on resume) |
| `vocab_size` | `cfg.vocab_size` |
| `train_loss` / `val_loss` | epoch mean (epoch-end ckpts) or partial-epoch mean (periodic) / `None` (periodic + safety ckpts) / last-good loss (safety ckpts) |
| `tokenizer_path` | absolute path of the sidecar `tokenizer.json` (None on manifest-only packed runs) |
| `tokenizer_fingerprint` | sha256 of the serialized tokenizer (or manifest-recorded sha on packed runs without sidecar) |
| `model_state_dict` | full weights — on a safety checkpoint these are the LAST-GOOD weights |
| `optimizer_state_dict` | AdamW state (resume-enabling, additive; old v1 ckpts lack it → resume refuses) |
| `rng_state` | torch CPU/GPU + numpy + python RNG snapshots (`capture_rng_state`, `:341-356`) |
| `epoch` | last completed epoch (`periodic`/safety ckpts record the epoch being trained) |
| `tokens_consumed` | **tokens since run start, including all prior resumed sessions**; on safety ckpts the pre-bad-step count |
| `run_metadata` | the SAME dict as `train_run_metadata.json` |
| `last_grad_norm` | pre-clip total grad norm of the most recent clipped step (additive; `None` when `--grad-clip` is off) |

**One writer, three cadences.** Epoch-end checkpoints (unchanged), periodic
intra-epoch checkpoints (`--save-every-tokens`), and NaN/Inf safety checkpoints
all call the SAME `save_checkpoint` — no duplicated save logic — and all writes
are **atomic** (`<path>.tmp` + `os.replace`).

### 13.2 `tokens_consumed` accounting (CURRENT)

- Incremented per GOOD step by `B*(S-1)` (a skipped bad step adds no tokens);
  persisted in every checkpoint.
- On resume, restored from the checkpoint; for **legacy** checkpoints without the counter the
  trainer estimates `step × batch × (seq-1)` (exact for constant-shape legacy runs) and warns.
- **Budget semantics**: `--token-budget` counts tokens since run start *including resumed ones*;
  if the budget is already consumed at resume, the run finishes with no new steps.
- **Mid-epoch resume — exact limitation, stated plainly.** The packed
  `PackedTokenDataset` (and the JSONL `StreamingTokenizedDataset`) is an
  **IterableDataset with no intra-epoch position restore**: after a resume from
  a mid-epoch checkpoint (periodic or safety), the **remainder of the partial
  epoch is NOT re-trained** — the run continues at the next epoch, whose stream
  re-starts from shard 0 (packed) / file start (JSONL). `tokens_consumed` is
  exact on both sides of the resume (restored counter, incremented per executed
  good step — a step is either fully trained or not counted), and checkpoint
  step numbers stay monotonic, so the budget stop and the final accounting are
  unaffected. The cost is data coverage: up to one epoch's tail can be skipped
  per disconnect; periodic checkpoints bound the *untrained* window, not the
  re-covering window. (Test-asserted:
  `tests/test_trainer_stability.py::test_resume_from_mid_epoch_periodic_checkpoint_keeps_token_accounting`.)

### 13.3 Directory resume + corruption fallback (CURRENT)

- `--resume <path>` accepts a **file** `step-<N>.pt` or a **directory**; directories are scanned
  for `step-<N>.pt` (regex `:208`) and sorted **numerically descending** (never lexicographic —
  regression-guarded, `:206-221`).
- Each candidate passes `validate_resume_checkpoint` (`:222-252`: format, preset resolution,
  exact `n_params`, optimizer presence); failing candidates are skipped with a warning and the
  next-newest is tried; if all fail, the run refuses to start.
- Resume also enforces: same data source, same packed-corpus identity
  (`manifest_identity` — seq/dtype/tokenizer/rows), same tokenizer fingerprint,
  and `--epochs` is then the **target total**, elapsing from resumed epoch+1.
- **After a NaN-abort** (`BadStepsAbort`), `--resume <run_dir>` picks the last
  safety checkpoint (numeric-newest VALID) and continues from the last-good
  state — test-asserted end-to-end in
  `tests/test_trainer_stability.py::test_train_run_abort_writes_metrics_metadata_and_is_recoverable`.

### 13.4 Write atomicity (CURRENT — hardened 2026-09-27)

- The **metadata JSON sidecars are atomic** (`tmp` + `os.replace`; `write_run_metadata`).
- The **`.pt` checkpoint is NOW atomic too** — `save_checkpoint` serializes to `<path>.tmp` and
  `os.replace`s it into place (`:732-734`), so a crash mid-save can never leave a truncated
  `step-N.pt` for the resume scanner to trip over. The corrupt-fallback scanner stays as a second
  line of defence for pre-hardening artifacts.
- **Intra-epoch ("super-save") checkpointing: CURRENT.** `--save-every-tokens N` writes periodic
  checkpoints at absolute multiples of N tokens since run start (restored on resume), so a
  disconnect loses at most ~N newly-consumed tokens. For the staged T4 budget (~2K tok/s), N = one
  epoch's tokens (or a few hours of wall time) is the natural choice; see §10.4 and §21.1.

### 13.5 Drive persistence (CURRENT as a manual step)

Nothing in the repo uploads to Drive. The Colab run must either write its `--out-dir` directly on
the mounted Drive or copy `step-*.pt` + `metrics.json` + `train_run_metadata.json` + `tokenizer.json`
to Drive after each epoch (notebook cell). The existing 1M/10M notebooks copy the whole run dir to
Drive at the end (`shared/colab/gen_notebook.py` persist step); multi-session resilience requires
per-epoch copy (**NEEDS IMPLEMENTATION** for the 100M notebook).

---

## 14. Google Drive Layout (PROPOSED)

Deployment convention from the plan — no code enforces it. Root: `/content/drive/MyDrive/Talos/Styx_100M/`

| Dir | Required before run? | Generated during run? | Contents |
|---|---|---|---|
| `tokenizer/` | **yes** — `tokenizer.json` (download from HF, verify sha256 `58e4ad40…`) | no | established tokenizer |
| `data/` | optional — if prep runs on Colab it writes here | yes | `prepare_corpus` `--out-dir`: `shard-*.npy`, `manifest.json`, `run_metadata.json` |
| `checkpoints/` | no | yes | trainer `--out-dir`: `step-<N>.pt`, `train_run_metadata.json`, `metrics.json`, `tokenizer.json` copy |
| `logs/` | no | yes | saved stdout/`nohup` logs per session (nothing writes here by default — the trainer's output is stdout only; capturing it is the operator's job) |
| `eval/` | no | yes | `eval_checkpoint --out-metrics` outputs, `generate` sample captures |
| `metadata/` | no | yes | copies of manifests/run_metadata if you want a single landing zone |

Disk budget (computed):
- packed corpus: 2 B real tokens ≈ **7.7 GB int32** / **3.9 GB uint16** (memo §3.1); a 15M-token val
  slice ≈ 60 MB int32. Fits Colab scratch (~78 GB) and Drive free tier with care.
- checkpoint: `step-N.pt` ≈ **~1.16 GB** fp32 (model state 96,482,304×4 B ≈ 386 MB + AdamW m/v
  ≈ 772 MB + config/RNG overhead). ~15 GB free Drive ⇒ **~12 checkpoints** (or convert to
  uint16/safetensors and prune old steps).
- tokenizer.json: 26 KB.

**Drive storage preflight: MISSING (no repo check exists).** Before the run, verify quota with
`df -h` on the mounted Drive (or the drive API) so the checkpoint cadence × size fits; the plan's
preflight gate §16 includes it as a NEEDS IMPLEMENTATION item.

---

## 15. Monitoring (what the trainer actually logs TODAY)

### 15.1 stdout

1. At start: banner lines — packed corpus summary (seq/dtype/train+val rows/tokens/shards), model
   preset line, expected params/vocab line (`expected: EXACTLY 96,482,304 params, vocab_size 1024`),
   data/seed/epochs/seq/batch/lr/device, budget line (`budget: N tokens | warmup W | lr-decay X`),
   run-metadata path (`train_oasst1.py:1101-1148, 1399`).
2. Per epoch (`:868-875`): `epoch %d/%d: steps=%d train_loss=%.4f val_loss=%s checkpoint=%s`.
3. At end (`:1491-1498`): final train/val loss, tokens consumed (+ `(REACHED)` if budget hit), wall
   time (s), peak RSS (MiB), checkpoint path, metrics path.

### 15.2 `metrics.json` fields (`:1684-1816`)

`format`, `params`, `vocab_size`, `tokenizer_vocab_size`, `tokenizer_merges`,
`tokenizer_sha256`, `tokenizer_origin`, `tokenizer_json_arg`, `data_source`,
`train_docs`/`val_docs`/`split_seed` (jsonl) or `train_rows`/`val_rows` (packed), `resumed_from`,
`epochs[]` (`EpochRow`: epoch/steps/global_step/train_loss/val_loss/checkpoint/wall_s),
`final_train_loss`, `final_val_loss`, `tokens_processed`, `token_budget`, `tokens_consumed`,
`steps` (= attempted steps), `budget_reached`, `lr_schedule` (`{type, warmup_tokens, token_budget,
min_lr_ratio}`), `wall_s`, `peak_rss_mb` (**host RAM**, not GPU), `device`, `checkpoint`,
`tokenizer_path`, `run_metadata_file` —

**plus, from the hardening pass:** `grad_clip_max_norm` (the flag in force; `None` = off),
`nan_inf_detection` (`True`), `max_consecutive_bad_steps`, `save_every_tokens`, `last_grad_norm`
(pre-clip norm of the most recent clipped step; `None` when off), `last_good_loss`,
`bad_steps[]` (every skipped NaN/Inf step: `{step, tokens_consumed, tensor, stat, loss,
consecutive_bad_steps, checkpoint}`), `total_bad_steps`, `consecutive_bad_steps`,
`aborted` (`null` or `{reason, last_checkpoint}`), `periodic_checkpoint` (most recent
`--save-every-tokens` checkpoint path).

### 15.3 What is NOT logged (MISSING — unchanged)

- **GPU memory** (VRAM) — only host `peak_rss_mb` (RAM) is recorded; `torch.cuda.max_memory…` is
  never called.
- **ETA / tok/s live** — wall time is per-run and per-epoch; no live throughput line and no ETA.
- **Per-step loss** — only the epoch mean is stored (stdout line per epoch; a partial-epoch budget
  stop records the partial epoch mean).
- Any of these require a small addition (new trainer flags or an operator-side
  `torch.cuda.memory` sampler) — mark **NEEDS IMPLEMENTATION**.

---

## 16. Preflight Gate (the owner's 32-point checklist — reconstructed; script MISSING)

> **Source note:** the owner's 32-point checklist was not found verbatim in the repo or
> `/home/team/shared` on 2026-09-27 (search for "preflight"/"checklist"/"32-point" — no hits). The
> gate below is reconstructed by the team from the repo's actual guards + the plan + the 100M
> audit's P0/P1 fix list (`shared/talos-100m-audit.md` §16). Treat the mapping as authoritative;
> the original owner wording may differ.
>
> **Implementation status: `scripts/preflight.py` is MISSING** (proposed; none exists). Every row
> below that maps to existing code can be run today with the cited one-liner; the NEEDS
> IMPLEMENTATION rows are the reason a preflight script should be written.

| # | Check | Status | How it is satisfied today / proposed |
|---|---|---|---|
| 1 | GPU is a T4 and torch sees CUDA | PARTIAL (no script; trivially runnable) | `python -c "import torch; print(torch.cuda.get_device_name(0))"`; notebook asserts T4 (shared/colab) |
| 2 | CUDA version / driver compatible with torch | PARTIAL | `torch.version.cuda` + `nvidia-smi` — operator check |
| 3 | Repo checkout at the intended commit, clean | CURRENT | `git_repo_state()` records commit/branch/dirty in metadata (`train_oasst1.py:256-281`); preflight should assert `d8bfbdd` |
| 4 | No uncommitted drift | CURRENT (recorded) | same `git_repo_state()`; trainer does not refuse on dirty — an explicit preflight `git status --porcelain` assert is NEEDS IMPLEMENTATION |
| 5 | Deps importable (torch, numpy, datasets, tokenizers) | PARTIAL | `pip check`-style; `datasets` needed only for prep |
| 6 | `tokenizer.json` exists at the chosen path | CURRENT | `--tokenizer-json` raises `FileNotFoundError` (`train_oasst1.py:977-981`) |
| 7 | Tokenizer sha256 == `58e4ad40…` | CURRENT | `tokenizer_file_sha256` + compare; on resume checked against checkpoint fp (`:1178-1196`) |
| 8 | Tokenizer vocab (1024) ≤ model vocab (1024) | CURRENT | `:1200-1207` raises on violation |
| 9 | Model builds with `--preset tiny_100m` | CURRENT | `build_preset_model` (`:546-557`) |
| 10 | Param count == 96,482,304 | CURRENT | `check_preset_compat` (`:503-533`); resume re-checks (`:222-252`) |
| 11 | vocab == 1024 from the single source | CURRENT | registry + `configs.vocab.VOCAB_SIZE`; `tests/test_vocab_seam.py` |
| 12 | Packed corpus manifest loads & validates | CURRENT | `data/packed.py:87-221` (`load_packed_manifest`) — exact `expected_seq/vocab/tokenizer_sha256` |
| 13 | Manifest seq == trainer seq (no silent reshape) | CURRENT | `:1066-1079` errors on mismatch |
| 14 | Manifest dtype ∈ {int32, uint16} | CURRENT | `VALID_DTYPES` + `_check_rows_and_ids` dtype check (`data/packed.py:58-86`) |
| 15 | Shard files exist and match manifest rows/shape | CURRENT | header mmap check per shard at stream time (`data/packed.py:52-86`) |
| 16 | Val region disjoint from train region | CURRENT (by construction) | `--val-skip-docs` carve (`prepare_corpus.py:25-31` docstring); manifest `val_region`/`train_region` records |
| 17 | Target tokens ≥ val tokens + safety margin | PARTIAL | no code check; the stop condition is `val+train >= target` (`prepare_corpus.py`); operator arithmetic — NEEDS IMPLEMENTATION as validation |
| 18 | Token budget ≥ warmup tokens (schedule sanity) | CURRENT | `TokenSchedule` validation (`train_oasst1.py:118-135`) |
| 19 | Checkpoint dir writable / Drive mounted & has space | **MISSING** | no free-space check anywhere — preflight.py NEEDS IMPLEMENTATION (`df`/drive API) |
| 20 | `--out-dir` writable and empty-or-resumable | PARTIAL | trainer mkdirs (`:1115-1120`); no "refuse to clobber a different run" guard — NEEDS IMPLEMENTATION |
| 21 | Resume checkpoint valid (format/params/preset) if resuming | CURRENT | `validate_resume_checkpoint` (`:222-252`) + numeric scan (`:206-221`) |
| 22 | Resume data-source identity matches CLI | CURRENT | `:1060-1065`; packed identity `:1086-1105` |
| 23 | Resume tokenizer fingerprint matches | CURRENT | `:1178-1208` |
| 24 | Budget not already exhausted at resume | CURRENT | `:785-800` warns + no-op finish |
| 25 | `--lr-decay cosine` only with `--token-budget` | CURRENT | `TokenSchedule` validation (`:118-127`) |
| 26 | Batch×seq fits VRAM (preflight benchmark) | **MISSING/NEEDS IMPLEMENTATION** | no trainer benchmark mode — §17 spec; manual 20-step smoke is the current substitute |
| 27 | FP32 mode forced (no AMP anywhere) | CURRENT by absence | no autocast in repo; preflight asserts no env var forces AMP |
| 28 | Token-ID bounds proven on a live batch | CURRENT | `validate_token_ids` at data source + batch + model seam; guards tested (`tests/test_training.py::test_invalid_token_id_rejected`-style) |
| 29 | Generation length guard (CLI) | CURRENT | `scripts/generate.py` truncation (`:179-198` per audit); library `generate()` guard fixed in P0 (#25) |
| 30 | Before-run smoke: N steps train + val + checkpoint write | PARTIAL | use `--max-steps-per-epoch N --val-max-steps M` (existing flags); an automated smoke cell/mode is NEEDS IMPLEMENTATION |
| 31 | All run config recorded (dataset/tokenizer/model/train + token counts) | CURRENT | `train_run_metadata.json` + manifest metadata (`train_oasst1.py:282-340`; `prepare_corpus.py:344-548`) |
| 32 | Owner approval gate for the full run | PROPOSED | plan requirement (2026-09-27): training report → owner approval → launch. No code involved. |

**Proposed `scripts/preflight.py` shape (NEEDS IMPLEMENTATION, do not claim it exists):**
`python -m scripts.preflight --preset tiny_100m --packed-dir <dir> --tokenizer-json <path>
--checkpoint-dir <dir> [--budget N --batch B --seq S --epochs E]` → runs every CURRENT row above
against the real repo, reports per-row PASS/FAIL with file:line evidence, and exits non-zero on any
FAIL. Rows marked MISSING above are the ones it must add (disk free, GPU mem probe, smoke-run,
out-dir clobber guard).

---

## 17. Benchmark (NEEDS IMPLEMENTATION — spec)

There is **no benchmark mode in the trainer today** (no flag; `--max-steps-per-epoch` + wall time
is the manual substitute). The plan requires a short measured benchmark on the new dataset before
any long run. Spec for the mode to add (do not claim it exists):

**Proposed invocation:** `python -m scripts.train_oasst1 --preset tiny_100m --packed-dir <dir>
--token-budget <B*(S-1)*N> --max-steps-per-epoch N --epochs 1 --batch B --seq S --device cuda
[--benchmark]` — or a separate `--benchmark-steps N` flag that stops after N optimizer steps
without val.

**Procedure (works today without the flag):**
1. Run N=20-50 steps at the chosen (B, S) with `--max-steps-per-epoch N --epochs 1` on the T4.
2. Measure: `s/step` (epoch `wall_s / steps` from `metrics.json`), `tok/s =
   steps*B*(S-1)/wall_s`, **peak VRAM** via `torch.cuda.max_memory_allocated()` + `nvidia-smi`
   (operator-side; the trainer records host RSS only), loss stability (first vs last epoch mean,
   and absence of NaN — remember §12: no automatic NaN detection).
3. Prefer S=512 over S=64 for the *final* shape decision: the audit's VRAM model says S=512 is the
   tighter configuration (7.7-9.2 GiB), and the owner-reported B=4/S=512 numbers (§11) are the only
   T4 datapoints at that shape.
4. Wall-clock estimate for the real run: `budget / measured_tok_s` (+ val + checkpoint overhead ≈
   val-tokens/tok_s + one ~1.16 GB checkpoint write per epoch).

**Deliverable of the benchmark:** the owner-report numbers for §16 rows 1-2, 26 and §26 — a
measured tok/s for tiny_100m on a T4, which currently does not exist anywhere.

---

## 18. Training Procedure (25 steps, fresh runtime)

All commands assume a fresh Colab T4 runtime, repo cloned at `/content/talos`, python available,
and the packed corpus prepared (§8) either on Drive or in Colab scratch.

1. **Mount Drive** (if using `Styx_100M`): Colab left-panel → Drive mount; confirm
   `/content/drive/MyDrive/Talos/Styx_100M/` exists with `tokenizer/` and `data/` populated.
2. **Assert GPU**: `import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))`
   → must print a T4.
3. **Check disk**: `df -h /content /content/drive` — corpus (≤8 GB int32 per 2B tokens) +
   checkpoints (~1.16 GB each) + scratch must fit (Colab ~78 GB scratch; Drive quota is yours).
4. **Clone the repo** at the exact commit: `git clone https://github.com/Promethean-Studios/talos
   /content/talos && cd /content/talos && git checkout d8bfbdd`.
5. **Install deps**: `pip install torch==2.13.* numpy safetensors datasets` (CPU wheel is fine for
   prep; Colab's torch is CUDA-enabled by default — verify `torch.cuda.is_available()` after).
6. **Verify tokenizer**: `sha256sum <Styx_100M>/tokenizer/tokenizer.json` ==
   `58e4ad40b174c9fde3cec8e86188115bcdee173e7fcd188e7ba139a04b549e48`.
7. **Inspect the corpus manifest**: `python -c "import json;m=json.load(open('<packed-dir>/manifest.json'));print(m['seq_len'],m['dtype'],m['metadata']['counts'])"` — confirm `seq_len=512`,
   `dtype=int32` (or uint16), tokenizer sha256 matches step 6.
8. **Choose the token budget** `N` (owner-approved; §25 math) and the step shape `B × S`.
9. **Compute steps**: `steps = N // (B*(S-1))` (budget is primary; epochs only bound the loop).
10. **Set `--epochs` large enough that the budget stops the run first** (e.g. `--epochs` such that
    `epochs × train_rows/B >= steps`); the budget hits mid-epoch and still saves a checkpoint.
11. **Smoke run (owner-mandated before the real run):**
    ```bash
    python -m scripts.train_oasst1 --preset tiny_100m --packed-dir <data_dir> \
      --tokenizer-json <tok_path> --out-dir <run_dir> \
      --token-budget 41000 --max-steps-per-epoch 20 --val-max-steps 2 \
      --batch 4 --seq 512 --device cuda --seed 0
    ```
    20 steps × 4×511 = 40,880 tokens — the packed analog of the existing 20-step smoke (shared/colab);
    assert the run prints `expected: EXACTLY 96,482,304 params, vocab_size 1024`, writes a
    `step-20.pt`, and `metrics.json` has `final_val_loss` finite.
12. **Benchmark** (§17) at your chosen (B, S) for ~30-50 steps; record measured tok/s + peak VRAM.
13. **Compute expected wall-clock** = `N / tok_s`; sanity-check against §26 milestones; if > 12 h,
    plan multi-session with resume (each Colab session ends with a checkpoint on Drive).
14. **Launch the real run:**
    ```bash
    nohup python -m scripts.train_oasst1 --preset tiny_100m \
      --packed-dir <data_dir> --tokenizer-json <tok_path> \
      --out-dir /content/drive/MyDrive/Talos/Styx_100M/checkpoints/fw-edu-run \
      --token-budget N --warmup-tokens W --lr-decay cosine \
      --batch B --seq 512 --device cuda --seed 0 \
      > /content/drive/MyDrive/Talos/Styx_100M/logs/run1.log 2>&1 &
    ```
    (warmup example for a 200M budget: `--warmup-tokens 2000000` = 1 % — tune per plan; fixed-LR
    runs omit `--warmup-tokens`/`--lr-decay`.)
15. **Watch epoch lines**: `epoch k/E: steps=… train_loss=… val_loss=… checkpoint=…`; confirm loss is
    finite and decreasing-ish; **there is no automatic NaN abort (§12) — monitor manually.**
16. **Copy to Drive after each epoch** (if `--out-dir` is on scratch): `cp -r <run_dir>/step-*.pt
    <run_dir>/metrics.json <run_dir>/train_run_metadata.json …/checkpoints/` (per-epoch automation:
    NEEDS IMPLEMENTATION, §13.5).
17. **On disconnect/timeout/new session**: resume (§19) —
    `python -m scripts.train_oasst1 … --resume <run_dir> --epochs <same target> …`
    (the directory scan picks the numerically-newest valid checkpoint).
18. **On budget stop** (`tokens N / budget N (REACHED)` in stdout): training is done; do not extend
    `--token-budget` without owner approval (gate §16 row 32).
19. **Eval the final checkpoint on the held-out val slice**: §27.
20. **Generation samples**: `python -m scripts.generate --checkpoint <run_dir> --prompt "…"`
    (greedy, deterministic; 5 fixed prompts per the 10M notebook convention).
21. **Consistency guard** (mirror the 10M notebook cell): load `step-<N>.pt`, assert
    `ck["format"]=="talos-training-checkpoint-v1"`, `ck["n_params"]==96_482_304`,
    `ck["vocab_size"]==1024`, `ck["tokens_consumed"]==N`.
22. **Export safetensors** (release format): `python -m scripts.export_safetensors --checkpoint
    <ckpt> --out-dir <release>/talos-mini-100m-fw-edu` (§27/§30).
23. **Write the run report** from `metrics.json` + `train_run_metadata.json` + `manifest.json`
    (dataset, token counts, budget, LR config, wall, tok/s, milestone achieved).
24. **Owner decision artifacts**: eval numbers + generation samples + report → decide SFT (OASST1,
    §24) or budget extension.
25. **Clean up**: prune old `step-*.pt` beyond the last few on Drive (each ≈1.16 GB) — keep the
    final one + `metrics.json` + metadata forever.

---

## 19. Resume / Recovery

### 19.1 Failure matrix — exact commands

| Event | What survives | Recovery command (all flags must match the original run EXCEPT `--resume`/`--epochs`) |
|---|---|---|
| Colab disconnect / runtime reset | last epoch's `step-*.pt` on Drive (+ in-checkpoint optimizer/RNG/counters) | `python -m scripts.train_oasst1 … --resume <run_dir> --epochs <SAME TOTAL>` (dir scan → numeric-newest valid) |
| Session OOM / killed mid-epoch | all prior epoch checkpoints; current epoch lost | same as above (resume from last completed epoch's checkpoint) |
| NaN/Inf loss or gradient | **auto-handled** (§12): step skipped, safety checkpoint written immediately, `metrics.json` records the event; after 3 consecutive bad steps the run ABORTS (exit code 3) with the last-good checkpoint on disk | `--resume <run_dir>` continues from the abort (numeric-newest = the last safety checkpoint); inspect `metrics.json → bad_steps[]`; lower `--lr`/raise `--grad-clip`/`--max-consecutive-bad-steps 0` if it recurs |
| Corrupt/truncated `step-N.pt` (crash mid-save, non-atomic `.pt`) | older valid checkpoints | `--resume <run_dir>` — the scanner skips the corrupt candidate with a warning and uses the next-newest (`:993-1013`) |
| Tokenizer file missing on resume | none (refuses) | restore `tokenizer.json` to the recorded `tokenizer_path`, or unpacked-run: pass `--tokenizer-json` matching the recorded fingerprint (`:1178-1208`) |
| Drive unavailable at resume | the run dir is unreachable | re-mount Drive; if checkpoints were staged on scratch they are gone — this is why per-epoch Drive copy matters |
| Budget already consumed at resume | — | run finishes with a warning and no new steps (`:785-800`); raise `--token-budget` only with approval |
| Epochs already reached (resume) | — | `ValueError: …nothing left to train; raise --epochs` (`:773-775`) |
| Wrong preset passed | — | `ValueError` from `validate_resume_checkpoint` (`:230-242`) |

### 19.2 What resume REJECTS (all CURRENT, all loud `ValueError`)

- **Wrong architecture/preset**: checkpoint resolves to a different canonical preset than
  `--preset` (`:230-242`); `n_params` ≠ canonical count → "corrupted or tampered checkpoint".
- **Wrong tokenizer**: file sha256 ≠ checkpoint's recorded fingerprint; `--tokenizer-json` on
  resume must match the fingerprint; on the packed path the manifest sha256 must match too
  (`:1178-1208`).
- **Wrong dataset**: `data_provenance.source` differs (jsonl vs packed) (`:1060-1065`); on packed,
  `manifest_identity` (seq/dtype/tokenizer/rows) must equal the recorded identity (`:1086-1105`).
- **Wrong seq**: explicit `--seq` ≠ manifest `seq_len` on the packed path (`:1066-1079`).
- **Pre-resume-support checkpoints** (no `optimizer_state_dict`): resume refuses (`:755-758`).

### 19.3 Override policy (documented)

- `--epochs` is the **target total** (not "additional"); resume runs `resume.epoch+1…epochs`
  (`:741-747`).
- `--lr` from the CLI **overrides** the checkpoint's stored LR after resume (all param groups
  reset to the CLI value, `:762-763`); warmup/decay/tokens-consumed continue from restored
  counters.
- Resume is **bit-exact vs. an uninterrupted run on CPU** (test-asserted,
  `tests/test_training.py::test_resume_bit_exact`; ckpt RNG snapshot/restore `:341-366`) — on CUDA
  see §23 (not bit-reproducible).

---

## 20. APIs (exact signatures from code — no invented APIs)

### 20.1 `scripts/prepare_corpus.py`

- `prepare_corpus(reader, tokenizer, *, out_dir, target_tokens, val_tokens, tokenizer_path,
  seq_len, dtype, val_skip_docs, stream_buffer_docs, rows_per_shard, text_field, dataset_meta,
  args_echo=None) -> dict` (metadata) — `:333`.
- CLI: `python -m scripts.prepare_corpus` — full flag table §8.1 (`:557-591`).

### 20.2 `scripts/train_oasst1.py` (the trainer)

- `make_arg_parser() -> argparse.ArgumentParser` — `:886` (all flags in §21).
- `train_run(args: argparse.Namespace) -> dict` (the metrics dict; also writes `metrics.json`) —
  `:993`.
- `train_epochs(model, train_ds, val_ds, *, out_dir, tokenizer_path, lr, epochs, device, seed,
  max_steps_per_epoch=None, val_max_steps=None, resume_from=None, token_budget=None,
  warmup_tokens=0, lr_decay="none", run_metadata=None, save_every_tokens=0, grad_clip=0.0,
  max_consecutive_bad_steps=3) -> TrainingHistory` — `:742`. Raises `BadStepsAbort` after K
  consecutive NaN/Inf steps (carrying the partial `TrainingHistory`).
- `save_checkpoint(path, model, step, train_loss, val_loss, tokenizer_path, *, optimizer=None,
  rng_state=None, epoch=None, tokens_consumed=None, run_metadata=None, tokenizer_fingerprint=None,
  last_grad_norm=None)` — `:661` (atomic write; the SINGLE writer for epoch-end, periodic and
  safety checkpoints). `load_checkpoint(path) -> dict` — `:737`.
- `list_checkpoint_candidates(checkpoint_dir) -> List[str]` — `:206`.
  `validate_resume_checkpoint(ckpt, preset) -> None` — `:222`.
- `evaluate(model, dataset, device, *, max_steps=None) -> Optional[float]` — `:588`.
- `build_preset_model(preset) -> TalosGPT` — `:546` (calls `check_preset_compat` `:503`).
- `split_jsonl(src_path, out_dir, ratio=0.9, seed=0, max_docs=None) -> SplitResult` — `:382`
  (JSONL path only).
- `TokenSchedule(lr, warmup_tokens, decay, budget, min_lr_ratio=MIN_LR_RATIO)` with
  `lr_at(tokens) -> float` and `to_dict()` — `:127`.
- `capture_rng_state() -> dict` / `restore_rng_state(state)` — `:341/:358`.
- `write_run_metadata(out_dir, run_metadata) -> str` — `:312` (atomic).
- `tokenizer_file_sha256(path) -> str` — `tokenizer/tokenizer.py:35`.

### 20.3 Data layer

- `data/packed.py`:
  - `load_packed_manifest(dir, *, expected_seq=None, expected_vocab=None,
    expected_tokenizer_sha256=None) -> dict` — `:87`.
  - `manifest_identity(manifest) -> dict` — `:223` (stable identity: format/seq/dtype/tokenizer
    sha/row counts).
  - `packed_phase_shard_paths(manifest, dir, phase) -> List[str]` — `:258`.
  - `PackedTokenDataset(shard_paths, *, seq_len, batch_size=1, expected_dtype="int32",
    max_id=None, drop_last=True)` — `:270` (IterableDataset, yields `(B, seq)` int32 tensors).
- `data/readers.py`: `HuggingFaceReader(dataset_id, split, text_field, streaming=True,
  config=None)`; also `JSONLReader(paths, source, text_field)`, `ParquetReader(...)`.
- `data/tokenized.py`: `StreamingTokenizedDataset(path, tokenizer, seq_len, batch_size, mode,
  eos, max_id, dtype, …)` (JSONL path only).

### 20.4 Model

- `tiny_100m_config() -> ModelConfig` — `configs/presets.py:96`; `.derive()` — `model/config.py:92`;
  `TalosGPT(config, attention_backend=None)`, `.forward(input_ids, position_ids=None,
  use_cache=False, cache=None) -> (logits, cache)`, `.num_parameters(trainable_only=True) -> int`
  — `model/gpt.py:24, 68, 168`. `ModelConfig(**asdict)` round-trips checkpoints (`:226` in trainer).

### 20.5 Checkpoint load/validate + eval entrypoints

- `scripts/eval_checkpoint.py::make_arg_parser()` — `:47`; `evaluation.harness.run_eval(
  checkpoint_path, data=None, *, train_data=None, seq_len=64, batch_size=4, drop_last=True,
  max_steps=None, seed=0, device=None, out_metrics=None) -> EvalResult` — `:300`.
- `scripts/generate.py::make_arg_parser()` — `:265`.
- `scripts/export_safetensors.py::make_arg_parser()` — `:309`.

---

## 21. CLI — every flag

### 21.1 `scripts/train_oasst1.py` (`make_arg_parser`, `:1082-1194`)

| Flag | Type | Default | Required | Valid values / notes |
|---|---|---|---|---|
| `--data` | path | None | one of `--data`/`--packed-dir` | JSONL corpus (OASST1 path) |
| `--packed-dir` | dir | None | one of `--data`/`--packed-dir` | `prepare_corpus` output; rows verbatim |
| `--out-dir` | dir | — | **yes** | run dir (split/tokenizer/checkpoints/metrics/metadata) |
| `--seed` | int | 0 | no | split+training seed |
| `--preset` | str | `"tiny"` | no | any of the 4 canonical presets; use `tiny_100m` |
| `--resume` | path | None | no | `step-<N>.pt` file or dir (numeric-newest valid; corrupt skip) |
| `--split-ratio` | float | 0.9 | no | JSONL only |
| `--split-max-docs` | int | None | no | JSONL only |
| `--bpe-num-merges` | int | None | no | JSONL BPE path only (default 764) |
| `--bpe-minfreq` | int | 2 | no | JSONL BPE path only |
| `--bpe-max-docs` | int | None | no | JSONL BPE path only |
| `--bpe-max-chars` | int | None | no | JSONL BPE path only |
| `--tokenizer-json` | path | None | on packed: recommended | existing tokenizer.json; fingerprint machinery §6 |
| `--epochs` | int | 1 | no | target TOTAL epochs (incl. resumed) |
| `--seq` | int | None (JSONL: 64) | no | packed: must equal manifest `seq_len` or error |
| `--batch` | int | 4 | no | batch size |
| `--lr` | float | 3e-3 | no | peak LR |
| `--token-budget` | int | None | no | PRIMARY stop target: tokens since run start incl. resumed |
| `--warmup-tokens` | int | 0 | no | linear warmup span (must be < budget if budget set) |
| `--lr-decay` | str | `"none"` | no | `none` \| `cosine` (cosine needs `--token-budget`) |
| `--save-every-tokens` | int | 0 | no | **intra-epoch checkpoint cadence**: v1 checkpoint at every absolute multiple of N tokens since run start (restored on resume); 0 = epoch-end only (old behavior). At ~2K tok/s, N = one epoch's tokens is the natural T4 value (§13.4) |
| `--grad-clip` | float | 0.0 | no | max_grad_norm for `clip_grad_norm_` after backward; 0 = off (old behavior, step untouched). Pre-clip norm recorded in metrics/checkpoints (§12) |
| `--max-consecutive-bad-steps` | int | 3 | no | abort (exit 3) after K consecutive NaN/Inf steps; 0 = never abort (steps still skipped + safety-checkpointed). Detection is ALWAYS on (§12) |
| `--max-steps-per-epoch` | int | None | no | CI/smoke cap |
| `--val-max-steps` | int | None | no | CI/smoke cap on val batches |
| `--device` | str | None (auto) | no | auto → `cuda` if available else `cpu` |

**Still missing (NEEDS IMPLEMENTATION — do not try to pass them):** `--benchmark`,
`--fp16`/`--amp` (FORBIDDEN, §11), `--eta`, `--vit`… none of these exist. The three hardening
flags above (`--save-every-tokens`, `--grad-clip`, `--max-consecutive-bad-steps`) are the ones
this doc's earlier revisions listed as `--super-save-every`/`--nan-abort` (§29 pruned accordingly).

### 21.2 `scripts/prepare_corpus.py` — full table in §8.1 (11 flags; `--tokenizer-json`,
`--target-tokens`, `--out-dir` required).

### 21.3 `scripts/eval_checkpoint.py` (`:47-85`)

`--checkpoint` (file or dir; required) · `--data` (val JSONL; default = ckpt dir's
`data/val.jsonl`) · `--train-data` (optional train loss) · `--seq` (64) · `--batch` (4) ·
`--keep-partial` (flag) · `--max-steps` (None) · `--seed` (0) · `--device` (auto) ·
`--out-metrics` (default `<ckpt dir>/eval-metrics.json`).

### 21.4 `scripts/generate.py` (`:265-291`)

`--checkpoint` (file or dir; required) · `--prompt` (required) · `--max-new-tokens` (32) ·
`--temperature` (None = greedy argmax, deterministic) · `--seed` (None; mandatory with
`--temperature`) · `--device` (auto).

### 21.5 `scripts/export_safetensors.py`

`--checkpoint` (required) · `--out-dir` (required).

---

## 22. Configuration Precedence

| Level | Wins over | Notes |
|---|---|---|
| **Code default** | — | e.g. `--lr` default 3e-3, `--batch` 4, preset `tiny` |
| **Preset** | code defaults | `--preset tiny_100m` fixes the model config from `ALL_PRESETS`; the config registry (`configs/canonical.py`) defines the exact count |
| **CLI** | preset + code defaults | `--lr`, `--batch`, `--seq`, `--epochs`, `--token-budget`, … |

**No hidden overrides exist.** Three documented, intentional exceptions (all in code, all loud):

1. **Packed `seq`**: the manifest's `seq_len` is authoritative; `--seq` may only *equal* it
   (`:1066-1079`). (`effective_seq` is the manifest value unless `--seq` matches.)
2. **Resume LR**: the CLI `--lr` replaces the checkpoint's stored LR after resume (`:762-763`).
3. **Resume epochs**: `--epochs` is the target total, not additional (`:741-747`).

Everything a run *did* is recorded (not enforced) in `train_run_metadata.json`
(`training_config` + `model_config` + `data_provenance` + `tokenizer` + `git`) and in the manifest
metadata (`args` echo) — so a post-hoc audit can always tell which precedence was in effect.

---

## 23. Reproducibility

### 23.1 What is verified

- **CPU bit-reproducibility is a VERIFIED repo fact**: fixed seed, no dropout, no RNG in the data
  path, deterministic batch order; resume is bit-exact vs. an uninterrupted run (test-asserted
  `tests/test_training.py::test_resume_bit_exact`). Same `(src, ratio, seed)` → identical JSONL
  split (`split_jsonl`); same packed rows → identical train order (rows stream verbatim).
- Per-checkpoint RNG snapshots (torch CPU/GPU + numpy + python random) are captured and restored
  (`capture_rng_state`/`restore_rng_state`, `:341-366`).
- Identity machinery: tokenizer sha256 (file content), `manifest_identity` (format/seq/dtype/
  tokenizer sha/rows), checkpoint `n_params`/`vocab_size`, `git_repo_state`
  (`{commit, branch, dirty}`, `:256-281`) in the run metadata, and `train_run_metadata.json`
  embedded in every checkpoint.

### 23.2 What is NOT (say it plainly)

- **CUDA is NOT bit-reproducible.** torch CUDA kernels (esp. reduction/matmul) are not
  deterministic across runs/hardware; resume on a T4 is *state-continuous* (optimizer + RNG
  restored), not bit-identical to an uninterrupted T4 run. Do not assert bit-equality for any
  T4-vs-T4 or T4-vs-CPU comparison; compare losses at ~1e-4 tolerance or better, report
  `device` in every artifact (the repo does).
- The repo's determinism tests are CPU; the packed corpus prep is deterministic given an identical
  HF stream (HF revision is recorded, not pinned — a dataset-fetch drift between prep runs changes
  the stream; the manifest records the revision/sha it saw).

---

## 24. Data Preservation (OASST1 → SFT later)

- **OASST1 is preserved untouched for the SFT stage** (owner directive 2026-09-27): the existing
  OASST1-derived subset (2,000 docs, 1,137,265-byte JSONL ≈ ~1.1M Talos tokens at the
  measured ~1 tok/char) and its JSONL pipeline (`--data` path, `split_jsonl`, BPE
  training, `StreamingTokenizedDataset`) remain intact and are NOT consumed by the pretraining
  pipeline. `--packed-dir` and `--data` are exclusive inputs; nothing mixes them.
- The canonical OASST1 subset identity for the SFT phase: rows 0-1999, content-sha
  `bfe3285da9dd1e250822449ae956b0bcec4921231179876ac983fd0c508d1f6c` (NUL-joined texts; the file
  sha varies with JSONL serialization) — per `shared/colab/README.md` and `benchmarks/phase-b/
  data-provenance.json`.
- **Pipeline (fixed order):** FineWeb-Edu packed corpus → `tiny_100m` pretraining → OASST1 SFT
  (later, separate run) → evaluation. **Never mix OASST1 into the base corpus.**
- The research memo's runner-up (fineweb general) is a *corpus choice*, not an SFT mix; an 80/20
  edu/web mix is allowed at the corpus stage if the owner approves the report (§16 row 32).

---

## 25. Token-Budget Training

- **`--token-budget` is the PRIMARY stop target.** The trainer stops mid-epoch as soon as
  `tokens_consumed >= budget`, then still runs validation and writes the epoch checkpoint
  (`:828-856`). `--epochs` only bounds the loop; whichever limit comes first wins.
- Step-based mode is documented separately and still exists: omitting `--token-budget` gives the
  legacy fixed-LR/epochs behavior (optionally `--max-steps-per-epoch`), exactly the machinery the
  1M/10M ladder runs used.
- **Budget → steps math**: `steps_needed = ceil(tokens_remaining / (B * (S-1)))` where
  `tokens_remaining = budget - tokens_consumed` (from the resumed checkpoint).
  Examples: B=32/S=64 → 2,016 tok/step; B=4/S=512 → 2,044 tok/step.
- **Resumed tokens are included**: the budget counts tokens since RUN START (all sessions);
  legacy checkpoints without the counter are estimated as `step × B × (S-1)` (exact for
  constant-shape runs) with a warning (`:765-784`).
- LR schedule is token-indexed: warmup spans `[0, W)`, cosine decays over `[W, budget)` to
  10 % — both computed from `tokens_consumed` pre-step (`TokenSchedule.lr_at`, `:137-153`).

---

## 26. 100M Scale Recommendation (heuristics vs. measured — labeled)

**This is NOT a proven optimum.** No tiny_100m training result exists yet; every number below is an
estimate or an extrapolation of smaller-model measurements. Use it to size the run, not to justify
a specific budget as optimal.

| Milestone | Tokens | Approx. wall-clock on T4 (ESTIMATE) | Basis |
|---|---|---|---|
| Staged first milestone (plan's suggestion) | 200–250 M | **~20–40 T4-h fp32** | audit §13: "a 100K-step, B=32, S=64 run = 204.8M tokens ≈ 20-40 GPU-hours on a T4 fp32" (`shared/talos-100m-audit.md` §13) — heuristic FLOP model, not measured |
| Chinchilla-anchored (20 tok/param) | ~1.93 B | **~90–380 T4-h** | research memo §6: derived from audit's 2.0–6.1K tok/s range at B=32/S=64 ("1–3 steps/s realistic") × 1.93 B |
| "More tokens than Chinchilla" | 4–6 B | ~180–1,100 T4-h (linear from above) | memo §6 — not one free-session material; requires merged-`--resume` multi-session |

Sources of the underlying throughput range (all clearly labeled, none measured on a T4):
- **No measured T4 tok/s exists anywhere** (audit + research memo both state this). The 2.0–6.1K
  tok/s range is the audit's *theoretical* estimate; the owner-reported B=4/S=512 numbers in §11
  imply ~3.7K tok/s at that shape, which is within the estimate band.
- CPU reference points (real measurements, different hardware): 254K ≈ 12.5K tok/s;
  tiny_1m b32 ≈ 14.5K tok/s (`docs/SCALING.md:35,137`).

**Ladder evidence (measured, from the merged runs) — read carefully:**
- tiny_1m needed **≥ ~15.5M tokens** to beat tiny (254K) at all: the full-budget A/B gave
  val 1.8697 (1M) vs 1.9275 (254K); at smaller budgets the bigger model lost
  (`benchmarks/phase-b/metrics-tiny-1m-15m.json`, `metrics-tiny-15m.json`).
- **Do not judge tiny_100m at tiny budgets.** The same crossover logic says a 96.5M model trained
  on 15M tokens is expected to be *worse* than the smaller models were at that budget; the staged
  200–250M first milestone is the smallest budget at which a widening is plausible, and a serious
  verdict needs the Chinchilla-scale budget.
- Recommendation for the training-report gate (§16 row 32): propose the **staged 200–250M milestone
  first**, run the T4 benchmark to convert it to wall-clock, and let the owner decide whether to
  extend toward 1.93 B in resumed sessions.

---

## 27. Evaluation

### 27.1 `scripts/eval_checkpoint.py` (CURRENT)

```bash
python -m scripts.eval_checkpoint --checkpoint <run_dir> \
    --data <data_dir>/val.jsonl_or_shard_path_equivalent \
    [--train-data ...] [--seq 512] [--batch 32] [--keep-partial] \
    [--max-steps N] [--seed 0] [--device cuda] [--out-metrics <path>]
```

Important nuances (from `scripts/eval_checkpoint.py:18-85` + `evaluation/harness.py`):
- **`--checkpoint`** accepts a file or a directory (newest `step-<N>.pt` wins).
- **Data**: for the packed path the trainer's val shards are consumed via `PackedTokenDataset`
  inside a *training-side* val pass; the eval harness reads **JSONL**. For a packed-corpus run the
  official per-checkpoint loss is the trainer's recorded `val_loss` (per-epoch, §15.2); to re-eval
  on the held-out slice with the harness, materialize the val region as JSONL or write a small
  packed→jsonl adapter (NEEDS IMPLEMENTATION if exact harness parity on packed data is required).
- **Default batch(4)/seq(64) match the training defaults** so recomputed loss matches the recorded
  `val_loss`; for a B=32/S=512 run pass `--batch 32 --seq 512`. The tiny_10m notebook convention
  requires `--batch 32` for bit-exactness of comparisons (shared/colab README) — same applies here.
- Reported (from `EvalResult`, `evaluation/harness.py:80-99`): `val_loss`, **`val_perplexity`**
  (= exp(val_loss)), `val_accuracy`, optional `train_loss`, `tokens_processed`, `eval_wall_s`,
  `throughput_tok_per_s`, `peak_rss_mb`, checkpoint identity fields (`params`, `vocab_size`,
  `tokenizer_vocab_size/merges`, `checkpoint_format`, `checkpoint_step`). Same checkpoint + split +
  seed → identical numbers (test-asserted).

### 27.2 Generation samples (CURRENT)

```bash
python -m scripts.generate --checkpoint <run_dir> --prompt "The capital of France is" \
    [--max-new-tokens 32] [--temperature 0.8 --seed 0] [--device cuda]
```

Greedy (no `--temperature`) is deterministic and RNG-free; temperature sampling requires `--seed`.
The CLI truncates long contexts correctly (audit §5; library guard fixed in PR #25).

### 27.3 What is NOT claimed

- **No benchmark-suite scores** (MMLU/HELM/etc.) are computed or claimed anywhere for this preset —
  none exist. Held-out val loss/perplexity on the disjoint val slice is the only objective number;
  generation samples are qualitative. `data/contamination.py` exists but is not wired into anything
  for this run.

---

## 28. Known Issues

| Issue | Detail | Status |
|---|---|---|
| **FP16 AMP overflow (do NOT enable AMP)** | Owner-reported, verbatim: `RuntimeError: value cannot be converted to type c10::Half without overflow`. Code path: `model/attention.py:182-184` — `mask = torch.zeros_like(scores); mask.masked_fill(~allowed, NEG_INF)` under fp16 autocast overflows filling `-inf` in half precision. No AMP code exists in the repo; the plan forbids enabling it until the mask path is proven safe. BF16 is unavailable on T4 (SM7.5). | FORBIDDEN (plan) + not implemented |
| NaN/Inf detection, grad clipping, abort-with-checkpoint | **FIXED (2026-09-27, hardening pass)** — detection always-on in `train_epochs`; bad steps skip the optimizer update, log structured events into `metrics.json`, write a safety checkpoint, abort after 3 consecutive failures; `--grad-clip` bounds the norm. Metadata now records `nan_inf_detection: True` + the real `gradient_clip_type`. | CURRENT (§12, `tests/test_trainer_stability.py`) |
| Non-atomic `.pt` writes | **FIXED (2026-09-27)** — `save_checkpoint` writes `<path>.tmp` + `os.replace` for every cadence (epoch-end, periodic, safety); the corrupt-fallback scanner remains for pre-hardening artifacts. | CURRENT (§13.4) |
| Local-box memory limits (dev box, not Colab) | The build box is memory-starved (~3.9 GB total, <2.2 GB usable); `tests/conftest.py` skips 100M-scale tests under ~900 MB MemAvailable. CI runners (7 GB) run everything. Not a Colab-T4 concern. | documented in `.github/workflows/tests.yml` |
| `PyGILState_Release` shutdown error | Cosmetic interpreter-shutdown message after `datasets` streaming on this box (research memo §7). Harmless; may appear in Colab too. | cosmetic |
| No measured T4 tok/s recorded anywhere | Owner's published T4 run has config + losses only; all repo throughput records are CPU. Every T4 number in §11/§26 is estimate or owner-reported. | gap (benchmark §17 fixes it) |
| Owner's 10M-run artifacts unreachable | The audit (2026-09-25) found no repo/artifact backing the owner-described 10M run (val 1.1524, safetensors release, 100K steps); decision to request publishing pending (business plan "owner decisions pending"). Unrelated to this doc's correctness — relevant when publishing claims. | pending owner decision |
| Owner's 32-point preflight list not in repo | §16 — reconstructed from repo guards; original wording unavailable. | gap documented in §16 |

---

## 29. Missing / Required Implementation (consolidated)

Ordered by criticality for a long unattended T4 run. **Items 1–4 of the previous
revision (NaN/Inf detection + checkpoint-before-abort, gradient clipping,
super-save checkpoints, atomic `.pt` writes) are DONE** — see §12/§13; the list
below is what remains.

1. **Benchmark mode** (§17) — `--benchmark-steps N` (or a sibling script) measuring s/step, tok/s,
   `torch.cuda.max_memory_allocated()` VRAM peak, loss stability, printed + saved (e.g.
   `benchmarks/t4-tiny100m-benchmark-<shape>.json`).
2. **`scripts/preflight.py`** (§16) — the 32-point gate as a runnable script; must add: disk-free
   check (Drive + scratch), out-dir clobber guard, GPU probe, cargo-cult the repo's existing
   guards (param count, vocab, manifest validation, tokenizer sha, resume validity).
3. **GPU-memory + ETA logging** (§15.3) — per-epoch `torch.cuda.max_memory_allocated()` + live
   tok/s/ETA line in stdout and `metrics.json`.
4. **100M Colab notebook** — a `shared/colab/talos_100m_colab.ipynb` (pattern exists for
   tiny_1m/tiny_10m; 100M does not) with per-epoch Drive copy. The §12 guards are now
   trainer-side, so the notebook only needs the standard flags
   (`--save-every-tokens`, `--grad-clip`, `--max-consecutive-bad-steps`).
5. **Packed→JSONL eval adapter or a packed-aware eval path** (§27.1) — so
   `scripts/eval_checkpoint` can score the packed val shards directly (currently harness reads
   JSONL; the packed val loss comes from the trainer's per-epoch `val_loss`).
6. **Expose AdamW knobs** (`--weight-decay`, `--betas`, `--eps`) if a non-default optimizer config
   is ever wanted (§10.1 — currently not configurable).

Not required for the run itself: FP16/AMP (forbidden), gradient accumulation (not needed at these
shapes), FlashAttention (plain backend suffices; Flash backend exists but neither is required on
T4 at S=512).

---

## 30. Final Execution Checklist

**Before the run (owner gate required for the real budget):**
- [ ] Repo at `d8bfbdd`, clean tree; `git_repo_state` will record it.
- [ ] `tokenizer.json` present, sha256 == `58e4ad40…` (§6).
- [ ] Packed corpus prepared: `manifest.json` seq=512, dtype int32/uint16, tokenizer sha matches,
  val region disjoint (prepare_corpus logs) (§8).
- [ ] Model+guard sanity: `python -c` build → 96,482,304 params; trainer smoke (20 steps) prints
  `expected: EXACTLY 96,482,304 params, vocab_size 1024` (§5.1, §18.11).
- [ ] T4 benchmark done; wall-clock for the budget accepted (§17, §26).
- [ ] Training report delivered to the owner: dataset, token count, token budget, batch/seq,
  LR config, checkpoint schedule, expected T4 runtime, dataset passes (§16 row 32).
- [ ] Owner approval recorded (the gate — the run must not start before it).
- [ ] NaN/clip/abort guard ON: `--grad-clip 1.0 --save-every-tokens <N> --max-consecutive-bad-steps 3`
  (trainer-side, CURRENT — §12/§13; the operator wrapper of the old §12.4 is no longer needed and
  must NOT be relied on instead of the trainer's own guards).
- [ ] Drive free space ≥ corpus + checkpoints budget + margin (§14); per-epoch copy planned.
- [ ] FP32 only: no AMP anywhere; no `--fp16` flag exists to pass.

**During the run:**
- [ ] Watch per-epoch stdout line (§15.1); verify loss finite + directionally decreasing.
- [ ] Confirm each epoch writes `step-<N>.pt` + metadata on Drive (§13.5).
- [ ] On disconnect: resume with the SAME flags + `--resume <run_dir>` + same `--epochs` (§19).

**After the budget stop (`(REACHED)` printed):**
- [ ] Final `metrics.json` + `train_run_metadata.json` captured.
- [ ] `scripts/eval_checkpoint` on the final checkpoint (batch/seq matching the run) → val loss,
  perplexity, throughput (§27.1); generation samples (`scripts/generate.py`).
- [ ] Consistency guard: `format==talos-training-checkpoint-v1`, `n_params==96_482_304`,
  `vocab_size==1024`, `tokens_consumed==budget` (§18.21).
- [ ] Safetensors release export (§18.22) if publishing.
- [ ] Report to owner: milestone achieved (token budget consumed), measured tok/s, val perplexity,
  samples, recommendation for SFT (OASST1, §24) or budget extension — no invented scores (§27.3).

---

*Generated 2026-09-27 by senior-ml-engineer (rev. 2, 2026-09-27): the trainer hardening pass PR — this revision updates the status table rows 22/23/26, rewrites §12 (now IMPLEMENTED), prunes §29 items 1-4 (done), and adds the three new flags to §10/§21. Base repo: `main` @ `46f9858`. All file:line references verified by reading the code + the green synthetic suites (`tests/test_trainer_stability.py`, trainer suites) on that date. Anything marked MISSING/PROPOSED is not implemented and must not be described as existing.*
