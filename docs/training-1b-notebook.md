# Talos 1B — the two-stage training workflow

The 1B model is trained **from its own random initialisation**. No pretrained
weights, no external foundation model, no distillation from someone else's
model. That decision is what makes this a two-stage run rather than one: a model
that starts from random weights has to learn language before it can learn to
chat, and the second thing is not a substitute for the first.

- **Notebook:** `notebooks/talos_colab_train.ipynb` (14 cells, Colab-native).
- **Supporting modules:** `notebooks/talos_stages.py` (budget, schedule, stage
  contract), `notebooks/talos_data.py` (streams, packing, loss masks),
  `notebooks/talos_hardware.py` (GPU detection + plan),
  `notebooks/talos_eval.py` (honest evaluation),
  `notebooks/talos_torch_utils.py` (loss/precision/optimizer glue),
  `notebooks/talos_training_lib.py` (metrics rows, HTML table, metrics sync,
  public tracker).
- **Tests:** `tests/test_notebook_{stages,data,hardware,eval,torch_utils,structure}.py`
  plus the existing `tests/test_training_lib.py`. All CPU-only.

## 1. Objective and the two stages

| | **Stage A — language pretraining** | **Stage B — conversational tuning** |
|---|---|---|
| model init | random | the final Stage-A checkpoint |
| data | `allenai/c4`, config `en`, splits `train` / `validation` | `HuggingFaceH4/ultrachat_200k`, splits `train_sft` / `test_sft` |
| objective | next-token prediction, every token supervised | next-token prediction on **assistant turns only** (user/system turns masked out of the loss) |
| budget | derived from the compute you have (`STAGE_A_HOURS × tokens/sec`) | measured tokens/epoch × `STAGE_B_EPOCHS` (1–3) |
| LR | 3e-4 peak, cosine to 10 % | 5e-5 peak, cosine to 10 % |
| checkpoints | `<out>/stage-a/step-<N>.pt` | `<out>/stage-b/step-<N>.pt` |

Both stages are resumable, and the stage is recorded **inside** each checkpoint
(`run_metadata.stage`). `validate_stage_resume()` refuses a Stage-B checkpoint
resumed as Stage A and vice versa: resuming one stage as the other trains the
wrong objective on the wrong data.

Stage B **will not start from random weights**. If no Stage-A checkpoint is
found the notebook raises, naming the fix; the only way to run Stage B cold is
the explicit `ALLOW_STAGE_B_FROM_RANDOM_INIT = True` ablation switch.

## 2. Why `allenai/c4` for Stage A

Verified against the HuggingFace datasets-server on 2026-11-17:

| dataset | config | splits (real names) | rows |
|---|---|---|---|
| `allenai/c4` | `en` | `train`, `validation` | 364,608 documents in `validation`; `train` is ~305B English tokens (C4 paper) |
| `HuggingFaceH4/ultrachat_200k` | `default` | `train_sft`, `test_sft`, `train_gen`, `test_gen` | 207,865 / 23,110 / 256,032 / 28,304 |
| `HuggingFaceFW/fineweb` (alternative) | `sample-10BT` | `train` only | 14,868,862 |

C4 was chosen because it has a **real, disjoint validation split** — no
carve-out from the training stream, so no risk of leaking validation documents
into training — and because ~305B tokens means a Colab-scale budget is
compute-bound, never data-bound (no epoch re-cycling). `fineweb` `sample-10BT`
(10B tokens, train split only) is the documented alternative; it would need a
carve-out and is not the default.

Measured with the `talos_1b_49k` tokenizer over 200 `train_sft` documents:
**1,226.1 tokens/document** (median 1,083), so one epoch of `train_sft` is
`207,865 × 1,226.1 ≈ 254.9M` tokens. The notebook re-measures this at runtime
(`MEASURE_DOCS = 100`) rather than trusting this number.

Our own FineWeb-Edu packed corpus (`/home/team/shared/corpus-1b-49k`) is **not**
touched or read by this notebook.

## 3. Token budget: justified by compute, not imposed

There is no 20B default. Stage A's budget is

```
budget = STAGE_A_HOURS × 3600 × estimated tokens/sec(GPU)
```

and the notebook prints it, with the estimate it used, before training starts.
The throughput estimate is FLOPs arithmetic, not a measurement:
`tokens/sec ≈ peak_TFLOPS × 1e12 × MFU / (8 × 1,036,060,672)`, where the `8N`
(not `6N`) accounts for the extra forward pass that gradient checkpointing
performs, and MFU is assumed 0.30 for Ada/Hopper-class parts and 0.22 for
T4/V100. **The ±40 % spread is real: this team has no GPU and has measured
none of these numbers.**

| Colab GPU | VRAM | precision | micro × accum | tokens/step | est. tokens/sec | tokens in 10 h | steps | tokens/param |
|---|---|---|---|---|---|---|---|---|
| A100 SXM4-40GB | 40 GB | bf16 | 8 × 8 | 131,072 | ~11,300 | ~407M | 3,102 | 0.39 |
| L4 | 24 GB | bf16 | 4 × 16 | 131,072 | ~4,380 | ~158M | 1,203 | 0.15 |
| T4 | 16 GB | fp16 + scaler | 2 × 32 | 131,072 | ~1,730 | ~62M | 474 | 0.06 |

Every GPU gets the **same 131,072 tokens per optimizer step**, so the schedule
(steps, warmup, LR curve) is identical across hardware and only wall-clock
differs. Stage B at 2 epochs costs ~510M tokens, i.e. ~12.5 h on an A100, ~32 h
on an L4, ~82 h on a T4 — Stage B is the cheaper stage per token seen but is
usually run for far fewer tokens than Stage A.

**What that buys, honestly.** Chinchilla-scale is ~20 tokens/parameter; these
budgets are 0.06–0.4. At that ratio the model is an early-training artifact: it
produces locally plausible word sequences, mimics the `User:/Assistant:` format,
and fails at facts, arithmetic and long-range coherence. Stage B adds the chat
*behaviour* on top of that weak base — it does not add knowledge. To get a base
model that can actually hold a conversation you need orders of magnitude more
compute than a Colab session; the notebook is built so that a bigger budget is a
one-line change (`STAGE_A_HOURS`, or `STAGE_A_TOKENS_OVERRIDE`) once you have it.

## 4. Dataset pipeline

- **Splits** are the real ones (above); Stage A validates on `c4/en validation`,
  Stage B on `ultrachat_200k test_sft`. Validation rows are drawn once into a
  fixed set, so val loss is comparable across checkpoints and sessions.
- **Formatting.** Stage A: `<bos> … <eos>` per document, every token supervised.
  Stage B: `User: …` / `Assistant: …` turns; each assistant turn ends with the
  tokenizer's EOS **and that EOS is supervised**, so the model learns when to
  stop. `loss_mask` is 0 on user/system tokens, 1 on assistant tokens.
- **Packing.** Documents are concatenated and sliced into `seq_len` rows — no
  padding between documents. Every id is bounds-checked against the 49,152-vocab
  at the data source, so a tokenizer mismatch fails with the document in context
  rather than as a device-side assert mid-step.
- **Shuffling.** Streaming shuffle with `seed = SEED + epoch` and a 10,000-row
  buffer; deterministic for a given (dataset revision, seed, buffer size), which
  is what makes resume possible.
- **Repeat exposure.** `assert_repeat_exposure()` refuses a budget that would
  cycle the corpus more than the stage allows (1 pass for c4/en, 3 epochs for
  ultrachat). The old notebook cycled ultrachat_200k ~100× to reach 20B tokens;
  that path is now an error, not a default.

## 5. Training-loop audit (what was wrong, what it is now)

| item | before | now |
|---|---|---|
| token accounting | counted `batch.numel()` per micro-batch, no cap | `TokenAccountant` with the budget invariant; every add is checked |
| budget overshoot | the loop could start a full step with less than one step of budget left | `plan_step()` returns at most what the remainder can pay for, `0` = stop |
| final partial step | loss divided by `GRAD_ACCUM` even when fewer micro-batches ran | divided by the **actual** micro-batch count |
| LR schedule | recomputed from `step`, but a resumed run restarted the counters | `lr_at(absolute_step)` — identical for a resumed and an uninterrupted run |
| warmup | fixed 2,000 steps regardless of budget | `warmup_ratio × total_steps` (min 20, clamped into the run) |
| gradient clipping | `clip_grad_norm_` result discarded | pre-clip norm recorded in the checkpoint (`last_grad_norm`) |
| fp16 | none (bf16 assumed everywhere) | T4/V100 run fp16 with `GradScaler`, bf16 elsewhere |
| validation | every 500 steps on a re-shuffled stream | every `EVAL_EVERY` steps on a **fixed** row set, masked like training |
| checkpoint timing | every 1,000 steps only | every `CHECKPOINT_EVERY` steps **and** every `CHECKPOINT_EVERY_MIN` minutes (Colab drops sessions) |
| resume | restored weights/optimizer/RNG, then re-streamed the dataset from its start | restores epoch, documents consumed and the packer's leftover ids — the exact token boundary |
| RNG | restored, but the schedule restarted | restored, and the schedule is a function of the absolute step |
| dataset iterator | restarted at epoch 1 on every resume | `RowStream` resumes mid-epoch at the same token boundary |
| loss memory | full `(tokens × 49,152)` fp32 copy | chunked masked cross-entropy (`chunk_tokens = 2048`) |

## 6. Checkpoints

Written by the repo's own `save_checkpoint` — format
`talos-training-checkpoint-v1`, unchanged — so `scripts/eval_checkpoint.py`
reads them directly. Each checkpoint carries: model weights, optimizer state,
step, tokens consumed, epoch, RNG state (torch/CUDA/numpy/python), tokenizer
path + sha256 fingerprint, the model config, and a `run_metadata` sidecar with
the **stage**, the **dataset identity** (`id`/`config`/splits/`seq_len`), the
**dataset position** (epoch, documents consumed, rows in epoch, packer leftover
ids/mask), the optimisation settings, the hardware plan and the last grad norm.

On resume the notebook validates, in order: format, preset + parameter count +
tensor shapes (repo contract), stage, dataset identity, tokenizer fingerprint,
presence of optimizer and RNG state, and that the recorded token count fits the
budget. A corrupt newest checkpoint is skipped with the reason printed and the
next-newest valid one is used.

## 7. Tracker and metrics

- `metrics.jsonl` in the stage directory: one row per log interval with step,
  epoch, tokens, tokens/sec, ETA, train/val loss, checkpoint, GPU, model, params,
  vocab, seq_len, timestamp and **stage**; pushed (throttled to 5 min) to the
  metrics dataset repo.
- `status.json` + `history.json` in the owner's
  `PrometheanStudio/talos-training-tracker` schema: the owner's 15 keys in their
  order and types, plus one additive `stage` key at the end. Loss, tokens,
  tokens/sec, ETA, GPU, checkpoint info and stage are all present.
- Both are failure-isolated: every upload path catches everything, records
  `last_error` and returns `False`. A tracker or network failure cannot stop
  training.

## 8. Evaluation (why a finished notebook is not a chatbot)

Cell 11 runs six general-purpose conversational prompts in greedy **and**
sampled decoding and reports, per generation: token count, whether it stopped on
EOS, distinct-1, distinct-2, the most-repeated 3-gram and a degeneracy flag;
plus validation loss and perplexity for the stage. `honest_verdict()` then
states the tokens-per-parameter actually trained, what that implies, and the
measured rates — and says plainly that the notebook completing is not evidence
of a working chatbot. Nothing in the notebook claims otherwise.

## 9. Hardware and what was actually tested

- **Runs on:** any Colab GPU. Detection is automatic (`detect_gpu()`), the plan
  is chosen from VRAM and compute capability, and `check_cuda_torch()` stops the
  run with instructions if no GPU is visible — it never silently continues on
  the CPU, and it never installs torch (which is what turns a CUDA runtime into
  a CPU one).
- **OOM:** `oom_guidance()` prints the ordered fix list; halving `MICRO_BATCH`
  and doubling `GRAD_ACCUM` keeps tokens/step (and therefore the schedule) the
  same.
- **Storage:** Google Drive when it mounts (`MyDrive/Talos/Talos_1B`), otherwise
  a local directory with a loud warning that Colab's local disk is deleted when
  the runtime ends. Each checkpoint is ~4.2 GB (bf16 weights + optimizer state).
- **Tested:** the whole pipeline on CPU — budget/schedule arithmetic, the
  overshoot guard, packing/masking, mid-epoch resume alignment, checkpoint stage
  validation, tracker/metrics payloads and failure isolation, the loss and
  precision helpers, and the notebook's structure (every cell compiles, every
  imported name exists). `pytest tests/test_notebook_*.py`.
- **Not tested:** no training step has been executed on a GPU, no checkpoint has
  been written by a real run, and no throughput number here is a measurement.
  Those are the owner's runs on Colab.

## 8. Argo audit (2026-11-19) — budget, precision, NaN/Inf, pins, smoke run

**Model identity.** "Argo" is the display name of the Talos 1B model
(`MODEL_DISPLAY_NAME = "Argo 1B"` in `notebooks/talos_training_lib.py`): the
notebook intro, the live display table and the tracker's `model_name` say Argo.
Code identifiers (`talos_1b`, `talos_1b_49k`, `step-<N>.pt`) are unchanged, and
the tracker schema keeps every key — only the `model_name` value changes.

**Stage A budget — what the old rule actually produced.** With
`tokens/sec = peak_TFLOPS × 1e12 × MFU / (8 × 1,036,060,672)`:

| GPU (MFU) | est. tokens/sec | old default 10 h | tokens/param | corrected 4B target | 2B floor |
|---|---|---|---|---|---|
| T4 16 GB (0.22) | 1,725 | 62.1M | 0.06 | 644 h (54 × 12 h sessions) | 322 h |
| L4 24 GB (0.30) | 4,379 | 157.6M | 0.15 | 254 h (21 sessions) | 127 h |
| A100 40 GB (0.30) | 11,293 | 406.6M | 0.39 | 98 h (8 sessions) | 49 h |

That is the owner's "severely insufficient": all three land far below **1**
token/parameter. The budget rule is now, in order: `STAGE_A_TOKENS_OVERRIDE` >
`STAGE_A_HOURS` (time-box) > `STAGE_A_TOKENS_DEFAULT = 4B`. Anything below
`STAGE_A_TOKENS_FLOOR = 2B` prints a loud WARNING before training. Chinchilla
(~20.7B, ≈20 tokens/param) stays the reference point and is **not** imposed:
~1,833 A100-hours is not a Colab run. What each budget buys, from
`budget_capability()`: <1 tok/param = babble; 2-8 = grammatical, locally
coherent English with unreliable facts; 8-15 = readable paragraphs;
>=15 = a genuine small base model.

**Precision / optimizer memory.** Parameters stay fp32 (master weights) with
bf16 (L4/A100) or fp16 + GradScaler (T4) autocast; casting the *parameters* to
bf16/fp16 (the old behaviour) rounds away sub-resolution updates and
bitsandbytes' AdamW8bit refuses fp16 parameters. Static memory for 1.036B:
weights 4.14 GB + fp32 grads 4.14 GB + optimizer state 2.07 GB (8-bit) or
8.29 GB (fp32 AdamW) = 10.35 GB or 16.6 GB, before activations. A checkpoint is
~6.4 GiB (8-bit) or ~12.6 GiB (plain AdamW).

**NaN/Inf and skipped updates.** The loss is checked before `backward()` and the
pre-clip gradient norm before the optimizer step. A non-finite update is
skipped: gradients dropped, no optimizer step, no LR advance, no step counter
advance, `skipped_updates` incremented (display row, metrics jsonl,
`status.json`, checkpoint metadata). The offending micro-batch's tokens are still
counted, because the forward pass consumed them — that matches the partial-step
accounting and guarantees forward progress. 20 consecutive skips raise
`NonFiniteAbort` after writing a checkpoint.

**Stage B accounting + dataset pins.** One epoch emits
`floor(tokens_per_epoch / seq_len) × seq_len` tokens (the packer drops the
remainder at the epoch boundary), so the Stage-B budget is quantised to whole
packed rows and the loop reports `epochs_completed`. Both "tokens consumed" and
"tokens the loss actually saw" (assistant-only) are reported. Both datasets are
pinned to Hub commit SHAs — `allenai/c4` `1588ec45…`, `HuggingFaceH4/ultrachat_200k`
`8049631c…` (resolved 2026-11-19 via `https://huggingface.co/api/datasets/<id>`)
 and the revision is part of the checkpoint's dataset identity, so a resume on
a different revision is refused.

**49K tokenizer selection — what the bug was.** (1) The notebook printed
`tokenizer.num_merges`, an attribute the tokenizer class does not have (it is
`merge_count`), so cell 4 died with `AttributeError` before training. (2) The
checkpoint fingerprint came from a hand-built path
(`REPO_DIR/assets/tokenizers/<name>.json`) rather than the file the loader
verified, so it recorded "whatever is at that path". (3) No expected digest and
no realized-vocab check were passed, leaving only the manifest between a run and
a different 49K/32K/1024-vocab asset. `notebooks/talos_tokenizer.py` now
resolves the asset through the manifest, verifies the sha256 against the code pin
`c7d26ace…`, the manifest and the file, checks vocab 49,152 and 48,892 merges,
enforces the preset pairing and the realized-vocab contract, requires the path to
be the repo's own asset directory, and raises `TokenizerSelectionError` — there is
no fallback.

**Repo pin.** `REPO_PIN_SHA` (one constant at the top of the setup cell) is
checked out with `git checkout --force`, `HEAD` is verified against it, and the
SHA + commit date are printed. A re-run can never reuse a stale checkout;
`REPO_PIN_SHA = ""` tracks main with a warning that the run is not reproducible.

**End-to-end CPU smoke run.** `python notebooks/talos_smoke.py --out /tmp/smoke`
(also `tests/test_notebook_smoke.py`) runs the same functions the 1B run uses —
`RowStream` packing, `train_stage`, `save_checkpoint`/`validate_resume_checkpoint`/
`restore_rng_state`, `inference.generate.generate`, `TrackerPublisher` /
`MetricsSync` — with the canonical **tiny** preset (254,272 parameters, vocab
1,024) on a synthetic corpus: 4 steps, a checkpoint, a resume that lands on the
same token and step count, validation perplexity, a greedy generation, tracker
payloads (`model_name = "Argo 1B"`), then a Stage-B run where prompt masking is
verified (312 of 512 tokens supervised) and the Stage-B checkpoint is refused by
a Stage-A resume. No GPU, no network, ~2 s.
