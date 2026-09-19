# Talos tiny_1m (1,000,320-param) — real OASST1 training run & 254K-vs-1M A/B

*Phase B, 2026-09-19 · branch `phase-b-tiny-1m-oasst1` (PR #22) · main tip `47dde8f` when branched.*
*Every number below is traceable to the committed artifacts in this directory
(`metrics-*.json`, `eval-*.json`, `generation_samples.md`, `data-provenance.json`)
and to run logs under `/tmp/phaseb/`. No estimates are placed in result slots.*

## 1. What this run is

The `tiny_1m` preset (**exactly 1,000,320 params**; hidden 128, 3 layers, 8
heads, 4 KV heads, dense FFN 512, vocab 1024, max_seq 512 — `docs/SCALING.md`
§3.1) trained end-to-end through the **existing** pipeline
(`scripts/train_oasst1.py --preset`, Phase-A preset-aware) on a **2,000-doc
real OASST1 subset**, compared against the canonical `tiny` preset
(**exactly 254,272 params**) on the **identical corpus, split, tokenizer,
sequence order and step budget** (3 full epochs each = 11,541 steps, seed 0).
Phase B adds no code.

## 2. Data provenance (exact)

- **Source dataset:** `OpenAssistant/oasst1`, split `train`, config `default`
  (84,437 rows; `text` field used; rows API).
- **Reused 1,800 docs:** `/tmp/oasst1_real.jsonl` (997,138 B; sha256
  `8a29212c…7b3de`) == dataset rows **0–1799** (first/last row byte-verified
  against the rows API).
- **Fetched 200 docs:** rows **1800–1999** via the same rows API, two
  100-row chunk requests (API caps `length=100`).
- **Corpus:** `/tmp/phaseb/oasst1_2000.jsonl` — 2,000 docs, 1,137,265 B,
  sha256 `e75b10aa…69a69`; full record in `data-provenance.json`.
- **Split:** `split_jsonl(ratio=0.9, seed=0)` (the owner-published
  methodology) → **1,800 train / 200 val**, disjoint, byte-reproducible.
- **Cross-check:** this run's train-split tokenizer is **byte-identical** to
  the owner's published `tokenizer.json` (HF
  `PrometheanStudio/talos-mini-254k-oasst1`) — the local train split equals
  the owner's, and the BPE trainer is deterministic. Both runs therefore
  share the tokenizer (764 merges, vocab 1024; `tokenizer_vocab ≤
  model_vocab` holds: 1024 ≤ 1024).

## 3. Method (identical for both models)

| knob | value |
|---|---:|
| data | 2,000 docs → 1,800 train / 200 val, seed 0, ratio 0.9 |
| tokenizer | Talos-native byte-level BPE, **train split only**, vocab 1024 (256 bytes + 4 specials + **764 merges**), ~105 s per run |
| seq / batch / lr / opt | 64 / 4 / 3e-3 / AdamW (established local methodology) |
| epochs / steps | 3 epochs / **11,541 steps** (3,847/epoch) per model |
| seed | 0 (split + training) |
| guards | canonical param-count + vocab guard before training (254,272 / 1,000,320) |
| eval | per-epoch val loss; final checkpoint re-evaluated with `scripts/eval_checkpoint` (bit-exact vs recorded) |
| generation | `scripts/generate`, greedy, 5 fixed prompts, 32 new tokens |

## 4. Owner's comparison table — local A/B (identical data + tokenizer + steps)

| Metric | tiny 254,272 | tiny_1m 1,000,320 |
|---|---:|---:|
| Parameters | **254,272** | **1,000,320** (3.93×) |
| Layers | 2 | 3 |
| Hidden size | 64 | 128 |
| Vocab | 1024 | 1024 |
| Train loss (final, epoch 3) | 2.1610 | 2.1826 |
| Validation loss (final, epoch 3) | **2.3704** | 2.3950 |
| Tokens per sec (train phase, mean of epoch walls) | **~25.5K** | ~9.2K |
| Training time (3 epochs) | 114.3 s train phase · 221.4 s run total¹ | 317.5 s train phase · 422.5 s run total¹ |
| Peak memory (CPU peak RSS — **no GPU exists on this box**) | 364.4 MiB train · 231.2 MiB eval | 378.2 MiB train · 237.8 MiB eval |
| Checkpoint size (final `step-11541.pt`) | 1,026,751 B² | 4,014,773 B |
| Eval throughput (val 102,816 tok) | 78,096 tok/s | 31,220 tok/s |
| Val perplexity / next-token accuracy | 10.7013 / 0.3414 | 10.9681 / 0.3309 |
| Decode latency (greedy KV-cache, CLI) | 1.3 ms/token | 2.0 ms/token |

¹ `wall_s` in `metrics.json` includes the ~105 s train-split BPE + split + per-epoch
val + checkpoints (same for both runs). ² tiny final checkpoint size measured from
`/tmp/phaseb/run_tiny/step-11541.pt` (the 254K fp32 weights ≈ 0.97 MB + config).

Per-epoch detail:

| model | epoch 1 train / val | epoch 2 train / val | epoch 3 train / val | epoch walls |
|---|---|---|---:|---:|
| tiny | 2.6297 / 2.6844 | 2.3172 / 2.4863 | 2.1610 / 2.3704 | 37.3 / 38.9 / 38.1 s |
| tiny_1m | 2.6426 / 2.6759 | 2.3516 / 2.5017 | 2.1826 / 2.3950 | 104.0 / 106.2 / 107.2 s |

## 5. Reference — owner's published T4 baseline (listed separately, not comparable)

HF `PrometheanStudio/talos-mini-254k-oasst1` (`step-7680.pt` + `metrics.json`):
254,272 params, 2 layers, hidden 64, vocab 1024, **7,680 steps, batch 32**,
seq 64, lr 3e-3, device **T4 GPU**, train 1.7859 / val 1.9177, 2,000-doc
subset (1,800/200) **with its own tokenizer/split pipeline** (tokenizer
byte-identical to ours — verified). Not directly comparable to the local A/B:
different hardware, ~5.3× more tokens processed (7,680 × 32 × 63 ≈ 15.5M vs
our 2.91M), different batch dynamics.

## 6. Evaluation (final checkpoints; `scripts/eval_checkpoint`, seed 0, seq 64, batch 4)

- **tiny**: val loss **2.3703659961723 == recorded 2.3703659961723** (bit-exact) ·
  ppl 10.7013 · acc 0.3414 · 78,096 tok/s · peak RSS 231.2 MiB — canonical
  guard OK (254,272 / vocab 1024).
- **tiny_1m**: val loss **2.3949950500474033 == recorded** (bit-exact) · ppl
  10.9681 · acc 0.3309 · 31,220 tok/s · peak RSS 237.8 MiB — canonical guard
  OK (1,000,320 / vocab 1024).

## 7. Generation

All 5 prompts × 3 checkpoints (local tiny, local tiny_1m, **published HF
baseline**) are in `generation_samples.md`: greedy, deterministic, 32 tokens;
all three checkpoints passed the pre-generation consistency guards (format,
canonical params, vocab, tokenizer compat). Decode (CLI, fresh process):
**tiny 1.3 ms/token · tiny_1m 2.0 ms/token · published 1.1 ms/token**. Output
is unstructured babble everywhere (including the published T4 model) — the
honest expectation at this scale; the deliverable is pipeline determinism +
guards, not quality.

## 8. Environment & provenance

- CPU: **Intel(R) Xeon(R) Processor @ 2.90GHz**, 2 cores (`nproc=2`,
  `siblings=2`, no HT) · **no GPU** (nothing on this box measures GPU memory;
  "peak memory" = CPU peak RSS, `ru_maxrss`, per `docs/SCALING.md` §5).
- RAM: 4,027,308 kB total; load average ≤ 0.8 during runs (no competing load).
- torch **2.13.0+cpu** (`torch.get_num_threads()=2`); python 3.12.
- git: branch off `47dde8f`, no code changes in Phase B.
- seeds: 0; all runs deterministic (same command → identical metrics).

## 9. Honest footnotes / deviations

- **No superiority claim for the 1M model:** at this identical 3-epoch /
  11,541-step budget on this corpus, tiny_1m's final val loss (2.3950) is
  slightly **worse** than tiny's (2.3704), and its accuracy is slightly lower
  (0.3309 vs 0.3414). The bigger model underfits this step/data budget (same
  lr, 4× params, 2.8× slower per step). This is a valid measured negative at
  this scale, not a trend.
- Throughput: `tok/s` = tokens/wall, per `metrics.json` and §5 conventions;
  both whole-run and train-phase rates are given (identical overhead per run,
  so the ratio is fair).
- The published T4 row (train 1.7859 / val 1.9177) is **not** a comparator —
  ~5.3× more tokens and GPU hardware; listed as owner reference only.
- Generation ms/token: a batch driver that reloaded checkpoints in one Python
  process inflated timings (32–51 ms/token); clean per-process CLI
  measurements are reported here and in `generation_samples.md`.
- Probe: a 1,200-step / ~31 s `tiny_1m` probe (9,832 tok/s) preceded the full
  run; it projected ~5.5 min/3-epoch — far below the 25-min gate, so the full
  3-epoch budget was kept for **both** models.
- Checkpoints stay in `/tmp/phaseb/run_{tiny,tiny_1m}/` (weight artifacts are
  not committed; see `benchmarks/` convention — JSON/MD only).