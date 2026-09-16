# Talos

**Talos** is an open-source, research-grade codebase for building and training
decoder-only transformer LLMs. It is developed prototype-first: one
architecture (`TalosGPT`), one code path, from a 254,272-parameter dev model
toward a long-term **~400B-total / ~30B-active MoE** research target with a
**128K-token context window**.

**Status in one paragraph.** Talos is currently a *proof-of-concept research
prototype* at the `tiny` scale (254,272 parameters). The full loop — model,
tokenizer, data pipeline, training, checkpointing, evaluation, and KV-cache
generation — is real, tested, and reproducible. The trained prototype is
**not** a capable general-purpose LLM: at this scale it produces incoherent
text (see the honest examples below). The large-model ambitions in this README
are **targets**, not demonstrated results; nothing larger than 254K parameters
has been trained.

---

## Demonstrated — what actually runs and is tested

Everything in this section is backed by committed artifacts (`benchmarks/`) and
a green regression suite (**255 passed / 2 skipped**):

- **Tiny dense decoder-only model** — exactly **254,272 parameters**, vocab
  1024, hidden 64, 2 layers, 4 attention heads, 2 KV heads (GQA), RoPE, dense
  FFN, max sequence length 512. Architectural building blocks (RMSNorm, SwiGLU,
  RoPE/YaRN, attention backends, MoE path, KV cache) are fully implemented,
  typed, and unit-tested — no `pass` stubs.
- **Byte-level BPE tokenizer with a memory-efficient streaming trainer**
  (`tokenizer/`, PR #16) — the ~1,800-doc / 764-merge workload that previously
  crashed memory-constrained VMs now trains at **peak RSS 62.4 MiB**, with
  knobs `max_docs` / `max_chars` / `num_merges` / `minfreq`, atomic saves, and
  a byte-compatible tokenizer file format. Round-trips English and tiny-vocab
  text (regression-tested).
- **OASST1-style JSONL training pipeline** (`scripts/train_oasst1.py`, PR #17)
  — deterministic fixed-seed train/val split (disjoint files), train-split-only
  tokenizer training, a hard fail-fast guard that the model is exactly 254,272
  params, per-epoch train + validation loss, and `talos-training-checkpoint-v1`
  checkpoints with sidecar `tokenizer.json` + `metrics.json`.
- **Checkpoint-reproducible evaluation** (`evaluation/harness.py`,
  `scripts/eval_checkpoint.py`, PR #18) — validation/train loss, perplexity,
  next-token accuracy, parameter count, tokens processed, throughput, and peak
  RSS, reproducible from any checkpoint; the recomputed validation loss matches
  the checkpoint's recorded value **bit-for-bit**, and a re-run is bit-identical.
- **Consistent generation CLI** (`scripts/generate.py`, PR #19) — loads a
  checkpoint, verifies tokenizer/model/vocab consistency *before* generating,
  then continues a prompt through the KV-cache decode path (prefill logits were
  verified exactly equal to KV-cache decode logits in the earlier validation
  cycle). Greedy decode is deterministic; sampling requires an explicit seed.
- **KV-cache inference** — prefill + incremental decode with correct GQA shapes;
  ~1.1 ms/token greedy decode on CPU at this scale.
- **int8 quantization experiment** (`quantization/`, PR #12) — symmetric
  int8/bf16 weight-only: **~99.97–99.99% logit similarity** to fp32, greedy
  generations identical, **3.77–3.98× smaller**. FP8/4-bit production work is
  deferred.
- **Performance baseline** (`tools/benchmark.py` → `benchmarks/`) — train
  ~27.9K tok/s, prefill ~1.75 ms, decode ~0.99 ms/token on the recorded CPU;
  determinism-tested.

### Quickstart

```bash
# 1) synthetic OASST1-style corpus (no downloads; a real OASST1 JSONL also works)
python -m tools.make_synthetic_oasst1 --docs 400 --seed 0 --output /tmp/oasst1.jsonl

# 2) train: deterministic split -> BPE (train split only) -> 3 epochs -> checkpoints
python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir /tmp/talos_run \
    --epochs 3 --seq 64 --batch 4 --seed 0

# 3) evaluate: loss / perplexity / accuracy / throughput / RSS from the checkpoint
python -m scripts.eval_checkpoint --checkpoint /tmp/talos_run --seed 0

# 4) generate (expect incoherent babble at this scale — see "Honest expectations")
python -m scripts.generate --checkpoint /tmp/talos_run \
    --prompt "The capital of France is" --max-new-tokens 32

# run the whole suite (torch + pytest only)
python -m pytest -q
```

### Honest expectations for the tiny prototype

A 254K-parameter model trained a few epochs on a synthetic corpus is a
pipeline-validation vehicle, not a language model. Greedy outputs from the
verification run (PR #19) demonstrate this plainly:

```
"The capital of France is"  -> " beproducibility: the key idea i"
"Once upon a time"          -> "n prate three main steps: the vo"
"How do I bake a cake?"     -> " Please be concrete.\n### Assista"
```

These are expected, and they are the point: the reliable train → eval →
generate loop is verified end-to-end; model quality is a different, later
problem.

## Repository layout

```
model/          core model (RMSNorm, SwiGLU, RoPE/YaRN, attention backends,
                MoE, decoder layer, TalosGPT, KV cache, masking)
configs/        ModelConfig + presets (tiny … 400B targets) + compute estimates
tokenizer/      byte-level BPE tokenizer + streaming trainer
data/           memory-bounded streaming data pipeline
training/       streaming/synthetic training data components + plan
scripts/        developer CLIs: train_oasst1, eval_checkpoint, generate, cli
evaluation/     checkpoint-reproducible eval harness (loss/ppl/acc/throughput/RSS)
inference/      prefill + KV-cache decode path (generate.py)
quantization/   int8/bf16 weight-only experiment (prototype verdicts)
experiments/    DDM KV-tier experiment (prototype-scale verdict, see below)
distributed/    (Phase 2 design) DP/TP/PP/EP, checkpoints, fault tolerance
tools/          benchmarks + experiment drivers + synthetic-data generators
docs/           architecture, tokenizer, scaling documentation
benchmarks/     committed measurement artifacts (baseline / quant / ddm / report)
tests/          unit + integration tests (255 passed / 2 skipped)
examples/       runnable end-to-end examples (tiny_train loop, inference)
```

## Current experiments

Prototype-scale work whose verdicts are measured and committed:

- **The 254K prototype** — verified run: train loss **1.0431 → 0.2521**, val
  loss **0.3447 → 0.2473** over 3 epochs (PR #17), val perplexity **1.2805**,
  accuracy **0.918** (PR #18), deterministic under seed. Artifacts under
  `benchmarks/`; consolidated report generated by `tools/make_report.py`
  (`benchmarks/report-tiny.md`).
- **Quantization** — int8 is viable at prototype scale (logit-similarity and
  greedy-identical results above); full FP8 / 4-bit production suites deferred.
- **DDM KV-tiering** (PR #13) — pre-registered **negative** result: the tiered
  KV mechanism is correct but offers **no acceptable benefit at this scale**
  (87.7% persistent-KV RAM cut, but throughput 0.8152× resident fails the
  ≥0.90 gate; quality bit-exact). Per the experiment's own rules this line
  stops at prototype scale rather than being retried on bigger models.

## Future targets — not yet demonstrated

These are research targets. **No results exist for any of them**; nothing larger
than 254K parameters has been trained.

- **Parameter progression 254K → ~1M → ~10M → ~100M** — dense scaling runs,
  each with the same measured metrics (loss, perplexity, accuracy, throughput,
  memory, convergence). Only the 254K row has real numbers; the other rows are
  honest placeholders with estimates. See **`docs/SCALING.md`**.
- **~400B-total / ~30B-active MoE, 128K context** — the long-term architecture
  target: top-k MoE routing + load-balancing auxiliary loss, shared/grouped
  experts, GQA + RoPE with YaRN long-context scaling, hybrid
  sliding-window/periodic-full attention, and DP/TP/PP/EP distributed
  infrastructure. Implemented and unit-tested at the architecture level
  (`model/`, `distributed/`, `configs/` — see `docs/architecture.md`), but
  **not trained, not benchmarked, and explicitly deferred** until the
  progression above produces measured evidence and the owner green-lights it.
- The same code path builds every size — but *having a config is not having a
  model*.

## Dependencies

Core: **torch**, **numpy**. Optional: **flash-attn** (optimized attention
kernel), **transformers** (later HF export). Dev: **pytest**.

## Design notes

Configuration is a `ModelConfig` dataclass (see `configs/presets.py`). The
compute helper (`configs.compute`) reports total/active parameters, BF16
weights memory, and per-token FLOPs for every config — estimates, not results.
Hardware-specific optimizations (FlashAttention, FP8, NCCL) are abstracted
behind interfaces with functional CPU/generic fallbacks. See
`docs/architecture.md` for the full design rationale and `docs/SCALING.md` for
the measured-vs-targeted status of every scale.

## License

Apache-2.0