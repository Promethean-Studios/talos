# Talos tiny prototype — performance baselines

Per-item performance baselines for the **271K `tiny` prototype** (254,272
params), captured with the reproducible harness at `tools/benchmark.py` before
quantization / DDM experiments begin. **This is a CPU baseline** — numbers are
specific to the machine and thread count recorded inside the artifact.

## Re-run with one command

```bash
cd /home/team/repos/talos
python tools/benchmark.py          # -> benchmarks/baseline-tiny-cpu.json + .md
```

The harness prints the rendered summary to stdout and writes:

- `baseline-tiny-cpu.json` — machine-readable baseline (schema_version 1):
  model size, train throughput + loss trajectory, prefill/decode latency, peak
  RSS, plus full environment provenance (torch/python versions, CPU model,
  torch thread count, git commit, seed, timestamp).
- `baseline-tiny-cpu.md` — the human-readable rendering of the same data.

Useful flags (see `python tools/benchmark.py --help` for all):

| flag | meaning | default |
|---|---|---|
| `--device` | `auto` (CUDA when available, else CPU), `cpu`, `cuda` | `auto` |
| `--seed` | deterministic seed for init/corpus/prompt | `0` |
| `--train-steps`, `--batch`, `--seq`, `--lr` | train-throughput config | 100, 4, 64, 1e-3 |
| `--prompt-len`, `--decode-tokens`, `--prefill-repeats` | inference config | 64, 64, 20 |
| `--threads` | pin torch CPU threads | torch default |
| `--in-process` | run sections in-process instead of isolated children | off |
| `--out` | JSON output path (`.md` written alongside) | `benchmarks/baseline-tiny-cpu.json` |

## What is measured

1. **Model size** — total / trainable / *active* parameter count (active is
   measured by hooking a real forward pass, so it is generic to MoE where
   active < total; for the dense tiny preset it equals total), plus fp32/bf16
   model size and the fp32 AdamW optimizer-state estimate.
2. **Train throughput** — tokens/sec over a fixed number of AdamW steps on the
   deterministic synthetic recurrent corpus (`training.synthetic`), with the
   aligned causal-LM objective, warmup excluded from timing, and the loss
   trajectory recorded so the baseline is tied to a *learned* model.
3. **Inference** — prefill latency for a fixed prompt length, per-token decode
   latency through the KV cache, and `inference.generate.generate()`
   end-to-end, all via the one canonical prefill/decode path.
4. **Peak RSS** — train and inference run in isolated child processes so each
   section's `resource.ru_maxrss` high-water mark is its own (includes the
   interpreter + torch import).

Determinism: fixed seed → identical loss trajectories and generated tokens
run-to-run on the same machine (asserted by `tests/test_benchmark.py`).
