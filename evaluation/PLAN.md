# evaluation
Scope: reproducible checkpoint evaluation harness, then (long-term)
LM-Eval-style benchmarks, long-context benchmarks, perplexity.

## Implemented (reliability cycle, 2026-09)
- `harness.py` — reproducible evaluation of a `talos-training-checkpoint-v1`
  artifact + sidecar `tokenizer.json` against a JSONL split:
  validation loss (natural-log mean cross-entropy), perplexity
  (`exp(loss)`), next-token argmax accuracy, parameter count (fail-loudly
  check vs. the checkpoint's recorded `n_params`), tokens processed, eval
  throughput, peak RSS (`ru_maxrss`, process metric). Deterministic
  (fixed seed, file-order batches); running twice yields identical metrics.
  CLI: `python -m scripts.eval_checkpoint --checkpoint <dir-or-file>`.

## Not implemented (future work)
- LM-Eval-style benchmark tasks (knowledge/reasoning suites).
- Long-context benchmarks for the 128K-target architecture.
- Perplexity on held-out corpora beyond the JSONL-split harness above.