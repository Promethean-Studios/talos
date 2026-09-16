"""Talos evaluation module.

Contains the reproducible checkpoint evaluation harness
(:mod:`evaluation.harness`, exposed via ``python -m scripts.eval_checkpoint``):
given a ``talos-training-checkpoint-v1`` artifact + its sidecar tokenizer and a
JSONL split, it recomputes validation loss, perplexity, next-token accuracy,
parameter count (guarded against the checkpoint's recorded ``n_params``),
tokens processed, throughput and peak RSS. See PLAN.md for the long-context /
LM-Eval-style scope that remains future work.
"""