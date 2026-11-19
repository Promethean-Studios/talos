# Talos Notebooks

This directory holds the ready-to-run training notebook and its testable
helpers.

## `talos_colab_train.ipynb` — Talos 1B, 20B-token Chinchilla run

Trains the project's primary 1B model (`talos_1b`: **1,036,060,672 params**,
vocab **49,152**, context 2,048, 16Q/4KV GQA, head_dim 128, RMSNorm pre-norm,
RoPE, no biases, tied embeddings) for a **20,000,000,000-token** Chinchilla
budget on a **Colab GPU** — the only supported GPU path (T4: fp16 +
GradScaler; L4/A100: bf16).

- **Tokenizer:** `assets/tokenizers/talos_1b_49k.json`, sha256 recorded in every
  checkpoint and metrics row.
- **Data:** `HuggingFaceH4/ultrachat_200k`, streamed, linearized to
  `User: … / Assistant: …` documents and packed into seq-2048 rows. Epochs loop
  until the token budget is met. This is the chat / continued-pretraining path;
  the 220M-token FineWeb-Edu packed corpus is the separate pretraining corpus.
- **Loop:** bf16 autocast, gradient checkpointing, gradient accumulation,
  8-bit AdamW (bitsandbytes; plain AdamW fallback), bf16 (no GradScaler).
- **Display:** one live HTML table updated every `LOG_EVERY` steps — token
  progress out of 20B, step, tok/s, elapsed, ETA, train/val loss, checkpoint
  filename, GPU, model, parameters, vocab, sequence length, epoch, UTC timestamp.
- **Metrics sync:** every log interval appends a row to `OUT_DIR/metrics.jsonl`
  and pushes it (throttled) to a HF **dataset** repo,
  default `<whoami>/talos-1b-training-metrics`. A failed push is logged and
  swallowed — it never stops training.
- **Checkpoints:** `step-<N>.pt` every `CHECKPOINT_EVERY` steps in the repo's
  `talos-training-checkpoint-v1` schema (weights + optimizer + RNG + losses +
  tokenizer sha256 + token counters), so `scripts/eval_checkpoint.py` reads them
  and `RESUME = True` continues bit-exactly.

### Running it

1. Open the notebook in Colab on a T4 (fp16 + GradScaler) or L4/A100 (bf16)
   runtime.
2. Cell 1 installs the Python extras; torch is left alone (a `pip install torch`
   would replace the CUDA build with a CPU wheel).
3. Set `HF_TOKEN` (or run `huggingface-cli login`) so the metrics dataset repo
   can be created and pushed. With no token the rows still land in
   `OUT_DIR/metrics.jsonl`; set `SYNC_DRY_RUN = True` to skip the network.
4. Run the cells top to bottom. Cell 8 is the training loop; interrupt it
   whenever — it writes a checkpoint and flushes metrics before stopping.

A 20B-token budget on a Colab GPU is a long run: the live table reports measured
tok/s and a real ETA, and the run can be stopped and resumed at any checkpoint.

## `talos_training_lib.py`

The notebook's pure-Python helpers — metrics row construction, ETA math, the
HTML display table, the chat→packed-token adapter and the HF Dataset sync.
No torch, no GPU, no network at import time, so it is unit-tested on CI by
`tests/test_training_lib.py` (which also validates the notebook's JSON, its
code cells, and the config/display fields it must carry).
