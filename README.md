# Talos

**Talos** is an open-source, research-grade codebase for building and training a
modern decoder-only **mixture-of-experts (MoE)** foundation LLM — roughly
**400B total / ~30B active parameters**, a **128K-token context window**, and
full distributed-training + efficient-inference-ready infrastructure.

This is the **Phase 1 MVP**: a complete, working repository where the *same*
code paths scale from a single-GPU dev model up to the ~400B config. Every
Phase 1 module is real, typed, documented and tested — no `pass` stubs or
pseudocode.

## Highlights

- **One architecture, six sizes.** The same `TalosGPT` model builds the `tiny`
  dev model or the `~400B` MoE; only the numbers change.
- **GQA attention** (`num_kv_heads < num_attention_heads`) with **RoPE**
  (optional **YaRN** long-context scaling).
- **Dense ↔ MoE interchangeable feed-forward** behind one interface: top-k
  routing + load-balancing auxiliary loss, shared/grouped experts.
- **Hybrid long-context attention**: sliding-window layers + periodic full
  attention, running near-linear memory at 128K (see `docs/architecture.md`).
- **Pluggable attention backend**: fully functional `PlainAttentionBackend`
  (correct on CPU/any GPU, with a *chunked* execution mode that bounds peak
  memory) and an optional `FlashAttentionBackend` (used automatically when
  `flash-attn` is installed, otherwise falls back to plain).
- **KV cache** for prefill + decode inference with correct GQA shapes and
  sliding-window support.

## Repository layout

```
model/          core model (RMSNorm, SwiGLU, RoPE/YaRN, attention backends,
                MoE, decoder layer, TalosGPT, KV cache, masking)
configs/        ModelConfig + 6 presets (tiny…400B) + compute estimates
training/       (Phase 2) distributed training
data/           (Phase 3) data pipeline
tokenizer/      (Phase 2) tokenizer training
inference/      (Phase 2) inference server
distributed/    (Phase 2) DP/TP/PP/EP, checkpoints, fault tolerance
evaluation/     (Phase 4) eval harness
scripts/        developer CLIs
tests/          unit + integration tests
docs/           architecture decisions
examples/       runnable end-to-end examples
```

## Quickstart

```bash
# tiny: forward + backward, end to end, on CPU or any GPU
python -m scripts.cli --steps 3 --seq 64

# print parameter / memory / FLOP estimates for all configs
python -m configs.compute --summary
python -m configs.compute 400b

# run the whole test suite (torch + pytest only)
python -m pytest
```

```python
from model import TalosGPT, ModelConfig
from configs.presets import tiny_config

cfg = tiny_config().derive()
model = TalosGPT(cfg)
logits, _ = model(torch.randint(0, 1024, (2, 32)))   # (2, 32, 1024)
```

## Dependencies

Core: **torch**, **numpy**. Optional: **flash-attn** (optimized attention
kernel), **transformers** (later HF export). Dev: **pytest**.

## Design notes

Configuration is a `ModelConfig` dataclass (see `configs/presets.py` for the
six presets). The compute helper (`configs.compute`) reports total/active
parameters, BF16 weights memory, and per-token FLOPs for every config. See
`docs/architecture.md` for the full design rationale and where
hardware-specific optimizations are abstracted with functional fallbacks.

## License

Apache-2.0
