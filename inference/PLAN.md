# inference (Phase 2)
Scope: OpenAI-compatible server, continuous batching, streaming, KV-cache reuse,
sampling/stopping. The Phase 1 KV cache + prefill/decode path is the foundation.

**Status (2026-11-06):** the core is IMPLEMENTED — `inference/generate.py` is the
canonical prefill/decode path (KV-cache decode is bit-identical to a full
prefill) with greedy and temperature/top-k/top-p/repetition-penalty sampling,
driven by `scripts/generate.py`. **The serving layer (HTTP server, continuous
batching, streaming) is still not implemented.**
