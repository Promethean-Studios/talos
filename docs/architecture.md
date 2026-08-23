# Talos Architecture

This document explains **every major design decision** in the Phase 1 MVP and
honestly notes where a hardware-specific optimization is abstracted behind a
functional fallback.

The guiding principle: **one code path** builds and runs a model from the tiny
dev size to the ~400B MoE. The `ModelConfig` dataclass (`model/config.py`) is
the single source of truth; `model/` instantiates whatever those numbers
describe.

---

## 1. Why MoE instead of dense

A dense 400B-parameter model needs ~2.4 TB+ of weights at BF16 and executes all
of them for every token (enormous per-token FLOPs). A **MoE** model keeps the
same *total* knowledge capacity (many small experts) but only activates a few
experts per token, cutting active compute by an order of magnitude.

- `ffn_type = "moe"` instantiates `MoEFFNBlock`; `"dense"` instantiates
  `DenseFFNBlock`. **Both implement the same `FFNInterface`** (`forward(x) ->
  (output, aux_loss)`), so a config chooses dense or MoE per size with zero code
  change (`model/ffn.py`, `model/moe.py`).
- Routing is **top-k over softmax router logits** with a
  **load-balancing auxiliary loss** (Switch/Mixtral style) that pushes tokens
  uniformly across experts and prevents routing collapse.
- **Shared experts** (`num_shared_experts`) run on *every* token and always
  count as active — a cheap "always-on" dense path.
- **Grouped experts** (`grouped_experts`) partition the router so selection is
  constrained within groups (DeepSeek-V3/Wiki-MoE style); the router emits a
  per-group logit and Talos max-pools over groups.
- The dispatch loop uses `index_add` over `index_select`ed tokens, which keeps
  the MoE **sparse** (each token only passes through its chosen experts) while
  remaining **autograd-compatible** so router and expert gradients both flow.

**Estimate:** the `~400B` config reports 400.7B total / 30.0B active per token
(see §9).

## 2. Mixture-of-experts vs. "800B dense for the same budget"

We compared MoE against trying to fit the same capability in a dense model.
Dense attention/experts scale ~linearly with parameters but *all* parameters
are active; the total FLOPs and per-GPU memory to *train* a hypothetical dense
400B are beyond Phase-1 budgets and require ~2.4 TB of weights versus ~750 GB
for the 400B/30B MoE. MoE gives frontier capability at a fraction of active
compute — the deciding factor.

## 3. GQA attention

**Grouped-query attention** lets all heads *share* a smaller number of KV
heads (`num_kv_heads < num_attention_heads`). This shrinks the KV cache and the
K/V projection weights substantially at long context — critical at 128K — while
preserving most of the quality of full MHA.

- `SelfAttention` projects K/V with `num_kv_heads` heads, applies RoPE, then
  repeats them to the query head count immediately before the attention kernel.
- The **KV cache** stores only `num_kv_heads` rows (`model/cache.py`), so
  memory/bandwidth scale with KV heads, not attention heads.
- GQA validation requires `num_attention_heads % num_kv_heads == 0`.

## 4. RoPE positional embeddings (+ YaRN)

**Rotary positional embeddings** add position via an orthonormal rotation of
each query/key vector pair. This gives a *relative* position prior that
generalizes to unseen sequence lengths better than absolute embeddings.

- Implemented once and exposed on every layer (`model/rotary.py`): precompute a
  `cos`/`sin` cache per pair-of-channels; rotate in `apply_rotary_pos_emb`.
- **YaRN** (`rope_scaling = {"type": "yarn", ...}`) is available and **off by
  default** so short contexts behave identically to plain RoPE. When enabled it
  rescales the base frequencies in a frequency-dependent way to extend a model
  trained at one length to a much longer context with minimal degradation.
- Decode caching rotates K/V *before* caching, so the cache holds already-rotated
  K/V and per-step gather uses the precomputed cache.

## 5. SwiGLU feed-forward

The feed-forward is a **gated** MLP: `down(SiLU(gate(x)) * up(x))`. The gate
gives the network an adaptive non-linear filter and is the modern standard for
LLMs. `model/activations.py` provides a `SwiGLU` block used by both the dense
FFN and every MoE expert.

## 6. RMSNorm (pre-norm)

Every transformer block uses **pre-norm RMSNorm** (`model/rms_norm.py`):
normalize `x` by its root-mean-square and scale by a learned gain, then apply
the residual connection around it. RMSNorm has no mean-centering or bias,
which makes it cheap and stable at scale. The same module is used for the final
head norm.

## 7. Long context: hybrid sliding-window + periodic full attention

Full quadratic attention at 128K would materialize a `131072 x 131072` attention
matrix — infeasible. Talos uses a **hybrid** scheme (`attention_type="hybrid"`):

- Most layers are **sliding-window attention** (`sliding_window_size`, e.g.
  16K): each query attends only the last `window` keys. Cost/memory drop from
  `O(L²)` to `O(L·w)`.
- A sparse subset (**periodic** `periodic_full_every`, or an explicit
  `full_attention_layers` list) are **full attention**, giving every token a
  global look at some layers and preserving long-range information that pure
  sliding windows would lose.
- `full_layer_indices()` computes which layers are full; each layer's
  `window_size` derives from it, so the model automatically alternates.

### Near-linear memory on the functional fallback

The plain backend runs **chunked attention** when `attention_chunk_size > 0`
and shorter than the sequence: queries are processed in blocks, and each block
only materializes `O(chunk × keys)` logits. Sliding layers touch `O(chunk ×
(chunk + window))` and full layers `O(chunk × seq)`. **No full `O(L²)` matrix is
ever materialized while the hybrid scheme + chunking are on**, which is what
lets the functional backend run a 16K sequence here and scale toward 128K on
CPU. The deployment configs additionally rely on FlashAttention for the sparse
full layers on GPU (see §10).

Masking (`model/masking.py`) supports both dense additive masks (for
short sequences and tests) and per-row key ranges (for the chunked path).

## 8. Vocab & context extension

- Vocab is configurable (`vocab_size`), with **~128K tokens** for the largest
  config (`131072`, powers-of-two friendly per the project's 125K→128K target).
- Context extension to 128K combines the hybrid attention (bounded memory) with
  **YaRN** RoPE scaling and a **large RoPE base** (`rope_theta=500000`).
- The KV cache is pre-sized to `max_seq_len` so decode never reallocates.

## 9. The ~400B / ~30B config (and honest tuning note)

The `scale_400b_config`:
```
hidden=6144, layers=46, heads=48 (head_dim=128), kv_heads=8, vocab=131072,
MoE: 128 routed experts, top-k=6, 2 shared, per-expert intermediate=3584,
hybrid attention (window 16K, full every 8), YaRN, 128K context.
```
The compute helper reports `400.7B total / 29.97B active` (7.48% active).

**Tuning note / deviation.** The brief listed "~256 experts, top-k ~6" *and*
"~400B total / ~30B active". These two are mathematically inconsistent: with
256 experts and top-6 routing, active params would be ~2.3% of total (~9–13B),
not 30B. We kept top-k ≈ 6 and **reduced the expert count to 128** so that
(6 routed + 2 shared) / (128 + 2) ≈ 6.2% of the FFN is active — the fraction
that yields ~30B active from ~400B total. `moe_intermediate_size=3584` per
expert gives a total MoE width of `128 × 3584 = 458,752 ≫` a dense FFN's
`4 × 6144 = 24,576` ("MoE FFN dimension larger than dense" under the total-
capacity reading). The helper *reports the honest numbers* for whatever it is
given; the presets are chosen so the reported numbers hit the targets.

## 10. Attention abstraction & where hardware-specific code is abstracted

`AttentionInterface` (`model/attention.py`) is the clean contract every backend
implements. `build_attention_backend("auto")` selects:

- **`PlainAttentionBackend`** — fully functional, differentiable, runs on CPU
  and any GPU; supports direct and chunked execution.
- **`FlashAttentionBackend`** — uses the `flash-attn` kernel. Its **import is
  guarded** (`FlashAttentionBackend.available()`); if `flash-attn` is not
  installed, `"auto"` transparently falls back to the plain path. **`flash-attn`
  is NOT a dependency** of `pyproject.toml` — it is an optional extra.

**Honest limitation.** FlashAttention (GPU) is the hardware-specific acceleration
for the sparse full-attention layers at 128K; FP8 matmul is a planned
distributed-training optimization (Phase 2). The functional fallback is
correct and demonstrated at long (16K) length, but on *very* long sequences the
plain backend is slower than the kernel — that is exactly what FlashAttention
is for, and the abstraction makes the swap a one-line config choice.

## 11. KV cache for prefill + decode

`KVCache` (`model/cache.py`) is a pre-allocated buffer
`(layers, batch, kv_heads, max_seq, head_dim)`. `update()` writes a new block
and returns the whole cached prefix; `get_window()` returns the last `window`
positions for sliding decode; `reset()` clears between sequences. `TalosGPT`
uses it for prefill (all tokens at once) and for incremental one-token-at-a-time
decode (position-aware RoPE + cache), with the prefill/decode equivalence
verified by tests.

## 12. Dense-vs-MoE interface & determinism

- Dense and MoE FFNs share `FFNInterface` (above); the decoder layer calls it
  uniformly and accumulates the MoE auxiliary loss.
- `model/utils.py` provides a `set_seed()` helper and structured logging; tests
  seed globally for reproducibility. Numerically deterministic where practical
  (RoPE rotation, attention softmax are deterministic on CPU/GPU).

## 13. Tests

`tests/` covers every core module (norm, SwiGLU, RoPE/YaRN, GQA attention,
MoE routing + load-balancing loss, KV cache, sliding/full masking), the
dense-vs-MoE interface, a tiny forward+backward integration, a prefill↔decode
equivalence test, compute-estimate checks, and a **long-context (16K) hybrid
test** using the functional chunked backend. All run on CPU with only
`torch` + `pytest` (`python -m pytest`).
