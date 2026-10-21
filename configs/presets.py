"""Pre-canned model configs for every scale, sharing one code path.

These are the six configs that drive development, training and deployment.
They differ only in *numbers*, never in architecture. All are
:class:`~model.config.ModelConfig` dataclasses; the compute helper
(``configs.compute``) turns them into parameter/FLOP estimates.

For long-context scaling the configs opt into the **hybrid** attention scheme
(sliding window + periodic full attention) and YaRN RoPE scaling. The ``tiny``
config is the dev-sized model that runs on one consumer GPU.
"""
from __future__ import annotations

from configs.vocab import VOCAB_SIZE
from model.config import ModelConfig
from tokenizer.vocab import TokenizerConfig


def tiny_config() -> ModelConfig:
    """Tiny dev model — fits on a single consumer GPU, CPU-instantiable."""
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_type="dense",
        intermediate_size=256,
        max_seq_len=512,
        attention_type="full",
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
    )


def tiny_tokenizer_config() -> TokenizerConfig:
    """Tokenizer sized to the tiny prototype model.

    The model↔tokenizer contract (``tokenizer/model_compat.py``) requires
    ``tokenizer_vocab_size <= model_vocab_size``: the tokenizer's ids map 1:1
    onto embedding rows and the model keeps the rest as padding. The tiny model
    holds **1024** rows, while ``TokenizerConfig`` defaults to 32768 — a
    tokenizer trained with the default overflows the tiny model (ids ≥ 1024
    cannot be embedded; specials would sit at 32764+).

    This config fixes ``vocab_size=VOCAB_SIZE`` (1024): 256 base byte tokens
    + 4 special tokens (ids 1020..1023, LLaMA-style at the top) + **764 merge
    slots** (``VOCAB_SIZE - 256 - 4``). Training learns at most 764 merges, so
    every token id stays inside the model. Byte-level encoding needs no merges
    to be lossless, so English round-trips exactly even before/at any merge
    budget.
    """
    return TokenizerConfig(vocab_size=VOCAB_SIZE)


def tiny_1m_config() -> ModelConfig:
    """The ~1M scaling step: ``tiny`` scaled up on the exact same architecture.

    Same architectural ratios as ``tiny`` — 2:1 GQA (8/4 heads), head_dim 16,
    dense FFN at 4× hidden (512/128), full attention, un-tied embeddings, no
    biases — with the model size ~3.93× tiny's. ``vocab_size`` and
    ``max_seq_len`` are **unchanged** (1024 / 512) so the existing vocab-1024
    tokenizer (``tiny_tokenizer_config``) is used unchanged and the tokenizer↔
    model compat contract is untouched. The exact parameter count, verified
    programmatically, is **1,000,320** (see docs/SCALING.md §3.1 for the full
    arithmetic).

    Depth is 3 layers rather than the 4 suggested during design: keeping the
    tiny FFN ratio at exactly 4× hidden, 4 layers would land at 1,246,336
    params (~25% over the 1.1M ceiling), while 3 layers at the same ratios
    lands at 1,000,320 — squarely in the 0.9–1.1M target. This preserves every
    ratio of the 254K model instead of re-tuning the FFN width, so the ~4×
    parameter scaling comes purely from hidden 64→128 and layers 2→3.
    """
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=128,
        num_layers=3,
        num_attention_heads=8,
        num_kv_heads=4,
        head_dim=16,
        ffn_type="dense",
        intermediate_size=512,
        max_seq_len=512,
        attention_type="full",
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
    )


def tiny_1m_tokenizer_config() -> TokenizerConfig:
    """Tokenizer for ``tiny_1m``: identical to ``tiny``'s.

    ``tiny_1m`` keeps ``vocab_size=1024`` unchanged (see
    :func:`tiny_1m_config`), so the existing vocab-1024 tokenizer contract is
    reused verbatim — no new tokenizer, no retrained merges, same byte-level
    BPE budget (256 base bytes + 4 specials + up to 764 merges).
    """
    return tiny_tokenizer_config()


def tiny_10m_config() -> ModelConfig:
    """The ~10M scaling step: ``tiny_1m`` widened on the exact same architecture.

    Same architectural ratios as ``tiny``/``tiny_1m`` — 2:1 GQA (28/14 heads),
    head_dim 16, dense FFN at 4× hidden (1792/448), full attention, un-tied
    embeddings, no biases — scaled **purely in width**: same 3 layers as
    ``tiny_1m``, hidden 128→448 (3.5×). ``vocab_size`` and ``max_seq_len`` are
    **unchanged** (1024 / 512) so the existing vocab-1024 tokenizer
    (``tiny_tokenizer_config``) is used unchanged and the tokenizer↔model
    compat contract is untouched. The exact parameter count, verified
    programmatically, is **9,952,320** (see docs/SCALING.md §3 for the full
    arithmetic).

    Design note: width was chosen over depth so the ~10× parameter jump from
    ``tiny_1m`` (9,952,320 / 1,000,320 ≈ 9.95) is attributable purely to
    ``hidden 128→448`` (and the head/FFN counts that scale with it) — the
    cleanest apples-to-apples comparison, mirroring how ``tiny_1m`` scaled
    from ``tiny``. Keeping every ratio (2:1 GQA, head_dim 16, FFN 4× hidden)
    means the same code path builds all three presets and only the numbers
    change.
    """
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=448,
        num_layers=3,
        num_attention_heads=28,
        num_kv_heads=14,
        head_dim=16,
        ffn_type="dense",
        intermediate_size=1792,
        max_seq_len=512,
        attention_type="full",
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
    )


def tiny_10m_tokenizer_config() -> TokenizerConfig:
    """Tokenizer for ``tiny_10m``: identical to ``tiny``'s.

    ``tiny_10m`` keeps ``vocab_size=VOCAB_SIZE`` (1024) unchanged (see
    :func:`tiny_10m_config`), so the existing vocab-1024 tokenizer contract is
    reused verbatim — no new tokenizer, no retrained merges, same byte-level
    BPE budget (256 base bytes + 4 specials + up to 764 merges).
    """
    return tiny_tokenizer_config()


def tiny_100m_config() -> ModelConfig:
    """The ~100M scaling step: ``tiny_10m`` scaled to the audit's §13 candidate.

    Same architectural ratios as ``tiny``/``tiny_1m``/``tiny_10m`` — 2:1 GQA
    (64/32 heads), head_dim 16, dense FFN at 4× hidden (4096/1024), full
    attention, un-tied embeddings, no biases — scaled to **hidden 1024 × 6
    layers**. ``vocab_size=VOCAB_SIZE`` (1024) and ``max_seq_len`` 512 are
    **unchanged**, so the existing vocab-1024 tokenizer
    (``tiny_tokenizer_config``) is reused verbatim and the tokenizer↔model
    compat contract is untouched. The exact parameter count, verified
    programmatically, is **96,482,304** (arithmetic below).

    Shape rationale (audit §13): of the two candidate shapes, hidden-1024 × 6
    layers (96,482,304) was chosen over hidden-896 × 8 layers (98,188,160)
    because (a) 1024 is a power of two — the cleanest hardware alignment and
    the audit's headline candidate, whose full fp32 T4 memory model (weights +
    grads + AdamW + activations at batch 32 × seq 64 **and** batch 32 × seq
    512) was already computed in §13; (b) parameter arithmetic stays
    round-numbered: per layer, attention = 3H² = 3,145,728, dense SwiGLU FFN =
    12H² = 12,582,912, norms = 2H; total = ``2·V·H + L·(15H² + 2H) + H`` =
    2·1024·1024 + 6·15,730,688 + 1024 = **96,482,304**, squarely inside the
    owner's 95–105M target with an exact, registry-pinned count (the
    ``tiny_1m``/``tiny_10m`` precedent). Depth doubles from 3 to 6 with the
    width jump (448 → 1024); every ratio of the family is preserved.
    """
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=1024,
        num_layers=6,
        num_attention_heads=64,
        num_kv_heads=32,
        head_dim=16,
        ffn_type="dense",
        intermediate_size=4096,
        max_seq_len=512,
        attention_type="full",
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
    )


def tiny_100m_tokenizer_config() -> TokenizerConfig:
    """Tokenizer for ``tiny_100m``: identical to ``tiny``'s (vocab 1024).

    ``tiny_100m`` keeps ``vocab_size=VOCAB_SIZE`` (1024) unchanged (see
    :func:`tiny_100m_config`), so the existing vocab-1024 tokenizer contract is
    reused verbatim — no new tokenizer, no retrained merges, same byte-level
    BPE budget (256 base bytes + 4 specials + up to 764 merges). The model's
    embedding rows stay exactly 1024, so every token id the tokenizer can emit
    is embeddable with zero padding.
    """
    return tiny_tokenizer_config()


# ------------------------------------------------------------------------------
# Talos 1B presets (owner-directed, 2026-10): a NEW architecture track.
# ------------------------------------------------------------------------------
# The 1B presets deliberately diverge from the tiny family on THREE knobs, each
# an explicit owner decision: vocab_size 32,000 (the real tokenizer track), a
# 4:1 GQA ratio (16/4 heads instead of 2:1), head_dim 128 instead of 16, and a
# SwiGLU FFN at 2.6875x hidden (5504/2048 instead of 4x). The per-layer shape
# is otherwise the classic small-LLM recipe: pre-norm RMSNorm, RoPE
# (theta 1e4), dense SwiGLU, no biases, TIED input/output embeddings.

#: The exact per-layer parameter arithmetic shared by both 1B presets
#: (see the docstrings below for the full derivation).
TALOS_1B_LAYER_PARAMS = 44_306_432  # attention + norms + SwiGLU FFN
TALOS_1B_EMBEDDING_PARAMS = 32_000 * 2048  # tied: one 32000x2048 matrix
TALOS_1B_FINAL_NORM_PARAMS = 2048


def talos_1b_config() -> ModelConfig:
    """The Talos ~1B preset — the owner's spec with depth adjusted to 21 layers.

    Owner's target architecture (verbatim): decoder-only, vocab 32,000, seq
    2,048, hidden 2,048, **16 heads / 4 KV heads (4:1 GQA), head_dim 128**,
    SwiGLU FFN intermediate **5,504**, RMSNorm pre-norm, RoPE, no biases,
    TIED input/output embeddings, standard stable init (``initializer_range``
    0.02, ``rope_theta`` 1e4).

    The owner pre-authorized adjusting dims to approach 1B; **depth is the
    chosen adjustment axis** (``num_layers=21``) so the per-layer shape stays
    exactly as specified. Parameter count, verified programmatically (see
    ``tests/test_talos_1b.py`` and the meta-device construction in
    docs/SCALING.md §7):

    * embedding (tied, single copy) = ``V·H`` = 32,000·2,048 = **65,536,000**
      (tied ⇒ no separate lm_head);
    * per layer: attention = ``H·H + H·(H/4) + H·(H/4) + H·H`` =
      2,048² + 2·2,048·512 + 2,048² = 4,194,304 + 2·1,048,576 + 4,194,304
      = **10,485,760**; 2 RMSNorms = 2·2,048 = **4,096**; SwiGLU FFN =
      3·H·I = 3·2,048·5,504 = **33,816,576**; per layer total
      = **44,306,432**;
    * 21 layers = 930,435,072; final norm = **2,048**;
    * **total = 65,536,000 + 930,435,072 + 2,048 = 995,973,120** (≈996M —
      the closest the owner's per-layer shape gets to 1B below the mark).

    ``vocab_size=32_000`` ≠ the canonical 1024 constant deliberately — see
    :func:`talos_1b_tokenizer_config` for the tokenizer pairing guard.
    """
    return ModelConfig(
        vocab_size=32_000,
        hidden_size=2048,
        num_layers=21,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        ffn_type="dense",
        intermediate_size=5504,
        max_seq_len=2048,
        attention_type="full",
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )


def talos_1b_spec16_config() -> ModelConfig:
    """The literal 16-layer variant of the owner's 1B spec (no dim adjustments).

    Same architecture and per-layer shape as :func:`talos_1b_config`
    (vocab 32,000, hidden 2,048, 16/4 GQA heads, head_dim 128, SwiGLU
    intermediate 5,504, no biases, tied embeddings, RoPE theta 1e4) with the
    spec's **exact 16 layers**. Parameter count, verified programmatically:

    * embedding (tied) = **65,536,000**;
    * 16 layers × 44,306,432 = **708,902,912**;
    * final norm = **2,048**;
    * **total = 65,536,000 + 708,902,912 + 2,048 = 774,440,960** (≈774M).

    The spec's literal shape lands ~23% under 1B; ``talos_1b`` (21 layers,
    995,973,120) is the preset that actually approaches the 1B target with the
    owner's per-layer shape untouched. This variant exists so a subsequent
    16-layer reference run is one config away and the arithmetic is pinned by
    tests either way.
    """
    return ModelConfig(
        vocab_size=32_000,
        hidden_size=2048,
        num_layers=16,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        ffn_type="dense",
        intermediate_size=5504,
        max_seq_len=2048,
        attention_type="full",
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )


def _assert_tokenizer_model_pairing(tok_config: TokenizerConfig, model: ModelConfig) -> None:
    """Enforce the *intended* tokenizer↔model pairing, not just the ≤ bound.

    ``tokenizer/model_compat.py`` enforces ``tokenizer_vocab_size <=
    model_vocab_size`` generically, but that alone would let a stray vocab-1024
    tokenizer config pair silently with a 1B model (1024 ≤ 32000 passes). Each
    1B preset is defined together with its 32,000-vocab tokenizer config; this
    guard makes the pairing **exact and loud**: the tokenizer config and its
    paired model config must agree on the vocab size, or constructing the
    tokenizer config raises. (The model keeps the standard ``<=`` contract for
    the *trained asset*, which may use fewer merge slots.)
    """
    if tok_config.vocab_size != model.vocab_size:
        raise ValueError(
            f"tokenizer config vocab_size {tok_config.vocab_size} does not match "
            f"its paired model config vocab_size {model.vocab_size} — a 1B preset "
            "must never run with a mismatched (e.g. the canonical 1024) tokenizer "
            "config; fix the pairing in configs/presets.py deliberately"
        )


def talos_1b_tokenizer_config() -> TokenizerConfig:
    """Tokenizer config for ``talos_1b``: vocab 32,000, paired and guarded.

    vocab 32,000 = 256 base bytes + 4 default specials + **31,740 merge slots**
    (LLaMA-style specials at the top). This is a **deliberate divergence** from
    the canonical 1024 contract (``configs.vocab.VOCAB_SIZE``) — the 1B track
    uses its own 32,000-vocab tokenizer asset (a later delegation trains it);
    :func:`_assert_tokenizer_model_pairing` refuses to construct this config
    beside a model config with any other vocab, so a silent reuse of the
    canonical 1024 tokenizer cannot happen at the config seam.
    """
    tok = TokenizerConfig(vocab_size=32_000)
    _assert_tokenizer_model_pairing(tok, talos_1b_config())
    return tok


def talos_1b_spec16_tokenizer_config() -> TokenizerConfig:
    """Tokenizer config for ``talos_1b_spec16``: identical (vocab 32,000).

    ``talos_1b_spec16`` shares ``talos_1b``'s vocabulary exactly (the two
    presets differ only in depth), so this reuses the same guarded 32,000-vocab
    config via :func:`_assert_tokenizer_model_pairing`.
    """
    tok = TokenizerConfig(vocab_size=32_000)
    _assert_tokenizer_model_pairing(tok, talos_1b_spec16_config())
    return tok


#: canonical preset name -> tokenizer-config builder. All four tiny-family
#: presets reuse the sole vocab-1024 contract (their rows stay 1024); the two
#: 1B presets deliberately register a 32,000-vocab contract. The registry
#: exists so a future preset with a different vocab is a one-line deliberate
#: change, not a silent re-wire — and ``tests/test_vocab_seam.py`` pins the
#: exact membership.
TOKENIZER_CONFIGS = {
    "tiny": tiny_tokenizer_config,
    "tiny_1m": tiny_1m_tokenizer_config,
    "tiny_10m": tiny_10m_tokenizer_config,
    "tiny_100m": tiny_100m_tokenizer_config,
    "talos_1b": talos_1b_tokenizer_config,
    "talos_1b_spec16": talos_1b_spec16_tokenizer_config,
}


def preset_tokenizer_config(preset: str) -> TokenizerConfig:
    """Tokenizer config for a *canonical* preset, by name (registry-driven).

    Raises ``ValueError`` for unknown presets so a typo or a not-yet-registered
    preset fails loudly at the training-script seam instead of silently using a
    default budget.
    """
    try:
        return TOKENIZER_CONFIGS[preset]()
    except KeyError:
        raise ValueError(
            f"no tokenizer config registered for preset {preset!r} "
            f"(registered: {', '.join(sorted(TOKENIZER_CONFIGS))})"
        ) from None


def small_config() -> ModelConfig:
    """Small research model (single-GPU / small-cluster fit)."""
    return ModelConfig(
        vocab_size=4096,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        num_kv_heads=4,
        head_dim=64,
        ffn_type="dense",
        intermediate_size=2048,
        max_seq_len=4096,
        attention_type="full",
        layer_norm_eps=1e-5,
    )


def medium_config() -> ModelConfig:
    """Medium model — multi-GPU training target for ablations."""
    return ModelConfig(
        vocab_size=16384,
        hidden_size=1024,
        num_layers=16,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=64,
        ffn_type="dense",
        intermediate_size=4096,
        max_seq_len=8192,
        attention_type="full",
        layer_norm_eps=1e-5,
    )


def large_config() -> ModelConfig:
    """Large dense model — long-context (hybrid attention) demonstration."""
    return ModelConfig(
        vocab_size=32768,
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="dense",
        intermediate_size=8192,
        max_seq_len=32768,
        attention_type="hybrid",
        sliding_window_size=8192,
        periodic_full_every=8,  # one full layer per 8
        attention_chunk_size=1024,
        rope_theta=500000.0,
        layer_norm_eps=1e-5,
    )


def scale_100b_config() -> ModelConfig:
    """~100B-total-parameter MoE (mid-scale training / inference target)."""
    return ModelConfig(
        vocab_size=65536,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="moe",
        num_experts=128,
        num_experts_per_tok=8,
        num_shared_experts=2,
        moe_intermediate_size=2048,
        grouped_experts=None,
        load_balance_coef=0.01,
        router_jitter_noise=0.0,
        max_seq_len=32768,
        attention_type="hybrid",
        sliding_window_size=8192,
        periodic_full_every=8,
        attention_chunk_size=1024,
        rope_theta=500000.0,
        layer_norm_eps=1e-5,
    )


def scale_400b_config() -> ModelConfig:
    """~400B-total / ~30B-active MoE with 128K context.

    Design (see docs/architecture.md): 46 layers, hidden 6144, GQA with 48
    query / 8 KV heads, 128 routed experts + 2 shared, top-6 routing. Tuning
    notes: the brief's "~256 experts, top-k ~6" and "~400B total / ~30B active"
    are mutually inconsistent — with 256 experts and top-6, active params would
    be only ~2.3% of total (~9B), not 30B. We keep top-k ~6 (as requested) and
    reduce the expert count to 128 so that 6 routed + 2 shared of 130 give the
    ~6% active fraction that yields ~30B active from ~400B total.
    """
    return ModelConfig(
        vocab_size=131072,          # 128K tokens
        hidden_size=6144,
        num_layers=46,
        num_attention_heads=48,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="moe",
        num_experts=128,
        num_experts_per_tok=6,
        num_shared_experts=2,
        moe_intermediate_size=3584,
        grouped_experts=None,
        load_balance_coef=0.01,
        router_jitter_noise=0.0,
        max_seq_len=131072,         # 128K context
        attention_type="hybrid",
        sliding_window_size=16384,  # 16K sliding window
        periodic_full_every=8,      # every 8th layer is full attention
        attention_chunk_size=1024,
        rope_theta=500000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "base": 500000.0,
        },
        layer_norm_eps=1e-5,
    )


ALL_PRESETS = {
    "tiny": tiny_config,
    "tiny_1m": tiny_1m_config,
    "tiny_10m": tiny_10m_config,
    "tiny_100m": tiny_100m_config,
    "talos_1b": talos_1b_config,
    "talos_1b_spec16": talos_1b_spec16_config,
    "small": small_config,
    "medium": medium_config,
    "large": large_config,
    "100b": scale_100b_config,
    "400b": scale_400b_config,
}
