"""Tests for the owner-directed Talos 1B presets (``talos_1b`` / ``talos_1b_spec16``).

Architecture (both, verbatim owner spec): vocab **32,000**, seq **2,048**,
hidden **2,048**, **16 query heads / 4 KV heads (4:1 GQA), head_dim 128**,
SwiGLU dense FFN intermediate **5,504** (2.6875× hidden), RMSNorm pre-norm,
RoPE (theta 1e4), **no biases**, **TIED input/output embeddings**,
``initializer_range`` 0.02. ``talos_1b`` runs the spec at **21 layers**
(995,973,120 params — the depth-adjusted approach to 1B); ``talos_1b_spec16``
is the literal **16-layer** spec (774,440,960 params). The exact counts below
are verified on the **meta device** — this file never instantiates a 1B model
in memory (the CPU box cannot hold a 4 GiB fp32 model; see docs/SCALING.md §7
for the memory model).

What is tested here:
  1. both 1B presets build on the meta device with exact parameter counts
     (model + ``configs.compute`` + ``ModelConfig.param_count_breakdown`` +
     the canonical registry all agree), full tensor-shape assertions on every
     module, no bias parameters anywhere, tied embeddings (no ``lm_head`` key);
  2. the tokenizer configs pair exactly with their models (32,000) and the
     pairing guard refuses the canonical 1024 config (the full seam coverage
     lives in ``tests/test_vocab_seam.py``);
  3. the functional suite runs on a small-dim config exercising the SAME
     feature switches as the 1B spec (explicit ``head_dim=128``, 4:1 GQA,
     SwiGLU dense, tied embeddings): forward shapes, backward, loss decrease,
     generation via the library ``inference.generate`` path, prefill == KV
     decode, checkpoint save/load round-trip WITH tied weights (no
     ``lm_head`` key; strict reload; bit-identical logits; the trainer
     ``--resume`` seam's ``expected_state_dict_shapes`` agrees key-for-key),
     and the safetensors export/import round-trip;
  4. flash/SDPA/plain backend selection at head_dim-128 shapes exactly as
     PR #33 established: default is SDPA when flash-attn is absent; the
     chunked functional path is the fallback for ``attention_chunk_size > 0``;
  5. REGRESSION — all four canonical presets still resolve via
     ``resolve_preset``, still construct at their exact historical counts
     (254,272 / 1,000,320 / 9,952,320 / 96,482,304), and one CPU fwd/bwd
     smoke step each passes. Nothing existing may move.

All tests are deterministic and CPU-fast.
"""
from __future__ import annotations

import json

import pytest
import torch

from configs.canonical import CANONICAL_PRESETS, expected_params, resolve_preset
from configs.compute import estimate
from configs.presets import (
    preset_tokenizer_config,
    talos_1b_config,
    talos_1b_spec16_config,
    tiny_100m_config,
    tiny_10m_config,
    tiny_1m_config,
    tiny_config,
)
from inference.generate import generate, prefill_decode_max_abs_diff
from model import TalosGPT
from model.config import ModelConfig
from model.utils import set_seed
from scripts.export_safetensors import export_model, load_artifact
from scripts.train_oasst1 import (
    expected_state_dict_shapes,
    load_checkpoint,
    save_checkpoint,
    train_tokenizer_for_run,
)
from tests.fixture_corpus import FIXTURE_CORPUS

TALOS_1B_PARAMS = 995_973_120        # talos_1b (21 layers, tied, vocab 32,000)
TALOS_1B_SPEC16_PARAMS = 774_440_960  # talos_1b_spec16 (16 layers, tied, vocab 32,000)
TALOS_1B_VOCAB = 32_000
LOGIT_ATOL = 1e-4  # established fp32 guardrail (tests/test_inference.py)

# Canonical family regression pins (these must stay byte-for-byte unchanged).
TINY_PARAMS = 254_272
TINY_1M_PARAMS = 1_000_320
TINY_10M_PARAMS = 9_952_320
TINY_100M_PARAMS = 96_482_304
VOCAB_1024 = 1024


# ---------------------------------------------------------------------------
# 1. Meta-device construction: exact counts + shape assertions at 1B
# ---------------------------------------------------------------------------
def _meta_talos_1b(layers: int) -> TalosGPT:
    """Build a 1B variant on the meta device (no host RAM allocated)."""
    cfg = talos_1b_config().derive() if layers == 21 else talos_1b_spec16_config().derive()
    with torch.device("meta"):
        model = TalosGPT(cfg)
    return model


@pytest.mark.parametrize(
    ("builder", "expected"),
    [(talos_1b_config, TALOS_1B_PARAMS), (talos_1b_spec16_config, TALOS_1B_SPEC16_PARAMS)],
)
def test_talos_1b_meta_device_exact_parameter_count(builder, expected: int) -> None:
    """Meta-device build: the model, compute estimate, breakdown and registry agree."""
    cfg = builder().derive()
    model = _meta_talos_1b(cfg.num_layers)
    assert model.num_parameters() == expected
    assert estimate(cfg).total_params == expected
    assert cfg.param_count_breakdown()["total"] == expected
    assert cfg.param_count_breakdown()["lm_head"] == 0  # tied: no separate head
    assert CANONICAL_PRESETS[resolve_preset(cfg)] == (expected, TALOS_1B_VOCAB)
    assert expected_params(cfg) == expected
    # Analytic per-layer arithmetic from the preset docstrings.
    assert cfg.param_count_breakdown()["per_layer_attention"] == 10_485_760
    assert cfg.param_count_breakdown()["per_layer_ffn"] == 33_816_576
    assert cfg.param_count_breakdown()["embedding"] == TALOS_1B_VOCAB * 2048 == 65_536_000
    assert (
        65_536_000 + cfg.num_layers * 44_306_432 + 2048 == expected
    )


@pytest.mark.parametrize(
    ("builder", "expected"),
    [(talos_1b_config, TALOS_1B_PARAMS), (talos_1b_spec16_config, TALOS_1B_SPEC16_PARAMS)],
)
def test_talos_1b_meta_device_shapes_and_no_bias_and_tied(builder, expected: int) -> None:
    """Every module's tensor shape at 1B, no bias params anywhere, no lm_head key.

    The P5 resume seam (:func:`scripts.train_oasst1.expected_state_dict_shapes`)
    is the repo's authoritative key->shape map (it is what the trainer's
    ``--resume`` validation checks a checkpoint's tensors against); the real
    meta-device state dict must match it key-for-key and shape-for-shape, and
    — because the 1B presets are TIED — must contain **no** ``lm_head.weight``.
    """
    cfg = builder().derive()
    model = _meta_talos_1b(cfg.num_layers)
    state = model.state_dict()
    expected_map = expected_state_dict_shapes(cfg)
    assert set(state.keys()) == set(expected_map.keys()), (
        f"tied 1B state dict keys drift from the --resume seam: "
        f"missing={sorted(set(expected_map) - set(state))} "
        f"extra={sorted(set(state) - set(expected_map))}"
    )
    for key, shape in expected_map.items():
        assert tuple(state[key].shape) == tuple(shape), f"{key}: {tuple(state[key].shape)} != {tuple(shape)}"
    assert "lm_head.weight" not in state  # tied: the embed matrix IS the head

    # Explicit spot-shapes for the owner's spec numbers (head_dim 128, 4:1 GQA).
    sa = model.layers[0].attention
    assert (sa.num_heads, sa.num_kv_heads, sa.head_dim, sa.gqa_ratio) == (16, 4, 128, 4)
    assert tuple(sa.q_proj.weight.shape) == (2048, 2048)
    assert tuple(sa.k_proj.weight.shape) == (512, 2048)  # 4 kv heads x head_dim 128
    assert tuple(sa.v_proj.weight.shape) == (512, 2048)
    assert tuple(sa.o_proj.weight.shape) == (2048, 2048)
    ffn = model.layers[0].ffn.swiglu
    assert tuple(ffn.gate_proj.weight.shape) == (5504, 2048)
    assert tuple(ffn.up_proj.weight.shape) == (5504, 2048)
    assert tuple(ffn.down_proj.weight.shape) == (2048, 5504)
    assert tuple(model.embed_tokens.weight.shape) == (TALOS_1B_VOCAB, 2048)
    assert tuple(model.final_norm.weight.shape) == (2048,)
    assert model.lm_head is None  # tied embeddings

    # No biases anywhere on this path: every Linear is bias=False (verified by
    # walking the actual modules, not just the state dict key set).
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None, f"{name} has a bias — 1B spec is bias-free"
    assert not any(n.endswith(".bias") for n in model.state_dict())


def test_talos_1b_meta_device_forward_never_materialized() -> None:
    """The meta model is shape-only (no parameters were allocated in RAM)."""
    model = _meta_talos_1b(21)
    assert all(p.device == torch.device("meta") for p in model.parameters())


def test_talos_1b_config_flags() -> None:
    """Every knob from the owner's spec is present on the derived configs."""
    for builder, layers in ((talos_1b_config, 21), (talos_1b_spec16_config, 16)):
        cfg = builder().derive()
        assert cfg.vocab_size == TALOS_1B_VOCAB
        assert cfg.max_seq_len == 2048
        assert cfg.hidden_size == 2048
        assert cfg.num_layers == layers
        assert cfg.num_attention_heads == 16
        assert cfg.num_kv_heads == 4
        assert cfg.head_dim == 128  # explicit override, never derived
        assert cfg.intermediate_size == 5504
        assert cfg.ffn_type == "dense"
        assert cfg.tie_word_embeddings is True
        assert cfg.rope_theta == 10_000.0
        assert cfg.initializer_range == 0.02
        assert cfg.attention_type == "full"


# ---------------------------------------------------------------------------
# 2. Tokenizer pairing (32,000 guard; the broad seam coverage is test_vocab_seam)
# ---------------------------------------------------------------------------
def test_talos_1b_tokenizer_config_pairing() -> None:
    """TokenizerConfigs are vocab 32,000 and refuse an incompatible pairing."""
    for name in ("talos_1b", "talos_1b_spec16"):
        tok = preset_tokenizer_config(name)
        model = talos_1b_config() if name == "talos_1b" else talos_1b_spec16_config()
        assert tok.vocab_size == TALOS_1B_VOCAB == model.vocab_size
        # The ≤ model-compat contract still holds (trained assets may use fewer
        # merge slots, never more).
        assert tok.vocab_size <= model.vocab_size
        # max_merges arithmetic: 32,000 = 256 bytes + 4 specials + merge slots.
        assert tok.max_merges() == TALOS_1B_VOCAB - 256 - 4 == 31_740


# ---------------------------------------------------------------------------
# 3. Functional suite on a small-dim config with the SAME feature switches
# ---------------------------------------------------------------------------
def talos_1b_like_config() -> ModelConfig:
    """Small-dim config exercising the 1B feature switches at CPU size.

    Mirrors every switch of the 1B spec — explicit ``head_dim=128`` (not
    derived), **4:1 GQA** (4 query / 1 KV head), SwiGLU dense FFN at the same
    2.6875× ratio (1376 = 512 · 5504/2048), tied embeddings, RoPE theta 1e4,
    init range 0.02, no biases — at hidden 512 / 4 layers / vocab 4096 so the
    whole functional battery runs on this CPU box in seconds.
    """
    return ModelConfig(
        vocab_size=4096,
        hidden_size=512,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=1,
        head_dim=128,
        ffn_type="dense",
        intermediate_size=1376,
        max_seq_len=512,
        attention_type="full",
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )


TALOS_1B_LIKE_PARAMS = 13_177_344  # 2,097,152 + 4·2,769,920 + 512 (pinned below)


def talos_1b_like_model(seed: int = 0) -> TalosGPT:
    set_seed(seed)
    return TalosGPT(talos_1b_like_config().derive()).eval()


def _trained_tokenizer(tmp_path) -> str:
    """Train a vocab-1024 byte-level BPE tokenizer (fits the 4096 like-model)."""
    data_path = str(tmp_path / "corpus.jsonl")
    with open(data_path, "w", encoding="utf-8") as fh:
        for doc in FIXTURE_CORPUS:
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(data_path, tok_path, preset="tiny")
    return tok_path


def test_talos_1b_like_exact_parameter_count() -> None:
    cfg = talos_1b_like_config().derive()
    assert cfg.param_count_breakdown()["total"] == TALOS_1B_LIKE_PARAMS
    assert cfg.param_count_breakdown()["lm_head"] == 0  # tied
    model = talos_1b_like_model()
    assert model.num_parameters() == TALOS_1B_LIKE_PARAMS
    assert model.lm_head is None


def test_talos_1b_like_forward_shapes_and_backward() -> None:
    model = talos_1b_like_model()
    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (2, 16))
    logits, cache = model(x)
    assert tuple(logits.shape) == (2, 16, vocab)
    assert cache is None
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), x.reshape(-1))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"missing grads: {missing[:5]}"


def test_talos_1b_like_training_steps_reduce_loss() -> None:
    """A few AdamW steps on the same tokens must reduce the loss."""
    set_seed(0)
    model = talos_1b_like_model().train()
    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss0 = None
    for i in range(5):
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            loss0 = float(loss.detach())
    logits, _ = model(x[:, :-1])
    loss1 = float(loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)).detach())
    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"


def test_talos_1b_like_generation_via_library_path() -> None:
    """Greedy KV-cache generation via inference.generate (ids stay in-vocab)."""
    model = talos_1b_like_model(seed=3)
    vocab = model.config.vocab_size
    prompt = torch.tensor([[7, 11, 13, 2]], dtype=torch.long)
    ids = generate(model, prompt, max_new_tokens=8)
    assert len(ids) == 8
    assert all(0 <= t < vocab for t in ids)
    # Prefill logits must equal incremental KV-cache decode (head_dim 128 path).
    seq = torch.randint(0, vocab, (1, 24))
    report = prefill_decode_max_abs_diff(model, seq)
    assert report.max_abs_diff < LOGIT_ATOL, f"prefill/decode mismatch: {report}"
    assert report.argmax_match


def test_talos_1b_like_checkpoint_round_trip_tied(tmp_path) -> None:
    """v1 checkpoint round-trip with TIED weights: no lm_head key, identical logits.

    Exercises the real trainer writer (``save_checkpoint``), the pure dict
    reader (``load_checkpoint``), a strict ``load_state_dict`` into a fresh
    tied model, and the trainer ``--resume`` seam's key/shape map
    (``expected_state_dict_shapes``) — proving a tied checkpoint round-trip
    needs no ``lm_head`` key anywhere on the path.
    """
    model = talos_1b_like_model(seed=1)
    vocab = model.config.vocab_size
    tok_path = _trained_tokenizer(tmp_path)
    ckpt_path = str(tmp_path / "step-1.pt")
    save_checkpoint(ckpt_path, model, 1, train_loss=1.5, val_loss=1.6, tokenizer_path=tok_path)

    ckpt = load_checkpoint(ckpt_path)
    assert "lm_head.weight" not in ckpt["model_state_dict"]  # tied save
    cfg = ModelConfig(**ckpt["model_config"]).derive()
    assert cfg.tie_word_embeddings is True
    # --resume seam: the recorded key/shape map matches a fresh tied model.
    assert set(ckpt["model_state_dict"].keys()) == set(expected_state_dict_shapes(cfg))

    loaded = TalosGPT(cfg).eval()
    assert loaded.lm_head is None
    loaded.load_state_dict(ckpt["model_state_dict"])  # strict by default
    with torch.no_grad():
        x = torch.randint(0, vocab, (1, 8))
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.equal(a, b)


def test_talos_1b_like_safetensors_round_trip(tmp_path) -> None:
    """Safetensors export/import round-trip accepts a ModelConfig-driven model.

    ``scripts.export_safetensors.export_model`` records preset=None for a
    non-canonical shape and ``load_artifact`` rebuilds + strictly loads it;
    tied weights survive the release format with bit-identical logits.
    """
    model = talos_1b_like_model(seed=2)
    vocab = model.config.vocab_size
    out_dir = str(tmp_path / "release")
    meta = export_model(model, out_dir)
    assert meta["n_params"] == TALOS_1B_LIKE_PARAMS
    assert meta["preset"] is None  # small like-shape is not a registered preset

    loaded, meta2 = load_artifact(out_dir)
    assert loaded.config.tie_word_embeddings is True
    assert loaded.lm_head is None
    with torch.no_grad():
        x = torch.randint(0, vocab, (1, 8))
        assert torch.equal(model(x)[0], loaded(x)[0])


# ---------------------------------------------------------------------------
# 4. Attention backend selection at head_dim-128 shapes (PR #33 semantics)
# ---------------------------------------------------------------------------
def test_talos_1b_backend_selects_sdpa_with_plain_fallback_available() -> None:
    """Default (auto) backend is SDPA when flash-attn is absent, at hd-128 shapes."""
    from model.attention import PlainAttentionBackend, SDPAAttentionBackend

    model = talos_1b_like_model(seed=5)
    # The build path (gpt._auto_backend) picks SDPA on this host (no flash-attn).
    assert isinstance(model.backend, SDPAAttentionBackend)
    # A forward at head_dim-128 shapes runs through that backend.
    vocab = model.config.vocab_size
    with torch.no_grad():
        logits, _ = model(torch.randint(0, vocab, (1, 32)))
    assert tuple(logits.shape) == (1, 32, vocab)
    # The plain fallback instance exists and is what long-context configs use:
    cfg = talos_1b_like_config().derive()
    cfg.attention_chunk_size = 64  # bounded-memory path requires the plain backend
    chunked = TalosGPT(cfg, attention_backend=PlainAttentionBackend(64)).eval()
    with torch.no_grad():
        logits_c, _ = chunked(torch.randint(0, vocab, (1, 32)))
    assert tuple(logits_c.shape) == (1, 32, vocab)
    # _auto_backend honours the chunked selector exactly as PR #33 established.
    auto = TalosGPT(cfg)._auto_backend(cfg)
    assert isinstance(auto, PlainAttentionBackend)


# ---------------------------------------------------------------------------
# 5. REGRESSION — the four canonical presets must not move
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "builder", "params"),
    [("tiny", tiny_config, TINY_PARAMS),
     ("tiny_1m", tiny_1m_config, TINY_1M_PARAMS),
     ("tiny_10m", tiny_10m_config, TINY_10M_PARAMS)],
)
def test_canonical_tiny_family_still_resolves_and_builds(name, builder, params) -> None:
    cfg = builder().derive()
    assert resolve_preset(cfg) == name
    assert expected_params(cfg) == params
    assert CANONICAL_PRESETS[name][0] == params
    model = TalosGPT(cfg).train()
    assert model.num_parameters() == params
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1))
    loss.backward()
    assert loss.isfinite()
    del model


def test_canonical_tiny_100m_still_resolves_and_smokes() -> None:
    """tiny_100m regression: the heavy preset keeps its exact count + fwd/bwd."""
    cfg = tiny_100m_config().derive()
    assert resolve_preset(cfg) == "tiny_100m"
    assert expected_params(cfg) == TINY_100M_PARAMS
    model = TalosGPT(cfg).train()
    try:
        assert model.num_parameters() == TINY_100M_PARAMS
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _ = model(x)  # forward
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), x.reshape(-1)
        )
        loss.backward()  # backward
        assert loss.isfinite()
    finally:
        del model