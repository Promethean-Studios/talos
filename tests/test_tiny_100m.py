"""Tests for the ``tiny_100m`` (~100M-parameter) preset — scaling step 4.

``tiny_100m`` is the audit §13 candidate (``hidden-1024 × 6 layers``) on the
**exact same architecture** as ``tiny``/``tiny_1m``/``tiny_10m`` (2:1 GQA
64/32 heads, head_dim 16, dense FFN at 4x hidden 4096/1024, full attention,
un-tied embeddings, no biases; ``vocab_size`` 1024 — the single source of
truth ``configs.vocab.VOCAB_SIZE`` — and ``max_seq_len`` 512 unchanged). Its
exact, programmatically-verified parameter count is **96,482,304** — see
``configs.presets.tiny_100m_config`` for the full arithmetic.

These tests pin:
  1. the exact 96,482,304-parameter count (model + canonical registry) and its
     place inside the owner's 95–105M target;
  2. forward pass + loss/backward and loss-decreasing training steps;
  3. checkpoint save/load round-trip in the ``talos-training-checkpoint-v1``
     format through the eval harness (which enforces the 100M canonical count);
  4. generation smoke from a (lightly-trained) 100M checkpoint via
     ``scripts.generate``, including the context-length guarantee (generated
     sequences never exceed ``max_seq_len``);
  5. tokenizer compatibility (vocab 1024 <= model vocab) and a
     save/load-identical-IDs round-trip through the trained tokenizer;
  6. KV-cache equivalence — prefill logits == cached-decode logits;
  7. the P0 token-ID bounds guard at the embedding seam: min/max valid ids
     pass, invalid ids (1024, -1) are rejected with the clean named error;
  8. ``tiny``/``tiny_1m``/``tiny_10m`` still construct at exactly
     254,272 / 1,000,320 / 9,952,320 (they must remain completely unchanged).

All tests are deterministic and CPU-fast.
"""
from __future__ import annotations

import json
import random

import pytest
import torch

from configs.canonical import CANONICAL_PRESETS, expected_params, resolve_preset
from configs.presets import (
    tiny_100m_config,
    tiny_10m_config,
    tiny_1m_config,
    tiny_config,
)
from evaluation.harness import load_checkpoint_artifacts
from inference.generate import generate, prefill_decode_max_abs_diff
from model import TalosGPT
from model.utils import set_seed, validate_token_ids
from scripts.generate import generate_from_checkpoint
from scripts.train_oasst1 import (
    build_preset_model,
    check_preset_compat,
    load_checkpoint,
    save_checkpoint,
    train_tokenizer_for_run,
)
from tests.fixture_corpus import FIXTURE_CORPUS

TINY_100M_PARAMS = 96_482_304  # canonical tiny_100m parameter count
TINY_10M_PARAMS = 9_952_320    # canonical tiny_10m parameter count (must stay intact)
TINY_1M_PARAMS = 1_000_320     # canonical tiny_1m parameter count (must stay intact)
TINY_PARAMS = 254_272          # canonical tiny parameter count (must stay intact)
VOCAB = 1024
LOGIT_ATOL = 1e-4              # established fp32 guardrail (tests/test_inference.py)
PROMPT = "The capital of France is"


@pytest.fixture(autouse=True)
def _restore_rng_state() -> None:
    """Keep these tests RNG-neutral so earlier/later suite state is unchanged.

    Several pre-existing tests construct random-init models without seeding
    their own init, so their outcome depends on the global RNG state accumulated
    from every test that ran before them. Snapshot the torch + Python RNG state
    before each of our tests and restore it afterwards, so this file perturbs
    nothing for the rest of the suite (this preserves the baseline ordering, in
    which every existing test passes).
    """
    torch_state = torch.random.get_rng_state()
    py_state = random.getstate()
    yield
    torch.random.set_rng_state(torch_state)
    random.setstate(py_state)


def tiny_100m_model(seed: int = 0) -> TalosGPT:
    set_seed(seed)
    return TalosGPT(tiny_100m_config().derive()).eval()


def _write_synthetic_jsonl(path, seed: int = 0) -> None:
    """Deterministic small JSONL corpus for tokenizer training (no copyrighted data)."""
    with open(path, "w", encoding="utf-8") as fh:
        for doc in FIXTURE_CORPUS:
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _trained_tokenizer(tmp_path) -> str:
    """Train a vocab-1024 byte-level BPE tokenizer for the 100M preset; return its path."""
    data_path = str(tmp_path / "corpus.jsonl")
    _write_synthetic_jsonl(data_path)
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(data_path, tok_path, preset="tiny_100m")
    return tok_path


def _briefly_train(model: TalosGPT, tokenizer, steps: int = 12) -> None:
    """Lightly train on the encoded fixture corpus so greedy stays in-vocab.

    A fully random-init model's greedy argmax can land on token ids the small
    trained tokenizer has no decoding for (its vocab is ~396 here, not 1024),
    which would make ``tokenizer.decode`` KeyError. Training a handful of steps
    on the tokenizer's own corpus makes the greedy output dominated by
    in-vocab corpus tokens — exactly how the existing tiny/1M/10M generation
    tests keep decode deterministic. The purpose is a smoke of the 100M
    generation path, not model quality.
    """
    cfg = model.config
    seqs = [tokenizer.encode(d) for d in FIXTURE_CORPUS]
    flat = [t for s in seqs for t in s]
    batch = min(4, len(flat) // 8)
    n = batch * 8
    x = torch.tensor(flat[:n], dtype=torch.long).view(batch, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def _save_100m_checkpoint(tmp_path, model: TalosGPT, tok_path: str, step: int = 1) -> str:
    ckpt_path = str(tmp_path / f"step-{step}.pt")
    save_checkpoint(ckpt_path, model, step, train_loss=1.5, val_loss=1.6, tokenizer_path=tok_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# 1. Exact parameter count (ladder i + ii) + registry
# ---------------------------------------------------------------------------
def test_tiny_100m_exact_parameter_count() -> None:
    model = tiny_100m_model()
    assert model.num_parameters() == TINY_100M_PARAMS
    # The canonical registry agrees with the constructed model.
    cfg = tiny_100m_config().derive()
    assert resolve_preset(cfg) == "tiny_100m"
    assert expected_params(cfg) == TINY_100M_PARAMS
    assert CANONICAL_PRESETS["tiny_100m"] == (TINY_100M_PARAMS, VOCAB)
    # The analytic breakdown (configs.presets.tiny_100m_config docstring)
    # reproduces the exact count.
    b = cfg.param_count_breakdown()
    assert b["total"] == TINY_100M_PARAMS
    assert b["embedding"] == VOCAB * 1024 == 1_048_576
    assert b["lm_head"] == 1_048_576
    assert b["per_layer_attention"] == 3_145_728
    assert b["per_layer_ffn"] == 12_582_912
    assert (
        b["embedding"]
        + b["lm_head"]
        + 6 * (b["per_layer_attention"] + 2 * 1024 + b["per_layer_ffn"])
        + 1024
        == TINY_100M_PARAMS
    )


def test_tiny_100m_params_inside_100m_target() -> None:
    """Ladder (ii): the exact count must sit inside the owner's 95–105M target."""
    assert 95_000_000 <= TINY_100M_PARAMS <= 105_000_000
    model = tiny_100m_model()
    assert 95_000_000 <= model.num_parameters() <= 105_000_000


def test_tiny_100m_preset_guard_enforces_exact_count() -> None:
    # build_preset_model asserts the 100M contract; a drift is rejected loudly.
    model = build_preset_model("tiny_100m")
    assert model.num_parameters() == TINY_100M_PARAMS
    cfg = tiny_100m_config().derive()
    with pytest.raises(ValueError) as exc:
        check_preset_compat("tiny_100m", cfg, TINY_100M_PARAMS - 1)
    assert "96,482,304" in str(exc.value) and "tiny_100m" in str(exc.value)


def test_tiny_tiny1m_tiny10m_presets_still_exact() -> None:
    """tiny (254,272), tiny_1m (1,000,320) and tiny_10m (9,952,320) must stay."""
    for name, builder, params in (("tiny", tiny_config, TINY_PARAMS),
                                  ("tiny_1m", tiny_1m_config, TINY_1M_PARAMS),
                                  ("tiny_10m", tiny_10m_config, TINY_10M_PARAMS)):
        model = TalosGPT(builder().derive())
        assert model.num_parameters() == params
        assert resolve_preset(builder().derive()) == name
        assert CANONICAL_PRESETS[name][0] == params


# ---------------------------------------------------------------------------
# 2. Forward pass + loss/backward (ladder iv — small batch)
# ---------------------------------------------------------------------------
def test_tiny_100m_forward_and_loss_backward() -> None:
    torch.manual_seed(0)
    model = tiny_100m_model()
    x = torch.randint(0, VOCAB, (2, 16))
    logits, cache = model(x)
    assert tuple(logits.shape) == (2, 16, VOCAB)
    assert cache is None  # no cache by default
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"missing grads: {missing[:5]}"


def test_tiny_100m_training_steps_reduce_loss() -> None:
    """A few AdamW steps on the same tokens must reduce the loss."""
    set_seed(0)
    model = tiny_100m_model().train()
    x = torch.randint(0, VOCAB, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss0 = None
    for i in range(5):
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            loss0 = float(loss.detach())
    logits, _ = model(x[:, :-1])
    loss1 = float(loss_fn(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1)).detach())
    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"


# ---------------------------------------------------------------------------
# 3. Checkpoint save/load round-trip in the v1 format (via eval harness)
# ---------------------------------------------------------------------------
def test_tiny_100m_checkpoint_round_trip(tmp_path) -> None:
    model = tiny_100m_model(seed=1)
    tok_path = _trained_tokenizer(tmp_path)
    ckpt_path = _save_100m_checkpoint(tmp_path, model, tok_path)

    # Rebuild + validate through the eval harness (enforces the 100M canonical count).
    ckpt, loaded, tokenizer = load_checkpoint_artifacts(ckpt_path)
    assert ckpt["format"] == "talos-training-checkpoint-v1"
    assert loaded.num_parameters() == TINY_100M_PARAMS
    assert ckpt["n_params"] == TINY_100M_PARAMS
    assert loaded.config.vocab_size == VOCAB
    assert tokenizer.vocab_size <= VOCAB
    # state_dict round-trips: loaded model is bit-identical to the saved one.
    with torch.no_grad():
        x = torch.randint(0, VOCAB, (1, 8))
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 4. Generation smoke from a tiny-trained 100M checkpoint (ladder v)
# ---------------------------------------------------------------------------
def test_tiny_100m_generation_smoke(tmp_path) -> None:
    model = tiny_100m_model(seed=2)
    tok_path = _trained_tokenizer(tmp_path)
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    _briefly_train(model, tokenizer)  # tiny-trained so greedy stays in-vocab
    ckpt_path = _save_100m_checkpoint(tmp_path, model, tok_path)

    result = generate_from_checkpoint(ckpt_path, PROMPT, max_new_tokens=8)
    assert result.params == TINY_100M_PARAMS
    assert result.vocab_size == VOCAB
    assert len(result.token_ids) == 8
    assert all(0 <= t < result.vocab_size for t in result.token_ids)
    assert result.tokenizer_vocab_size <= result.vocab_size
    assert result.checkpoint_format == "talos-training-checkpoint-v1"
    assert result.text  # decodable continuation


def test_tiny_100m_generation_respects_context_length(tmp_path) -> None:
    """Generated sequences never exceed max_seq_len (P0 context-length policy).

    The library path left-truncates an over-long prompt to
    ``max_seq_len - max_new_tokens`` and decodes exactly ``max_new_tokens``
    more, so prompt + continuation can never exceed 512 or drive the RoPE
    position table out of bounds.
    """
    model = tiny_100m_model(seed=3)
    tok_path = _trained_tokenizer(tmp_path)
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    _briefly_train(model, tokenizer)
    ckpt_path = _save_100m_checkpoint(tmp_path, model, tok_path)

    max_seq_len = model.config.max_seq_len
    assert max_seq_len == 512
    # An over-long prompt (encoded to more than max_seq_len tokens) with a
    # large --max-new-tokens must still produce a sequence that fits 512.
    long_prompt = ("word " * 600).strip()  # ~1200 bytes -> clearly > 512 tokens
    result = generate_from_checkpoint(ckpt_path, long_prompt, max_new_tokens=64)
    assert len(result.token_ids) == 64
    assert result.prompt_truncated is True
    assert result.prompt_tokens <= max_seq_len - 64
    # Direct library call on the raw model: same guarantee, ids all in-vocab.
    ids = torch.randint(0, VOCAB, (1, 510))
    generated = generate(model, ids, max_new_tokens=8)
    assert len(generated) == 8
    assert all(0 <= t < VOCAB for t in generated)


def test_tiny_100m_checkpoint_rejects_tampered_config(tmp_path) -> None:
    """A tampered 100M checkpoint is rejected by the canonical guard BEFORE decode."""
    model = tiny_100m_model(seed=4)
    tok_path = _trained_tokenizer(tmp_path)
    ckpt_path = _save_100m_checkpoint(tmp_path, model, tok_path)
    ckpt = load_checkpoint(ckpt_path)
    cfg = dict(ckpt["model_config"])
    cfg["vocab_size"] = 2048  # tamper: rebuilt model != recorded n_params
    ckpt["model_config"] = cfg
    tampered = str(tmp_path / "tampered-100m.pt")
    torch.save(ckpt, tampered)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint(tampered, PROMPT, max_new_tokens=8)
    assert "n_params mismatch" in str(exc.value) and "96,482,304" in str(exc.value)


# ---------------------------------------------------------------------------
# 5. Tokenizer compatibility + save/load identical IDs
# ---------------------------------------------------------------------------
def test_tiny_100m_tokenizer_compatibility(tmp_path) -> None:
    tok_path = _trained_tokenizer(tmp_path)
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    assert tokenizer.vocab_size <= VOCAB  # tokenizer/model compat contract
    model = tiny_100m_model()
    assert model.config.vocab_size == VOCAB
    text = FIXTURE_CORPUS[0]
    ids = tokenizer.encode(text)
    assert all(0 <= i < VOCAB for i in ids)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    assert tuple(logits.shape) == (1, len(ids), VOCAB)
    # Byte-level BPE round-trips English exactly with the unchanged tokenizer.
    assert tokenizer.decode(tokenizer.encode("The quick brown fox")) == "The quick brown fox"


def test_tiny_100m_tokenizer_save_load_identical_ids(tmp_path) -> None:
    """Re-saving + reloading the trained tokenizer yields identical IDs.

    The 100M preset reuses the vocab-1024 tokenizer verbatim; the serialized
    tokenizer must be byte-faithful (same merge list, same special ids), so
    every encode call returns the exact same ids before and after the
    save/load round-trip.
    """
    tok_path = _trained_tokenizer(tmp_path)
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    first = ByteLevelBPETokenizer.from_file(tok_path)
    # Re-save (round-trip through the serialization seam) and reload.
    reloaded_path = str(tmp_path / "tokenizer-reloaded.json")
    first.save(reloaded_path)
    second = ByteLevelBPETokenizer.from_file(reloaded_path)
    assert second.vocab_size == first.vocab_size
    assert second.merge_count == first.merge_count
    assert second.bos_id == first.bos_id == VOCAB - 4  # 1020, LLaMA-style top of budget
    assert second.eos_id == first.eos_id == VOCAB - 3  # 1021
    for doc in FIXTURE_CORPUS[:5]:
        ids_a = first.encode(doc, bos=True, eos=True)
        ids_b = second.encode(doc, bos=True, eos=True)
        assert ids_a == ids_b, f"tokenizer reload changed ids for {doc!r}"
        assert all(0 <= i < VOCAB for i in ids_a)


# ---------------------------------------------------------------------------
# 6. KV-cache equivalence for the 100M config
# ---------------------------------------------------------------------------
def test_tiny_100m_prefill_matches_kv_decode() -> None:
    set_seed(5)
    model = tiny_100m_model(seed=5)
    ids = torch.randint(0, VOCAB, (1, 24))
    report = prefill_decode_max_abs_diff(model, ids)
    assert report.max_abs_diff < LOGIT_ATOL, f"prefill/decode mismatch: {report}"
    assert report.argmax_match, f"greedy choices diverge: {report}"


# ---------------------------------------------------------------------------
# 7. P0 token-ID bounds guard at the embedding seam (for the 100M model)
# ---------------------------------------------------------------------------
def test_tiny_100m_valid_id_boundaries_pass() -> None:
    """ids 0 (min) and 1023 (max) embed cleanly through the guard."""
    model = tiny_100m_model(seed=6)
    for bad_free in (0, VOCAB - 1):
        x = torch.tensor([[bad_free]], dtype=torch.long)
        validate_token_ids(x, model.config.vocab_size, where="test")
        with torch.no_grad():
            logits, _ = model(x)
        assert tuple(logits.shape) == (1, 1, VOCAB)
        assert torch.isfinite(logits).all()


def test_tiny_100m_invalid_ids_rejected_cleanly() -> None:
    """ids 1024 and -1 fail with the named P0 guard error, never an IndexError.

    The embedding-seam guard (model.utils.validate_token_ids, called inside
    TalosGPT.forward) converts the raw ``IndexError``/CUDA device-assert class
    of failure into a clear ValueError naming the offending id and vocab bound.
    """
    model = tiny_100m_model(seed=7)
    for bad in (VOCAB, -1, VOCAB + 100):
        x = torch.tensor([[bad]], dtype=torch.long)
        with pytest.raises(ValueError) as exc:
            model(x)
        assert str(bad) in str(exc.value)
        assert f"[0, {VOCAB})" in str(exc.value)
        assert "model input_ids" in str(exc.value)
    # The standalone helper rejects a mixed batch with the first bad id named.
    mixed = torch.tensor([[5, 4], [9, VOCAB]], dtype=torch.long)
    with pytest.raises(ValueError) as exc:
        validate_token_ids(mixed, VOCAB, where="100m test batch")
    assert "100m test batch" in str(exc.value)
    assert str(VOCAB) in str(exc.value)