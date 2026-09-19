"""Tests for the ``tiny_1m`` (~1M-parameter) preset — the Phase A scaling step.

``tiny_1m`` is the 254K ``tiny`` preset scaled up on the **exact same
architecture** (same 2:1 GQA ratio, head_dim 16, dense FFN at 4× hidden,
full attention, un-tied embeddings, no biases; ``vocab_size`` 1024 and
``max_seq_len`` 512 unchanged). Its exact, programmatically-verified parameter
count is **1,000,320** — see ``docs/SCALING.md`` §3.1 for the full arithmetic.

These tests pin:
  1. the exact 1,000,320-parameter count (model + canonical registry);
  2. forward pass + loss/backward;
  3. checkpoint save/load round-trip in the existing ``talos-training-checkpoint-v1``
     format through the eval harness (which enforces the 1M canonical count);
  4. generation smoke from a (random-init) 1M checkpoint via ``scripts.generate``;
  5. tokenizer compatibility (vocab 1024 <= model vocab 1024, unchanged tokenizer);
  6. KV-cache equivalence — prefill logits == cached-decode logits;
  7. the DDM disk-tiered KV-cache path exercised for the 1M config;
  8. the 254K ``tiny`` preset still constructs at exactly 254,272.

All tests are deterministic and CPU-fast.
"""
from __future__ import annotations

import json
import random

import pytest
import torch

from configs.canonical import CANONICAL_PRESETS, expected_params, resolve_preset
from configs.presets import tiny_1m_config, tiny_config
from evaluation.harness import load_checkpoint_artifacts
from experiments.ddm_kv_cache import DiskTieredKVCache
from inference.generate import decode_step, prefill, prefill_decode_max_abs_diff
from model import TalosGPT
from model.utils import set_seed
from scripts.generate import generate_from_checkpoint
from scripts.train_oasst1 import (
    build_preset_model,
    check_preset_compat,
    load_checkpoint,
    save_checkpoint,
    train_tokenizer_for_run,
)
from tests.fixture_corpus import FIXTURE_CORPUS

TINY_1M_PARAMS = 1_000_320  # canonical tiny_1m parameter count
TINY_PARAMS = 254_272       # canonical tiny parameter count (must stay intact)
VOCAB = 1024
LOGIT_ATOL = 1e-4           # established fp32 guardrail (tests/test_inference.py)
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


def tiny_1m_model(seed: int = 0) -> TalosGPT:
    set_seed(seed)
    return TalosGPT(tiny_1m_config().derive()).eval()


def _write_synthetic_jsonl(path, seed: int = 0) -> None:
    """Deterministic small JSONL corpus for tokenizer training (no copyrighted data)."""
    with open(path, "w", encoding="utf-8") as fh:
        for doc in FIXTURE_CORPUS:
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _trained_tokenizer(tmp_path) -> str:
    """Train a vocab-1024 byte-level BPE tokenizer; return its JSON path."""
    data_path = str(tmp_path / "corpus.jsonl")
    _write_synthetic_jsonl(data_path)
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(data_path, tok_path)
    return tok_path


def _briefly_train(model: TalosGPT, tokenizer, steps: int = 60) -> None:
    """Lightly train on the encoded fixture corpus so greedy stays in-vocab.

    A fully random-init model's greedy argmax can land on token ids the small
    trained tokenizer has no decoding for (its vocab is ~396 here, not 1024),
    which would make ``tokenizer.decode`` KeyError. Training a handful of steps
    on the tokenizer's own corpus makes the greedy output dominated by
    in-vocab corpus tokens — exactly how the existing tiny generation tests
    keep decode deterministic. The purpose is a smoke of the 1M generation
    path, not model quality.
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


def _save_1m_checkpoint(tmp_path, model: TalosGPT, tok_path: str, step: int = 1) -> str:
    ckpt_path = str(tmp_path / f"step-{step}.pt")
    save_checkpoint(ckpt_path, model, step, train_loss=1.5, val_loss=1.6, tokenizer_path=tok_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# 1. Exact parameter count
# ---------------------------------------------------------------------------
def test_tiny_1m_exact_parameter_count() -> None:
    model = tiny_1m_model()
    assert model.num_parameters() == TINY_1M_PARAMS
    # The canonical registry agrees with the constructed model.
    cfg = tiny_1m_config().derive()
    assert resolve_preset(cfg) == "tiny_1m"
    assert expected_params(cfg) == TINY_1M_PARAMS
    assert CANONICAL_PRESETS["tiny_1m"] == (TINY_1M_PARAMS, VOCAB)
    # The analytic breakdown (docs/SCALING.md §3.1) reproduces the exact count.
    b = cfg.param_count_breakdown()
    assert b["total"] == TINY_1M_PARAMS
    assert b["embedding"] == VOCAB * 128 == 131_072
    assert b["lm_head"] == 131_072
    assert b["per_layer_attention"] == 49_152
    assert b["per_layer_ffn"] == 196_608
    assert (
        b["embedding"] + b["lm_head"] + 3 * (b["per_layer_attention"] + 2 * 128 + b["per_layer_ffn"]) + 128
        == TINY_1M_PARAMS
    )


def test_tiny_1m_preset_guard_enforces_exact_count() -> None:
    # build_preset_model asserts the 1M contract; a drift is rejected loudly.
    model = build_preset_model("tiny_1m")
    assert model.num_parameters() == TINY_1M_PARAMS
    cfg = tiny_1m_config().derive()
    with pytest.raises(ValueError) as exc:
        check_preset_compat("tiny_1m", cfg, TINY_1M_PARAMS - 1)
    assert "1,000,320" in str(exc.value) and "tiny_1m" in str(exc.value)


def test_tiny_preset_still_exactly_254272() -> None:
    """The tiny 254K preset must remain byte-for-byte unchanged at 254,272."""
    model = TalosGPT(tiny_config().derive())
    assert model.num_parameters() == TINY_PARAMS
    assert resolve_preset(tiny_config().derive()) == "tiny"


# ---------------------------------------------------------------------------
# 2. Forward pass + loss/backward
# ---------------------------------------------------------------------------
def test_tiny_1m_forward_and_loss_backward() -> None:
    torch.manual_seed(0)
    model = tiny_1m_model()
    x = torch.randint(0, VOCAB, (2, 16))
    logits, cache = model(x)
    assert tuple(logits.shape) == (2, 16, VOCAB)
    assert cache is None  # no cache by default
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"missing grads: {missing[:5]}"


# ---------------------------------------------------------------------------
# 3. Checkpoint save/load round-trip in the v1 format (via eval harness)
# ---------------------------------------------------------------------------
def test_tiny_1m_checkpoint_round_trip(tmp_path) -> None:
    model = tiny_1m_model(seed=1)
    tok_path = _trained_tokenizer(tmp_path)
    ckpt_path = _save_1m_checkpoint(tmp_path, model, tok_path)

    # Rebuild + validate through the eval harness (enforces the 1M canonical count).
    ckpt, loaded, tokenizer = load_checkpoint_artifacts(ckpt_path)
    assert ckpt["format"] == "talos-training-checkpoint-v1"
    assert loaded.num_parameters() == TINY_1M_PARAMS
    assert ckpt["n_params"] == TINY_1M_PARAMS
    assert loaded.config.vocab_size == VOCAB
    assert tokenizer.vocab_size <= VOCAB
    # state_dict round-trips: loaded model is bit-identical to the saved one.
    with torch.no_grad():
        x = torch.randint(0, VOCAB, (1, 8))
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 4. Generation smoke from a tiny-trained 1M checkpoint
# ---------------------------------------------------------------------------
def test_tiny_1m_generation_smoke(tmp_path) -> None:
    model = tiny_1m_model(seed=2)
    tok_path = _trained_tokenizer(tmp_path)
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    _briefly_train(model, tokenizer)  # tiny-trained so greedy stays in-vocab
    ckpt_path = _save_1m_checkpoint(tmp_path, model, tok_path)

    result = generate_from_checkpoint(ckpt_path, PROMPT, max_new_tokens=8)
    assert result.params == TINY_1M_PARAMS
    assert result.vocab_size == VOCAB
    assert len(result.token_ids) == 8
    assert all(0 <= t < result.vocab_size for t in result.token_ids)
    assert result.tokenizer_vocab_size <= result.vocab_size
    assert result.checkpoint_format == "talos-training-checkpoint-v1"
    assert result.text  # decodable continuation


def test_tiny_1m_checkpoint_rejects_tampered_config(tmp_path) -> None:
    """A tampered 1M checkpoint is rejected by the canonical guard BEFORE decode."""
    model = tiny_1m_model(seed=3)
    tok_path = _trained_tokenizer(tmp_path)
    ckpt_path = _save_1m_checkpoint(tmp_path, model, tok_path)
    ckpt = load_checkpoint(ckpt_path)
    cfg = dict(ckpt["model_config"])
    cfg["vocab_size"] = 2048  # tamper: rebuilt model != recorded n_params
    ckpt["model_config"] = cfg
    tampered = str(tmp_path / "tampered-1m.pt")
    torch.save(ckpt, tampered)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint(tampered, PROMPT, max_new_tokens=8)
    assert "n_params mismatch" in str(exc.value) and "1,000,320" in str(exc.value)


# ---------------------------------------------------------------------------
# 5. Tokenizer compatibility (vocab 1024 <= model vocab, unchanged tokenizer)
# ---------------------------------------------------------------------------
def test_tiny_1m_tokenizer_compatibility(tmp_path) -> None:
    tok_path = _trained_tokenizer(tmp_path)
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    assert tokenizer.vocab_size <= VOCAB  # tokenizer/model compat contract
    model = tiny_1m_model()
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


# ---------------------------------------------------------------------------
# 6. KV-cache equivalence for the 1M config
# ---------------------------------------------------------------------------
def test_tiny_1m_prefill_matches_kv_decode() -> None:
    set_seed(4)
    model = tiny_1m_model(seed=4)
    ids = torch.randint(0, VOCAB, (1, 24))
    report = prefill_decode_max_abs_diff(model, ids)
    assert report.max_abs_diff < LOGIT_ATOL, f"prefill/decode mismatch: {report}"
    assert report.argmax_match, f"greedy choices diverge: {report}"


# ---------------------------------------------------------------------------
# 7. DDM disk-tiered KV-cache path exercised for the 1M config
# ---------------------------------------------------------------------------
def test_tiny_1m_ddm_tiered_logits_match_resident() -> None:
    set_seed(5)
    model = tiny_1m_model(seed=5)
    prompt = torch.randint(0, VOCAB, (1, 80), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        logits_r, cache_r = prefill(model, prompt)
        tok_r = logits_r[:, -1:, :].argmax(dim=-1)
        res_r = [decode_step(model, cache_r, tok_r, cache_r.length)]
        for _ in range(4):
            tok_r = res_r[-1][:, -1:, :].argmax(dim=-1)
            res_r.append(decode_step(model, cache_r, tok_r, cache_r.length))

        tier = DiskTieredKVCache.for_model(model)
        try:
            logits_t, cache_t = prefill(model, prompt, cache=tier)
            assert cache_t is tier and tier.length == 80
            assert float((logits_r - logits_t).abs().max()) <= LOGIT_ATOL
            tok_t = logits_t[:, -1:, :].argmax(dim=-1)
            for step_r in res_r:
                out_t = decode_step(model, cache_t, tok_t, cache_t.length)
                assert float((step_r - out_t).abs().max()) <= LOGIT_ATOL
                tok_t = out_t[:, -1:, :].argmax(dim=-1)
            # 80-token prompt completed a 64-token block and recalled it on decode.
            assert tier.ledger.blocks_read > 0
            assert tier.ledger.bytes_read_total > 0
        finally:
            tier.close()
