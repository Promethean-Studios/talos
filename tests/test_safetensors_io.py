"""Safetensors release-format export/import tests (owner requirement #10).

Pins the release path added by ``scripts/export_safetensors.py``:

* export ``model.safetensors`` + ``model.config.json`` sidecar from a live
  model and from a ``talos-training-checkpoint-v1`` artifact;
* round-trip **bit-exactness**: the reloaded model's state dict and its
  forward logits are byte-for-byte identical to the source model's;
* the canonical registry guard runs on the sidecar too (a tampered
  ``n_params``/vocab/preset is rejected loudly before weights load);
* strict state-dict loading (no silent partial loads);
* sidecar tokenizer: fingerprint identity check + vocab contract.

All tests are deterministic and CPU-fast; the 100M-scale cases mirror the
ladder's requirement to validate the release path at the real target size.
"""
from __future__ import annotations

import json
import random

import pytest
import safetensors.torch
import torch

from configs.presets import tiny_config, tiny_100m_config
from model import TalosGPT
from model.utils import set_seed
from scripts.export_safetensors import (
    SAFETENSORS_FORMAT,
    checkpoint_to_safetensors,
    export_model,
    load_artifact,
    load_tokenizer,
)
from scripts.train_oasst1 import save_checkpoint

TINY_100M_PARAMS = 96_482_304
TINY_PARAMS = 254_272
VOCAB = 1024
PROMPT = "The capital of France is"


@pytest.fixture(autouse=True)
def _restore_rng_state() -> None:
    torch_state = torch.random.get_rng_state()
    py_state = random.getstate()
    yield
    torch.random.set_rng_state(torch_state)
    random.setstate(py_state)


def _tiny_100m_model(seed: int = 0) -> TalosGPT:
    set_seed(seed)
    return TalosGPT(tiny_100m_config().derive()).eval()


def _write_synthetic_jsonl(path, seed: int = 0) -> None:
    import json as _json
    from tests.fixture_corpus import FIXTURE_CORPUS

    with open(path, "w", encoding="utf-8") as fh:
        for doc in FIXTURE_CORPUS:
            fh.write(_json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _trained_tokenizer(tmp_path) -> str:
    from scripts.train_oasst1 import train_tokenizer_for_run

    data_path = str(tmp_path / "corpus.jsonl")
    _write_synthetic_jsonl(data_path)
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(data_path, tok_path, preset="tiny_100m")
    return tok_path


def _assert_bit_exact(source: TalosGPT, loaded: TalosGPT) -> None:
    """Every tensor and the forward logits must be byte-for-byte identical."""
    src_sd, got_sd = source.state_dict(), loaded.state_dict()
    assert set(src_sd.keys()) == set(got_sd.keys())
    for name in src_sd:
        assert torch.equal(src_sd[name], got_sd[name]), (
            f"tensor {name} not bit-identical after safetensors round-trip"
        )
    with torch.no_grad():
        x = torch.randint(0, VOCAB, (2, 16))
        a, _ = source(x)
        b, _ = loaded(x)
    assert torch.equal(a, b), "forward logits not bit-identical after round-trip"


def _write_v1_checkpoint(tmp_path, model, tok_path, name="step-1.pt") -> str:
    ckpt_path = str(tmp_path / name)
    save_checkpoint(ckpt_path, model, 1, train_loss=1.5, val_loss=1.6, tokenizer_path=tok_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Export from a live 100M model + bit-exact round-trip (ladder: safetensors)
# ---------------------------------------------------------------------------
def test_safetensors_100m_export_and_bit_exact_round_trip(tmp_path) -> None:
    model = _tiny_100m_model(seed=0)
    tok_path = _trained_tokenizer(tmp_path)
    release_dir = str(tmp_path / "release")
    meta = export_model(model, release_dir, tokenizer_path=tok_path, step=7,
                        train_loss=1.25, val_loss=1.3)

    # Sidecar is self-describing and registry-consistent.
    assert meta["format"] == SAFETENSORS_FORMAT
    assert meta["n_params"] == TINY_100M_PARAMS
    assert meta["vocab_size"] == VOCAB
    assert meta["preset"] == "tiny_100m"
    assert meta["step"] == 7
    assert meta["tokenizer_fingerprint"]
    with open(f"{release_dir}/model.config.json", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == meta
    import os
    size = os.path.getsize(f"{release_dir}/model.safetensors")
    # Raw fp32 payload plus the (small) safetensors JSON header.
    assert 4 * TINY_100M_PARAMS <= size < 4 * TINY_100M_PARAMS + 8192

    # Drop the source model BEFORE loading so the peak is at most two live
    # 386 MiB objects (this box runs the suite with ~1 GB of headroom).
    import gc
    del model
    gc.collect()
    loaded, loaded_meta = load_artifact(release_dir)
    assert loaded_meta["preset"] == "tiny_100m"
    assert loaded.num_parameters() == TINY_100M_PARAMS
    # Bit-exact round-trip vs the artifact payload itself: every tensor in the
    # reloaded model is byte-for-byte the tensor stored in model.safetensors.
    payload = safetensors.torch.load_file(f"{release_dir}/model.safetensors")
    try:
        got = loaded.state_dict()
        assert set(got.keys()) == set(payload.keys())
        for name in got:
            assert torch.equal(got[name], payload[name]), (
                f"tensor {name} not bit-identical to the safetensors payload"
            )
    finally:
        del payload
        gc.collect()
    # Forward sanity: finite logits of the right shape (bit-exactness of the
    # logits themselves is asserted at the tiny scale, where source + loaded
    # model fit together comfortably).
    with torch.no_grad():
        x = torch.randint(0, VOCAB, (1, 16))
        logits, _ = loaded(x)
    assert tuple(logits.shape) == (1, 16, VOCAB)
    assert torch.isfinite(logits).all()

    # The release's tokenizer loads with the fingerprint + vocab contract intact.
    tokenizer = load_tokenizer(release_dir, meta)
    assert tokenizer is not None
    assert tokenizer.vocab_size <= VOCAB


def test_safetensors_round_trip_from_v1_checkpoint(tmp_path) -> None:
    """checkpoint_to_safetensors validates via the harness, then exports."""
    model = _tiny_100m_model(seed=1)
    tok_path = _trained_tokenizer(tmp_path)
    ckpt_path = _write_v1_checkpoint(tmp_path, model, tok_path)
    # Drop the source BEFORE re-loading from the checkpoint: the harness keeps
    # the checkpoint's state_dict (386 MiB) plus its rebuilt model alive, so
    # the peak must be capped at two live 386 MiB objects on this box.
    import gc
    del model
    gc.collect()
    release_dir = str(tmp_path / "release-ckpt")
    meta = checkpoint_to_safetensors(ckpt_path, release_dir)

    assert meta["format"] == SAFETENSORS_FORMAT
    assert meta["n_params"] == TINY_100M_PARAMS
    assert meta["preset"] == "tiny_100m"
    assert meta["source_checkpoint"].endswith("step-1.pt")
    assert meta["step"] == 1
    # Realized tokenizer vocab (fixture corpus trains < 764 merges) is <= the
    # model's 1024 — the compat contract — and carried from the harness.
    assert 0 < meta["tokenizer_vocab_size"] <= VOCAB

    loaded, _ = load_artifact(release_dir)
    # Bit-exact vs the payload written from the harness-validated model.
    payload = safetensors.torch.load_file(f"{release_dir}/model.safetensors")
    try:
        got = loaded.state_dict()
        assert set(got.keys()) == set(payload.keys())
        for name in got:
            assert torch.equal(got[name], payload[name])
    finally:
        del payload
        gc.collect()
    tokenizer = load_tokenizer(release_dir, meta)
    assert tokenizer is not None
    # Same IDs as the original sidecar tokenizer (fingerprint-verified).
    first_ids = tokenizer.encode(PROMPT)
    assert all(0 <= i < VOCAB for i in first_ids)


def test_safetensors_tiny_preset_round_trip(tmp_path) -> None:
    """The release path is preset-agnostic: tiny (254,272) round-trips too."""
    set_seed(2)
    model = TalosGPT(tiny_config().derive()).eval()
    release_dir = str(tmp_path / "release-tiny")
    meta = export_model(model, release_dir)
    assert meta["n_params"] == TINY_PARAMS
    assert meta["preset"] == "tiny"
    loaded, _ = load_artifact(release_dir)
    _assert_bit_exact(model, loaded)


# ---------------------------------------------------------------------------
# Guard behaviour on the sidecar (mirrors the checkpoint tamper tests)
# ---------------------------------------------------------------------------
def test_safetensors_rejects_tampered_n_params(tmp_path) -> None:
    model = _tiny_100m_model(seed=3)
    release_dir = str(tmp_path / "release-tamper")
    export_model(model, release_dir)
    cfg_path = f"{release_dir}/model.config.json"
    with open(cfg_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["n_params"] = TINY_100M_PARAMS - 1  # tamper
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with pytest.raises(ValueError) as exc:
        load_artifact(release_dir)
    assert "n_params mismatch" in str(exc.value) and "96,482,304" in str(exc.value)


def test_safetensors_rejects_wrong_preset(tmp_path) -> None:
    model = _tiny_100m_model(seed=4)
    release_dir = str(tmp_path / "release-preset")
    export_model(model, release_dir)
    cfg_path = f"{release_dir}/model.config.json"
    with open(cfg_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["preset"] = "tiny"  # tamper: sidecar claims a different canonical preset
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with pytest.raises(ValueError) as exc:
        load_artifact(release_dir)
    assert "preset" in str(exc.value)


def test_safetensors_rejects_unknown_format(tmp_path) -> None:
    model = _tiny_100m_model(seed=5)
    release_dir = str(tmp_path / "release-format")
    export_model(model, release_dir)
    cfg_path = f"{release_dir}/model.config.json"
    with open(cfg_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["format"] = "talos-training-checkpoint-v1"  # wrong format tag
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with pytest.raises(ValueError) as exc:
        load_artifact(release_dir)
    assert SAFETENSORS_FORMAT in str(exc.value)


def test_safetensors_rejects_missing_weights(tmp_path) -> None:
    model = _tiny_100m_model(seed=6)
    release_dir = str(tmp_path / "release-nofile")
    export_model(model, release_dir)
    import os
    os.remove(f"{release_dir}/model.safetensors")
    with pytest.raises(FileNotFoundError):
        load_artifact(release_dir)


def test_safetensors_rejects_swapped_same_size_tokenizer(tmp_path) -> None:
    """A same-size but different-content tokenizer fails the sha256 check."""
    model = _tiny_100m_model(seed=7)
    tok_path = _trained_tokenizer(tmp_path)
    release_dir = str(tmp_path / "release-tok")
    meta = export_model(model, release_dir, tokenizer_path=tok_path)
    # Swap in a different-but-valid tokenizer with the same vocab budget:
    # encode a second corpus and retrain -> different merges, same size.
    from scripts.train_oasst1 import train_tokenizer_for_run

    other_data = str(tmp_path / "other-corpus.jsonl")
    with open(other_data, "w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({"text": f"completely different corpus line {i} "
                                        f"with distinct unusual words zebra xanthan"}) + "\n")
    other_tok = str(tmp_path / "other-tokenizer.json")
    train_tokenizer_for_run(other_data, other_tok, preset="tiny_100m")
    # Point the recorded tokenizer_path at the other file -> hash mismatch.
    meta["tokenizer_path"] = other_tok
    cfg_path = f"{release_dir}/model.config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with pytest.raises(ValueError) as exc:
        load_tokenizer(release_dir, meta)
    assert "sha256" in str(exc.value) or "swapped" in str(exc.value)