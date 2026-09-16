"""Training smoke/integration tests: a tiny model must demonstrably learn.

These assert the core owner-priority property of the prototype — that the
forward/backward/optimizer pipeline actually *works*, i.e. loss decreases on a
genuine tiny model. We train the canonical ``tiny`` preset (254,272 params) on
a small fixed synthetic corpus (`training.synthetic`) and require the
causal-LM loss to fall well below its initial value. Deterministic via a fixed
seed; small enough (a few hundred steps, ~1s on CPU) to run in the normal
suite.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from configs.presets import tiny_config
from model import ModelConfig, TalosGPT
from model.utils import set_seed
from scripts.train_oasst1 import (
    EXPECTED_TINY_PARAMS,
    build_tiny_model,
    check_tiny_compat,
    evaluate,
    load_checkpoint,
    split_jsonl,
    train_epochs,
    train_tokenizer_for_run,
)
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tools.make_synthetic_oasst1 import generate
from training.synthetic import build_recurrent_corpus

# Loss must drop by at least this many nats from its step-0 value. The tiny
# model overfits the fixed corpus from ~log(1024) (~6.9) to well under 1, so a
# margin of 1.5 is comfortably met while remaining robust to small param drift.
DROP_MARGIN = 1.5


def test_tiny_training_decreases_loss() -> None:
    set_seed(0)
    cfg = tiny_config().derive()
    model = TalosGPT(cfg).train()
    corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=32, seed=0)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    n_seq = corpus.shape[0]

    initial: float | None = None
    final: float | None = None
    batch, steps = 4, 120
    for step in range(steps):
        idx = torch.randint(0, n_seq, (batch,))
        x = corpus[idx]
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss = float(loss.detach())
        if step == 0:
            initial = loss
        final = loss

    assert initial is not None and final is not None
    assert torch.isfinite(torch.tensor(initial)) and torch.isfinite(torch.tensor(final))
    # Loss must start high (~log vocab) and fall substantially on the fixed corpus.
    assert initial > 5.0, f"expected a high initial loss near log(vocab), got {initial:.3f}"
    assert final < initial - DROP_MARGIN, (
        f"loss did not decrease enough: {initial:.3f} -> {final:.3f} "
        f"(need a drop > {DROP_MARGIN})"
    )


def test_tiny_training_is_deterministic() -> None:
    def run_once() -> float:
        set_seed(0)
        cfg = tiny_config().derive()
        model = TalosGPT(cfg).train()
        corpus = build_recurrent_corpus(cfg.vocab_size, n_sequences=8, seq_len=16, seed=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        final = None
        for _ in range(30):
            x = corpus[torch.randint(0, corpus.shape[0], (2,))]
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            final = float(loss.detach())
        return float(final)

    a, b = run_once(), run_once()
    assert a == b, f"training not deterministic with fixed seed: {a} vs {b}"


# ---------------------------------------------------------------------------
# OASST1-style JSONL -> tokenizer -> tiny training path (reliability pass)
# End-to-end: deterministic split -> BPE (train split only) -> 254,272-param
# guard -> streamed train -> checkpoint -> reload -> validation loss.
# ---------------------------------------------------------------------------
def _write_synthetic_jsonl(path, n_docs: int, seed: int = 0) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i, doc in enumerate(generate(n_docs, seed)):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _read_docs(path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def test_oasst1_split_is_deterministic_and_disjoint(tmp_path) -> None:
    src = tmp_path / "src.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=20, seed=0)

    out1 = tmp_path / "split1"
    out2 = tmp_path / "split2"
    a = split_jsonl(str(src), str(out1), ratio=0.9, seed=7)
    b = split_jsonl(str(src), str(out2), ratio=0.9, seed=7)

    assert a.train_docs == 18 and a.val_docs == 2 and a.total_docs == 20
    train_docs = set(_read_docs(a.train_path))
    val_docs = set(_read_docs(a.val_path))
    # No leakage: every source doc appears exactly once, no doc in both files.
    assert len(train_docs) == 18 and len(val_docs) == 2
    assert not (train_docs & val_docs)
    assert len(train_docs | val_docs) == 20
    # Determinism: same seed -> identical files.
    assert _read_docs(a.train_path) == _read_docs(b.train_path)
    assert _read_docs(a.val_path) == _read_docs(b.val_path)


def test_oasst1_e2e_split_tokenize_train_checkpoint_reload(tmp_path) -> None:
    src = tmp_path / "src.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=20, seed=0)

    # 1) deterministic split (18 train / 2 val), no leakage
    split = split_jsonl(str(src), str(tmp_path / "data"), ratio=0.9, seed=0)

    # 2) Talos-native BPE on the TRAIN split only, vocab-1024 budget
    tok_path = str(tmp_path / "tokenizer.json")
    tokenizer = train_tokenizer_for_run(split.train_path, tok_path)
    # Realized vocab may stop short of 1024 when minfreq exhausts pairs
    # (256 base bytes + 4 specials + up to 764 merges); it must never exceed
    # the model's 1024 embedding rows.
    assert 256 + 4 <= tokenizer.vocab_size <= 1024
    assert tokenizer.merge_count <= 1024 - 256 - 4  # 764-slot merge budget
    assert os.path.isfile(tok_path)
    tokenizer_reloaded = ByteLevelBPETokenizer.from_file(tok_path)
    assert tokenizer_reloaded.vocab_size == tokenizer.vocab_size

    # 3) canonical tiny model with the hard 254,272-param guard
    model = build_tiny_model()
    assert model.num_parameters() == EXPECTED_TINY_PARAMS == 254_272
    assert model.config.vocab_size == 1024

    from data.tokenized import StreamingTokenizedDataset

    train_ds = StreamingTokenizedDataset(
        split.train_path, tokenizer, seq_len=32, batch_size=2, mode="pack", eos=True
    )
    val_ds = StreamingTokenizedDataset(
        split.val_path, tokenizer, seq_len=32, batch_size=2, mode="pack", eos=True
    )

    # 4) train a couple of steps (seconds-scale) with per-epoch val + ckpt
    history = train_epochs(
        model, train_ds, val_ds,
        out_dir=str(tmp_path),
        tokenizer_path=tok_path,
        lr=3e-3,
        epochs=1,
        device=torch.device("cpu"),
        seed=0,
        max_steps_per_epoch=3,
        val_max_steps=3,
    )
    row = history.row(1)
    assert row.steps == 3 and row.global_step == 3
    assert torch.isfinite(torch.tensor(row.train_loss))
    assert row.val_loss is not None and torch.isfinite(torch.tensor(row.val_loss))
    assert os.path.isfile(row.checkpoint)

    # 5) reload the checkpoint: weights + config + step + losses + tokenizer path
    ckpt = load_checkpoint(row.checkpoint)
    assert ckpt["format"] == "talos-training-checkpoint-v1"
    assert ckpt["step"] == 3
    assert ckpt["n_params"] == 254_272
    assert ckpt["vocab_size"] == 1024
    assert ckpt["train_loss"] == pytest.approx(row.train_loss)
    assert ckpt["val_loss"] == pytest.approx(row.val_loss)
    assert os.path.isfile(ckpt["tokenizer_path"])

    # 6) reload the model from the config and recompute validation loss
    cfg = ModelConfig(**ckpt["model_config"]).derive()
    reloaded = TalosGPT(cfg)
    reloaded.load_state_dict(ckpt["model_state_dict"])
    val_ds2 = StreamingTokenizedDataset(
        split.val_path, tokenizer_reloaded, seq_len=32, batch_size=2,
        mode="pack", eos=True,
    )
    val_loss = evaluate(reloaded, val_ds2, device=torch.device("cpu"), max_steps=3)
    assert val_loss is not None and torch.isfinite(torch.tensor(val_loss))
    assert val_loss == pytest.approx(row.val_loss)


def test_tiny_compat_guard_rejects_config_drift() -> None:
    # Canonical config passes the guard.
    cfg = tiny_config().derive()
    model = TalosGPT(cfg)
    check_tiny_compat(cfg, model.num_parameters())  # no raise

    # Any drift that changes the parameter count or vocab must fail fast.
    drifted = [
        ModelConfig(
            vocab_size=2048, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_kv_heads=2, head_dim=16,
            ffn_type="dense", intermediate_size=256, max_seq_len=512,
        ),
        ModelConfig(
            vocab_size=1024, hidden_size=128, num_layers=2,
            num_attention_heads=4, num_kv_heads=2, head_dim=32,
            ffn_type="dense", intermediate_size=256, max_seq_len=512,
        ),
    ]
    for bad in drifted:
        bad_model = TalosGPT(bad.derive())
        try:
            check_tiny_compat(bad, bad_model.num_parameters())
        except ValueError as exc:
            assert "254,272" in str(exc) and "1024" in str(exc)
        else:
            raise AssertionError("config drift was not rejected by the compat guard")
