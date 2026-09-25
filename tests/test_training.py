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

from configs.canonical import CANONICAL_PRESETS
from configs.presets import tiny_config
from model import ModelConfig, TalosGPT
from model.utils import set_seed
from scripts.train_oasst1 import (
    build_tiny_model,
    check_tiny_compat,
    evaluate,
    load_checkpoint,
    split_jsonl,
    train_epochs,
    train_tokenizer_for_run,
)
from scripts.generate import generate_from_checkpoint
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tools.make_synthetic_oasst1 import generate
from training.synthetic import build_recurrent_corpus
from evaluation.harness import run_eval

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
    assert model.num_parameters() == CANONICAL_PRESETS["tiny"][0] == 254_272  # noqa: E501
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


# ---------------------------------------------------------------------------
# Checkpoint evaluation harness (reproducibility cycle)
# e2e pattern of PR #17: small JSONL -> short train -> checkpoint -> eval.
# ---------------------------------------------------------------------------
def _short_trained_run(tmp_path, n_docs: int = 20, steps: int = 3) -> tuple:
    """Shared fixture: deterministic split -> BPE -> 254,272-param train ->
    checkpoint; returns (row, val_path, train_path)."""
    src = tmp_path / "src.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=n_docs, seed=0)
    split = split_jsonl(str(src), str(tmp_path / "data"), ratio=0.9, seed=0)
    tok_path = str(tmp_path / "tokenizer.json")
    tokenizer = train_tokenizer_for_run(split.train_path, tok_path)
    model = build_tiny_model()
    from data.tokenized import StreamingTokenizedDataset

    train_ds = StreamingTokenizedDataset(
        split.train_path, tokenizer, seq_len=32, batch_size=2, mode="pack", eos=True
    )
    val_ds = StreamingTokenizedDataset(
        split.val_path, tokenizer, seq_len=32, batch_size=2, mode="pack", eos=True
    )
    history = train_epochs(
        model, train_ds, val_ds,
        out_dir=str(tmp_path),
        tokenizer_path=tok_path,
        lr=3e-3,
        epochs=1,
        device=torch.device("cpu"),
        seed=0,
        max_steps_per_epoch=steps,
    )
    return history.row(1), split.val_path, split.train_path


def test_eval_checkpoint_full_metrics_short_run(tmp_path) -> None:
    row, val_path, train_path = _short_trained_run(tmp_path)

    result = run_eval(
        row.checkpoint,
        data=val_path,
        train_data=train_path,
        seq_len=32,
        batch_size=2,
        seed=0,
    )
    # Parameter count must match the recorded (and canonical 254,272) count.
    assert result.params == CANONICAL_PRESETS["tiny"][0] == 254_272  # noqa: E501
    assert result.vocab_size == 1024
    # Perplexity = exp(natural-log loss): finite and positive.
    assert torch.isfinite(torch.tensor(result.val_loss))
    assert result.val_perplexity > 0.0 and torch.isfinite(
        torch.tensor(result.val_perplexity)
    )
    # Next-token argmax accuracy is a rate in [0, 1].
    assert 0.0 <= result.val_accuracy <= 1.0
    # Recomputed val loss (same batch layout as training) matches the loss the
    # checkpoint recorded at save time.
    assert result.checkpoint_val_loss == pytest.approx(row.val_loss)
    assert result.val_loss == pytest.approx(row.val_loss)
    # Train loss reported when a train split is supplied.
    assert result.train_loss is not None and torch.isfinite(
        torch.tensor(result.train_loss)
    )
    # Tokens processed, throughput and wall time are positive and sensible.
    assert result.tokens_processed > 0
    assert result.throughput_tok_per_s > 0.0
    assert result.eval_wall_s >= 0.0
    # Peak RSS is labelled as the process metric it is (ru_maxrss, MiB).
    assert result.peak_rss_mb > 0.0
    # Metrics file written next to the checkpoint, loadable + comparable.
    assert result.metrics_path == str(tmp_path / "eval-metrics.json")
    with open(result.metrics_path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["format"] == "talos-oasst1-eval-metrics-v1"
    assert saved["val_loss"] == result.val_loss
    assert saved["val_perplexity"] == result.val_perplexity
    assert saved["val_accuracy"] == result.val_accuracy


def test_eval_checkpoint_is_deterministic(tmp_path) -> None:
    """Two eval runs on the same checkpoint + split are bit-identical."""
    row, val_path, _ = _short_trained_run(tmp_path)
    kwargs = dict(checkpoint_path=row.checkpoint, data=val_path,
                  seq_len=32, batch_size=2, seed=0)
    a = run_eval(**kwargs)
    b = run_eval(**kwargs)
    # Substantive numbers must be IDENTICAL (wall time / peak RSS are machine
    # metrics and are excluded by design).
    for field in ("val_loss", "val_perplexity", "val_accuracy",
                  "tokens_processed", "params", "checkpoint_val_loss",
                  "train_loss", "tokenizer_vocab_size"):
        assert getattr(a, field) == getattr(b, field), (
            f"eval field {field} not deterministic: {getattr(a, field)} vs "
            f"{getattr(b, field)}"
        )


def test_eval_checkpoint_default_val_split(tmp_path) -> None:
    """No --data: the checkpoint's own val split is found and used."""
    row, val_path, _ = _short_trained_run(tmp_path)
    result = run_eval(row.checkpoint, seq_len=32, batch_size=2, seed=0)
    assert os.path.abspath(result.data) == os.path.abspath(val_path)
    assert result.val_loss == pytest.approx(row.val_loss)


def test_eval_checkpoint_nparams_mismatch_fails(tmp_path) -> None:
    """A checkpoint whose recorded n_params disagrees fails loudly."""
    row, val_path, _ = _short_trained_run(tmp_path)
    ckpt = load_checkpoint(row.checkpoint)
    ckpt["n_params"] = 999
    tampered = str(tmp_path / "tampered.pt")
    torch.save(ckpt, tampered)
    with pytest.raises(ValueError) as exc:
        run_eval(tampered, data=val_path, seq_len=32, batch_size=2, seed=0)
    assert "n_params" in str(exc.value) and "999" in str(exc.value)


# ---------------------------------------------------------------------------
# Generation smoke + negative tests (reproducibility cycle, owner ask):
# checkpoint -> consistency checks -> greedy KV-cache decode. Reuses the same
# e2e pattern as the eval tests: 20-doc run, a few steps, checkpoint, generate.
# ---------------------------------------------------------------------------
PROMPT = "How do I bake a cake?"
GENERATED_LEN = 20


def test_generate_from_checkpoint_smoke(tmp_path) -> None:
    """Train a tiny run, load the checkpoint, generate; the full contract holds."""
    row, _, _ = _short_trained_run(tmp_path, steps=3)
    result = generate_from_checkpoint(
        row.checkpoint, PROMPT, max_new_tokens=GENERATED_LEN
    )

    # No exception; the continuation has exactly the requested number of tokens.
    assert len(result.token_ids) == GENERATED_LEN
    assert result.max_new_tokens == GENERATED_LEN
    # Every generated id is a valid model-vocab id (in-bounds for the embedding).
    assert all(0 <= t < result.vocab_size for t in result.token_ids)
    assert result.params == CANONICAL_PRESETS["tiny"][0] == 254_272  # noqa: E501
    assert result.vocab_size == 1024
    # tokenizer/model compat: tokenizer vocab never exceeds model vocab.
    assert result.tokenizer_vocab_size <= result.vocab_size
    assert result.vocab_padding == result.vocab_size - result.tokenizer_vocab_size
    # Generations are non-trivial: 20 ids decode to non-empty text.
    assert result.text
    # Prompt echo integrity: byte-level BPE round-trips the prompt exactly, and
    # the echoed full text starts with the original prompt bytes.
    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(os.path.dirname(row.checkpoint), "tokenizer.json")
    )
    assert tokenizer.decode(tokenizer.encode(PROMPT)) == PROMPT
    assert result.full_text.startswith(PROMPT)
    assert result.prompt_tokens == len(tokenizer.encode(PROMPT))
    assert result.prompt_truncated is False
    # Report fields are populated and sane.
    assert result.mode == "greedy" and result.seed is None
    assert result.wall_s >= 0.0 and result.device
    assert result.checkpoint_step == 3


def test_generate_from_checkpoint_is_deterministic(tmp_path) -> None:
    """Strongest smoke signal: same checkpoint, greedy -> identical ids + text."""
    row, _, _ = _short_trained_run(tmp_path, steps=3)
    a = generate_from_checkpoint(row.checkpoint, PROMPT, max_new_tokens=GENERATED_LEN)
    b = generate_from_checkpoint(row.checkpoint, PROMPT, max_new_tokens=GENERATED_LEN)
    assert a.token_ids == b.token_ids
    assert a.text == b.text
    assert a.full_text == b.full_text
    assert a.prompt_tokens == b.prompt_tokens


def test_generate_argument_errors_are_clear() -> None:
    """Bad arguments fail loudly (no checkpoint needed): never a silent default."""
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint("nope.pt", "", max_new_tokens=8)
    assert "empty" in str(exc.value).lower()
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint("nope.pt", "hi", max_new_tokens=0)
    assert "max_new_tokens" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint("nope.pt", "hi", max_new_tokens=8,
                                 temperature=0.8)  # no --seed
    assert "--seed" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint("nope.pt", "hi", max_new_tokens=8,
                                 temperature=0.0, seed=0)
    assert "temperature" in str(exc.value)


def test_generate_rejects_tampered_model_config(tmp_path) -> None:
    """A checkpoint whose recorded model_config was tampered (vocab) is rejected
    BEFORE any token is generated: the rebuilt model's params no longer match
    the recorded n_params."""
    row, _, _ = _short_trained_run(tmp_path, steps=3)
    ckpt = load_checkpoint(row.checkpoint)
    cfg = dict(ckpt["model_config"])
    cfg["vocab_size"] = 2048  # tamper: rebuilt model != recorded artifact
    ckpt["model_config"] = cfg
    tampered = str(tmp_path / "tampered-config.pt")
    torch.save(ckpt, tampered)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint(tampered, PROMPT, max_new_tokens=8)
    msg = str(exc.value)
    # The rebuilt model (vocab 2048) no longer matches the recorded n_params
    # (254,272): the loader rejects with the clean guard error, BEFORE any
    # weight-copying shape error could surface.
    assert "n_params mismatch" in msg and "254,272" in msg
    assert "385,344" in msg  # == params of the tampered rebuild


def test_generate_rejects_recorded_vocab_mismatch(tmp_path) -> None:
    """A checkpoint whose recorded vocab_size disagrees with its own rebuilt
    model_config is rejected loudly (recorded values must match the rebuild)."""
    row, _, _ = _short_trained_run(tmp_path, steps=3)
    ckpt = load_checkpoint(row.checkpoint)
    ckpt["vocab_size"] = 2048  # tamper the recorded value (config still 1024)
    tampered = str(tmp_path / "tampered-vocab.pt")
    torch.save(ckpt, tampered)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint(tampered, PROMPT, max_new_tokens=8)
    msg = str(exc.value)
    assert "vocab_size mismatch" in msg and "2048" in msg


def test_generate_rejects_tokenizer_vocab_overflow(tmp_path) -> None:
    """A sidecar tokenizer whose vocab exceeds the model vocab is rejected:
    ids would be unembeddable (the tokenizer/model_compat.py contract)."""
    row, _, _ = _short_trained_run(tmp_path, steps=3)
    out_dir = os.path.dirname(row.checkpoint)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Inflate the realized vocab (base bytes + specials + merges) past the
    # model's 1024 rows while keeping the file loadable: raise the config
    # budget and add extra distinct special tokens (vs. the realized vocab of
    # a 20-doc corpus — up to 981 — 50 extras guarantee overflow).
    payload["config"]["vocab_size"] = 4096
    payload["config"]["extra_special_tokens"] = [
        f"<|x{i}|>" for i in range(50)
    ]
    overflow_tok = os.path.join(out_dir, "tokenizer-overflow.json")
    with open(overflow_tok, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    assert ByteLevelBPETokenizer.from_file(overflow_tok).vocab_size > 1024

    ckpt = load_checkpoint(row.checkpoint)
    ckpt["tokenizer_path"] = overflow_tok
    tampered = str(tmp_path / "tampered-tokenizer.pt")
    torch.save(ckpt, tampered)
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint(tampered, PROMPT, max_new_tokens=8)
    msg = str(exc.value)
    assert "tokenizer vocab" in msg and "1024" in msg


# ---------------------------------------------------------------------------
# Tokenizer identity fingerprint (audit P0 fix 4): sha256 of the serialized
# tokenizer.json recorded in the checkpoint AND metrics.json, validated on
# every load (eval harness + generate CLI) so a same-size, different-content
# tokenizer — the audit's silent-swap hole — fails loudly with both hashes.
# ---------------------------------------------------------------------------
def test_checkpoint_and_metrics_record_tokenizer_fingerprint(tmp_path) -> None:
    """The fingerprint is written to the checkpoint and metrics.json and both
    match the sha256 of the actual sidecar tokenizer.json."""
    from scripts.train_oasst1 import train_run
    from tokenizer.tokenizer import tokenizer_file_sha256

    out_dir = str(tmp_path / "run")
    args, _ = _train_run_args(tmp_path, out_dir=out_dir, epochs=1,
                              n_docs=20, steps_per_epoch=3)
    metrics = train_run(args)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    expected_fp = tokenizer_file_sha256(tok_path)
    assert metrics["tokenizer_sha256"] == expected_fp

    ckpt = load_checkpoint(os.path.join(out_dir, "step-3.pt"))
    assert ckpt["tokenizer_fingerprint"] == expected_fp
    # Resume-enabling state is present (audit P0 fix 3).
    assert "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None
    assert "rng_state" in ckpt and set(("torch", "numpy", "python")) <= set(ckpt["rng_state"])
    assert ckpt["epoch"] == 1 and ckpt["step"] == 3

    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["tokenizer_sha256"] == expected_fp


def test_generate_rejects_swapped_same_size_tokenizer(tmp_path) -> None:
    """A same-size, different-content sidecar tokenizer is rejected on load
    with both hashes printed (audit: a 512-vocab tokenizer next to a 1024
    model used to load silently — the exact hole this closes)."""
    from tokenizer.tokenizer import tokenizer_file_sha256

    row, _, _ = _short_trained_run(tmp_path, steps=3)
    out_dir = os.path.dirname(row.checkpoint)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    recorded_fp = tokenizer_file_sha256(tok_path)

    with open(tok_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Same size, same validity, different content: swap two *byte-only* merge
    # pairs (both components < 256, so neither depends on merge rank order —
    # merge tokens referencing earlier merge ids would break under reordering).
    merges = list(payload["merges"])
    byte_only = [
        i for i, m in enumerate(merges)
        if int(m[0]) < 256 and int(m[1]) < 256
    ]
    assert len(byte_only) >= 2, "corpus too small: need two byte-only merges"
    i, j = byte_only[0], byte_only[1]
    merges[i], merges[j] = merges[j], merges[i]
    payload["merges"] = merges
    swapped = os.path.join(out_dir, "tokenizer-swapped.json")
    with open(swapped, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    swapped_tok = ByteLevelBPETokenizer.from_file(swapped)
    assert swapped_tok.vocab_size == ByteLevelBPETokenizer.from_file(tok_path).vocab_size
    actual_fp = tokenizer_file_sha256(swapped)
    assert actual_fp != recorded_fp

    ckpt = load_checkpoint(row.checkpoint)
    ckpt["tokenizer_path"] = swapped
    tampered = str(tmp_path / "swapped-tokenizer.pt")
    torch.save(ckpt, tampered)

    # generate CLI path: rejected by the shared loader, both hashes printed.
    with pytest.raises(ValueError) as exc:
        generate_from_checkpoint(tampered, PROMPT, max_new_tokens=8)
    msg = str(exc.value)
    assert "tokenizer identity mismatch" in msg
    assert recorded_fp in msg and actual_fp in msg  # both hashes in the error

    # eval harness path: same loader, same rejection.
    val_path = os.path.join(out_dir, "data", "val.jsonl")
    with pytest.raises(ValueError) as exc:
        run_eval(tampered, data=val_path, seq_len=32, batch_size=2, seed=0)
    assert "tokenizer identity mismatch" in str(exc.value)


def test_checkpoint_without_fingerprint_still_loads(tmp_path) -> None:
    """Backward compatibility: pre-fingerprint checkpoints (no recorded hash)
    still load and generate — the fingerprint check is opt-in per artifact."""
    row, _, _ = _short_trained_run(tmp_path, steps=3)
    ckpt = load_checkpoint(row.checkpoint)
    ckpt.pop("tokenizer_fingerprint")
    legacy = str(tmp_path / "legacy.pt")
    torch.save(ckpt, legacy)
    result = generate_from_checkpoint(legacy, PROMPT, max_new_tokens=8)
    assert len(result.token_ids) == 8


# ---------------------------------------------------------------------------
# Resume-from-checkpoint (audit P0 fix 3): optimizer + RNG + step/epoch
# counters persisted, and a resumed run is BIT-EXACT to an uninterrupted one.
# ---------------------------------------------------------------------------
def _train_run_args(tmp_path, *, out_dir, epochs, n_docs=100, steps_per_epoch=30):
    """Build the Namespace for train_run (fresh or resumed) on a 100-doc corpus."""
    src = tmp_path / "src.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=n_docs, seed=0)
    from types import SimpleNamespace

    return SimpleNamespace(
        data=str(src), out_dir=out_dir, seed=0, preset="tiny",
        resume=None, split_ratio=0.9, split_max_docs=None,
        bpe_num_merges=None, bpe_minfreq=2, bpe_max_docs=None, bpe_max_chars=None,
        epochs=epochs, seq=32, batch=2, lr=3e-3, max_steps_per_epoch=steps_per_epoch,
        val_max_steps=3, device="cpu",
    ), src


def test_resume_bit_exact_matches_uninterrupted(tmp_path) -> None:
    """A 30-step run resumed at step 30 is BIT-IDENTICAL to a 60-step run that
    never stopped: same weights, same train/val loss at every checkpoint.

    Two full ``train_run`` flows (split -> tokenizer -> train -> checkpoint)
    in separate directories: A runs 2 epochs x 30 steps uninterrupted; B runs
    1 epoch (30 steps), then resumes with --resume to reach the same 2 epochs.
    """
    from scripts.train_oasst1 import train_run
    from tokenizer.tokenizer import tokenizer_file_sha256

    out_a = str(tmp_path / "run_a")
    out_b = str(tmp_path / "run_b")

    # --- run A: uninterrupted 60-step run -----------------------------------
    args_a, _ = _train_run_args(tmp_path, out_dir=out_a, epochs=2)
    metrics_a = train_run(args_a)
    ckpt_a = load_checkpoint(os.path.join(out_a, "step-60.pt"))
    assert ckpt_a["epoch"] == 2 and ckpt_a["step"] == 60

    # --- run B: 30 steps, checkpoint, then resume to 60 ----------------------
    args_b1, _ = _train_run_args(tmp_path, out_dir=out_b, epochs=1)
    metrics_b1 = train_run(args_b1)
    step30 = os.path.join(out_b, "step-30.pt")
    assert os.path.isfile(step30)
    ckpt30 = load_checkpoint(step30)
    assert ckpt30["epoch"] == 1 and ckpt30["step"] == 30
    # Same corpus + seed => same tokenizer in both runs (BPE determinism);
    # the resume checkpoint's recorded fingerprint must match the metrics'
    # recorded hash and run A's.
    assert ckpt30["tokenizer_fingerprint"] == metrics_b1["tokenizer_sha256"]
    assert metrics_a["tokenizer_sha256"] == metrics_b1["tokenizer_sha256"]

    args_b2, _ = _train_run_args(tmp_path, out_dir=out_b, epochs=2)
    args_b2.resume = step30
    metrics_b2 = train_run(args_b2)
    ckpt_b = load_checkpoint(os.path.join(out_b, "step-60.pt"))
    assert ckpt_b["epoch"] == 2 and ckpt_b["step"] == 60
    assert metrics_b2["resumed_from"] == os.path.abspath(step30)

    # --- bit-exactness: every tensor in the final state dict is identical ----
    sa, sb = ckpt_a["model_state_dict"], ckpt_b["model_state_dict"]
    assert set(sa) == set(sb)
    for key in sa:
        assert torch.equal(sa[key], sb[key]), (
            f"model weight {key} diverged between uninterrupted and resumed runs"
        )
    # Optimizer state is identical too (m/v moments + step counters).
    oa, ob = ckpt_a["optimizer_state_dict"], ckpt_b["optimizer_state_dict"]
    for group_a, group_b in zip(oa["param_groups"], ob["param_groups"]):
        assert group_a == group_b
    for idx in oa["state"]:
        for k in oa["state"][idx]:
            va, vb = oa["state"][idx][k], ob["state"][idx][k]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), f"optimizer state {idx}.{k} diverged"
            else:
                assert va == vb, f"optimizer state {idx}.{k} diverged ({va} vs {vb})"
    # Losses: the resumed epoch-2 row equals the uninterrupted epoch-2 row.
    # (Run A recorded epochs 1+2; the resumed session only records epoch 2.)
    row_a2 = metrics_a["epochs"][1]
    row_b2 = metrics_b2["epochs"][0]
    assert row_a2["epoch"] == row_b2["epoch"] == 2
    assert row_a2["train_loss"] == row_b2["train_loss"]
    assert row_a2["val_loss"] == row_b2["val_loss"]
    assert row_b2["global_step"] == 60
    # RNG states stored are the exact ones each run had at its final checkpoint.
    assert set(("torch", "numpy", "python")) <= set(ckpt_a["rng_state"])
    assert set(("torch", "numpy", "python")) <= set(ckpt_b["rng_state"])


def test_resume_rejects_checkpoint_without_optimizer_state(tmp_path) -> None:
    """An old v1 checkpoint (no optimizer/RNG state) cannot be resumed: clear
    error, not a silent fresh-restart."""
    from scripts.train_oasst1 import train_epochs, save_checkpoint
    from scripts.train_oasst1 import split_jsonl, build_tiny_model
    from data.tokenized import StreamingTokenizedDataset

    src = tmp_path / "src.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=50, seed=0)
    split = split_jsonl(str(src), str(tmp_path / "data"), ratio=0.9, seed=0)
    tok_path = str(tmp_path / "tokenizer.json")
    train_tokenizer_for_run(split.train_path, tok_path)
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    model = build_tiny_model()
    train_ds = StreamingTokenizedDataset(
        split.train_path, tokenizer, seq_len=32, batch_size=2, mode="pack", eos=True,
    )
    legacy_path = str(tmp_path / "legacy.pt")
    save_checkpoint(  # no optimizer/rng/epoch keys (old v1 format)
        legacy_path, model, 30, 3.0, None, tok_path,
    )
    legacy = load_checkpoint(legacy_path)
    legacy.pop("optimizer_state_dict")
    with pytest.raises(ValueError) as exc:
        train_epochs(
            model, train_ds, None, out_dir=str(tmp_path),
            tokenizer_path=tok_path, lr=3e-3, epochs=2,
            device=torch.device("cpu"), seed=0,
            max_steps_per_epoch=5, resume_from=legacy,
        )
    assert "optimizer_state_dict" in str(exc.value)
