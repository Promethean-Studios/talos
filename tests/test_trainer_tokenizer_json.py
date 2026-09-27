"""Tests for the trainer's --tokenizer-json load path (scripts/train_oasst1.py).

The trainer normally trains a BPE tokenizer from the corpus; the corpus-
pretraining path instead LOADS the established Talos tokenizer (published
artifact, sha256 58e4ad40…, vocab 1024 / merges 764). The published artifact is
not vendored in the repo, so per the delegation brief a deterministic stand-in
is generated in tmp_path *from the tokenizer module* (same round-trip contract:
load from a saved tokenizer.json must reproduce byte-identical tokenization).
No network, no real corpus.
"""
from __future__ import annotations

import json
import os

import pytest

from scripts.train_oasst1 import load_checkpoint, train_run
from tokenizer.tokenizer import (
    ByteLevelBPETokenizer,
    tokenizer_file_sha256,
)
from tokenizer.vocab import TokenizerConfig
from tools.make_synthetic_oasst1 import generate

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! 你好，世界。🚀",
    "def f(x): return x ** 2  # a tiny snippet",
    "Talos is a research-grade open-source foundation model.",
]


def _write_synthetic_jsonl(path, n_docs: int = 40, seed: int = 11) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for doc in generate(n_docs, seed):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _established_tokenizer(path, src, num_merges: int = 100) -> ByteLevelBPETokenizer:
    """Train + save a deterministic stand-in for the established tokenizer."""
    from scripts.train_oasst1 import train_tokenizer_for_run

    return train_tokenizer_for_run(src, path, preset="tiny", num_merges=num_merges)


def _train_args(data, out_dir, tokenizer_json, **extra):
    p = __import__("scripts.train_oasst1", fromlist=["make_arg_parser"]).make_arg_parser()
    argv = [
        "--data", data, "--out-dir", out_dir,
        "--tokenizer-json", tokenizer_json,
        "--epochs", "1", "--seq", "32", "--batch", "2", "--seed", "0",
        "--max-steps-per-epoch", "2", "--val-max-steps", "2", "--preset", "tiny",
    ]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return p.parse_args(argv)


def test_trainer_loads_tokenizer_json_identical_tokenization(tmp_path) -> None:
    src = str(tmp_path / "docs.jsonl")
    _write_synthetic_jsonl(src)
    est_path = str(tmp_path / "established-tokenizer.json")
    established = _established_tokenizer(est_path, src)
    original_sha = tokenizer_file_sha256(est_path)

    out_dir = str(tmp_path / "run")
    metrics = train_run(_train_args(src, out_dir, est_path))

    # 1) The established tokenizer is copied to the sidecar location ...
    sidecar = os.path.join(out_dir, "tokenizer.json")
    assert os.path.isfile(sidecar)
    assert tokenizer_file_sha256(sidecar) == original_sha  # byte-identical copy
    # 2) ... wired through the fingerprint machinery (metrics + checkpoints) ...
    assert metrics["tokenizer_origin"] == "loaded"
    assert metrics["tokenizer_sha256"] == original_sha
    assert metrics["tokenizer_vocab_size"] == established.vocab_size
    assert metrics["tokenizer_merges"] == established.merge_count
    assert metrics["tokenizer_json_arg"] == est_path
    ckpt = load_checkpoint(os.path.join(out_dir, "step-2.pt"))
    assert ckpt["tokenizer_fingerprint"] == original_sha
    assert ckpt["tokenizer_path"] == sidecar
    # 3) ... and produces IDENTICAL tokenization to the established artifact.
    reloaded = ByteLevelBPETokenizer.from_file(sidecar)
    for text in SAMPLES:
        assert reloaded.encode(text) == established.encode(text)
    assert reloaded.encode_with_special("x") == established.encode_with_special("x")
    assert reloaded.eos_id == established.eos_id
    assert reloaded.pad_id == established.pad_id


def test_trainer_tokenizer_json_missing_file_fails(tmp_path) -> None:
    src = str(tmp_path / "docs.jsonl")
    _write_synthetic_jsonl(src)
    out_dir = str(tmp_path / "run")
    missing = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="--tokenizer-json"):
        train_run(_train_args(src, out_dir, missing))


def test_trainer_rejects_tokenizer_that_cannot_be_embedded(tmp_path) -> None:
    src = str(tmp_path / "docs.jsonl")
    _write_synthetic_jsonl(src)
    # A tokenizer whose *realized* vocab exceeds the 1024-row model: config
    # vocab_size 2048 + 800 merges -> vocab_size 256+4+800 = 1060 > 1024.
    big_path = str(tmp_path / "too-big.json")
    big = ByteLevelBPETokenizer(
        TokenizerConfig(vocab_size=2048),
        merges=[[i, i + 1] for i in range(800)],
    )
    assert big.vocab_size > 1024
    big.save(big_path)
    out_dir = str(tmp_path / "run")
    with pytest.raises(ValueError, match="exceeds"):
        train_run(_train_args(src, out_dir, big_path))


def test_trainer_tokenizer_json_resume_fingerprint_mismatch(tmp_path) -> None:
    src = str(tmp_path / "docs.jsonl")
    _write_synthetic_jsonl(src)
    est_path = str(tmp_path / "est-tokenizer.json")
    _established_tokenizer(est_path, src, num_merges=90)
    out_dir = str(tmp_path / "run")
    train_run(_train_args(src, out_dir, est_path))
    ckpt = os.path.join(out_dir, "step-2.pt")

    # A different tokenizer (different merges) under the same flag must fail on
    # resume rather than silently mis-tokenizing the continued run.
    other_path = str(tmp_path / "other-tokenizer.json")
    _established_tokenizer(other_path, src, num_merges=60)
    assert tokenizer_file_sha256(other_path) != tokenizer_file_sha256(est_path)
    args = _train_args(src, out_dir, other_path, resume=ckpt, epochs=2)
    with pytest.raises(ValueError, match="does not match"):
        train_run(args)

    # The matching tokenizer is accepted (fingerprint equality, sha256-verified).
    args_ok = _train_args(src, out_dir, est_path, resume=ckpt, epochs=2)
    metrics = train_run(args_ok)
    assert metrics["tokenizer_origin"] == "resumed"
    assert metrics["resumed_from"] == ckpt
    assert metrics["tokenizer_sha256"] == tokenizer_file_sha256(est_path)