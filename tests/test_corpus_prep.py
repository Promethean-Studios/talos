"""Tests for the corpus-prep pipeline (scripts/prepare_corpus.py).

Synthetic-data-only, no network, no real corpus downloads: the packing path is
exercised with a local JSONL reader (any :class:`data.readers.DatasetReader`
works — the CLI's HuggingFaceReader is only the remote default), so the full
pipeline runs on stdlib + numpy + torch in CI.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from configs.vocab import VOCAB_SIZE
from data.readers import JSONLReader
from scripts.prepare_corpus import (
    METADATA_SCHEMA,
    SCRIPT_VERSION,
    _shard_name,
    make_arg_parser,
    prepare_corpus,
)
from scripts.train_oasst1 import train_tokenizer_for_run
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256
from tools.make_synthetic_oasst1 import generate

SEQ = 16
VAL_SKIP = 5


def _write_synthetic_jsonl(path, n_docs: int = 40, seed: int = 0) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for doc in generate(n_docs, seed):
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")


def _read_doc_texts(path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


@pytest.fixture()
def corpus(tmp_path):
    """Synthetic JSONL corpus + a deterministic established tokenizer."""
    src = tmp_path / "docs.jsonl"
    _write_synthetic_jsonl(str(src), n_docs=40, seed=4)
    tok_path = str(tmp_path / "tokenizer.json")
    tokenizer = train_tokenizer_for_run(
        str(src), tok_path, preset="tiny", num_merges=150
    )
    return {
        "src": str(src),
        "tok_path": tok_path,
        "tokenizer": tokenizer,
        "docs": _read_doc_texts(str(src)),
    }


def _run(corpus, tmp_path, **overrides):
    """Run prepare_corpus with sensible tiny defaults and return the metadata."""
    kwargs = dict(
        out_dir=str(tmp_path / "out"),
        target_tokens=2500,
        val_tokens=800,
        seq_len=SEQ,
        dtype="int32",
        val_skip_docs=VAL_SKIP,
        stream_buffer_docs=6,
        rows_per_shard=4,
    )
    kwargs.update(overrides)
    return prepare_corpus(
        JSONLReader(corpus["src"]),
        corpus["tokenizer"],
        tokenizer_path=corpus["tok_path"],
        dataset_meta={"dataset": "synthetic", "config": "sample", "split": "train"},
        args_echo={"smoke": True},
        **kwargs,
    )


def _load_phase(out_dir: str, phase: str) -> np.ndarray:
    """Concatenate every shard of one phase, rows -> (rows, seq) int array."""
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = [e["rows"] for e in manifest["shards"] if e["phase"] == phase]
    arrays = []
    for index in range(len(rows)):
        arrays.append(np.load(os.path.join(out_dir, _shard_name(phase, index))))
    return np.concatenate(arrays, axis=0) if arrays else np.empty((0, SEQ))


def _expected_stream(docs: list[str], tokenizer, eos_id: int) -> list[int]:
    """The packed token stream the convention dictates: each doc + EOS."""
    out: list[int] = []
    for d in docs:
        out.extend(tokenizer.encode(d))
        out.append(eos_id)
    return out


def test_prepare_corpus_packs_rows_eos_and_counts(tmp_path, corpus) -> None:
    meta = _run(corpus, tmp_path)
    counts = meta["counts"]
    assert not counts["truncated"]
    assert meta["val_region"]["first_doc"] == VAL_SKIP

    for phase in ("val", "train"):
        data = _load_phase(str(tmp_path / "out"), phase)
        # Row shape + dtype contract.
        assert data.ndim == 2 and data.shape[1] == SEQ
        assert data.dtype == np.int32
        region = meta["val_region"] if phase == "val" else meta["train_region"]
        assert data.shape[0] == counts[f"{phase}_rows"] == meta["counts"][f"{phase}_rows"]
        docs = corpus["docs"][region["first_doc"] : region["last_doc"] + 1]
        expected = _expected_stream(docs, corpus["tokenizer"], meta["packing"]["eos_id"])
        # Real tokens == sum(len(encode(doc)) + 1).
        assert counts[f"{phase}_tokens"] == len(expected)
        # Every stored token (incl. row padding) is a valid id.
        assert int(data.min()) >= 0 and int(data.max()) < VOCAB_SIZE
        # Removing only the pad padding from the tail reproduces the doc stream
        # exactly — this proves rows are contiguous seq-length cuts of the
        # EOS-separated document stream (correct packing + EOS placement).
        flat = data.reshape(-1)
        pad = meta["packing"]["pad_id"]
        stripped = flat[flat != pad]  # pad ids appear only in the padded tail
        assert stripped.tolist() == expected
        # The ONLY padding is the tail: everything except the pad run is kept.
        assert int(flat.size) - int((flat != pad).sum()) == counts[f"{phase}_pad_tokens"]


def test_prepare_corpus_uint16_roundtrip(tmp_path, corpus) -> None:
    meta16 = _run(corpus, tmp_path, dtype="uint16")
    assert meta16["packing"]["dtype"] == "uint16"
    data16 = _load_phase(str(tmp_path / "out"), "train")
    assert data16.dtype == np.uint16
    # Ids survive a byte-width round-trip exactly (vocab 1024 < 65536).
    assert int(data16.max()) == max(meta16["packing"]["eos_id"], meta16["packing"]["pad_id"])

    # Same content as the int32 layout.
    meta32 = _run(corpus, tmp_path, dtype="int32", out_dir=str(tmp_path / "out32"))
    data32 = _load_phase(str(tmp_path / "out32"), "train")
    assert data16.tolist() == data32.tolist()
    assert meta16["counts"]["train_tokens"] == meta32["counts"]["train_tokens"]


def test_prepare_corpus_enforces_token_id_bounds(tmp_path, corpus) -> None:
    tok = corpus["tokenizer"]
    real_encode = tok.encode

    def poisoned(text):
        ids = real_encode(text)
        return ids + [VOCAB_SIZE]  # one id the canonical model cannot embed

    tok.encode = poisoned  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="out of range"):
        _run(corpus, tmp_path)


def test_prepare_corpus_manifest_complete_and_consistent(tmp_path, corpus) -> None:
    out_dir = str(tmp_path / "out")
    meta = _run(corpus, tmp_path)
    with open(os.path.join(out_dir, "run_metadata.json"), encoding="utf-8") as fh:
        run_meta = json.load(fh)
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    # The top-level run_metadata.json IS the manifest's embedded metadata.
    assert manifest["metadata"] == run_meta
    # Every schema-required field is present.
    assert run_meta["schema"] == METADATA_SCHEMA
    assert run_meta["script_version"] == SCRIPT_VERSION
    assert isinstance(run_meta["timestamp"], str) and run_meta["timestamp"]
    assert run_meta["args"] == {"smoke": True}
    for key in ("dataset", "config", "split"):
        assert run_meta["dataset"][key] is not None
    for key in ("revision", "sha"):  # best-effort HF fields must exist
        assert key in run_meta["dataset"]
    tok = run_meta["tokenizer"]
    for key in ("path", "sha256", "vocab_size", "merge_count"):
        assert key in tok
    assert tok["sha256"] == tokenizer_file_sha256(corpus["tok_path"])
    assert tok["vocab_size"] == corpus["tokenizer"].vocab_size
    assert tok["merge_count"] == corpus["tokenizer"].merge_count
    packing = run_meta["packing"]
    for key in ("seq", "eos_id", "pad_id", "dtype", "vocab_size_bound"):
        assert key in packing
    assert packing["seq"] == SEQ
    assert packing["eos_id"] == corpus["tokenizer"].eos_id
    assert packing["dtype"] == "int32"
    assert packing["vocab_size_bound"] == VOCAB_SIZE
    c = run_meta["counts"]
    for key in (
        "target_tokens", "docs_streamed", "skipped_gap_docs",
        "val_docs", "val_chars", "val_tokens", "val_rows", "val_pad_tokens",
        "val_tokens_incl_padding",
        "train_docs", "train_chars", "train_tokens", "train_rows",
        "train_pad_tokens", "train_tokens_incl_padding",
        "total_tokens", "total_rows", "total_chars", "truncated",
    ):
        assert key in c, f"missing counts field {key}"

    # Internal consistency: counts add up, shards add up, budget honoured.
    assert c["total_tokens"] == c["val_tokens"] + c["train_tokens"]
    assert c["total_chars"] == c["val_chars"] + c["train_chars"]
    assert c["total_rows"] == c["val_rows"] + c["train_rows"]
    assert c["val_tokens_incl_padding"] == c["val_rows"] * SEQ
    assert c["train_tokens_incl_padding"] == c["train_rows"] * SEQ
    assert c["total_tokens"] >= c["target_tokens"] or c["truncated"]
    assert manifest["format"].startswith("talos-packed-tokens-v")
    assert manifest["num_shards"] == len(manifest["shards"])
    assert manifest["num_shards"] == len(
        [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    )
    assert sum(e["rows"] for e in manifest["shards"]) == c["total_rows"]
    for e in manifest["shards"]:
        assert os.path.isfile(os.path.join(out_dir, e["shard"]))
        assert e["rows"] > 0
        assert e["tokens_incl_padding"] == e["rows"] * SEQ


def test_prepare_corpus_val_is_disjoint_from_train(tmp_path, corpus) -> None:
    meta = _run(corpus, tmp_path)
    vr, tr = meta["val_region"], meta["train_region"]
    # The val region sits at a fixed stream offset after the skip gap ...
    assert vr["first_doc"] == VAL_SKIP
    # ... and the train region starts strictly after it: no doc overlap.
    assert tr["first_doc"] == vr["last_doc"] + 1
    assert tr["first_doc"] > vr["last_doc"] >= 0
    # Skip gap + val + train == exactly the streamed docs (nothing double-used).
    c = meta["counts"]
    assert c["skipped_gap_docs"] == VAL_SKIP
    assert (
        c["skipped_gap_docs"] + vr["docs"] + tr["docs"]
        == c["docs_streamed"]
    )
    # Val location is independent of the training budget (same val_tokens).
    meta2 = _run(corpus, tmp_path, target_tokens=3000)
    assert meta2["val_region"] == vr


def test_prepare_corpus_sharding_and_buffer_flush(tmp_path, corpus) -> None:
    out_dir = str(tmp_path / "out")
    meta = _run(corpus, tmp_path, rows_per_shard=4, stream_buffer_docs=3)
    assert meta["counts"]["total_rows"] > 4
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert len(manifest["shards"]) > 1  # forced frequent flushes

    # Sharded output is byte-for-byte identical to an unsharded run.
    big = _run(corpus, tmp_path, rows_per_shard=10_000, stream_buffer_docs=10_000,
               out_dir=str(tmp_path / "out-big"))
    for phase in ("val", "train"):
        small = _load_phase(out_dir, phase)
        big_arr = _load_phase(str(tmp_path / "out-big"), phase)
        assert small.tolist() == big_arr.tolist()
    assert meta["counts"] == big["counts"]


def test_prepare_corpus_cli_defaults() -> None:
    p = make_arg_parser()
    ns = p.parse_args(["--tokenizer-json", "x.json", "--target-tokens", "100",
                       "--out-dir", "/tmp/x"])
    assert ns.dataset == "HuggingFaceFW/fineweb-edu"
    assert ns.config == "sample-10BT"
    assert ns.split == "train"
    assert ns.seq == 512
    assert ns.dtype == "int32"
    assert ns.val_tokens == 15_000_000
    assert ns.val_skip_docs == 10_000
    assert ns.stream_buffer_docs == 1_000
    assert ns.rows_per_shard == 16_384


def test_prepare_corpus_rejects_bad_args(tmp_path, corpus) -> None:
    with pytest.raises(ValueError, match="strictly less"):
        _run(corpus, tmp_path, target_tokens=100, val_tokens=1000)
    with pytest.raises(ValueError, match="seq_len"):
        _run(corpus, tmp_path, seq_len=1)


def test_prepare_corpus_truncates_when_stream_ends(tmp_path, corpus) -> None:
    # Budget far beyond the 40-doc corpus: the run must stop cleanly and flag it.
    meta = _run(corpus, tmp_path, target_tokens=10_000_000, val_tokens=800)
    assert meta["counts"]["truncated"] is True
    assert meta["counts"]["total_tokens"] < meta["counts"]["target_tokens"]


def test_hf_reader_accepts_config_kwarg_with_dataset_object() -> None:
    """HuggingFaceReader forwards config for string ids; objects still work."""
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"text": ["hello world", "second doc"]})
    from data.readers import HuggingFaceReader, reader_from_config

    reader = HuggingFaceReader(ds, split="train", config="sample-10BT",
                               revision="main")
    records = list(reader)
    assert [r["text"] for r in records] == ["hello world", "second doc"]

    via_config = reader_from_config(
        {"type": "huggingface", "dataset": ds, "config": "sample-10BT"}
    )
    assert [r["text"] for r in via_config] == ["hello world", "second doc"]
    # String ids go through load_dataset(..., config=...): assert the kwarg
    # reaches the loader by stubbing it (no network).
    calls: dict = {}

    def fake_load(*args, **kwargs):
        calls.update(args=args, kwargs=kwargs)
        return ds

    datasets.load_dataset = fake_load  # type: ignore[method-assign]
    HuggingFaceReader("SomeOrg/fake", config="sample-10BT")
    assert calls["args"][0] == "SomeOrg/fake"
    assert calls["kwargs"]["config"] == "sample-10BT"
    assert calls["kwargs"]["streaming"] is True