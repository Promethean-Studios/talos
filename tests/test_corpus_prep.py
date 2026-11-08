"""Tests for the corpus-prep pipeline (scripts/prepare_corpus.py).

Synthetic-data-only, no network, no real corpus downloads: the packing path is
exercised with a local JSONL reader (any :class:`data.readers.DatasetReader`
works — the CLI's HuggingFaceReader is only the remote default), so the full
pipeline runs on stdlib + numpy + torch in CI.

The token-id guard is exercised for both tracks: the canonical vocab-1024
tokenizers (unchanged behaviour) and the 1B-track 32K/49K assets, whose own
``vocab_size`` is now the packer's bound. The real ``talos_1b_49k`` asset is
read straight from the repo (a committed file — no HF network access).
"""
from __future__ import annotations

import inspect
import json
import os

import numpy as np
import pytest

from configs.vocab import VOCAB_SIZE
from data.packed import load_packed_manifest
from data.readers import JSONLReader
from scripts.prepare_corpus import (
    METADATA_SCHEMA,
    SCRIPT_VERSION,
    _shard_name,
    check_tokenizer_dtype,
    main as prepare_corpus_main,
    make_arg_parser,
    prepare_corpus,
    tokenizer_id_bound,
)
from scripts.train_oasst1 import train_tokenizer_for_run
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256
from tokenizer.vocab import TokenizerConfig
from tools.make_synthetic_oasst1 import generate

SEQ = 16
VAL_SKIP = 5

#: Committed 1B-track tokenizer asset (the one the owner's 1B corpus uses).
_ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "tokenizers"
)
TALOS_1B_49K_PATH = os.path.join(_ASSETS, "talos_1b_49k.json")
TALOS_1B_49K_SHA256 = "c7d26acebad67be04af9b16cb72a812156cf3a708121a8ee9bf794d6a95c8684"
TALOS_1B_49K_VOCAB = 49_152
TALOS_1B_49K_MERGES = 48_892
#: Largest vocab whose every id still fits in 16 bits (max id 65,535).
UINT16_MAX_VOCAB = 65_536


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


def test_hf_reader_accepts_config_kwarg_with_dataset_object(monkeypatch) -> None:
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
    # String ids go through load_dataset(..., name=...): assert the kwarg
    # reaches the loader by stubbing it (no network).
    calls: dict = {}

    def fake_load(*args, **kwargs):
        calls.update(args=args, kwargs=kwargs)
        return ds

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    HuggingFaceReader("SomeOrg/fake", config="sample-10BT")
    assert calls["args"][0] == "SomeOrg/fake"
    assert calls["kwargs"]["name"] == "sample-10BT"
    assert "config" not in calls["kwargs"]
    assert calls["kwargs"]["streaming"] is True


# --------------------------------------------------------------------------- #
# Tokenizer-vocab guard: derived from the LOADED tokenizer, not a constant.
#
# The packer used to hard-wire the canonical ``VOCAB_SIZE`` (1024) as its token-id
# bound, which blocked the 1B-track corpora (talos_1b_32k v2 = 32,000 ids,
# talos_1b_49k = 49,152 ids). The bound is now the loaded ``tokenizer.json``'s own
# ``config.vocab_size``; every other safety property is unchanged (out-of-range
# ids still fail loudly, uint16 still needs every id to fit in 16 bits).
# --------------------------------------------------------------------------- #

def _wide_tokenizer(
    tmp_path, vocab_size: int, *, name: str = "tok"
) -> tuple[ByteLevelBPETokenizer, str]:
    """A local tokenizer declaring ``vocab_size`` ids (a few merges, saved to disk).

    Same JSON format and ``gpt2`` pre-tokenizer as the 1B-track assets; the
    special ids land at the top of the declared range exactly like they do in
    ``assets/tokenizers/talos_1b_49k.json``.
    """
    config = TokenizerConfig(vocab_size=vocab_size, pre_tokenize="gpt2")
    tok = ByteLevelBPETokenizer(
        config, [(97, 98), (99, 100), (101, 102), (105, 106)], exact_bpe=True
    )
    path = str(tmp_path / f"{name}.json")
    tok.save(path)
    return tok, path


def _synthetic_src(tmp_path, *, n_docs: int = 40, seed: int = 4) -> str:
    src = str(tmp_path / "docs.jsonl")
    if not os.path.isfile(src):
        _write_synthetic_jsonl(src, n_docs=n_docs, seed=seed)
    return src


def _pack_with(tmp_path, tokenizer, tokenizer_path, src, *, dtype="uint16",
               out="out", **overrides):
    """prepare_corpus on a synthetic JSONL corpus, with the tiny test defaults."""
    kwargs = dict(
        out_dir=str(tmp_path / out),
        target_tokens=2500,
        val_tokens=800,
        seq_len=SEQ,
        dtype=dtype,
        val_skip_docs=VAL_SKIP,
        stream_buffer_docs=6,
        rows_per_shard=4,
    )
    kwargs.update(overrides)
    return prepare_corpus(
        JSONLReader(src),
        tokenizer,
        tokenizer_path=tokenizer_path,
        dataset_meta={"dataset": "synthetic", "config": "sample", "split": "train"},
        args_echo={"smoke": True},
        **kwargs,
    )


def test_canonical_vocab_1024_bound_is_unchanged(tmp_path, corpus) -> None:
    """The canonical track keeps its old bound of 1024 — nothing moved for it."""
    tok = corpus["tokenizer"]
    assert tokenizer_id_bound(tok) == VOCAB_SIZE
    for dtype in ("int32", "uint16"):
        assert check_tokenizer_dtype(tok, dtype) == VOCAB_SIZE
    meta = _pack_with(
        tmp_path, tok, corpus["tok_path"], corpus["src"], dtype="uint16"
    )
    assert meta["packing"]["vocab_size_bound"] == VOCAB_SIZE
    assert meta["tokenizer"]["configured_vocab_size"] == VOCAB_SIZE
    data = _load_phase(str(tmp_path / "out"), "train")
    assert data.dtype == np.uint16
    assert int(data.max()) < VOCAB_SIZE


def test_packer_accepts_49k_vocab_tokenizer_uint16(tmp_path) -> None:
    """A 49,152-vocab tokenizer packs to uint16 shards with every id < 49,152."""
    tok, tok_path = _wide_tokenizer(tmp_path, TALOS_1B_49K_VOCAB)
    assert tokenizer_id_bound(tok) == TALOS_1B_49K_VOCAB
    assert check_tokenizer_dtype(tok, "uint16") == TALOS_1B_49K_VOCAB
    src = _synthetic_src(tmp_path)
    meta = _pack_with(tmp_path, tok, tok_path, src, dtype="uint16")

    # The bound recorded for downstream pairing is the tokenizer's, not 1024.
    assert meta["packing"]["vocab_size_bound"] == TALOS_1B_49K_VOCAB
    assert meta["tokenizer"]["configured_vocab_size"] == TALOS_1B_49K_VOCAB

    out = str(tmp_path / "out")
    docs = _read_doc_texts(src)
    for phase in ("val", "train"):
        data = _load_phase(out, phase)
        assert data.dtype == np.uint16
        assert data.ndim == 2 and data.shape[1] == SEQ
        assert int(data.min()) >= 0
        assert int(data.max()) < TALOS_1B_49K_VOCAB
        region = meta["val_region"] if phase == "val" else meta["train_region"]
        expected = _expected_stream(
            docs[region["first_doc"]: region["last_doc"] + 1],
            tok,
            meta["packing"]["eos_id"],
        )
        flat = data.reshape(-1)
        pad = meta["packing"]["pad_id"]
        # Contiguous rows + EOS placement survive the wide id range untouched.
        assert flat[flat != pad].tolist() == expected
        # The top-of-range EOS (49,149) really is stored — no 16-bit truncation.
        assert meta["packing"]["eos_id"] > 256 and meta["packing"]["eos_id"] in flat.tolist()

    # Same token stream as the int32 layout (dtype is storage only).
    meta32 = _pack_with(tmp_path, tok, tok_path, src, dtype="int32", out="out32")
    assert meta32["packing"]["vocab_size_bound"] == TALOS_1B_49K_VOCAB
    for phase in ("val", "train"):
        assert _load_phase(out, phase).tolist() == _load_phase(
            str(tmp_path / "out32"), phase
        ).tolist()
    assert meta32["counts"]["total_tokens"] == meta["counts"]["total_tokens"]


def test_real_49k_asset_packs_uint16_and_records_identity(tmp_path) -> None:
    """The committed talos_1b_49k asset packs, and its identity is recorded.

    This is the owner's 1B-corpus scenario end to end: load the asset, pack
    uint16, and leave a metadata record the corpus can later be verified
    against (sha256 / vocab / merges / encoder).
    """
    if not os.path.isfile(TALOS_1B_49K_PATH):
        pytest.skip("assets/tokenizers/talos_1b_49k.json not present")
    tok = ByteLevelBPETokenizer.from_file(TALOS_1B_49K_PATH)
    assert tok.vocab_size == TALOS_1B_49K_VOCAB
    assert tok.merge_count == TALOS_1B_49K_MERGES

    src = _synthetic_src(tmp_path)
    meta = _pack_with(tmp_path, tok, TALOS_1B_49K_PATH, src, dtype="uint16")

    tk = meta["tokenizer"]
    assert tk["sha256"] == TALOS_1B_49K_SHA256
    assert tk["sha256"] == tokenizer_file_sha256(TALOS_1B_49K_PATH)
    assert tk["vocab_size"] == TALOS_1B_49K_VOCAB
    assert tk["configured_vocab_size"] == TALOS_1B_49K_VOCAB
    assert tk["merge_count"] == TALOS_1B_49K_MERGES
    assert tk["exact_bpe"] is True
    assert meta["packing"]["vocab_size_bound"] == TALOS_1B_49K_VOCAB

    out = str(tmp_path / "out")
    # run_metadata.json carries the same provenance block.
    with open(os.path.join(out, "run_metadata.json"), encoding="utf-8") as fh:
        run_meta = json.load(fh)
    assert run_meta["tokenizer"] == tk

    data = _load_phase(out, "train")
    assert data.dtype == np.uint16
    assert int(data.max()) < TALOS_1B_49K_VOCAB

    # Downstream pairing: the manifest validates for a 49,152-vocab model ...
    manifest = load_packed_manifest(
        out,
        expected_seq=SEQ,
        expected_vocab=TALOS_1B_49K_VOCAB,
        expected_tokenizer_sha256=TALOS_1B_49K_SHA256,
    )
    assert manifest["metadata"]["tokenizer"]["sha256"] == TALOS_1B_49K_SHA256
    # ... and is refused for the canonical vocab-1024 model: the recorded bound
    # is the tokenizer's own, so a 1B corpus cannot be fed to a tiny preset.
    with pytest.raises(ValueError, match="vocab_size_bound"):
        load_packed_manifest(out, expected_vocab=VOCAB_SIZE)


@pytest.mark.parametrize(
    "bad_id", [TALOS_1B_49K_VOCAB, TALOS_1B_49K_VOCAB + 500, -1]
)
def test_packer_still_rejects_out_of_range_ids_at_wide_bound(tmp_path, bad_id) -> None:
    """Out-of-range ids still fail loudly — the guard was generalized, not weakened."""
    tok, tok_path = _wide_tokenizer(tmp_path, TALOS_1B_49K_VOCAB)
    real_encode = tok.encode

    def poisoned(text):
        return real_encode(text) + [bad_id]

    tok.encode = poisoned  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="out of range"):
        _pack_with(tmp_path, tok, tok_path, _synthetic_src(tmp_path))


def test_packer_accepts_max_legal_id_at_wide_bound(tmp_path) -> None:
    """The guard is exactly as tight as the tokenizer: id vocab_size-1 is legal."""
    tok, tok_path = _wide_tokenizer(tmp_path, TALOS_1B_49K_VOCAB)
    real_encode = tok.encode

    def with_last_id(text):
        return real_encode(text) + [TALOS_1B_49K_VOCAB - 1]

    tok.encode = with_last_id  # type: ignore[method-assign]
    meta = _pack_with(tmp_path, tok, tok_path, _synthetic_src(tmp_path))
    data = _load_phase(str(tmp_path / "out"), "train")
    assert int(data.max()) == TALOS_1B_49K_VOCAB - 1
    assert meta["packing"]["vocab_size_bound"] == TALOS_1B_49K_VOCAB


def test_uint16_capacity_boundary(tmp_path) -> None:
    """uint16 is accepted iff every id fits in 16 bits (vocab_size <= 65,536)."""
    at_limit, at_limit_path = _wide_tokenizer(
        tmp_path, UINT16_MAX_VOCAB, name="tok-65536"
    )
    assert tokenizer_id_bound(at_limit) == UINT16_MAX_VOCAB
    assert check_tokenizer_dtype(at_limit, "uint16") == UINT16_MAX_VOCAB
    # 65,536 ids => max id 65,535 => exactly representable in uint16.
    meta = _pack_with(
        tmp_path, at_limit, at_limit_path, _synthetic_src(tmp_path), dtype="uint16"
    )
    assert meta["packing"]["vocab_size_bound"] == UINT16_MAX_VOCAB
    assert _load_phase(str(tmp_path / "out"), "train").dtype == np.uint16

    over, _ = _wide_tokenizer(tmp_path, UINT16_MAX_VOCAB + 1, name="tok-65537")
    with pytest.raises(ValueError, match="cannot hold token ids"):
        check_tokenizer_dtype(over, "uint16")
    # int32 is always available for the same tokenizer.
    assert check_tokenizer_dtype(over, "int32") == UINT16_MAX_VOCAB + 1


@pytest.mark.parametrize("vocab", [UINT16_MAX_VOCAB + 1, 70_000])
def test_packer_refuses_uint16_above_65535_vocab(tmp_path, vocab) -> None:
    """> 65,535 ids: uint16 is refused with a clear message, int32 still packs."""
    tok, tok_path = _wide_tokenizer(tmp_path, vocab, name=f"tok-{vocab}")
    src = _synthetic_src(tmp_path)
    with pytest.raises(ValueError, match="cannot hold token ids"):
        _pack_with(tmp_path, tok, tok_path, src, dtype="uint16")
    # Nothing half-written: the failure happens before the first shard.
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "manifest.json"))

    meta = _pack_with(tmp_path, tok, tok_path, src, dtype="int32")
    assert meta["packing"]["dtype"] == "int32"
    assert meta["packing"]["vocab_size_bound"] == vocab
    assert int(_load_phase(str(tmp_path / "out"), "train").max()) < vocab


def test_cli_refuses_uint16_for_oversized_vocab_before_streaming(
    tmp_path, capsys
) -> None:
    """The CLI pre-flight fails (exit 2) with the real numbers — no network, no run."""
    _, tok_path = _wide_tokenizer(tmp_path, 70_000, name="tok-70000")
    out = str(tmp_path / "cli-out")
    rc = prepare_corpus_main([
        "--tokenizer-json", tok_path,
        "--target-tokens", "1000",
        "--val-tokens", "100",
        "--seq", str(SEQ),
        "--dtype", "uint16",
        "--out-dir", out,
        "--no-hf-metadata",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "cannot hold token ids" in err
    assert "70000" in err
    assert "int32" in err
    assert not os.path.exists(os.path.join(out, "manifest.json"))


def test_dtype_help_text_states_the_real_rule() -> None:
    """The stale '--dtype uint16: vocab 1024 < 65536' help text is gone."""
    help_text = make_arg_parser()._option_string_actions["--dtype"].help
    assert "vocab 1024 < 65536" not in help_text
    assert "65536" in help_text
    assert "loaded tokenizer" in help_text
    assert "int32" in help_text


def test_cli_49k_uint16_end_to_end(tmp_path, monkeypatch) -> None:
    """The owner's 1B command line runs: 49K asset + uint16 through the real CLI.

    Only the reader is swapped for a local JSONL source: the CLI is invoked with
    ``--reader datasets`` so the patched ``HuggingFaceReader`` is actually used
    (the production default is the remote parquet path, which would hit the real
    HF dataset — a network dependency CI must not have). Every other step — arg
    parsing, tokenizer load, guard, packing, manifests — is the production path.
    """
    import scripts.prepare_corpus as prepare_corpus_module

    src = _synthetic_src(tmp_path)
    monkeypatch.setattr(
        prepare_corpus_module, "HuggingFaceReader", lambda *a, **k: JSONLReader(src)
    )
    out = str(tmp_path / "cli-out")
    rc = prepare_corpus_module.main([
        "--tokenizer-json", TALOS_1B_49K_PATH,
        "--target-tokens", "2500",
        "--val-tokens", "800",
        "--seq", str(SEQ),
        "--dtype", "uint16",
        "--val-skip-docs", str(VAL_SKIP),
        "--out-dir", out,
        "--no-hf-metadata",
        "--reader", "datasets",
    ])
    assert rc == 0
    manifest = load_packed_manifest(
        out,
        expected_seq=SEQ,
        expected_vocab=TALOS_1B_49K_VOCAB,
        expected_tokenizer_sha256=TALOS_1B_49K_SHA256,
    )
    meta = manifest["metadata"]
    assert meta["args"]["dtype"] == "uint16"
    assert meta["packing"]["vocab_size_bound"] == TALOS_1B_49K_VOCAB
    assert meta["tokenizer"]["sha256"] == TALOS_1B_49K_SHA256
    assert meta["tokenizer"]["merge_count"] == TALOS_1B_49K_MERGES
    data = _load_phase(out, "train")
    assert data.dtype == np.uint16
    assert int(data.max()) < TALOS_1B_49K_VOCAB


# --------------------------------------------------------------------------- #
# load_dataset kwarg contract: the config name is forwarded as ``name=``.
#
# ``datasets.load_dataset()`` has no ``config`` parameter — the dataset
# configuration is ``name``. Forwarding ``config=`` leaked the stray kwarg into
# the builder config and killed the owner's pre-flight run with
# ``BuilderConfig ParquetConfig(name='default', ...) doesn't have a 'config' key``
# on both datasets 5.0.1 and 5.1.0, so it was never version drift.
# No network here: ``load_dataset`` is replaced by a spy.
# --------------------------------------------------------------------------- #

def test_hf_reader_forwards_config_as_load_dataset_name(monkeypatch) -> None:
    """config -> ``name=`` (never ``config=``); revision/streaming unchanged."""
    datasets = pytest.importorskip("datasets")
    from data.readers import HuggingFaceReader

    seen: list[tuple[tuple, dict]] = []

    def spy_load_dataset(*args, **kwargs):
        seen.append((args, dict(kwargs)))
        return datasets.Dataset.from_dict({"text": ["hello world"]})

    monkeypatch.setattr(datasets, "load_dataset", spy_load_dataset)
    reader = HuggingFaceReader(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        config="sample-10BT",
        revision="abc123",
    )

    assert len(seen) == 1
    args, kwargs = seen[0]
    assert args == ("HuggingFaceFW/fineweb-edu",)
    assert kwargs["name"] == "sample-10BT"
    assert "config" not in kwargs
    assert kwargs["revision"] == "abc123"
    assert kwargs["split"] == "train"
    assert kwargs["streaming"] is True
    assert [r["text"] for r in reader] == ["hello world"]


def test_hf_reader_omits_optional_load_dataset_kwargs(monkeypatch) -> None:
    """No config/revision means no name=/revision= keys at all."""
    datasets = pytest.importorskip("datasets")
    from data.readers import HuggingFaceReader

    seen: list[tuple[tuple, dict]] = []

    def spy_load_dataset(*args, **kwargs):
        seen.append((args, dict(kwargs)))
        return datasets.Dataset.from_dict({"text": ["x"]})

    monkeypatch.setattr(datasets, "load_dataset", spy_load_dataset)
    HuggingFaceReader("HuggingFaceFW/fineweb-edu", streaming=False)

    _, kwargs = seen[0]
    assert "name" not in kwargs
    assert "config" not in kwargs
    assert "revision" not in kwargs
    assert kwargs["streaming"] is False


def test_load_dataset_signature_takes_name_not_config() -> None:
    """Pin the upstream contract the fix relies on (catches a future datasets bump)."""
    datasets = pytest.importorskip("datasets")
    params = inspect.signature(datasets.load_dataset).parameters
    assert "name" in params
    assert "config" not in params

