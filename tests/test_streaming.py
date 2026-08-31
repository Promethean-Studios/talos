"""Tests for the memory-bounded streaming data path.

Covers: streaming readers (TextReader paragraphs, Parquet/HF when the optional
deps are installed), eager writer flushing, bounded dedup, the streaming
tokenized-dataset loader (pack + padded modes), and the stream-gzip download
helper.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pytest

from data.dedup import ExactDedupFilter, MinHashNearDupFilter
from data.tokenized import (
    StreamingTokenizedDataset,
    iter_documents,
    iter_token_arrays,
    resolve_shard_paths,
)
from data.writers import MAX_BUFFER_RECORDS, ShardedWriter
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.vocab import TokenizerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture()
def tokenizer():
    # Byte-fallback tokenizer: ids are byte values + specials; deterministic.
    return ByteLevelBPETokenizer(TokenizerConfig(vocab_size=1024))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def test_text_reader_streams_paragraphs(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("first para\nmore text\n\nsecond para\n\n\nthird para\n", "utf-8")
    from data.readers import TextReader

    recs = list(TextReader(str(p), split_paragraphs=True))
    assert [r["text"] for r in recs] == [
        "first para\nmore text",
        "second para",
        "third para",
    ]


def test_text_reader_whole_file_mode_unchanged(tmp_path):
    p = tmp_path / "one.txt"
    p.write_text("  hello world  \n", "utf-8")
    from data.readers import TextReader

    recs = list(TextReader(str(p)))
    assert len(recs) == 1 and recs[0]["text"] == "hello world"


def test_parquet_reader_streams_batches(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 25_000  # > default batch_size 10_000: forces multiple batches
    table = pa.table({
        "text": [f"document number {i} with some text" for i in range(n)],
        "id": list(range(n)),
    })
    path = str(tmp_path / "data.parquet")
    pq.write_table(table, path)
    from data.readers import ParquetReader

    reader = ParquetReader(path, batch_size=10_000)
    assert reader.num_records() == n
    recs = list(reader)
    assert len(recs) == n
    assert recs[0]["text"].startswith("document number 0")
    assert recs[-1]["id"] == n - 1
    # columns subset must include the text column
    with pytest.raises(ValueError):
        ParquetReader(path, text_column="text", columns=["id"])


# ---------------------------------------------------------------------------
# Writer: eager flush / bounded buffer
# ---------------------------------------------------------------------------
def test_writer_default_shard_size_bounded():
    assert ShardedWriter("/nonexistent", write_manifest=False).shard_size == 10_000


def test_writer_flushes_buffer_in_bytes_mode(tmp_path):
    """bytes-mode with an absurd shard_size must still flush at MAX_BUFFER_RECORDS."""
    out = tmp_path / "bytes-cap"
    with ShardedWriter(str(out), shard_size=10**12, size_by="bytes") as w:
        for i in range(MAX_BUFFER_RECORDS + 500):
            w.write({"text": f"doc {i}", "source": "s"})
    manifest = json.load(open(out / "manifest.json"))
    assert manifest["num_shards"] >= 1
    assert manifest["total_records"] == MAX_BUFFER_RECORDS + 500


# ---------------------------------------------------------------------------
# Dedup: bounded memory
# ---------------------------------------------------------------------------
def test_exact_dedup_max_seen_caps_growth():
    f = ExactDedupFilter(max_seen=2)
    assert f.process({"text": "alpha"}) is not None
    assert f.process({"text": "beta"}) is not None
    assert len(f._seen) == 2
    # capped: new unique docs are kept but not remembered
    assert f.process({"text": "gamma"}) is not None
    assert len(f._seen) == 2
    # previously-seen docs are still dropped
    assert f.process({"text": "alpha"}) is None
    # ...but "gamma" (never stored) would pass again
    assert f.process({"text": "gamma"}) is not None
    with pytest.raises(ValueError):
        ExactDedupFilter(max_seen=0)


def test_minhash_max_seen_bands_caps_growth():
    f = MinHashNearDupFilter(permutations=8, bands=2, rows_per_band=4, max_seen_bands=4)
    kept = 0
    for i in range(20):
        if f.process({"text": f"totally distinct document body {i} " * 5}) is not None:
            kept += 1
    assert len(f._seen_bands) <= 4
    assert kept == 20  # capped: nothing new is remembered, so nothing is dropped


# ---------------------------------------------------------------------------
# Streaming tokenized loader
# ---------------------------------------------------------------------------
def _make_shards(tmp_path, docs):
    out = tmp_path / "shards"
    with ShardedWriter(str(out), shard_size=2) as w:
        for d in docs:
            w.write({"text": d, "source": "t"})
    return str(out)


def test_resolve_shard_paths_manifest_and_dir(tmp_path):
    docs = ["alpha", "beta", "gamma", "delta"]
    shard_dir = _make_shards(tmp_path, docs)
    paths = resolve_shard_paths(shard_dir)
    assert len(paths) == 2 and all(p.endswith(".jsonl") for p in paths)
    assert resolve_shard_paths(os.path.join(shard_dir, "manifest.json")) == paths
    with pytest.raises(FileNotFoundError):
        resolve_shard_paths(str(tmp_path / "missing-dir"))


def test_iter_documents_streams_and_limits(tmp_path):
    docs = [f"doc {i}" for i in range(5)]
    shard_dir = _make_shards(tmp_path, docs)
    paths = resolve_shard_paths(shard_dir)
    assert list(iter_documents(paths)) == docs
    assert list(iter_documents(paths, limit_docs=2)) == docs[:2]


def test_iter_token_arrays_pack_mode_shapes_and_content(tokenizer, tmp_path):
    # byte-fallback tokenizer: "ab" -> [97, 98]; eos id appended per doc
    docs = ["ab", "cd", "ef", "gh", "ij"]  # 3 tokens each -> 15 tokens total
    shard_dir = _make_shards(tmp_path, docs)
    paths = resolve_shard_paths(shard_dir)
    eos = tokenizer.eos_id
    stream: list = []
    for d in docs:
        stream.extend(tokenizer.encode(d, eos=True))
    assert stream == [97, 98, eos, 99, 100, eos, 101, 102,
                      eos, 103, 104, eos, 105, 106, eos]

    # batch_size=3: the 3 full blocks exactly fill one batch. drop_last=True
    # drops the trailing partial *block* (3 leftover tokens) and any partial
    # *batch*; with 3 blocks + batch 3, flat is exactly stream[:12].
    batches = list(iter_token_arrays(paths, tokenizer, seq_len=4, batch_size=3,
                                     mode="pack", eos=True, drop_last=True))
    flat = [int(t) for b in batches for row in b for t in row]
    assert flat == stream[:12]
    assert flat[:4] == [97, 98, eos, 99]  # first block, concrete
    assert all(b.dtype == np.int32 and b.shape == (3, 4) for b in batches)

    full = list(iter_token_arrays(paths, tokenizer, seq_len=4, batch_size=3,
                                  mode="pack", eos=True, drop_last=False))
    all_tokens = [int(t) for b in full for row in b for t in row]
    # Nothing lost: final 3-token block padded to seq_len with pad_id.
    assert all_tokens == stream + [tokenizer.pad_id]
    assert full[-1].shape == (1, 4)  # short final batch kept
    assert all(b.dtype == np.int32 for b in full)


def test_iter_token_arrays_padded_mode(tokenizer, tmp_path):
    docs = ["abc", "xy"]
    shard_dir = _make_shards(tmp_path, docs)
    paths = resolve_shard_paths(shard_dir)
    pad = tokenizer.pad_id
    batches = list(iter_token_arrays(paths, tokenizer, seq_len=5, batch_size=4,
                                     mode="padded", eos=True, drop_last=False))
    assert len(batches) == 1
    b = batches[0]
    assert b.shape == (2, 5) and b.dtype == np.int32
    assert b[0].tolist() == [97, 98, 99, tokenizer.eos_id, pad]
    assert b[1].tolist() == [120, 121, tokenizer.eos_id, pad, pad]


def test_iter_token_arrays_padded_mode_truncates(tokenizer, tmp_path):
    docs = ["a" * 10]  # longer than seq_len -> truncated
    shard_dir = _make_shards(tmp_path, docs)
    paths = resolve_shard_paths(shard_dir)
    batches = list(iter_token_arrays(paths, tokenizer, seq_len=4, batch_size=1,
                                     mode="padded", eos=True))
    assert batches[0].shape == (1, 4)
    assert batches[0][0].tolist() == [97, 97, 97, 97]


def test_iter_token_arrays_matches_materialized_total(tokenizer, tmp_path):
    """Streaming must account for every token the materialized path produces."""
    rng = np.random.default_rng(0)
    docs = ["".join(chr(97 + int(x)) for x in rng.integers(0, 26, size=40))
            for _ in range(30)]
    shard_dir = _make_shards(tmp_path, docs)
    paths = resolve_shard_paths(shard_dir)
    materialized = sum(
        len(tokenizer.encode(d, eos=True)) for d in iter_documents(paths)
    )
    # drop_last=True: only the trailing partial block / partial batch are lost
    streamed_drop_last = sum(
        int(b.size)
        for b in iter_token_arrays(paths, tokenizer, seq_len=8, batch_size=4,
                                   mode="pack", eos=True, drop_last=True)
    )
    assert 0 <= materialized - streamed_drop_last < 8 * 4
    # drop_last=False: every real token is accounted for (final block padded)
    batches_all = list(iter_token_arrays(paths, tokenizer, seq_len=8, batch_size=4,
                                         mode="pack", eos=True, drop_last=False))
    streamed_all = sum(int(b.size) for b in batches_all)
    n_pad = sum(int((b == tokenizer.pad_id).sum()) for b in batches_all)
    assert streamed_all - n_pad == materialized


def test_streaming_dataset_yields_torch_batches(tokenizer, tmp_path):
    import torch

    docs = ["hello world", "second document here"]
    shard_dir = _make_shards(tmp_path, docs)
    ds = StreamingTokenizedDataset(shard_dir, tokenizer, seq_len=8, batch_size=2,
                                   mode="pack", eos=True, drop_last=False)
    batches = list(ds)
    assert len(batches) >= 1
    b = batches[0]
    assert isinstance(b, torch.Tensor)
    assert b.dtype == torch.int32
    assert b.shape[1] == 8


def test_streaming_dataset_accepts_single_jsonl_file(tokenizer, tmp_path):
    p = tmp_path / "single.jsonl"
    _write_jsonl(p, [{"text": "abc"}, {"text": "def"}])
    ds = StreamingTokenizedDataset(str(p), tokenizer, seq_len=4, batch_size=1,
                                   mode="padded", eos=False, drop_last=False)
    batches = list(ds)
    assert len(batches) == 2
    assert batches[0].shape == (1, 4)


# ---------------------------------------------------------------------------
# Download: stream-gzip helper (file:// URL so no network is needed)
# ---------------------------------------------------------------------------
def test_iter_url_lines_streams_gzip(tmp_path):
    from data.setup.download import _iter_url_lines

    payload = "\n".join(json.dumps({"text": f"doc {i}"}) for i in range(100))
    gz_path = tmp_path / "data.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        fh.write(payload)
    plain_path = tmp_path / "data.jsonl"
    plain_path.write_text(payload, "utf-8")

    lines_gz = list(_iter_url_lines("file://" + str(gz_path)))
    lines_plain = list(_iter_url_lines("file://" + str(plain_path)))
    assert len(lines_gz) == 100 and len(lines_plain) == 100
    assert json.loads(lines_gz[0])["text"] == "doc 0"
    assert json.loads(lines_plain[-1])["text"] == "doc 99"
