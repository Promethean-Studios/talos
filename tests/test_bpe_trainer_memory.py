"""BPE-trainer memory-fix tests (PR: fix/bpe-trainer-memory).

Covers the six acceptance criteria for the streaming / unique-word-frequency
trainer work:

* CLI wiring: ``--max-docs`` / ``--max-chars`` / ``--num-merges`` / ``--minfreq``
  reach the trainer and produce the requested model.
* Config validation: nonsensical arguments fail fast with clear errors.
* Atomic save: a mid-save failure never leaves a corrupted destination file.
* Format compatibility: the committed legacy-format fixture (written by the
  post-fix trainer, ByteLevelBPE format version 1) loads through the repo's
  existing loader and round-trips to byte-identical output.
* Equivalence: the new trainer reproduces the expected merge sequence on the
  shared fixture corpus and matches the naive oracle; encode/decode round-trips
  stay lossless (PR #7's pre-tokenization fix must remain intact).
* Memory: training on a corpus of many repeated long documents (the case that
  OOM-killed the old trainer) stays under a measured, generous RSS bound.

All corpora are synthetic — no copyrighted data.
"""
from __future__ import annotations

import json
import os
import resource
import time
from typing import Iterable, Iterator

import pytest

from tests.fixture_corpus import FIXTURE_CORPUS
from tokenizer import tokenizer as tok_module
from tokenizer import train as train_module
from tokenizer.bpe import train_bpe, train_bpe_naive
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.train import _limit_texts, train_tokenizer
from tokenizer.vocab import TokenizerConfig

LEGACY_FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "tokenizer_vocab512_legacy.json"
)


def _words(texts: Iterable[str]) -> list:
    """Byte-words for the naive/incremental BPE-level comparison."""
    return [list(t.encode("utf-8")) for t in texts]


def _write_jsonl(path: str, docs: Iterable[str]) -> str:
    text_field = "text"
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps({text_field: doc}, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------
def test_cli_wires_max_docs_max_chars_num_merges_minfreq(tmp_path) -> None:
    """All four trainer flags are exposed and reach the trainer via the CLI."""
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(str(corpus), FIXTURE_CORPUS * 3)  # 39 docs
    out = str(tmp_path / "trained.json")

    train_module.main([
        "--corpus", str(corpus),
        "--vocab-size", "1024",
        "--num-merges", "100",
        "--max-docs", "39",
        "--max-chars", str(10 * 1024 * 1024),
        "--minfreq", "2",
        "--output", out,
    ])

    assert os.path.exists(out)
    tok = ByteLevelBPETokenizer.from_file(out)
    assert tok.merge_count == 100
    assert tok.vocab_size == 256 + 4 + 100  # bytes + 4 specials + merges


def test_limit_texts_honours_doc_and_char_budgets() -> None:
    def gen() -> Iterator[str]:
        yield from ("doc-%d" % i for i in range(100))

    assert list(_limit_texts(gen(), max_docs=5, max_chars=None)) == [
        "doc-0", "doc-1", "doc-2", "doc-3", "doc-4",
    ]
    # 4 docs * ~17 chars = ~68 chars < max_chars=30 is cut inside doc-1.
    limited = list(_limit_texts(iter(["a" * 20, "b" * 20, "c" * 20]),
                                max_docs=None, max_chars=30))
    assert limited == ["a" * 20, "b" * 20]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"minfreq": 0}, "minfreq must be >= 1"),
        ({"num_merges": -1}, "num_merges must be >= 0"),
        ({"num_merges": 2048}, "exceeds vocab budget"),
        ({"max_docs": 0}, "max_docs must be >= 1"),
        ({"max_chars": 0}, "max_chars must be >= 1"),
    ],
)
def test_bad_train_args_raise_clear_errors(kwargs, fragment) -> None:
    config = TokenizerConfig(vocab_size=1024)
    with pytest.raises(ValueError, match=fragment):
        train_tokenizer(iter([]), config, **kwargs)


def test_num_merges_below_resume_count_raises() -> None:
    config = TokenizerConfig(vocab_size=1024)
    resume = [(101, 32), (32, 99), (99, 105)]
    with pytest.raises(ValueError, match="already-learned merges"):
        train_tokenizer(iter([]), config, num_merges=2, resume_merges=resume)


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------
def test_atomic_save_success_leaves_no_temp_files(tmp_path) -> None:
    tok = train_tokenizer(FIXTURE_CORPUS, TokenizerConfig(vocab_size=512)).tokenizer
    out = str(tmp_path / "tok.json")
    tok.save(out)
    assert ByteLevelBPETokenizer.from_file(out).vocab_size == tok.vocab_size
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == []


def test_atomic_save_failure_never_corrupts_destination(
    tmp_path, monkeypatch
) -> None:
    tok = train_tokenizer(FIXTURE_CORPUS, TokenizerConfig(vocab_size=512)).tokenizer

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated disk failure mid-write")

    target = tmp_path / "tok.json"

    # Case 1: destination did not exist -> must stay absent (no partial file).
    monkeypatch.setattr(tok_module.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="mid-write"):
        tok.save(str(target))
    assert not target.exists(), "a failed save must not leave a partial file"
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == [], "failed save must clean up its temp file"

    # Case 2: a valid file already exists -> must stay fully valid (unchanged).
    monkeypatch.undo()
    tok.save(str(target))
    good_bytes = target.read_bytes()
    monkeypatch.setattr(tok_module.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="mid-write"):
        tok.save(str(target))
    assert target.read_bytes() == good_bytes
    assert ByteLevelBPETokenizer.from_file(str(target)).vocab_size == tok.vocab_size


# ---------------------------------------------------------------------------
# Format compatibility with the committed legacy fixture
# ---------------------------------------------------------------------------
def test_format_compat_legacy_fixture_roundtrips(tmp_path) -> None:
    """The committed fixture loads through the repo's loader and a fresh save by
    the new trainer is byte-identical (same serializer, same data)."""
    assert os.path.exists(LEGACY_FIXTURE), "legacy fixture must be committed"
    legacy = json.loads(open(LEGACY_FIXTURE, encoding="utf-8").read())
    assert legacy["version"] == 1 and legacy["type"] == "ByteLevelBPE"
    assert len(legacy["merges"]) == 136

    tok = ByteLevelBPETokenizer.from_file(LEGACY_FIXTURE)
    assert tok.merge_count == 136
    assert tok.vocab_size == 396  # 256 bytes + 4 specials + 136 merges

    saved = str(tmp_path / "roundtrip.json")
    tok.save(saved)
    # Byte-compare the vocab/merges sections against the fixture.
    with open(saved, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["config"] == legacy["config"]
    assert payload["merges"] == legacy["merges"]
    # Same serializer + same data => identical serialized bytes.
    assert open(saved, "rb").read() == open(LEGACY_FIXTURE, "rb").read()


def test_new_trainer_output_matches_committed_fixture(tmp_path) -> None:
    """Re-training on the fixture corpus with the NEW trainer reproduces the
    committed legacy fixture exactly — the format contract holds end to end."""
    result = train_tokenizer(FIXTURE_CORPUS, TokenizerConfig(vocab_size=512))
    assert len(result.merges) == 136
    assert result.tokenizer.vocab_size == 396
    assert result.num_words == len(FIXTURE_CORPUS)

    out = str(tmp_path / "from_new_trainer.json")
    result.tokenizer.save(out)
    assert open(out, "rb").read() == open(LEGACY_FIXTURE, "rb").read()


# ---------------------------------------------------------------------------
# Equivalence: new trainer vs naive oracle + round-trip intactness
# ---------------------------------------------------------------------------
def test_new_trainer_matches_naive_oracle_on_fixture_corpus() -> None:
    words = _words(FIXTURE_CORPUS)
    expected = train_bpe_naive(words, 136)
    assert train_bpe(words, 136) == expected

    # Full pipeline: same merge sequence as the committed fixture.
    result = train_tokenizer(FIXTURE_CORPUS, TokenizerConfig(vocab_size=512))
    assert result.merges == expected


def test_trained_tokenizer_roundtrips_fixture_corpus() -> None:
    """PR #7 pre-tokenization fix stays intact: encode/decode is lossless."""
    tok = train_tokenizer(
        FIXTURE_CORPUS, TokenizerConfig(vocab_size=512, pre_tokenize="gpt2")
    ).tokenizer
    for text in FIXTURE_CORPUS:
        assert tok.decode(tok.encode(text)) == text
    # Byte-level path (default) is lossless for the whole corpus too.
    tok2 = train_tokenizer(FIXTURE_CORPUS, TokenizerConfig(vocab_size=512)).tokenizer
    for text in FIXTURE_CORPUS:
        assert tok2.decode(tok2.encode(text)) == text


# ---------------------------------------------------------------------------
# Memory bound
# ---------------------------------------------------------------------------
_REPEATED_DOC = (
    "the quick brown fox jumps over the lazy dog while the attention "
    "transformer model trains on byte level tokenizer merges efficiently. "
    "scaling laws suggest that capability grows smoothly with compute. "
) * 50  # ~8.5 KB per document


def _repeated_docs(n: int) -> Iterator[str]:
    for _ in range(n):
        yield _REPEATED_DOC


def test_training_memory_bounded_on_repeated_long_docs() -> None:
    """1500 x ~8.5KB duplicated docs (naive path > 12M byte-ints) must train
    within a generous RSS budget thanks to unique-word frequency compression."""
    n_docs = 1500
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB
    t0 = time.perf_counter()
    result = train_tokenizer(
        _repeated_docs(n_docs),
        TokenizerConfig(vocab_size=1024),
        num_merges=200,
        minfreq=2,
    )
    elapsed = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB
    delta_kib = rss_after - rss_before

    # Generator must be consumed once and compacted: corpus is ~12.5MB of text
    # but the word table holds 1 unique word + multiplicity.
    assert result.num_words == n_docs
    assert result.num_bytes >= 10_000_000

    # Generous bound derived from measurements on the CI VM (see PR notes):
    # the actual delta is a few tens of MB; allow 384 MiB of headroom.
    assert delta_kib < 384 * 1024, (
        f"training {n_docs} duplicated long docs grew RSS by "
        f"{delta_kib / 1024:.1f} MiB — the unique-word frequency table "
        f"compression is not working"
    )
    # And it must actually finish quickly enough to be a usable unit test.
    assert elapsed < 60.0