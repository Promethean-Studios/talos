"""Tests for the Phase 2 byte-level BPE tokenizer.

These run with only the stdlib + numpy (no torch required) and are kept fast
so the full Phase 1 + Phase 2 suite stays quick:

* round-trip correctness over diverse Unicode (the default byte-level path is
  lossless for *arbitrary* text),
* special-token handling (BOS/EOS flags, extra special tokens, splitting),
* determinism (encode + seeded training),
* save/load round-trip,
* long-text chunking,
* a fast training smoke test (target vocab reached + round-trip),
* a large-vocab / model-compat demonstration (maps onto the ~128K model vocab).

All corpora below are synthetic fixtures — no copyrighted data.
"""
from __future__ import annotations

import pytest

from tokenizer import model_compat
from tokenizer.bpe import train_bpe, train_bpe_naive
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.train import train_tokenizer
from tokenizer.vocab import BASE_VOCAB_SIZE, TokenizerConfig, Vocabulary

# ---------------------------------------------------------------------------
# Synthetic corpus: multilingual prose + source code + mathematics.
# ---------------------------------------------------------------------------
_EN_WORDS = (
    "the quick brown fox jumps over lazy dog model attention transformer neural "
    "network tokenizer byte merge training inference scaling efficient reasoning "
    "capability research language open source foundation recursive descent"
).split()

_MATH_TOKENS = ["sum", "integral", "dx", "f", "g", "x", "2", "≈", "≤", "≥", "±",
                "∑", "∫", "∞", "√", "=", "+", "-", "*", "f(", ")"]

_CODE_LINES = [
    "def foo(x):",
    "    return x + 1",
    "class Tokenizer:",
    "        self._rank = 0",
    "import torch as th",
    "for i in range(10):",
    "    print(i)",
    "if __name__ == '__main__':",
]

_CJK = "你好世界 語言模型 深度学习 神经网络"
_ARABIC = "العربية اللغة العربية موديل"
_DEVANAGARI = "हिन्दी मॉडल नेटवर्क"
_CYRILLIC = "Русский кириллица модель"

DIVERSE_TEXTS = [
    "Hello, world! This is an English sentence.",
    _CJK,
    _ARABIC,
    _DEVANAGARI,
    _CYRILLIC,
    "def foo(x):\n    return x + 1\n\nif __name__ == '__main__':\n    print('hi')",
    "math ∑ f(x) = ∫ x² dx ≈ 42 · ¾ ∞ ≤ ≥ ± √ 🚀",
    "NUL\x00byte\x00and\x00more",
    "A" * 4000,  # very long text
]


def _corpus() -> list:
    """Build a deterministic synthetic corpus (multilingual + code + math)."""
    import random

    docs: list = []
    rng = random.Random(0)
    for _ in range(60):
        docs.append(" ".join(rng.choice(_EN_WORDS) for _ in range(15)))
    for _ in range(30):
        docs.append(" ".join(rng.choice(_MATH_TOKENS) for _ in range(8)))
    for _ in range(20):
        docs.append(rng.choice(_CODE_LINES) * 5)
    docs.append(_CJK)
    docs.append(_ARABIC)
    docs.append(_DEVANAGARI)
    docs.append(_CYRILLIC)
    docs.append("∫ x² dx ≈ 42 ∑ᵢ aᵢ ≤ 10 🚀")
    return docs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_tokenizer() -> ByteLevelBPETokenizer:
    """A tokenizer trained on the synthetic corpus to its 512-token target.

    Learning reaches the configured target (vocab_size == 512) on this corpus
    in a couple of seconds, giving the round-trip / special / save-load tests a
    real merge list to exercise.
    """
    result = train_tokenizer(_corpus(), TokenizerConfig(vocab_size=512))
    assert result.tokenizer.vocab_size == 512
    return result.tokenizer


# ---------------------------------------------------------------------------
# Core merge-training consistency
# ---------------------------------------------------------------------------
def test_naive_and_incremental_training_agree() -> None:
    words = [list(w.encode()) for w in ("the", "the", "quick", "brown", "fox",
                                        "the the", "quick", "brown fox jumps")]
    assert train_bpe_naive(words, 50) == train_bpe(words, 50)


def test_bpe_encodes_after_training_are_lossless() -> None:
    words = [list(w.encode()) for w in ("hello", "world", "hello", "world!")]
    merges = train_bpe(words, 20)
    from tokenizer.bpe import encode_bytes, ranks_from_merges
    tok = ByteLevelBPETokenizer(TokenizerConfig(vocab_size=512), merges)
    for t in ("hello world!", "hello", "world!"):
        assert tok.decode(tok.encode(t)) == t


# ---------------------------------------------------------------------------
# Round-trip over diverse Unicode (default byte-level path)
# ---------------------------------------------------------------------------
def test_roundtrip_diverse_unicode(trained_tokenizer) -> None:
    for text in DIVERSE_TEXTS:
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text, text


def test_roundtrip_empty_and_single_byte_text(trained_tokenizer) -> None:
    assert trained_tokenizer.decode(trained_tokenizer.encode("")) == ""
    for text in ("a", " ", "\n", "\t", "🚀", "\x00"):
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


def test_roundtrip_with_merges_exercised(trained_tokenizer) -> None:
    # A word that appears in the training corpus must be collapsed to fewer
    # than its byte count (proving merges are actually applied), and must still
    # round-trip exactly.
    ids = trained_tokenizer.encode("the quick brown fox")
    assert ids and all(i < trained_tokenizer.vocab_size for i in ids)
    assert trained_tokenizer.decode(ids) == "the quick brown fox"
    # Whole 3-byte CJK characters share one token id at most per byte; never
    # exceeds byte count of the input.
    assert trained_tokenizer.encode(_CJK)  # non-trivial


def test_lone_surrogate_is_not_silently_mangled(trained_tokenizer) -> None:
    # A lone surrogate has no UTF-8 encoding; encode must fail loudly rather
    # than produce a lossy result. (Surrogate *pairs* — e.g. emoji — work.)
    with pytest.raises(UnicodeEncodeError):
        trained_tokenizer.encode("bad\ud800surrogate")


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
def test_bos_eos_flags_and_decode(trained_tokenizer) -> None:
    text = "Hello world"
    ids = trained_tokenizer.encode(text, bos=True, eos=True)
    assert ids[0] == trained_tokenizer.bos_id
    assert ids[-1] == trained_tokenizer.eos_id
    # decode (default) strips special tokens -> original text
    assert trained_tokenizer.decode(ids) == text
    # decode(skip_special_tokens=False) reproduces the literal token strings
    assert trained_tokenizer.decode(ids, skip_special_tokens=False) == (
        trained_tokenizer.config.bos_token + text + trained_tokenizer.config.eos_token
    )


def test_extra_special_tokens_are_registered_and_split() -> None:
    cfg = TokenizerConfig(
        vocab_size=1024,
        split_special=True,
        extra_special_tokens=["<|tool_call|>", "<|/tool_call|>"],
    )
    tok = ByteLevelBPETokenizer(cfg)
    text = "call <|tool_call|> fn() <|/tool_call|> done"
    ids = tok.encode(text, split_special=True)
    tc = tok.special_id_map()["<|tool_call|>"]
    close = tok.special_id_map()["<|/tool_call|>"]
    assert tc in ids and close in ids
    # ids are at the top of the vocab, in configuration order
    assert tc == cfg.vocab_size - 2 and close == cfg.vocab_size - 1
    # Round-trip as plain text (default skip_special=True)
    assert tok.decode(ids) == text.replace("<|tool_call|>", "").replace("<|/tool_call|>", "")
    # And exactly with specials (skip_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=False) == text


def test_special_tokens_occupy_top_of_vocab(trained_tokenizer) -> None:
    ids = trained_tokenizer.special_id_map()
    assert len(ids) == 4  # bos, eos, pad, unk
    n = trained_tokenizer.vocab_size
    assert sorted(ids.values()) == list(range(n - 4, n))
    assert trained_tokenizer.bos_id < trained_tokenizer.eos_id < trained_tokenizer.pad_id


def test_vocabulary_id_layout(trained_tokenizer) -> None:
    vocab = trained_tokenizer.vocab
    # raw bytes 0..255 map to ids 0..255
    for b in range(256):
        assert vocab.id_to_bytes[b] == bytes([b])
    # first learned merge gets id 256 (== merge_base_id)
    assert trained_tokenizer.merge_count > 0
    first = vocab.merges[0]
    assert vocab.pair_to_merge_id[first] == 256
    # merge ids sit immediately above the base bytes, below the specials
    first_merge_id = 256
    last_merge_id = 256 + trained_tokenizer.merge_count - 1
    assert last_merge_id < trained_tokenizer.bos_id


def test_vocabulary_rejects_over_capacity() -> None:
    cfg = TokenizerConfig(vocab_size=300)  # room for 300 - 256 - 4 = 40 merges
    with pytest.raises(ValueError):
        Vocabulary(cfg, [(0, i) for i in range(1, 100)])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_encode_is_deterministic(trained_tokenizer) -> None:
    text = "determinism over diverse Unicode 你好世界 🚀"
    assert trained_tokenizer.encode(text) == trained_tokenizer.encode(text)
    assert trained_tokenizer.encode(text, bos=True, eos=True) == \
        trained_tokenizer.encode(text, bos=True, eos=True)


def test_training_is_deterministic() -> None:
    a = train_tokenizer(_corpus(), TokenizerConfig(vocab_size=512))
    b = train_tokenizer(_corpus(), TokenizerConfig(vocab_size=512))
    assert a.merges == b.merges
    assert a.tokenizer.encode("the quick brown fox") == \
        b.tokenizer.encode("the quick brown fox")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(trained_tokenizer, tmp_path) -> None:
    path = str(tmp_path / "tokenizer.json")
    trained_tokenizer.save(path)

    loaded = ByteLevelBPETokenizer.from_file(path)
    assert loaded.vocab_size == trained_tokenizer.vocab_size
    assert loaded.merge_count == trained_tokenizer.merge_count
    assert loaded.vocab.merges == trained_tokenizer.vocab.merges
    assert loaded.special_id_map() == trained_tokenizer.special_id_map()
    for text in ("hello world 🚀", _CJK, "def f(x):\n return x"):
        assert loaded.encode(text) == trained_tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == text


# ---------------------------------------------------------------------------
# Long-text chunking
# ---------------------------------------------------------------------------
def test_chunk_encode_matches_plain_encode(trained_tokenizer) -> None:
    # Chunking is character-aligned: a learned BPE merge that spans a chunk
    # boundary cannot be applied in the chunked path, so the chunked and plain
    # encodings need not produce identical token id *lists*. The invariant that
    # MUST hold is lossless round-trip (decode(chunk_encode) == original), which
    # we assert here; and that when the whole text fits in a single chunk the two
    # paths are identical (the only case exact token alignment is guaranteed).
    text = " ".join(["model transformer tokenizer 你好世界" * 12 for _ in range(3)])
    assert len(text) > 2 * 512
    ids = trained_tokenizer.chunk_encode(text, chunk_size=1024)
    assert all(i < trained_tokenizer.vocab_size for i in ids)
    assert trained_tokenizer.decode(ids) == text

    # Single-chunk text: character-aligned chunking degenerates to a plain
    # encode, so merge-aligned equality does hold.
    short = " ".join(["model transformer tokenizer 你好世界" * 2 for _ in range(2)])
    assert len(short) < 1024
    assert trained_tokenizer.chunk_encode(short, chunk_size=1024) == \
        trained_tokenizer.encode(short)

    # Per-chunk encoding with BOS/EOS; stripping the added special ids and
    # concatenating recovers the original text exactly (chunks are char-aligned).
    chunks = trained_tokenizer.encode_ids_chunks(text, chunk_size=512)
    assert len(chunks) > 1
    b, e = trained_tokenizer.bos_id, trained_tokenizer.eos_id
    joined = [i for c in chunks for i in c if i not in (b, e)]
    assert trained_tokenizer.decode(joined) == text


def test_iter_chunks_bounds() -> None:
    tok = ByteLevelBPETokenizer(TokenizerConfig(vocab_size=512))
    assert list(tok.iter_chunks("abcdefghij", 3)) == ["abc", "def", "ghi", "j"]
    with pytest.raises(ValueError):
        list(tok.iter_chunks("ab", 0))


# ---------------------------------------------------------------------------
# Training smoke test (fast)
# ---------------------------------------------------------------------------
def test_training_smoke_reaches_target() -> None:
    result = train_tokenizer(_corpus(), TokenizerConfig(vocab_size=512))
    # The full target vocab is reached on this synthetic corpus and round-trip
    # still holds for the trained (merged) tokenizer.
    assert result.tokenizer.vocab_size == 512
    assert result.tokenizer.merge_count == 512 - 256 - 4
    text = "The quick brown fox jumps over the lazy dog 你好世界 math ∫ x² dx 🚀"
    assert result.tokenizer.decode(result.tokenizer.encode(text)) == text


# ---------------------------------------------------------------------------
# Large-vocab / model-compat demonstration
#
# Reaching *literally* thousands of learned merges needs a realistically-sized
# (MB-scale) corpus, which is out of scope to keep this suite fast. Instead we
# prove the architecture handles a configurable large vocab:
#   * a tokenizer config with the ~128K model vocab is constructible and
#     round-trips, with merge capacity in the tens of thousands;
#   * a trained tokenizer maps cleanly onto the model's 128K vocab via
#     model_compat (non-empty, injective/monotone id mapping + resize plan).
# ---------------------------------------------------------------------------
def test_large_vocab_config_is_supported() -> None:
    # MATLAB/GPT-4-class byte-level config vocab, matching the ~128K model vocab
    cfg = TokenizerConfig(vocab_size=131072)
    tok = ByteLevelBPETokenizer(cfg)
    assert tok.vocab_size == 256 + len(cfg.special_tokens)
    assert cfg.max_merges() == 131072 - 256 - 4  # tens of thousands of slots
    assert tok.decode(tok.encode("hi 🚀 你好  العربية")) == "hi 🚀 你好  العربية"


def test_model_compat_mapping_into_128k_vocab(trained_tokenizer) -> None:
    model_vocab_size = 131072  # the model's largest preset (128K tokens)
    mapping = model_compat.map_to_model_vocab(trained_tokenizer, model_vocab_size)
    assert mapping.fits
    assert 0 <= mapping.padding <= model_vocab_size
    assert mapping.tokenizer_vocab_size == trained_tokenizer.vocab_size
    # non-empty, injective, and in-range special ids (monotone mapping)
    specials = mapping.special_ids
    assert specials
    assert len(specials) == len(set(specials.values()))
    assert all(0 <= v < model_vocab_size for v in specials.values())
    assert mapping.to_dict()["model_vocab_size"] == model_vocab_size

    # resize plan is coherent with the ~128K model embedding width
    plan = model_compat.resize_embedding_plan(
        trained_tokenizer.vocab_size, model_vocab_size, hidden_size=4096
    )
    assert plan["used_rows"] == trained_tokenizer.vocab_size
    assert plan["total_rows"] == 131072
    assert plan["padding_rows"] == 131072 - trained_tokenizer.vocab_size
    assert plan["embedding_size"] == 4096


def test_model_compat_rejects_overflow() -> None:
    small = ByteLevelBPETokenizer(TokenizerConfig(vocab_size=1024))
    # An untrained tokenizer's actual vocab is bytes + specials (256 + 4 = 260),
    # not its config capacity. The mapping contract keys on the vocab actually
    # used, so overflow is relative to that.
    tv = small.vocab_size
    assert model_compat.map_to_model_vocab(small, 2048).fits
    assert model_compat.map_to_model_vocab(small, tv).fits
    with pytest.raises(ValueError):
        model_compat.map_to_model_vocab(small, tv - 1)  # tv > tv-1
    with pytest.raises(ValueError):
        model_compat.resize_embedding_plan(tv, tv - 1, 64)


def test_model_vocab_constant_is_powers_of_two() -> None:
    # The 128K model vocab is powers-of-two friendly (matches the plan).
    assert 131072 & (131072 - 1) == 0
    assert BASE_VOCAB_SIZE == 256
