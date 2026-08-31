"""Regression tests for the tokenizer↔tiny-model correctness fix.

Root cause of the "letters lost, punctuation kept" bug (e.g.
``"Hello! I'm John, who are you?"`` decoding to ``"! 'm ,   ?"``):

``gpt2_pattern()`` used its letter-range table (``_GPT2_ASCII``) **outside**
a character class, so the letter-run alternative could never match. Every
English letter then matched *no* pre-tokenization alternative and was
silently dropped by ``finditer`` (only the punctuation/whitespace clauses
matched). Two things are pinned here:

1. the fixed GPT-2 pattern matches letters and round-trips English exactly
   (with merges actually exercised),
2. pre-tokenization can never silently drop bytes: even a deliberately
   non-covering pattern keeps unmatched spans (``iter_words_with_gaps``).

It also pins the tiny-model compatibility contract
(``tokenizer/model_compat.py``): a tokenizer used with the ``tiny`` model
(vocab 1024) must have every id — bytes, merges and special tokens — inside
the model's embedding table, and the model must accept encoded text ids.

All corpora are synthetic fixtures — no copyrighted data.
"""
from __future__ import annotations

import random

import pytest

from configs.presets import tiny_config, tiny_tokenizer_config
from tokenizer import model_compat
from tokenizer.pre_tokenize import iter_words_with_gaps, resolve_pattern
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.train import train_tokenizer
from tokenizer.vocab import TokenizerConfig

# ---------------------------------------------------------------------------
# Synthetic corpus + required regression examples
# ---------------------------------------------------------------------------
_CORPUS_SENTENCES = [
    "Hello! I'm John, who are you?",
    "The quick brown fox jumps over the lazy dog.",
    "Talos is an open-source foundation model project.",
    "The year is 2026 and pi is approximately 3.14159.",
    "We're going to the store, aren't we?",
    "Don't stop believing in the power of open source.",
    "I can't believe you've done this to me.",
    "The model's vocabulary must match the tokenizer to avoid losing letters.",
]

#: The examples from the bug report that MUST round-trip exactly.
REQUIRED_EXAMPLES = [
    "Hello world",
    "Hello! I'm John.",
    "Hello! I'm John, who are you?",
    "The quick brown fox jumps over the lazy dog.",
    "Talos is an open-source AI project.",
    "The year is 2026 and pi is 3.14159.",
    "  leading, trailing, and  internal   whitespace  ",
    "don't I'm we're can't",
    "punctuation: !?.,;: '\"-() [] {}",
]


def _english_corpus(n_docs: int = 200, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [rng.choice(_CORPUS_SENTENCES) for _ in range(n_docs)]


@pytest.fixture(scope="module")
def tiny_gpt2_tokenizer() -> ByteLevelBPETokenizer:
    """Tokenizer trained *with GPT-2 pre-tokenization* to fit the tiny model.

    This is the configuration that used to drop every English letter.
    """
    result = train_tokenizer(
        _english_corpus(),
        TokenizerConfig(vocab_size=1024, pre_tokenize="gpt2"),
        minfreq=2,
    )
    return result.tokenizer


@pytest.fixture(scope="module")
def tiny_byte_tokenizer() -> ByteLevelBPETokenizer:
    """Tokenizer trained pure byte-level (no pre-tokenization) to fit the tiny model."""
    result = train_tokenizer(
        _english_corpus(),
        TokenizerConfig(vocab_size=1024),
        minfreq=2,
    )
    return result.tokenizer


# ---------------------------------------------------------------------------
# The headline bug: GPT-2 pre-tokenization must keep every letter
# ---------------------------------------------------------------------------
def test_regression_roundtrip_gpt2_pretokenized_english(tiny_gpt2_tokenizer) -> None:
    """The exact reported symptom: letters vanished, punctuation survived."""
    text = "Hello! I'm John, who are you?"
    decoded = tiny_gpt2_tokenizer.decode(tiny_gpt2_tokenizer.encode(text))
    assert decoded == text


def test_gpt2_pretokenization_matches_letters_and_merges(
    tiny_gpt2_tokenizer,
) -> None:
    """Letters must be matched by the pattern AND compressed by merges.

    Before the fix, ``encode("Hello world")`` produced only the whitespace id
    (max id 32); after the fix, letters are real merge tokens (ids >= 256).
    """
    ids = tiny_gpt2_tokenizer.encode("Hello world")
    assert ids
    assert max(ids) >= 256  # merges actually applied to letter runs
    # Every id is a byte (<=255) or a learned merge token inside the vocab.
    assert all(0 <= i < tiny_gpt2_tokenizer.vocab_size for i in ids)
    # The pattern itself matches letters now (GPT-2 style: the word may own
    # its leading space). The critical invariant is *coverage*: the words
    # join back to the original text — nothing is dropped.
    pattern = resolve_pattern("gpt2")
    words = iter_words_with_gaps("Hello world", pattern)
    assert "".join(words) == "Hello world"
    assert "Hello" in words and " world" in words


@pytest.mark.parametrize("text", REQUIRED_EXAMPLES)
def test_roundtrip_required_examples_gpt2_pretokenized(
    tiny_gpt2_tokenizer, text: str
) -> None:
    assert tiny_gpt2_tokenizer.decode(tiny_gpt2_tokenizer.encode(text)) == text


@pytest.mark.parametrize("text", REQUIRED_EXAMPLES)
def test_roundtrip_required_examples_byte_level(
    tiny_byte_tokenizer, text: str
) -> None:
    assert tiny_byte_tokenizer.decode(tiny_byte_tokenizer.encode(text)) == text


def test_roundtrip_untrained_tiny_sized_tokenizer() -> None:
    """Even with zero merges, a tiny-sized tokenizer round-trips exactly."""
    tok = ByteLevelBPETokenizer(tiny_tokenizer_config())
    for text in REQUIRED_EXAMPLES:
        assert tok.decode(tok.encode(text)) == text


def test_roundtrip_gpt2_pretokenized_unicode_fuzz() -> None:
    """The fixed GPT-2 pattern must not drop any Unicode, not just English."""
    tok = ByteLevelBPETokenizer(
        TokenizerConfig(vocab_size=1024, pre_tokenize="gpt2")
    )
    texts = [
        "café naïve résumé",
        "你好世界 — CJK text",
        "Привет мир",
        "مرحبا بالعالم",
        "emoji 🚀🔥 and math ∑ ∫ ≤ ≥",
        "tabs\tand\nnewlines\r\nmixed",
        "NUL\x00 inside",
        "mixed 123 numbers and 3.14 floats",
    ]
    for text in texts:
        assert tok.decode(tok.encode(text)) == text, text


# ---------------------------------------------------------------------------
# Pre-tokenization must never silently drop bytes (the bug class)
# ---------------------------------------------------------------------------
def test_non_covering_pattern_keeps_gaps() -> None:
    """A pattern matching only digits must NOT lose the other characters."""
    pattern = resolve_pattern(r"[0-9]+")
    tok = ByteLevelBPETokenizer(
        TokenizerConfig(vocab_size=1024, pre_tokenize=r"[0-9]+")
    )
    for text in ("abc 123 def", "Hello! I'm John, who are you?", "x9y"):
        # The letters survive as gap words even though the pattern ignores them.
        assert tok.decode(tok.encode(text)) == text, text
    words = iter_words_with_gaps("abc 123 def", pattern)
    # Coverage invariant: joined words reconstruct the text — no dropped bytes.
    assert "".join(words) == "abc 123 def"
    assert "123" in words  # the pattern-matched span
    assert any(w.startswith("abc") for w in words)  # kept as a gap word


def test_simple_preset_compiles_and_roundtrips() -> None:
    """The ``simple`` preset must compile under stdlib ``re`` and stay lossless.

    (It used ``\\p{...}`` which stdlib ``re`` rejects, so it could never be
    used at all.)
    """
    assert resolve_pattern("simple") is not None
    tok = ByteLevelBPETokenizer(
        TokenizerConfig(vocab_size=1024, pre_tokenize="simple")
    )
    for text in REQUIRED_EXAMPLES[:5]:
        assert tok.decode(tok.encode(text)) == text


# ---------------------------------------------------------------------------
# Tiny-model compatibility: the tokenizer must fit the model's vocab
# ---------------------------------------------------------------------------
def test_tiny_tokenizer_config_fits_tiny_model() -> None:
    model_vocab = tiny_config().vocab_size
    assert model_vocab == 1024
    cfg = tiny_tokenizer_config()
    assert cfg.vocab_size == model_vocab
    # 256 base bytes + 4 specials + merge slots, all inside the model table.
    assert cfg.max_merges() == model_vocab - 256 - 4


def test_trained_tiny_tokenizers_fit_tiny_model(
    tiny_gpt2_tokenizer, tiny_byte_tokenizer
) -> None:
    model_vocab = tiny_config().vocab_size
    for tok in (tiny_gpt2_tokenizer, tiny_byte_tokenizer):
        assert tok.vocab_size <= model_vocab
        mapping = model_compat.map_to_model_vocab(tok, model_vocab)
        assert mapping.fits
        # Special ids sit at the top of the tokenizer vocab, inside the model.
        assert all(0 <= v < model_vocab for v in tok.special_id_map().values())


def test_default_vocab_tokenizer_hazard_with_tiny_model() -> None:
    """Documents the hazard that ``tiny_tokenizer_config()`` exists to prevent.

    The tokenizer default is ``vocab_size=32768`` while the tiny model holds
    only 1024 embedding rows. An *untrained* default-config tokenizer uses just
    260 ids (bytes + specials), but its special tokens sit at the top of the
    configured 32768 vocab (ids 32764..32767) — far outside the model's table,
    so ``encode(..., bos=True)`` ids cannot be embedded. Once a default-config
    tokenizer is actually trained past 764 merges its real vocab exceeds 1024
    and the compat contract raises loudly (never silently mis-map).
    """
    tok = ByteLevelBPETokenizer(TokenizerConfig())  # default vocab_size=32768
    assert tok.vocab_size == 256 + 4  # untrained: bytes + specials only
    assert all(v >= 1024 for v in tok.special_id_map().values())

    big = ByteLevelBPETokenizer(
        TokenizerConfig(vocab_size=32768), [(0, i) for i in range(1, 801)]
    )  # 800 merges -> actual vocab 256+4+800 = 1060 > 1024
    assert big.vocab_size == 1060
    with pytest.raises(ValueError):
        model_compat.map_to_model_vocab(big, tiny_config().vocab_size)


# ---------------------------------------------------------------------------
# Model path: encoded text ids flow through the tiny model and back
# ---------------------------------------------------------------------------
def test_tiny_model_encodes_decodes_required_examples(tiny_gpt2_tokenizer) -> None:
    """Encode with the tokenizer, run the ids through the tiny model, decode.

    This pins the full prototype path: every id (bytes, merges, specials)
    must be a valid embedding row of the tiny model, the forward pass must
    accept them, and decode must return the original text.
    """
    import torch

    from model import TalosGPT

    model_vocab = tiny_config().vocab_size
    model = TalosGPT(tiny_config().derive())
    assert model.num_parameters() == 254_272

    for text in REQUIRED_EXAMPLES:
        ids = tiny_gpt2_tokenizer.encode(text, bos=True, eos=True)
        # Every id — including the added BOS/EOS — must be an embedding row.
        assert all(0 <= i < model_vocab for i in ids)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, len(ids), model_vocab)
        # Decoding the *input* ids (what training data / generation builds on)
        # recovers the text exactly; specials are skipped by default.
        assert tiny_gpt2_tokenizer.decode(ids) == text


def test_tiny_model_generation_tokens_decode_losslessly(tiny_gpt2_tokenizer) -> None:
    """Ids sampled from the model's vocab always decode to valid text."""
    import torch

    from model import TalosGPT

    model = TalosGPT(tiny_config().derive())
    torch.manual_seed(0)
    prompt = torch.tensor(
        [[tiny_gpt2_tokenizer.bos_id] + tiny_gpt2_tokenizer.encode("Hi")],
        dtype=torch.long,
    )
    with torch.no_grad():
        logits, _ = model(prompt)
    next_id = int(torch.argmax(logits[0, -1]))
    # A model-side id is decodable: within the tokenizer's byte/merge tables.
    assert 0 <= next_id < tiny_gpt2_tokenizer.vocab_size
    assert tiny_gpt2_tokenizer.decode(prompt[0].tolist() + [next_id]).startswith("Hi")
