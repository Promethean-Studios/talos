"""The 49,152-vocab tokenizer (``talos_1b_49k``) — the **paired primary** 1B asset.

``talos_1b_49k`` was built (2026-11-05) so the owner's vocab-upgrade decision
("raise the 1B embedding to 49,152 → ≈1,036M params") could be made on measured
numbers. **The owner made that decision on 2026-11-07**: the manifest now
registers it for ``"presets": ["talos_1b"]``, and ``talos_1b``'s model config
carries vocab 49,152 (1,036,060,672 params). Its two siblings —
``talos_1b_21l`` / ``talos_1b_spec16`` — keep vocab 32,000 and the
``talos_1b_32k`` asset; every cross-pairing (49K with a 32,000-row model, 32K
with the 49,152-row primary, the canonical 1024 anywhere on the 1B track) is
refused by both guards.

Measured compression: 4.6525 chars/token on the 1,000-row FineWeb-Edu held-out
(4.6411 on the v1-comparable 200-row prefix) vs the 32K v2's 4.5195/4.5039 —
~3% fewer tokens on identical text.
"""
from __future__ import annotations

import json
import os

import pytest

from tokenizer.assets import (
    ASSET_DIR,
    MANIFEST_FILENAME,
    assert_asset_preset_pairing,
    assert_tokenizer_vocab_pairs,
    asset_for_preset,
    get_asset,
    load_asset,
    preset_asset_map,
)
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256

ASSET_NAME = "talos_1b_49k"
VOCAB_49K = 49_152
MERGES_49K = 48_892
SHA_49K = "c7d26acebad67be04af9b16cb72a812156cf3a708121a8ee9bf794d6a95c8684"
PRIMARY_32K_SHA = "9777034aec31f864ddc2f14114556d8042ab91df92f08720980d1e45a4f27863"

DIVERSE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A tokenizer must be lossless on arbitrary UTF-8 byte sequences.",
    "Numbers 0123456789, punctuation !?.,;:'\"()[]{}-- and math 3+4=7 <= 8.",
    "Unicode: naïve café, 日本語のテキスト, Ελληνικά, русский текст, עברית.",
    "Emoji and symbols: 🚀🧪✨ 🏳️‍🌈 👩‍👩‍👧‍👦 → ≤ ≥ ≈ ∑ ∫ ½ ⅓",
    "Code:\n    def f(x: int) -> int:\n        return x * 2  # comment\n",
    "   leading and trailing whitespace   \n\n\ttabs\tand newlines\n",
    "<|endoftext|> literal special-token text stays lossless",
    "".join(chr(c) for c in range(0x20, 0x7F)),
    "Mixed: A1_b2-C3.d4/e5\\f6|g7~h8`i9@j0#k$l%m^n&o*p(q)r[s]t{u}v",
]


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPETokenizer:
    return load_asset(ASSET_NAME)


@pytest.fixture(scope="module")
def asset():
    return get_asset(ASSET_NAME)


# --------------------------------------------------------------------------
# the asset itself
# --------------------------------------------------------------------------
def test_manifest_pins_the_49k_asset(asset) -> None:
    assert asset.exists(), asset.path
    assert tokenizer_file_sha256(asset.path) == asset.sha256 == SHA_49K
    assert asset.vocab_size == VOCAB_49K and asset.merges == MERGES_49K
    with open(os.path.join(ASSET_DIR, MANIFEST_FILENAME), encoding="utf-8") as fh:
        raw = json.load(fh)
    entry = raw["assets"][ASSET_NAME]
    assert entry["sha256"] == SHA_49K
    # Owner decision 2026-11-07: this asset is now the PAIRED primary 1B
    # tokenizer. The asset file itself is unchanged (same sha256 pin above).
    assert entry["presets"] == ["talos_1b"]
    assert entry["paired_with_any_model"] is True
    assert "asset_only" not in entry and "why_unpaired" not in entry
    assert entry["training_sample"]["committed"] is False
    assert entry["training_sample"]["num_rows"] == 16_000
    assert entry["training_sample"]["sha256"] == (
        "2bafe0fb63999bf4f9c69b4f064fcaa6c692c24db29b5f03a9d235e79d815033"
    )


def test_49k_vocab_is_exact(tokenizer) -> None:
    assert tokenizer.vocab_size == VOCAB_49K
    assert tokenizer.merge_count == MERGES_49K
    assert 256 + MERGES_49K + 4 == VOCAB_49K  # 256 bytes + merges + 4 specials
    assert len(tokenizer.vocab.id_to_bytes) == 256 + MERGES_49K  # specials live beside it


def test_49k_special_ids_are_sane(tokenizer) -> None:
    ids = tokenizer.special_id_map()
    assert ids == {
        "<|beginoftext|>": VOCAB_49K - 4,
        "<|endoftext|>": VOCAB_49K - 3,
        "<|pad|>": VOCAB_49K - 2,
        "<|unk|>": VOCAB_49K - 1,
    }
    assert len(set(ids.values())) == 4
    assert max(ids.values()) < tokenizer.vocab_size


def test_49k_round_trip_is_lossless(tokenizer) -> None:
    for text in DIVERSE_TEXTS:
        assert tokenizer.decode(tokenizer.encode(text)) == text, repr(text[:40])


def test_32k_primary_asset_is_v2_retrained() -> None:
    """The 32K primary is the v2 asset retrained on the 16,000-row sample (held-out gate passed)."""
    path = os.path.join(ASSET_DIR, "talos_1b_32k.json")
    assert tokenizer_file_sha256(path) == PRIMARY_32K_SHA
    with open(os.path.join(ASSET_DIR, MANIFEST_FILENAME), encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["version"] == 1  # top-level = manifest SCHEMA version
    assert raw["assets"]["talos_1b_32k"]["sha256"] == PRIMARY_32K_SHA
    assert raw["assets"]["talos_1b_32k"]["training_sample"]["num_rows"] == 16000


# --------------------------------------------------------------------------
# the pairing guards: 49K pairs ONLY talos_1b, and every other pairing is refused
# --------------------------------------------------------------------------
def test_49k_is_paired_with_talos_1b_only() -> None:
    """Owner decision 2026-11-07: 49K owns ``talos_1b``; 32K keeps the siblings."""
    mapping = preset_asset_map()
    assert mapping["talos_1b"] == ASSET_NAME
    assert asset_for_preset("talos_1b") == ASSET_NAME
    for preset in ("talos_1b_21l", "talos_1b_spec16"):
        assert asset_for_preset(preset) == "talos_1b_32k"
    # exactly one asset owns each preset (the manifest cross-claims nothing)
    assert sum(1 for v in mapping.values() if v == ASSET_NAME) == 1


def test_49k_asset_is_accepted_for_talos_1b() -> None:
    assert_asset_preset_pairing(ASSET_NAME, "talos_1b")  # must not raise
    assert_tokenizer_vocab_pairs(VOCAB_49K, "talos_1b")  # must not raise


@pytest.mark.parametrize("preset", ["talos_1b_21l", "talos_1b_spec16"])
def test_49k_asset_is_refused_for_the_32k_siblings(preset: str) -> None:
    """The two 32,000-vocab presets must never load the 49K asset."""
    with pytest.raises(ValueError):
        assert_asset_preset_pairing(ASSET_NAME, preset)


@pytest.mark.parametrize("preset", ["tiny", "tiny_1m", "tiny_10m", "tiny_100m"])
def test_49k_asset_is_refused_for_canonical_presets(preset: str) -> None:
    with pytest.raises(ValueError, match="trains its tokenizer inside the run"):
        assert_asset_preset_pairing(ASSET_NAME, preset)


def test_49k_vocab_is_refused_by_the_realized_vocab_guard() -> None:
    """49,152 pairs exactly with ``talos_1b`` and with nothing else."""
    assert_tokenizer_vocab_pairs(VOCAB_49K, "talos_1b")  # the one valid pairing
    # The 32,000-row siblings refuse it...
    for preset in ("talos_1b_21l", "talos_1b_spec16"):
        with pytest.raises(ValueError, match="does not match"):
            assert_tokenizer_vocab_pairs(VOCAB_49K, preset)
    # ...and the primary now refuses the 32K asset's vocab and the 1024 one.
    with pytest.raises(ValueError, match="does not match"):
        assert_tokenizer_vocab_pairs(32_000, "talos_1b")
    with pytest.raises(ValueError, match="does not match"):
        assert_tokenizer_vocab_pairs(1024, "talos_1b")
    # The canonical 1024 tokenizer is refused everywhere on the 1B track.
    for preset in ("talos_1b", "talos_1b_21l", "talos_1b_spec16"):
        with pytest.raises(ValueError, match="does not match"):
            assert_tokenizer_vocab_pairs(1024, preset)


def test_49k_config_is_a_plain_tokenizer_config() -> None:
    """The asset's own layout, and the model-paired factory agreeing with it."""
    from tokenizer.vocab import TokenizerConfig

    cfg = TokenizerConfig(vocab_size=VOCAB_49K)
    assert cfg.vocab_size == VOCAB_49K and cfg.max_merges() == MERGES_49K
    # The model-paired factory now produces the same 49,152 contract for
    # ``talos_1b`` (and 32,000 for its two siblings).
    from configs.presets import (
        talos_1b_21l_tokenizer_config,
        talos_1b_spec16_tokenizer_config,
        talos_1b_tokenizer_config,
    )

    assert talos_1b_tokenizer_config().vocab_size == VOCAB_49K
    assert talos_1b_21l_tokenizer_config().vocab_size == 32_000
    assert talos_1b_spec16_tokenizer_config().vocab_size == 32_000


def test_trainer_refuses_a_third_vocab_size() -> None:
    from scripts.train_tokenizer_32k import merges_for_vocab

    assert merges_for_vocab(32_000) == 31_740
    assert merges_for_vocab(VOCAB_49K) == MERGES_49K
    with pytest.raises(ValueError):
        merges_for_vocab(40_000)
