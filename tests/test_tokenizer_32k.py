"""W3 — the dedicated 32,000-vocab tokenizer asset for the Talos 1B presets.

Covers the asset itself (vocab exactly 32,000, sane special ids, lossless
round-trip, sha256-pinned manifest), its pairing with ``talos_1b`` /
``talos_1b_spec16`` (including a meta-device embedding-shape check — the 1B
model is NEVER instantiated for real on this box), the mismatch guards, the
correct greedy encoder, and the frozen canonical 1024 tokenizer.
"""
from __future__ import annotations

import json
import random

import pytest

from configs.canonical import TALOS_1B_VOCAB_SIZE
from configs.presets import ALL_PRESETS, preset_tokenizer_config
from tokenizer.assets import (
    ASSET_DIR,
    MANIFEST_FILENAME,
    asset_for_preset,
    assert_asset_preset_pairing,
    assert_tokenizer_vocab_pairs,
    get_asset,
    load_asset,
    preset_asset_map,
)
from tokenizer.bpe import encode_bytes_exact
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256
from tokenizer.vocab import TokenizerConfig

ASSET_NAME = "talos_1b_32k"
PRESETS_1B = ("talos_1b", "talos_1b_spec16")

REPRESENTATIVE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are sequence models trained with next-token prediction.",
    "Numbers 0123456789 and punctuation !?.,;:'\"()[]{}-- and math 3+4=7.",
    "Unicode: naïve café, 日本語のテキスト, Ελληνικά, русский текст.",
    "Emoji and symbols: 🚀🧪✨ → ≤ ≥ ≈ ∑ ∫",
    "Code:\n    def f(x: int) -> int:\n        return x * 2  # comment\n",
    "   leading and trailing whitespace   \n\n\ttabs\tand newlines\n",
    "<|endoftext|> literal special-token text stays lossless",
]


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPETokenizer:
    return load_asset(ASSET_NAME)


@pytest.fixture(scope="module")
def asset():
    return get_asset(ASSET_NAME)


# --------------------------------------------------------------------------
# asset + manifest
# --------------------------------------------------------------------------
def test_manifest_pins_the_committed_asset(asset) -> None:
    """The asset file exists and its sha256 matches MANIFEST.json."""
    assert asset.exists(), asset.path
    assert tokenizer_file_sha256(asset.path) == asset.sha256
    with open(f"{ASSET_DIR}/{MANIFEST_FILENAME}", encoding="utf-8") as fh:
        raw = json.load(fh)
    entry = raw["assets"][ASSET_NAME]
    assert entry["sha256"] == asset.sha256
    # The training sample is documented and pinned, and deliberately NOT in git.
    sample = entry["training_sample"]
    assert sample["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert sample["sha256"] == (
        "f183e03bac859063a62b868a47f1410b738e42bd8f8c02681f040c7805d73072"
    )
    assert sample["committed"] is False
    assert "fetch_tokenizer_sample" in sample["fetch_command"]
    assert sample["num_rows"] == 1000 and sample["bytes"] > 4_000_000


def test_asset_sha256_mismatch_is_loud(tmp_path, asset) -> None:
    """A tampered copy fails verification instead of tokenizing silently."""
    path = tmp_path / "tampered.json"
    payload = json.loads(open(asset.path, encoding="utf-8").read())
    payload["merges"] = payload["merges"][:-1]  # drop one merge
    path.write_text(json.dumps(payload), encoding="utf-8")
    import tokenizer.assets as assets_mod

    fake = assets_mod.TokenizerAsset(
        name=asset.name, filename=str(path), vocab_size=asset.vocab_size,
        merges=asset.merges, sha256=asset.sha256, presets=asset.presets,
        special_tokens=asset.special_tokens, pre_tokenize=asset.pre_tokenize,
        minfreq=asset.minfreq, seed=asset.seed,
        training_sample=asset.training_sample, compression=asset.compression,
        trained_with=asset.trained_with, date=asset.date,
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fake.verify()


# --------------------------------------------------------------------------
# vocabulary contract
# --------------------------------------------------------------------------
def test_vocab_size_is_exactly_32000(tokenizer) -> None:
    assert tokenizer.vocab_size == TALOS_1B_VOCAB_SIZE == 32_000
    assert tokenizer.merge_count == 32_000 - 256 - 4 == 31_740
    assert tokenizer.config.max_merges() == tokenizer.merge_count
    # every merge slot is used: no unused ids, no id gaps below 32000
    assert tokenizer.vocab_report()["unused_slots"] == 0
    seen = set(range(32000))
    assert max(seen) == tokenizer.vocab_size - 1


def test_special_token_ids_are_sane(tokenizer) -> None:
    ids = tokenizer.special_id_map()
    assert ids == {
        "<|beginoftext|>": 31_996,
        "<|endoftext|>": 31_997,
        "<|pad|>": 31_998,
        "<|unk|>": 31_999,
    }
    assert len(set(ids.values())) == 4
    assert tokenizer.bos_id == 31_996
    assert tokenizer.eos_id == 31_997
    assert tokenizer.pad_id == 31_998
    assert tokenizer.unk_id == 31_999
    # specials sit ABOVE every byte and merge id (LLaMA-style, top of vocab)
    assert min(ids.values()) > 256 + tokenizer.merge_count - 1
    # and they never collide with a produced id
    for text in REPRESENTATIVE_TEXTS:
        assert set(tokenizer.encode(text)).isdisjoint(set(ids.values()))
    assert tokenizer.encode("hi", bos=True, eos=True)[0] == tokenizer.bos_id


@pytest.mark.parametrize("text", REPRESENTATIVE_TEXTS)
def test_encode_decode_round_trip_is_identity(tokenizer, text) -> None:
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    # with specials present, decode(..., skip_special_tokens=False) still
    # restores the text once the added specials are dropped
    full = tokenizer.encode(text, bos=True, eos=True)
    assert tokenizer.decode(full) == text
    assert tokenizer.decode(full, skip_special_tokens=False).startswith(
        "<|beginoftext|>"
    )


# --------------------------------------------------------------------------
# pairing with the 1B presets (meta device — never a real 1B model)
# --------------------------------------------------------------------------
def test_asset_is_registered_for_both_1b_presets() -> None:
    mapping = preset_asset_map()
    for preset in PRESETS_1B:
        assert asset_for_preset(preset) == ASSET_NAME
        assert mapping[preset] == ASSET_NAME
    # the canonical tiny family owns NO committed asset (trains in-run)
    for preset in ("tiny", "tiny_1m", "tiny_10m", "tiny_100m"):
        assert asset_for_preset(preset) is None


@pytest.mark.parametrize("preset", PRESETS_1B)
def test_model_compat_meta_device_embedding_shapes(preset, tokenizer) -> None:
    """32K vocab == the preset's embedding rows, checked on the meta device."""
    import torch

    from model import TalosGPT

    cfg = ALL_PRESETS[preset]()
    with torch.device("meta"):
        model = TalosGPT(cfg)
    assert tuple(model.embed_tokens.weight.shape) == (32_000, cfg.hidden_size)
    assert model.config.vocab_size == tokenizer.vocab_size
    assert tokenizer.vocab_size <= cfg.vocab_size
    assert_tokenizer_vocab_pairs(tokenizer.vocab_size, preset)  # must not raise


# --------------------------------------------------------------------------
# guards: the pairing is accepted, every mismatch is refused
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset", PRESETS_1B)
def test_guard_accepts_the_1b_pairing(preset) -> None:
    assert_asset_preset_pairing(ASSET_NAME, preset)  # must not raise
    assert_tokenizer_vocab_pairs(32_000, preset)     # must not raise


@pytest.mark.parametrize("preset", PRESETS_1B)
def test_guard_refuses_canonical_1024_tokenizer_for_1b(preset) -> None:
    """1024 <= 32000 passes the old "<=" contract — the exact guard refuses it."""
    canonical = preset_tokenizer_config("tiny").vocab_size
    assert canonical == 1024
    with pytest.raises(ValueError, match="does not match"):
        assert_tokenizer_vocab_pairs(canonical, preset)
    with pytest.raises(ValueError):
        assert_tokenizer_vocab_pairs(32_000 - 1, preset)  # off-by-one is refused


@pytest.mark.parametrize("preset", ("tiny", "tiny_1m", "tiny_10m", "tiny_100m"))
def test_guard_refuses_32k_asset_for_canonical_presets(preset) -> None:
    with pytest.raises(ValueError, match="trains its tokenizer inside the run"):
        assert_asset_preset_pairing(ASSET_NAME, preset)


def test_guard_refuses_unknown_asset_and_preset() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer asset"):
        get_asset("does_not_exist")
    with pytest.raises(ValueError, match="unknown preset"):
        asset_for_preset("not_a_preset")


# --------------------------------------------------------------------------
# encoder correctness (and the documented legacy divergence)
# --------------------------------------------------------------------------
def test_asset_uses_the_exact_encoder(tokenizer) -> None:
    assert tokenizer.exact_bpe is True
    # a word that IS in the vocabulary must encode to a single token
    for word in (" the", " place", " energy", " government"):
        ids = tokenizer.encode(word)
        assert len(ids) == 1, (word, [tokenizer.vocab.id_to_bytes[i] for i in ids])


def test_exact_encoder_matches_the_naive_reference(tokenizer) -> None:
    """Heap encoder == O(n^2) reference (lowest rank, leftmost tie-break)."""
    ranks, merge_id = tokenizer.vocab.pair_to_rank, tokenizer.vocab.pair_to_merge_id

    def naive(byte_ids):
        ids = list(byte_ids)
        while len(ids) > 1:
            best = None
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            _, i = best
            ids[i] = merge_id[(ids[i], ids[i + 1])]
            del ids[i + 1]
        return ids

    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz .,theof andin"
    for _ in range(60):
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24)))
        data = list(word.encode("utf-8"))
        assert encode_bytes_exact(data, ranks, merge_id) == naive(data), word


def test_legacy_encoder_still_round_trips_but_costs_more_tokens(tokenizer) -> None:
    """The frozen legacy encoder is kept; it is just measurably worse."""
    legacy = ByteLevelBPETokenizer(
        tokenizer.config, tokenizer.vocab.merges, exact_bpe=False
    )
    text = "The government announced the energy policy quickly."
    assert legacy.decode(legacy.encode(text)) == text  # still lossless
    assert len(legacy.encode(text)) > len(tokenizer.encode(text))
    assert legacy.exact_bpe is False


# --------------------------------------------------------------------------
# canonical 1024 tokenizer untouched
# --------------------------------------------------------------------------
def test_canonical_1024_tokenizer_is_unchanged() -> None:
    """The frozen canonical contract: 1024 vocab, legacy encoder, no new keys."""
    cfg = preset_tokenizer_config("tiny")
    assert cfg.vocab_size == 1024 and cfg.max_merges() == 764
    tok = ByteLevelBPETokenizer(cfg, [(104, 105), (256, 101)])
    assert tok.exact_bpe is False  # default preserved
    assert tok.encode("hie") == tok.encode("hie")  # deterministic
    assert tok.config.vocab_size == 1024  # configured rows; 262 ids realised from 2 merges
    # A legacy-encoder save writes NO exact_bpe key, so every existing
    # tokenizer.json keeps its byte-identical sha256 fingerprint.
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tok.json")
        tok.save(path)
        payload = json.loads(open(path, encoding="utf-8").read())
        assert "exact_bpe" not in payload
        assert set(payload) == {"version", "type", "config", "merges"}
        assert ByteLevelBPETokenizer.from_file(path).exact_bpe is False


def test_1b_config_still_guards_its_pairing() -> None:
    """W1's config-level guard is intact (configs/presets.py)."""
    from configs.presets import _assert_tokenizer_model_pairing

    for preset in PRESETS_1B:
        assert preset_tokenizer_config(preset).vocab_size == 32_000
    with pytest.raises(ValueError, match="does not match"):
        _assert_tokenizer_model_pairing(
            TokenizerConfig(vocab_size=1024), ALL_PRESETS["talos_1b"]()
        )


def test_trainer_cli_offers_the_asset_flag() -> None:
    """scripts/train_oasst1.py can select the committed 32K asset by name."""
    from scripts.train_oasst1 import make_arg_parser

    parser = make_arg_parser()
    assert parser.get_default("tokenizer_asset") is None
    args = parser.parse_args(
        ["--data", "x.jsonl", "--out-dir", "out", "--preset", "talos_1b",
         "--tokenizer-asset", ASSET_NAME]
    )
    assert args.tokenizer_asset == ASSET_NAME
