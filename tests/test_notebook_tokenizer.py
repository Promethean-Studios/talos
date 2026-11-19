"""Fix 6 — the notebook loads *the* 49K tokenizer or it fails loudly.

The audited defect: the notebook used to print ``tokenizer.num_merges`` (an
attribute the tokenizer class does not have), fingerprint the hand-built path
``assets/tokenizers/<name>.json`` rather than the file the loader verified, and
pass no expected digest or realized-vocab check — so a 32K or 1024-vocab asset
(or a Drive copy) could stand in for the 49K one without anything failing.

These tests pin the selection contract: the manifest-resolved asset, the sha256
pin, vocab 49,152, 48,892 merges, the preset pairing, and a hard error — never a
silent fallback — on every mismatch.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOTEBOOKS = os.path.join(_ROOT, "notebooks")
for _path in (_ROOT, _NOTEBOOKS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from talos_tokenizer import (TALOS_1B_49K_MERGES, TALOS_1B_49K_SHA256,  # noqa: E402
                            TALOS_1B_49K_VOCAB, TokenizerSelectionError,
                            load_pinned_tokenizer)
from tokenizer.assets import ASSET_DIR  # noqa: E402

ASSET = "talos_1b_49k"
PRESET = "talos_1b"


def _pinned(**overrides):
    kwargs = dict(expected_sha256=TALOS_1B_49K_SHA256,
                  expected_vocab=TALOS_1B_49K_VOCAB,
                  expected_merges=TALOS_1B_49K_MERGES)
    kwargs.update(overrides)
    return load_pinned_tokenizer(ASSET, PRESET, **kwargs)


# ---------------------------------------------------------------------------
# The good path: the committed 49K asset, verified end to end
# ---------------------------------------------------------------------------
def test_the_pinned_asset_loads_with_vocab_49152_and_the_pinned_digest():
    pinned = _pinned()
    assert pinned.name == ASSET
    assert pinned.vocab_size == 49_152
    assert pinned.merges == 48_892
    assert pinned.sha256 == TALOS_1B_49K_SHA256
    assert pinned.tokenizer.vocab_size == 49_152
    assert pinned.tokenizer.merge_count == 48_892
    # The recorded path is the file that was verified, inside the repo's assets.
    assert os.path.dirname(os.path.abspath(pinned.path)) == os.path.abspath(ASSET_DIR)
    assert os.path.basename(pinned.path) == f"{ASSET}.json"


def test_the_pinned_digest_matches_the_file_on_disk():
    import hashlib

    with open(os.path.join(ASSET_DIR, f"{ASSET}.json"), "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == TALOS_1B_49K_SHA256


# ---------------------------------------------------------------------------
# Every mismatch is a hard error — there is no fallback asset
# ---------------------------------------------------------------------------
def test_a_different_expected_digest_is_refused():
    with pytest.raises(TokenizerSelectionError, match="digest|sha"):
        _pinned(expected_sha256="0" * 64)


def test_a_truncated_vocab_pin_is_refused():
    with pytest.raises(TokenizerSelectionError, match="vocab"):
        _pinned(expected_vocab=32_000)


def test_a_wrong_merge_pin_is_refused():
    with pytest.raises(TokenizerSelectionError, match="merges"):
        _pinned(expected_merges=32_508)


def test_an_unknown_asset_name_is_refused_not_substituted():
    with pytest.raises(TokenizerSelectionError, match="manifest"):
        load_pinned_tokenizer("talos_1b_49k_v2", PRESET,
                             expected_sha256=TALOS_1B_49K_SHA256)


def test_the_32k_asset_cannot_stand_in_for_the_49k_model():
    # Asking for the 32K asset with the 49K contract must fail, and asking for
    # the 32K asset for the primary 1B preset must fail the pairing check.
    with pytest.raises(TokenizerSelectionError):
        load_pinned_tokenizer("talos_1b_32k", PRESET,
                             expected_vocab=TALOS_1B_49K_VOCAB)
    with pytest.raises(TokenizerSelectionError):
        load_pinned_tokenizer("talos_1b_32k", PRESET)


def test_the_49k_asset_cannot_be_paired_with_a_32k_preset():
    with pytest.raises(TokenizerSelectionError, match="pairing"):
        load_pinned_tokenizer(ASSET, "talos_1b_21l",
                             expected_sha256=TALOS_1B_49K_SHA256)


# ---------------------------------------------------------------------------
# No Drive copies: the asset must be the repo's own committed file
# ---------------------------------------------------------------------------
def test_a_working_directory_copy_is_refused(tmp_path):
    copy_dir = tmp_path / "tokenizers"
    copy_dir.mkdir()
    for name in ("MANIFEST.json", f"{ASSET}.json"):
        shutil.copy(os.path.join(ASSET_DIR, name), copy_dir / name)
    # A bare copy has no committed manifest, so it cannot be resolved at all.
    bare = tmp_path / "drive_copy"
    bare.mkdir()
    shutil.copy(os.path.join(ASSET_DIR, f"{ASSET}.json"), bare / f"{ASSET}.json")
    with pytest.raises(TokenizerSelectionError, match="manifest"):
        load_pinned_tokenizer(ASSET, PRESET, asset_dir=str(bare))


def test_missing_asset_file_is_refused(tmp_path):
    copy_dir = tmp_path / "tokenizers"
    copy_dir.mkdir()
    shutil.copy(os.path.join(ASSET_DIR, "MANIFEST.json"), copy_dir / "MANIFEST.json")
    with pytest.raises(TokenizerSelectionError, match="missing"):
        load_pinned_tokenizer(ASSET, PRESET, asset_dir=str(copy_dir))
