"""Deterministic, pinned tokenizer-asset selection for the Argo 1B notebook.

The 1B notebook must load **exactly** ``assets/tokenizers/talos_1b_49k.json`` —
vocab 49,152, paired with the ``talos_1b`` preset — and fail loudly if it cannot.
The audit found three ways the notebook got this wrong:

1. it printed ``tokenizer.num_merges``, an attribute that does not exist (the
   class exposes ``merge_count``), so cell 4 died with ``AttributeError`` before
   any training could start;
2. it computed the checkpoint's tokenizer fingerprint from a **hand-built path**
   (``REPO_DIR / "assets" / "tokenizers" / f"{name}.json"``) instead of the file
   the loader actually verified, so the fingerprint recorded in every checkpoint
   was "whatever that path happened to be" rather than the loaded asset's
   identity;
3. it passed **no** expected digest and never checked the realized
   tokenizer↔preset vocab contract, so nothing but the manifest stood between a
   run and a silently different 49K file (or a 32K/1024-vocab asset that the
   manifest pairing check alone would not catch once the name was edited).

:func:`load_pinned_tokenizer` fixes all three: it resolves the asset through the
committed manifest, verifies the sha256 **twice** (the caller's pin *and* the
manifest's recorded digest), checks the vocab size and merge count against the
pin, enforces the preset pairing and the realized-vocab contract, refuses any
path outside the repo's own asset directory (no Drive copy), and returns the
*verified* path together with its digest. There is no fallback: every failure
raises :class:`TokenizerSelectionError` naming the fix.

Torch-free and network-free, so it is unit-tested on CPU
(``tests/test_notebook_tokenizer.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

#: sha256 of ``assets/tokenizers/talos_1b_49k.json`` as recorded in
#: ``assets/tokenizers/MANIFEST.json`` (read 2026-11-19). Pinned in code as well
#: as in the manifest so that editing the manifest alone cannot swap the
#: tokenizer under a run: both values must agree with the file on disk.
TALOS_1B_49K_SHA256 = "c7d26acebad67be04af9b16cb72a812156cf3a708121a8ee9bf794d6a95c8684"

#: The 1B tokenizer's pinned vocab size (== the ``talos_1b`` model vocab).
TALOS_1B_49K_VOCAB = 49_152

#: The 1B tokenizer's pinned merge count.
TALOS_1B_49K_MERGES = 48_892


class TokenizerSelectionError(ValueError):
    """Raised when the requested tokenizer asset is not the pinned one."""


@dataclass(frozen=True)
class PinnedTokenizer:
    """A verified tokenizer asset: the object, the file it came from, its digest."""

    name: str
    path: str
    sha256: str
    vocab_size: int
    merges: int
    tokenizer: Any

    @property
    def bos_id(self) -> int:
        return int(self.tokenizer.bos_id)

    @property
    def eos_id(self) -> int:
        return int(self.tokenizer.eos_id)

    def encode(self, text: str, bos: bool = False, eos: bool = False):
        """The tokenizer's encoder (``talos_data``'s ``Encoder`` contract)."""
        return self.tokenizer.encode(text, bos=bos, eos=eos)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids))


def load_pinned_tokenizer(
    name: str,
    preset: str,
    *,
    expected_sha256: Optional[str] = None,
    expected_vocab: Optional[int] = None,
    expected_merges: Optional[int] = None,
    asset_dir: Optional[str] = None,
) -> PinnedTokenizer:
    """Load ``name`` from the repo's committed assets, verified against the pins.

    Args:
        name: asset name from ``assets/tokenizers/MANIFEST.json`` (``"talos_1b_49k"``).
        preset: the model preset it must be paired with (``"talos_1b"``).
        expected_sha256: caller pin; must equal the manifest's digest *and* the
            file's digest. ``None`` falls back to the manifest only.
        expected_vocab: required vocab size (49,152 for ``talos_1b_49k``).
        expected_merges: required merge count.
        asset_dir: override for tests; defaults to the repo's ``assets/tokenizers``.

    Raises:
        TokenizerSelectionError: on any mismatch, missing asset, or an asset that
            resolves outside the repo's own tokenizer directory.
    """
    try:  # imported lazily so this module stays importable without the repo root
        from tokenizer.assets import ASSET_DIR, assert_asset_preset_pairing, get_asset
        from tokenizer.assets import assert_tokenizer_vocab_pairs
        from tokenizer.tokenizer import tokenizer_file_sha256
    except Exception as exc:  # noqa: BLE001 - a broken checkout is reported, not raised raw
        raise TokenizerSelectionError(
            f"cannot import the repo's tokenizer modules ({type(exc).__name__}: {exc}) — "
            "is the repository root on sys.path and the checkout complete?"
        ) from exc

    try:
        asset = get_asset(name, asset_dir)
    except Exception as exc:  # noqa: BLE001 - unknown name / unreadable manifest
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} is not in the committed manifest "
            f"({type(exc).__name__}: {exc}) — expected 'talos_1b_49k' for the "
            "Argo 1B run"
        ) from exc

    # (1) the asset must be the repo's own file, not a Drive/working copy.
    resolved = os.path.abspath(asset.path)
    expected_dir = os.path.abspath(asset_dir or ASSET_DIR)
    if os.path.dirname(resolved) != expected_dir:
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} resolved to {resolved}, outside the repo's "
            f"asset directory {expected_dir} — refusing to train with a copy"
        )

    # (2) the caller's pin, the manifest's digest and the file on disk must agree.
    try:
        path = asset.verify(expected_sha256)
    except FileNotFoundError as exc:
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} is missing from the checkout: {exc}"
        ) from exc
    except ValueError as exc:
        raise TokenizerSelectionError(f"tokenizer asset {name!r}: {exc}") from exc
    actual_sha = tokenizer_file_sha256(path)
    if actual_sha != asset.sha256:
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r}: file digest {actual_sha} != manifest "
            f"{asset.sha256}"
        )

    # (3) vocab / merge counts against the pins (a truncated or re-trained file
    #     with the same name must not pass).
    if expected_vocab is not None and int(asset.vocab_size) != int(expected_vocab):
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} has vocab {asset.vocab_size:,} but the run "
            f"requires {int(expected_vocab):,} — refusing a different vocabulary"
        )
    if expected_merges is not None and int(asset.merges) != int(expected_merges):
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} has {asset.merges:,} merges but the run "
            f"requires {int(expected_merges):,}"
        )

    # (4) asset <-> preset pairing, in both directions, and the realized-vocab
    #     contract (a 1024-vocab tokenizer must never stand in for the 49K one).
    try:
        assert_asset_preset_pairing(name, preset, asset_dir)
        assert_tokenizer_vocab_pairs(int(asset.vocab_size), preset)
    except ValueError as exc:
        raise TokenizerSelectionError(
            f"tokenizer/model pairing rejected: {exc}"
        ) from exc

    # (5) load it and re-check what the loader actually built.
    try:
        tokenizer = asset.load(expected_sha256)
    except ValueError as exc:
        raise TokenizerSelectionError(f"tokenizer asset {name!r}: {exc}") from exc
    if int(tokenizer.vocab_size) != int(asset.vocab_size):
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} loaded as vocab {tokenizer.vocab_size:,} "
            f"but the manifest records {asset.vocab_size:,}"
        )
    if int(tokenizer.merge_count) != int(asset.merges):
        raise TokenizerSelectionError(
            f"tokenizer asset {name!r} loaded with {tokenizer.merge_count:,} merges "
            f"but the manifest records {asset.merges:,}"
        )
    return PinnedTokenizer(
        name=str(name),
        path=str(path),
        sha256=str(actual_sha),
        vocab_size=int(tokenizer.vocab_size),
        merges=int(tokenizer.merge_count),
        tokenizer=tokenizer,
    )


__all__ = [
    "PinnedTokenizer",
    "TALOS_1B_49K_MERGES",
    "TALOS_1B_49K_SHA256",
    "TALOS_1B_49K_VOCAB",
    "TokenizerSelectionError",
    "load_pinned_tokenizer",
]
