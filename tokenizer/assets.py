"""Committed tokenizer assets: registry, sha256 verification, pairing guards.

The canonical tiny-family presets (254K/1M/10M/100M) *train* their vocab-1024
tokenizer inside each run (``scripts/train_oasst1.py`` step 2). The Talos 1B
track does not: ``talos_1b`` / ``talos_1b_spec16`` use a **dedicated, committed,
32,000-vocab BPE asset** that every 1B run must load — so the tokenizer is
identical across runs, machines and checkpoints, and so the canonical 1024
tokenizer can never be silently reused for a 1B model.

Layout (see ``assets/tokenizers/``)::

    assets/tokenizers/MANIFEST.json        # provenance + sha256 pins (this module's source of truth)
    assets/tokenizers/talos_1b_32k.json    # the trained ByteLevelBPE asset

The manifest records, per asset: the file name, vocab size, merge count, the
sha256 of the tokenizer file, the special-token ids, the presets the asset is
*the* tokenizer for, and the provenance of the **public sample** it was trained
on (dataset, config, split, offset slice, row count, bytes, and the sample's
sha256). The sample itself is deliberately **not** committed — it is re-fetched
by :mod:`scripts.fetch_tokenizer_sample` and verified against the pinned digest,
so the repo carries no third-party text while the artifact stays reproducible.

Two guards live here:

* :func:`assert_asset_preset_pairing` — an asset may only be loaded for a preset
  it is registered for. Loading the 32K asset for ``tiny_100m`` (vocab 1024) or
  a vocab-1024 tokenizer for ``talos_1b`` fails loudly.
* :func:`assert_tokenizer_vocab_pairs` — the *realized* tokenizer's vocab must
  pair with the preset's model vocab: ``<=`` for the canonical tiny family
  (unchanged historical contract) and **exactly equal** for the 1B track, which
  is what stops a stray 1024-vocab tokenizer from slipping under a 32,000-row
  embedding matrix.

Both raise :class:`ValueError` with the offending sizes named; nothing here is
torch-dependent, so the checks run on any box (including the 2-core CPU box
where a 1B model cannot be instantiated).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from configs.canonical import CANONICAL_PRESETS, TALOS_1B_VOCAB_SIZE
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256

#: Repository root (this file lives in ``<root>/tokenizer/``).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Directory holding committed tokenizer assets and their manifest.
ASSET_DIR = os.path.join(REPO_ROOT, "assets", "tokenizers")

#: The manifest file name inside :data:`ASSET_DIR`.
MANIFEST_FILENAME = "MANIFEST.json"

#: Presets whose tokenizer is *trained inside the run* (canonical tiny family).
TRAINED_IN_RUN_PRESETS: Tuple[str, ...] = ("tiny", "tiny_1m", "tiny_10m", "tiny_100m")


@dataclass(frozen=True)
class TrainingSample:
    """Provenance of the public sample an asset was trained on."""

    dataset: str
    config: str
    split: str
    start_offset: int
    num_rows: int
    page_size: int
    chars: int
    bytes: int
    sha256: str
    license: str
    fetch_command: str

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "TrainingSample":
        return cls(
            dataset=str(d["dataset"]),
            config=str(d["config"]),
            split=str(d["split"]),
            start_offset=int(d["start_offset"]),
            num_rows=int(d["num_rows"]),
            page_size=int(d["page_size"]),
            chars=int(d["chars"]),
            bytes=int(d["bytes"]),
            sha256=str(d["sha256"]),
            license=str(d.get("license", "")),
            fetch_command=str(d.get("fetch_command", "")),
        )


@dataclass(frozen=True)
class TokenizerAsset:
    """One committed tokenizer asset, as recorded in ``MANIFEST.json``."""

    name: str
    filename: str
    vocab_size: int
    merges: int
    sha256: str
    presets: Tuple[str, ...]
    special_tokens: Mapping[str, int]
    pre_tokenize: Optional[str]
    minfreq: int
    seed: int
    training_sample: TrainingSample
    compression: Mapping[str, object]
    trained_with: str
    date: str

    # -- paths -------------------------------------------------------------
    @property
    def path(self) -> str:
        """Absolute path of the committed tokenizer file."""
        return os.path.join(ASSET_DIR, self.filename)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    # -- verification ------------------------------------------------------
    def verify(self, expected_sha256: Optional[str] = None) -> str:
        """Check the file exists and its sha256 matches the manifest.

        Args:
            expected_sha256: optional caller-supplied pin; must agree with the
                manifest's recorded digest as well (a caller pinning a different
                tokenizer than the repo ships is a loud error, not a fallback).

        Returns:
            The absolute path of the verified asset.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"tokenizer asset {self.name!r} is missing: {self.path} "
                f"(expected a committed asset; run `git lfs pull`/re-checkout "
                f"if the working tree is sparse)"
            )
        actual = tokenizer_file_sha256(self.path)
        if actual != self.sha256:
            raise ValueError(
                f"tokenizer asset {self.name!r} sha256 mismatch: file has "
                f"{actual}, manifest pins {self.sha256} — the asset was "
                f"modified or truncated; restore it from git"
            )
        if expected_sha256 is not None and expected_sha256 != self.sha256:
            raise ValueError(
                f"caller pinned sha256 {expected_sha256} for tokenizer asset "
                f"{self.name!r} but the manifest records {self.sha256} — "
                f"refusing to load a different tokenizer"
            )
        return self.path

    def load(self, expected_sha256: Optional[str] = None) -> ByteLevelBPETokenizer:
        """Verify then load the asset, checking vocab/merge counts match."""
        path = self.verify(expected_sha256)
        tok = ByteLevelBPETokenizer.from_file(path)
        if tok.vocab_size != self.vocab_size or tok.merge_count != self.merges:
            raise ValueError(
                f"tokenizer asset {self.name!r} loaded as vocab {tok.vocab_size} "
                f"({tok.merge_count} merges) but the manifest records vocab "
                f"{self.vocab_size} ({self.merges} merges) — manifest and asset "
                f"disagree"
            )
        return tok


def manifest_path(asset_dir: Optional[str] = None) -> str:
    """Absolute path of the asset manifest."""
    return os.path.join(asset_dir or ASSET_DIR, MANIFEST_FILENAME)


def load_manifest(asset_dir: Optional[str] = None) -> Dict[str, object]:
    """Read and parse ``MANIFEST.json`` (raw dict; see :func:`assets`)."""
    path = manifest_path(asset_dir)
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("version") != 1:
        raise ValueError(
            f"unsupported tokenizer asset manifest version {payload.get('version')} "
            f"in {path}"
        )
    return payload


_CACHE: Dict[str, Dict[str, TokenizerAsset]] = {}


def assets(asset_dir: Optional[str] = None) -> Dict[str, TokenizerAsset]:
    """``{asset name: TokenizerAsset}`` from the manifest (cached per dir)."""
    key = os.path.abspath(asset_dir or ASSET_DIR)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    payload = load_manifest(asset_dir)
    built: Dict[str, TokenizerAsset] = {}
    for name, entry in (payload.get("assets") or {}).items():
        e = dict(entry)  # type: ignore[arg-type]
        built[name] = TokenizerAsset(
            name=name,
            filename=str(e["file"]),
            vocab_size=int(e["vocab_size"]),
            merges=int(e["merges"]),
            sha256=str(e["sha256"]),
            presets=tuple(e.get("presets", ())),
            special_tokens=dict(e.get("special_ids", {})),
            pre_tokenize=e.get("pre_tokenize"),
            minfreq=int(e.get("minfreq", 1)),
            seed=int(e.get("seed", 0)),
            training_sample=TrainingSample.from_dict(e["training_sample"]),
            compression=dict(e.get("compression", {})),
            trained_with=str(e.get("trained_with", "")),
            date=str(e.get("date", "")),
        )
    _CACHE[key] = built
    return built


def asset_names(asset_dir: Optional[str] = None) -> List[str]:
    """Sorted names of every registered asset."""
    return sorted(assets(asset_dir))


def get_asset(name: str, asset_dir: Optional[str] = None) -> TokenizerAsset:
    """Look up one asset by name (raises ``ValueError`` for unknown names)."""
    registry = assets(asset_dir)
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"unknown tokenizer asset {name!r} (known: "
            f"{', '.join(sorted(registry)) or 'none'})"
        ) from None


def preset_asset_map(asset_dir: Optional[str] = None) -> Dict[str, str]:
    """``{preset: asset name}`` derived from the manifest's ``presets`` lists."""
    mapping: Dict[str, str] = {}
    for name, asset in assets(asset_dir).items():
        for preset in asset.presets:
            if preset in mapping:
                raise ValueError(
                    f"preset {preset!r} is claimed by two tokenizer assets "
                    f"({mapping[preset]!r} and {name!r}) — exactly one asset must "
                    f"own a preset"
                )
            mapping[preset] = name
    return mapping


def asset_for_preset(preset: str, asset_dir: Optional[str] = None) -> Optional[str]:
    """The asset name registered for ``preset``, or ``None`` if it trains in-run.

    The canonical tiny-family presets have no committed asset (they train their
    vocab-1024 tokenizer inside the run), so they return ``None``; the 1B presets
    return their 32,000-vocab asset.
    """
    mapping = preset_asset_map(asset_dir)
    if preset not in mapping and preset not in CANONICAL_PRESETS:
        raise ValueError(
            f"unknown preset {preset!r} (known canonical presets: "
            f"{', '.join(sorted(CANONICAL_PRESETS))})"
        )
    return mapping.get(preset)


def load_asset(name: str, expected_sha256: Optional[str] = None,
               asset_dir: Optional[str] = None) -> ByteLevelBPETokenizer:
    """Verify + load a committed asset by name."""
    return get_asset(name, asset_dir).load(expected_sha256)


def assert_asset_preset_pairing(
    asset_name: str,
    preset: str,
    asset_dir: Optional[str] = None,
) -> None:
    """Refuse to pair an asset with a preset it is not registered for.

    This is the *asset-level* guard: ``talos_1b``/``talos_1b_spec16`` are owned
    by the 32,000-vocab asset; the tiny family owns no committed asset at all.
    Both directions are loud:

    * a 32K asset requested for a tiny-family preset (vocab 1024) raises;
    * a preset that must use the 32K asset with a different asset raises.
    """
    asset = get_asset(asset_name, asset_dir)
    expected = asset_for_preset(preset, asset_dir)
    if expected is None:
        raise ValueError(
            f"preset {preset!r} trains its tokenizer inside the run (vocab "
            f"{CANONICAL_PRESETS[preset][1]}); committed tokenizer asset "
            f"{asset_name!r} (vocab {asset.vocab_size}) must not be loaded for it"
        )
    if expected != asset_name:
        raise ValueError(
            f"preset {preset!r} requires tokenizer asset {expected!r}, not "
            f"{asset_name!r} — refusing a mismatched tokenizer/model pairing"
        )


def assert_tokenizer_vocab_pairs(
    tokenizer_vocab_size: int,
    preset: str,
) -> None:
    """Enforce the *realized* tokenizer ↔ preset model vocab contract.

    * canonical tiny family — ``tokenizer_vocab_size <= model vocab`` (the
      historical contract: the model may keep unused embedding rows).
    * Talos 1B track — ``tokenizer_vocab_size == model vocab`` (exactly 32,000).
      A 1024-vocab tokenizer satisfies ``<=`` but is exactly the silent-reuse
      failure the owner forbade, so it is refused here.
    """
    if preset not in CANONICAL_PRESETS:
        raise ValueError(
            f"unknown preset {preset!r} (known canonical presets: "
            f"{', '.join(sorted(CANONICAL_PRESETS))})"
        )
    model_vocab = CANONICAL_PRESETS[preset][1]
    if preset in TRAINED_IN_RUN_PRESETS:
        if tokenizer_vocab_size > model_vocab:
            raise ValueError(
                f"tokenizer vocab_size {tokenizer_vocab_size} exceeds the "
                f"{preset} model's {model_vocab} — the tokenizer cannot be "
                f"embedded"
            )
        return
    # 1B track: exact equality only.
    if tokenizer_vocab_size != model_vocab:
        required = preset_asset_map().get(preset, "talos_1b_32k")
        raise ValueError(
            f"tokenizer vocab_size {tokenizer_vocab_size} does not match the "
            f"{preset} model's {model_vocab} — the Talos 1B track requires its "
            f"dedicated {TALOS_1B_VOCAB_SIZE:,}-vocab tokenizer asset "
            f"({required!r}); a smaller (e.g. the canonical 1024-vocab) "
            f"tokenizer is refused"
        )
