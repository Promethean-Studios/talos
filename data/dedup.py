"""Deterministic de-duplication for the Talos data pipeline.

Two independently usable processors:

* :class:`ExactDedupFilter` — drop documents whose normalised text hash was
  already seen (exact duplicates, incl. across sources).
* :class:`MinHashNearDupFilter` — drop documents that are *near* duplicates of
  an already-seen document (shingled MinHash with seeded PRNG permutations and
  a banded LSH so the implementation is the real algorithm, not a stub).

Both are stateful — they keep a growing set of seen hashes/band-signatures in
memory — and both are fully deterministic: all hash/permutation values derive
from :mod:`hashlib` and a seeded PRNG, never from global Python ``random``.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from data._logging import get_logger
from data.processors import Processor
from data.types import Record

log = get_logger("dedup")

#: Default was: MinHash with 64 permutations, banded LSH with 8 bands of 8 rows.
DEFAULT_PERMUTATIONS = 64
DEFAULT_BANDS = 8
DEFAULT_ROWS_PER_BAND = 8  # bands * rows_per_band == permutations
DEFAULT_NGRAM = 5  # character shingles
DEFAULT_THRESHOLD = 0.5  # estimated Jaccard similarity -> considered near-dup


def _normalize(text: str) -> str:
    """Lower-case and collapse whitespace for exact-duplicate hashing."""
    return re.sub(r"\s+", " ", text.strip().lower())


class ExactDedupFilter(Processor):
    """Drop exact duplicates.

    ``normalize`` collapses case/whitespace before hashing so near-identical
    variants (different spacing) are treated as duplicates. Use ``keep_first``
    to control whether the first or last occurrence survives (first is default).
    """

    def __init__(self, normalize: bool = True, keep_first: bool = True) -> None:
        self.normalize = normalize
        self.keep_first = keep_first
        self._seen: Set[str] = set()

    @property
    def name(self) -> str:
        return "exact_dedup"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        key = _normalize(text) if self.normalize else text
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if digest in self._seen:
            return None
        self._seen.add(digest)
        return record


def _shingles(text: str, ngram: int) -> Set[str]:
    """Character n-gram shingles of ``text`` (whitespace-normalised)."""
    norm = _normalize(text)
    if len(norm) < ngram:
        return {norm} if norm else set()
    return {norm[i : i + ngram] for i in range(len(norm) - ngram + 1)}


def minhash_signature(shingles: Set[str], permutations: int, seed: int) -> List[int]:
    """Compute a MinHash signature via seeded random hash permutations.

    Deterministic given ``shingles``, ``permutations`` and ``seed``: we derive
    a per-permutation ``a``/``b`` pair from ``random.Random(seed + i)``, then
    map each shingle hash through ``h(x) = (a*x + b) mod p`` and take the min
    over the permutation. The result is a length-``permutations`` list of min
    values approximating the Jaccard similarity between two sets.
    """
    p = (1 << 61) - 1  # Mersenne prime for the hash universe
    sig: List[int] = []
    for i in range(permutations):
        rng = random.Random(f"{seed}:{i}")
        a = rng.randrange(1, p)
        b = rng.randrange(0, p)
        best: Optional[int] = None
        for sh in shingles:
            h = int(hashlib.md5(sh.encode("utf-8")).hexdigest()[:8], 16)
            val = (a * h + b) % p
            if best is None or val < best:
                best = val
        sig.append(best if best is not None else 0)
    return sig


class MinHashNearDupFilter(Processor):
    """Drop near-duplicate documents using shingled MinHash + banded LSH.

    On each record it:
    1. builds the character n-gram shingle set,
    2. computes a seeded MinHash signature,
    3. splits it into ``bands`` of ``rows_per_band`` and, for each band, hashes
       the (band_index, values) tuple into a band bucket key,
    4. keeps the document only if none of its band keys was already seen.

    If any band matches a previously stored band key, the document is judged a
    near duplicate and dropped. Larger ``bands`` -> lower false-positive rate
    but fewer exact detections; ``threshold`` is surfaced as documentation /
    tuning guidance and the band geometry should be chosen so that documents
    above the desired similarity share at least one band key.
    """

    def __init__(
        self,
        permutations: int = DEFAULT_PERMUTATIONS,
        bands: int = DEFAULT_BANDS,
        rows_per_band: int = DEFAULT_ROWS_PER_BAND,
        ngram: int = DEFAULT_NGRAM,
        threshold: float = DEFAULT_THRESHOLD,
        seed: int = 0,
    ) -> None:
        if bands * rows_per_band != permutations:
            raise ValueError("bands * rows_per_band must equal permutations")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.permutations = permutations
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.ngram = ngram
        self.threshold = threshold
        self.seed = seed
        self._seen_bands: Set[Tuple[int, str]] = set()

    @property
    def name(self) -> str:
        return "minhash_near_dedup"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        sh = _shingles(text, self.ngram)
        if not sh:
            return None
        sig = minhash_signature(sh, self.permutations, self.seed)
        for b in range(self.bands):
            chunk = sig[b * self.rows_per_band : (b + 1) * self.rows_per_band]
            key = (b, hashlib.sha256(repr(chunk).encode()).hexdigest())
            if key in self._seen_bands:
                return None
            self._seen_bands.add(key)
        return record
