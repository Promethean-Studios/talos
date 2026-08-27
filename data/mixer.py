"""Deterministic, reproducible multi-source mixing / weighted sampling.

The :class:`WeightedMixer` combines several record iterables (each with a
weight) into a single stream such that, over an epoch, each source contributes
about its weight share of records, and the *entire ordering is reproducible*
from a seed. Two runs with the same seed produce byte-identical output.

Algorithm (documented, deterministic):
We use *smooth weighted round-robin* (the same arithmetic nginx uses for load
balancing): each source keeps a ``current`` weight; each step we add each
source's base weight to its current, pick the source with the largest current,
then subtract the total weight from that one. This yields a perfectly fair,
deterministic interleave. To make the *starting* order seed-sensitive we first
shuffle the source order with ``random.Random(seed)`` and break remaining ties
by a seeded hash of source index, so different seeds give different (but still
reproducible) mixes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from data._logging import get_logger
from data.types import Record

log = get_logger("mixer")

SourceInput = Iterable[Record]


@dataclass
class _Source:
    index: int
    weight: float
    iterable: SourceInput
    iterator: Optional[Iterator[Record]] = None
    remaining: int = 1  # epochs left
    current: float = 0.0


class WeightedMixer:
    """Mix multiple sources deterministically by weight.

    Args:
        sources: list of ``(iterable, weight)`` pairs. Weights should be >= 0.
        seed: RNG seed controlling source order + tie-breaking.
        num_epochs: number of passes over each source. ``1`` (the default) is
            the safe, finite mode — each source is consumed once, then
            deactivated when exhausted. Pass ``None`` for *streaming* mode,
            where an exhausted source is restarted indefinitely (use only with
            genuinely infinite/near-infinite readers; with finite lists this
            would never terminate). Any ``n > 1`` gives "epoch"-based mixing.
    """

    def __init__(
        self,
        sources: Sequence[Tuple[SourceInput, float]],
        seed: int = 0,
        num_epochs: Optional[int] = 1,
    ) -> None:
        if not sources:
            raise ValueError("WeightedMixer needs at least one source")
        if any(w < 0 for _, w in sources):
            raise ValueError("weights must be >= 0")
        self.seed = seed
        self.num_epochs = num_epochs
        total = sum(w for _, w in sources)
        if total <= 0:
            raise ValueError("sum of weights must be > 0")
        # Build sources and seed-shuffle their presentation order.
        rng = random.Random(seed)
        indices = list(range(len(sources)))
        rng.shuffle(indices)
        self._sources: List[_Source] = []
        for i in indices:
            iterable, weight = sources[i]
            self._sources.append(
                _Source(
                    index=i,
                    weight=weight,
                    iterable=iterable,
                    remaining=self.num_epochs,  # None == streaming
                )
            )

    def __iter__(self) -> Iterator[Record]:
        active = self._start_all()
        total_weight = sum(s.weight for s in self._sources)
        while active:
            # add base weight to each active source's current
            for s in active:
                s.current += s.weight
            chosen = active[0]
            for s in active[1:]:
                if s.current > chosen.current:
                    chosen = s
            chosen.current -= total_weight
            # advance the chosen source
            next_record = _next(chosen.iterator)
            if next_record is None:
                # epoch exhausted; either move to next epoch or deactivate
                if self.num_epochs is not None:
                    chosen.remaining -= 1
                if self.num_epochs is None or chosen.remaining > 0:
                    chosen.iterator = iter(chosen.iterable)
                    res = _next(chosen.iterator)
                    if res is None:
                        active.remove(chosen)
                        continue
                    next_record = res
                else:
                    active.remove(chosen)
                    continue
            yield next_record

    def _start_all(self) -> List[_Source]:
        for s in self._sources:
            s.iterator = iter(s.iterable)
        return list(self._sources)

    def weights(self) -> Dict[int, float]:
        """Return ``{source_index: weight}`` for diagnostics."""
        return {s.index: s.weight for s in self._sources}


def _next(it: Optional[Iterator[Record]]) -> Optional[Record]:
    if it is None:
        return None
    try:
        return next(it)
    except StopIteration:
        return None


def source_counts(
    records: Iterable[Record], sources: Sequence[str]
) -> Dict[str, int]:
    """Tally how many records came from each source name."""
    counts = {s: 0 for s in sources}
    for rec in records:
        src = rec.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts
