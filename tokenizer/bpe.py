"""Core byte-level BPE: pair statistics, merge training and greedy encoding.

This module is pure Python + stdlib. It operates entirely on lists of byte ids
(``0..255``); the tokenizer layer (``tokenizer/tokenizer.py``) is responsible
for id assignment (special tokens, merges) and text round-tripping.

Two training implementations are provided:

* :func:`train_bpe_naive` — a straightforward reference that recomputes the
  whole corpus pair histogram every round. It is clearly correct but ``O(V*N)``
  total work, so it is used only for tests and as an oracle.
* :func:`train_bpe` — the efficient incremental implementation that maintains a
  global pair→count table, a lazy max-heap of the most frequent pairs, and a
  pair→words index so every merge round touches only the words that contain the
  chosen pair. It scales to 100K+ merges. Its output is validated to match
  :func:`train_bpe_naive` by a unit test.

Both are fully deterministic: ties are broken by pair value (larger byte-tuple
first), there is no randomness, and the merge order depends only on the corpus
and ``minfreq``.
"""
from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Pair = Tuple[int, int]
Word = List[int]


def get_stats(word: Sequence[int]) -> Dict[Pair, int]:
    """Count adjacent byte-pair frequencies inside one ``word``."""
    stats: Dict[Pair, int] = {}
    for i in range(len(word) - 1):
        p = (word[i], word[i + 1])
        stats[p] = stats.get(p, 0) + 1
    return stats


def merge(word: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every occurrence of ``pair`` in ``word`` with ``new_id``."""
    new_word: List[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i + 1 < n and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


# --------------------------------------------------------------------------
# Reference (naive, clearly correct) training
# --------------------------------------------------------------------------
def train_bpe_naive(
    words: Iterable[Sequence[int]],
    num_merges: int,
    minfreq: int = 2,
) -> List[Pair]:
    """Train ``num_merges`` merges by recomputing the global histogram each round.

    Deterministic tie-break: the most frequent pair is chosen; ties are broken
    by the larger pair tuple (``(a, b)``). Stops early when every remaining pair
    has frequency ``< minfreq``.
    """
    words_: List[Word] = [list(w) for w in words]
    merges: List[Pair] = []
    for _ in range(num_merges):
        stats: Dict[Pair, int] = {}
        for w in words_:
            for p, c in get_stats(w).items():
                stats[p] = stats.get(p, 0) + c
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], p))
        if stats[pair] < minfreq:
            break
        new_id = 256 + len(merges)
        merges.append(pair)
        for i, w in enumerate(words_):
            words_[i] = merge(w, pair, new_id)
    return merges


# --------------------------------------------------------------------------
# Efficient training (incremental global counts + lazy heap + pair→words index)
# --------------------------------------------------------------------------
def train_bpe(
    words: Iterable[Sequence[int]],
    num_merges: int,
    minfreq: int = 2,
    report_every: Optional[int] = None,
    report: Optional[object] = None,
) -> List[Pair]:
    """Efficient, deterministic BPE merge training for large corpora/vocabs.

    Returns the ordered list of learned merges (index = merge rank). The
    algorithm:

    1. Build per-word pair counts, a global ``pair→count`` table, and a
       ``pair→{word indices}`` index so each round only touches affected words.
    2. Maintain a lazy min-heap of ``(-count, -a, -b, pair)`` so the heap's top
       is the most frequent pair (largest count, then largest pair — matching
       :func:`train_bpe_naive`).
    3. Pop the top, validate it is still current (lazy deletion), then merge
       that pair over every word containing it, updating counts and the index.

    The memory usage is proportional to the number of *distinct pairs* across
    words plus the words themselves, which is why it scales to 100K+ merges.
    """
    words_ = [list(w) for w in words]

    # Global pair -> count and pair -> set of word indices.
    global_counts: Dict[Pair, int] = {}
    pair_words: Dict[Pair, set] = {}
    for wi, w in enumerate(words_):
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            global_counts[p] = global_counts.get(p, 0) + 1
            pair_words.setdefault(p, set()).add(wi)

    # Lazy max-heap (min-heap of negated values). Tie-break by larger pair.
    heap: List[tuple] = []
    for p, c in global_counts.items():
        heap.append((-c, -p[0], -p[1], p))
    heapq.heapify(heap)

    merges: List[Pair] = []
    while len(merges) < num_merges:
        # Pop the current most frequent (valid) pair lazily. Stale entries whose
        # recorded count no longer matches are refreshed and re-pushed.
        best: Pair = None  # type: ignore[assignment]
        while heap:
            negc, _na, _nb, p = heapq.heappop(heap)
            c = global_counts.get(p, 0)
            if c == 0:
                continue  # stale entry (pair no longer present)
            if -negc != c:
                # Count changed but pair still present; refresh and re-push.
                heapq.heappush(heap, (-c, -p[0], -p[1], p))
                continue
            best = p
            break
        if best is None:
            break  # no pairs remain
        if global_counts[best] < minfreq:
            break

        pair = best
        new_id = 256 + len(merges)
        merges.append(pair)

        # Optional periodic progress / checkpoint hook.
        if callable(report) and report_every and len(merges) % report_every == 0:
            report(list(merges))

        # Merge every word that currently contains ``pair``, updating the global
        # counts incrementally and collecting every changed pair so a fresh heap
        # entry (with its current count) keeps the true maximum at the top.
        fresh: List[tuple] = []
        touched = list(pair_words.get(pair, ()))
        for wi in touched:
            w = words_[wi]
            if not _contains_pair(w, pair):
                continue
            before = get_stats(w)
            merged = merge(w, pair, new_id)
            after = get_stats(merged)
            for p2, c2 in before.items():
                new_top = global_counts.get(p2, 0) - c2
                global_counts[p2] = new_top
                fresh.append((-new_top, -p2[0], -p2[1], p2))
            for p2, c2 in after.items():
                new_top = global_counts.get(p2, 0) + c2
                global_counts[p2] = new_top
                fresh.append((-new_top, -p2[0], -p2[1], p2))
            _rebuild_index(words_, wi, merged, pair, pair_words)
            words_[wi] = merged
        for e in fresh:
            heapq.heappush(heap, e)

    return merges


def _contains_pair(word: Sequence[int], pair: Pair) -> bool:
    a, b = pair
    for i in range(len(word) - 1):
        if word[i] == a and word[i + 1] == b:
            return True
    return False


def _rebuild_index(
    words_: List[Word],
    wi: int,
    merged: Word,
    merged_pair: Pair,
    pair_words: Dict[Pair, set],
) -> None:
    """Refresh which words contain each pair after ``words_[wi]`` changed."""
    set_ = pair_words.get(merged_pair)
    if set_ is not None:
        set_.discard(wi)  # this word no longer contains the pair we just removed
    for i in range(len(merged) - 1):
        p = (merged[i], merged[i + 1])
        pair_words.setdefault(p, set()).add(wi)


# --------------------------------------------------------------------------
# Greedy byte-level encoding using the learned merges
# --------------------------------------------------------------------------
def ranks_from_merges(merges: Sequence[Pair]) -> Dict[Pair, int]:
    """Map each merge pair to its rank (lower = merged earlier)."""
    return {p: i for i, p in enumerate(merges)}


def encode_bytes(
    byte_ids: Sequence[int],
    ranks: Dict[Pair, int],
    merge_id: "Dict[Pair, int]",
) -> List[int]:
    """Greedily apply BPE merges to a byte-id sequence.

    Uses a priority queue keyed by merge rank (and then position) so the
    *globally* lowest-rank adjacent pair is always merged first — exactly the
    GPT-2 / tiktoken greedy rule. ``merge_id`` maps a pair to its final token id.
    """
    ids: List[int] = list(byte_ids)
    if len(ids) < 2:
        return ids
    heap: List[tuple] = []
    for i in range(len(ids) - 1):
        p = (ids[i], ids[i + 1])
        r = ranks.get(p)
        if r is not None:
            heapq.heappush(heap, (r, i, p))
    while heap:
        rank, i, pair = heapq.heappop(heap)
        if i + 1 >= len(ids):
            continue  # stale / out of range
        if ids[i] != pair[0] or ids[i + 1] != pair[1]:
            continue  # already merged
        new_id = merge_id[pair]
        ids[i] = new_id
        del ids[i + 1]
        for j in (i - 1, i):
            if 0 <= j < len(ids) - 1:
                npair = (ids[j], ids[j + 1])
                nr = ranks.get(npair)
                if nr is not None:
                    heapq.heappush(heap, (nr, j, npair))
    return ids
