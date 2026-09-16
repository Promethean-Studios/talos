"""Core byte-level BPE: pair statistics, merge training and greedy encoding.

This module is pure Python + stdlib. It operates entirely on lists of byte ids
(``0..255``); the tokenizer layer (``tokenizer/tokenizer.py``) is responsible
for id assignment (special tokens, merges) and text round-tripping.

Two training implementations are provided:

* :func:`train_bpe_naive` — a straightforward reference that recomputes the
  whole corpus pair histogram every round. It is clearly correct but ``O(V*N)``
  total work, so it is used only for tests and as an oracle.
* :func:`train_bpe` — the efficient incremental implementation. It collapses the
  corpus into a **unique-word frequency table** (``word_counts``), keeps a global
  pair→count table and an exact pair→words index so every merge round touches
  only the words that contain the chosen pair, and maintains a lazy max-heap
  whose growth is bounded by a periodic rebuild. Memory is proportional to the
  number of *distinct pairs* and *unique words* — not the raw corpus size. Its
  output is validated to match :func:`train_bpe_naive` by a unit test.

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
# Efficient training (unique-word freq table + exact pair index + bounded heap)
# --------------------------------------------------------------------------
def build_word_freq_table(
    words: Iterable[Sequence[int]],
) -> Tuple[List[Word], List[int]]:
    """Collapse a (possibly duplicated) word iterable into a frequency table.

    Returns ``(unique_words, counts)`` where ``counts[i]`` is how many times
    ``unique_words[i]`` occurs in the corpus. This single pass is what keeps
    training memory proportional to the *unique* vocabulary of the corpus: with
    ``word_counts``, every merge updates each unique word once with its
    multiplicity multiplier instead of once per raw occurrence.
    """
    freqs: Dict[tuple, int] = {}
    for w in words:
        key = tuple(w)
        freqs[key] = freqs.get(key, 0) + 1
    unique: List[Word] = []
    counts: List[int] = []
    for key, c in freqs.items():
        unique.append(list(key))
        counts.append(c)
    return unique, counts


def train_bpe(
    words: Iterable[Sequence[int]],
    num_merges: int,
    minfreq: int = 2,
    word_counts: Optional[Sequence[int]] = None,
    report_every: Optional[int] = None,
    report: Optional[object] = None,
) -> List[Pair]:
    """Efficient, deterministic BPE merge training for large corpora/vocabs.

    Returns the ordered list of learned merges (index = merge rank). The
    algorithm:

    1. Collapse the corpus into a unique-word frequency table (one pass; pass
       ``word_counts`` to supply an already-built table — see
       :func:`build_word_freq_table`). Each unique word keeps its multiplicity,
       so a pair present in ``freq`` copies of a word contributes ``freq`` to
       the global count for the price of one update.
    2. Build a global ``pair→count`` table and an *exact* ``pair→{word index}``
       index. The index is maintained incrementally: when a word is merged, it
       is dropped from every pair it used to contain and added to every pair in
       the merged word, so no stale memberships accumulate.
    3. Maintain a lazy max-heap of ``(-count, -a, -b, pair)`` (largest count,
       then larger pair — matching :func:`train_bpe_naive`). Each round pushes
       **one aggregated entry per changed pair** (never one per word), and the
       heap is rebuilt from the live pair table whenever stale entries make it
       grow past a small multiple of the live pair count, so memory stays
       bounded regardless of merge count.

    Choosing the merge only needs the true current global maximum, so stale
    heap entries are validated lazily at pop time (refresh + re-push) — the
    classic priority-queue-with-lazy-deletion pattern.

    Memory is proportional to the number of *distinct pairs* (≤ 256² for bytes)
    plus the *unique words*, not the raw document count — this is what lets the
    trainer run on corpora of thousands of documents in a few hundred MB.
    """
    if word_counts is None:
        words_, counts = build_word_freq_table(words)
    else:
        words_ = [list(w) for w in words]
        counts = [int(c) for c in word_counts]
        if len(words_) != len(counts):
            raise ValueError("word_counts must parallel words")

    # Global pair -> count and exact pair -> set of unique-word indices.
    global_counts: Dict[Pair, int] = {}
    pair_words: Dict[Pair, set] = {}
    for wi, w in enumerate(words_):
        freq = counts[wi]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            global_counts[p] = global_counts.get(p, 0) + freq
            pair_words.setdefault(p, set()).add(wi)

    # Lazy max-heap (min-heap of negated values). Tie-break by larger pair.
    heap: List[tuple] = []
    for p, c in global_counts.items():
        heap.append((-c, -p[0], -p[1], p))
    heapq.heapify(heap)

    merges: List[Pair] = []
    while len(merges) < num_merges:
        # Pop the true current maximum lazily; refresh stale entries in place.
        best: Pair = None  # type: ignore[assignment]
        while heap:
            negc, _na, _nb, p = heapq.heappop(heap)
            c = global_counts.get(p, 0)
            if c == 0:
                continue  # pair no longer present anywhere
            if -negc != c:
                heapq.heappush(heap, (-c, -p[0], -p[1], p))  # refresh
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

        # Merge every unique word containing ``pair`` (the index is exact, so
        # this set is complete and contains no stale memberships). All pair
        # count changes in the round are aggregated into ``changed``, whose
        # last write per pair is by construction the final post-round count,
        # so exactly one heap entry per changed pair is pushed.
        changed: Dict[Pair, int] = {}
        for wi in list(pair_words.get(pair, ())):
            w = words_[wi]
            freq = counts[wi]
            before = get_stats(w)
            merged = merge(w, pair, new_id)

            for p2, c2 in before.items():
                newc = global_counts[p2] - freq * c2
                global_counts[p2] = newc
                changed[p2] = newc
                pws = pair_words[p2]
                pws.discard(wi)
                if not pws:
                    del pair_words[p2]  # free the empty set
            for p2, c2 in get_stats(merged).items():
                newc = global_counts.get(p2, 0) + freq * c2
                global_counts[p2] = newc
                changed[p2] = newc
                pair_words.setdefault(p2, set()).add(wi)
            words_[wi] = merged

        for p2, c in changed.items():
            if c:
                heapq.heappush(heap, (-c, -p2[0], -p2[1], p2))

        # Bound heap growth: when stale entries push the heap well past the
        # number of live pairs, rebuild it from the live table (deterministic).
        if len(heap) > 4 * max(1, len(global_counts)):
            heap = [
                (-c, -p[0], -p[1], p) for p, c in global_counts.items() if c
            ]
            heapq.heapify(heap)

    return merges


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
