"""Memory measurement harness for the Talos data path (271K tiny prototype).

Measures **peak RSS** (``resource.getrusage`` high-water mark) and Python-heap
peak (``tracemalloc``) for each stage of the tiny-prototype data path:

* ``corpus``              — generate a deterministic "real-ish" text corpus
                            (``corpus.txt`` with blank-line-separated paragraphs
                            + sharded JSONL via the default ShardedWriter).
* ``pipeline``            — run_pipeline over the corpus (TextReader with
                            split_paragraphs -> length filter -> exact dedup ->
                            ShardedWriter with default settings).
* ``tokenize-materialize``— the OLD pattern: encode every document and keep the
                            whole corpus as ``list[list[int]]``, converting to
                            one int32 array only at the end.
* ``tokenize-stream``     — the NEW pattern: data.tokenized streaming loader
                            (tokenize doc-by-doc into int32, yield batches).

Each phase runs in its own process (peak RSS is a process-lifetime high-water
mark, so phases must be isolated) and prints one machine-readable line::

    MEMRESULT phase=<name> peak_rss_mb=<x> python_peak_mb=<y> tokens=<n> ...

Before/after comparison: run this same script against a worktree of ``main``
(the "before") and this branch (the "after") — the harness imports whatever
repo it is *run from* (cwd), so::

    git worktree add /tmp/talos-main main
    cd /tmp/talos-main && /opt/forge-venv/bin/python <branch>/tools/measure_memory.py --phase pipeline ...

Usage (from a repo root)::

    python tools/measure_memory.py --phase corpus --corpus-dir /tmp/talos-mem/corpus
    python tools/measure_memory.py --phase pipeline --corpus-dir /tmp/talos-mem/corpus
    python tools/measure_memory.py --phase tokenize-materialize --corpus-dir /tmp/talos-mem/corpus
    python tools/measure_memory.py --phase tokenize-stream --corpus-dir /tmp/talos-mem/corpus
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import resource
import sys
import tracemalloc
from typing import Callable, Dict, List, Optional

# Import whatever repo the process is *run from* (cwd) — lets the same script
# measure a `main` worktree ("before") vs a feature branch ("after").
sys.path.insert(0, os.getcwd())


def _rss_mb() -> float:
    """Peak RSS of this process in MB (Linux reports ru_maxrss in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _report(phase: str, body: Callable[[], Dict[str, object]]) -> None:
    gc.collect()
    tracemalloc.start()
    extra = body()
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    parts = " ".join(f"{k}={v}" for k, v in extra.items())
    print(f"MEMRESULT phase={phase} peak_rss_mb={_rss_mb():.1f} "
          f"python_peak_mb={python_peak / (1024 * 1024):.1f} {parts}", flush=True)


# ---------------------------------------------------------------------------
# Phase: corpus (deterministic, real-ish text)
# ---------------------------------------------------------------------------
_WORDS = (
    "the model trains on tokens from a large corpus of text and code "
    "attention heads attend to previous positions in the sequence gradient "
    "descent updates parameters loss decreases over steps data streams from "
    "shards tokenizer encodes bytes into ids mixture of experts routes "
    "hidden states through networks memory stays bounded while training"
).split()


def _make_doc(rng: random.Random, n_chars: int) -> str:
    words: List[str] = []
    n = 0
    while n < n_chars:
        w = rng.choice(_WORDS)
        words.append(w)
        n += len(w) + 1
    sentence = " ".join(words)
    return sentence[0].upper() + sentence[1:] + "."


def phase_corpus(corpus_dir: str, n_docs: int, doc_chars: int) -> None:
    from data.writers import ShardedWriter

    os.makedirs(corpus_dir, exist_ok=True)
    rng = random.Random(1234)
    txt_path = os.path.join(corpus_dir, "corpus.txt")
    jsonl_dir = os.path.join(corpus_dir, "jsonl")
    with open(txt_path, "w", encoding="utf-8") as fh:
        for _ in range(n_docs):
            fh.write(_make_doc(rng, doc_chars))
            fh.write("\n\n")  # paragraph separator for TextReader
    with ShardedWriter(jsonl_dir, prefix="shard") as writer:  # defaults
        for _ in range(n_docs):
            writer.write({"text": _make_doc(rng, doc_chars), "source": "synthetic"})
    size_mb = os.path.getsize(txt_path) / (1024 * 1024)
    print(f"corpus: {n_docs} docs x ~{doc_chars} chars at {corpus_dir} "
          f"({size_mb:.1f} MB text)", flush=True)


# ---------------------------------------------------------------------------
# Phase: pipeline (TextReader paragraphs -> length -> exact_dedup -> writer)
# ---------------------------------------------------------------------------
def phase_pipeline(corpus_dir: str, n_docs: int, doc_chars: int) -> None:
    from data.pipeline import run_pipeline

    config = {
        "sources": [
            {
                "reader": {
                    "type": "text",
                    "path": os.path.join(corpus_dir, "corpus.txt"),
                    "split_paragraphs": True,
                },
                "weight": 1.0,
            }
        ],
        "processors": [
            {"type": "length", "min_chars": 1},
            {"type": "exact_dedup"},
        ],
        "output": {"dir": os.path.join(corpus_dir, "pipeline-out")},
    }
    stats = run_pipeline(config, os.path.join(corpus_dir, "pipeline-out"), seed=0)
    return {"kept": stats.total_kept, "dropped": stats.total_dropped}


# ---------------------------------------------------------------------------
# Tokenizer shared by both tokenize phases (byte-fallback: no merges needed)
# ---------------------------------------------------------------------------
def _make_tokenizer():
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    from tokenizer.vocab import TokenizerConfig

    return ByteLevelBPETokenizer(TokenizerConfig(vocab_size=1024))


def _iter_jsonl_texts(jsonl_dir: str, limit_docs: Optional[int]):
    """Stream doc texts from the JSONL shards (same code on every revision)."""
    paths = sorted(
        os.path.join(jsonl_dir, f) for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")
    )
    seen = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj["text"]
                seen += 1
                if limit_docs is not None and seen >= limit_docs:
                    return


def phase_tokenize_materialize(corpus_dir: str, limit_docs: Optional[int]) -> None:
    import numpy as np

    tok = _make_tokenizer()
    # OLD pattern: materialize every document's ids as list[list[int]] and
    # convert to a tensor only at the end.
    corpus_ids: List[List[int]] = []
    for text in _iter_jsonl_texts(os.path.join(corpus_dir, "jsonl"), limit_docs):
        corpus_ids.append(tok.encode(text, eos=True))
    flat = np.asarray([i for ids in corpus_ids for i in ids], dtype=np.int32)
    total = int(flat.size)
    del corpus_ids, flat
    gc.collect()
    return {"tokens": total, "layout": "list[list[int]] then int32"}


def phase_tokenize_stream(corpus_dir: str, limit_docs: Optional[int]) -> None:
    from data.tokenized import iter_token_arrays

    tok = _make_tokenizer()
    paths = sorted(
        os.path.join(corpus_dir, "jsonl", f)
        for f in os.listdir(os.path.join(corpus_dir, "jsonl"))
        if f.endswith(".jsonl")
    )
    total = 0
    checksum = 0
    for batch in iter_token_arrays(
        paths, tok, seq_len=64, batch_size=256, mode="pack", eos=True,
        limit_docs=limit_docs,
    ):
        total += int(batch.size)
        checksum += int(batch.sum())
    return {"tokens": total, "checksum": checksum, "layout": "streamed int32 batches"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=("corpus", "pipeline", "tokenize-materialize", "tokenize-stream"))
    ap.add_argument("--corpus-dir", default="/tmp/talos-mem/corpus")
    ap.add_argument("--n-docs", type=int, default=30_000)
    ap.add_argument("--doc-chars", type=int, default=300)
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="cap docs for the tokenize phases (default: all)")
    args = ap.parse_args()

    if args.phase == "corpus":
        phase_corpus(args.corpus_dir, args.n_docs, args.doc_chars)
        return 0
    if args.phase == "pipeline":
        _report("pipeline", lambda: phase_pipeline(args.corpus_dir, args.n_docs, args.doc_chars))
    elif args.phase == "tokenize-materialize":
        _report("tokenize-materialize",
                lambda: phase_tokenize_materialize(args.corpus_dir, args.limit_docs))
    elif args.phase == "tokenize-stream":
        _report("tokenize-stream",
                lambda: phase_tokenize_stream(args.corpus_dir, args.limit_docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
