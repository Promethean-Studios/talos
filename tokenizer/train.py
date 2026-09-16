"""Tokenizer training: BPE merge learning + CLI + checkpoint/resume.

Training flow:

1. Read text documents from a configurable corpus (plain text or JSONL; any
   legal corpus the user points at — see :mod:`tokenizer.corpus`). Documents
   are streamed from disk; ``max_docs`` / ``max_chars`` bound consumption.
2. (Optional) pre-tokenize into words with a regex (off by default = pure
   byte-level). Each word becomes a list of byte ids — but instead of keeping
   a list per document, the words are collapsed **in one pass** into a
   unique-word frequency table (:func:`build_word_frequency_table`), so memory
   tracks the unique vocabulary, not the raw document count.
3. Learn BPE merges incrementally (:func:`tokenizer.bpe.train_bpe`) up to
   ``num_merges`` (default: the configured vocab-derived merge budget). Each
   round touches only the unique words containing the chosen pair and updates
   pair counts with the word's multiplicity, so 1,800-document corpora train
   comfortably on a few-hundred-MB budget.
4. Optionally persist intermediate checkpoints every ``checkpoint_every``
   merges so a large run can be resumed with ``--resume``.

The result is a :class:`ByteLevelBPETokenizer` whose vocab_size (bytes +
specials + merges) fits inside the requested ``vocab_size``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tokenizer._logging import get_logger
from tokenizer.bpe import train_bpe, Pair
from tokenizer.corpus import iter_text_documents
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.vocab import TokenizerConfig

log = get_logger("train")


@dataclass
class TrainResult:
    """Outcome of a :func:`train_tokenizer` run."""

    tokenizer: ByteLevelBPETokenizer
    merges: List[Pair] = field(default_factory=list)
    num_words: int = 0
    num_bytes: int = 0
    requested_vocab_size: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> dict:
        report = self.tokenizer.vocab_report()
        report.update(
            {
                "num_words": self.num_words,
                "num_bytes": self.num_bytes,
                "requested_vocab_size": self.requested_vocab_size,
                "elapsed_s": round(self.elapsed_s, 3),
            }
        )
        return report


def build_word_frequency_table(
    texts: Iterable[str],
    pre_tokenize: Optional[str] = None,
    block_bytes: int = 512 * 1024,
) -> Tuple[List[List[int]], List[int], int]:
    """Stream documents into a unique-word frequency table in ONE pass.

    ``texts`` is consumed lazily (generator-friendly): each document is
    utf-8-encoded and split into bounded byte blocks (pure byte-level mode) or
    regex words (pre-tokenized mode, unmatched spans kept as their own words so
    bytes are never silently dropped). Words are stored **once** and counted by
    multiplicity, so the returned structure is proportional to the number of
    *unique* words — not the raw corpus size. This is the memory fix for
    thousands-of-documents corpora.

    Returns ``(unique_words, counts, total_bytes)``.
    """
    from tokenizer.pre_tokenize import iter_words_with_gaps, resolve_pattern

    pattern = resolve_pattern(pre_tokenize)
    freqs: Dict[tuple, int] = {}
    total_bytes = 0
    for text in texts:
        data = text.encode("utf-8")
        total_bytes += len(data)
        if pattern is None:
            for blk in _byte_blocks(data, block_bytes):
                freqs[tuple(blk)] = freqs.get(tuple(blk), 0) + 1
        else:
            # iter_words_with_gaps keeps unmatched spans as their own words, so a
            # pattern that does not cover the corpus never silently drops bytes.
            for w in iter_words_with_gaps(text, pattern):
                if w:
                    key = tuple(w.encode("utf-8"))
                    freqs[key] = freqs.get(key, 0) + 1
    words: List[List[int]] = []
    counts: List[int] = []
    for key, c in freqs.items():
        words.append(list(key))
        counts.append(c)
    return words, counts, total_bytes


def build_words(
    texts: Iterable[str],
    pre_tokenize: Optional[str] = None,
    block_bytes: int = 512 * 1024,
) -> Tuple[List[List[int]], int]:
    """Convert documents into byte-id words for BPE training.

    Compatibility wrapper around :func:`build_word_frequency_table` that expands
    the frequency table back into one entry per occurrence (i.e. the legacy
    behaviour). Prefer the frequency-table path directly when memory matters.

    Returns ``(words, total_bytes)``.
    """
    words, counts, total_bytes = build_word_frequency_table(
        texts, pre_tokenize, block_bytes
    )
    expanded: List[List[int]] = []
    for w, c in zip(words, counts):
        expanded.extend(list(w) for _ in range(c))
    return expanded, total_bytes


def _byte_blocks(data: bytes, block_bytes: int) -> List[bytes]:
    """Split ``data`` into ≤ ``block_bytes`` blocks, preferring newline breaks."""
    if len(data) <= block_bytes:
        return [data]
    blocks: List[bytes] = []
    start = 0
    while start < len(data):
        end = min(start + block_bytes, len(data))
        if end < len(data):
            nl = data.rfind(b"\n", start, end)
            if nl != -1:
                end = nl + 1
        blocks.append(data[start:end])
        start = end
    return blocks


def preapply_merges(words: List[List[int]], merges: Sequence[Pair]) -> List[List[int]]:
    """Apply already-learned ``merges`` to byte words for resume.

    Ids ``256 + rank`` are assigned in rank order so ``train_bpe`` can continue
    producing fresh ranks on top without re-learning the earlier merges.
    """
    for rank, pair in enumerate(merges):
        new_id = 256 + rank
        for i, w in enumerate(words):
            a, b = pair
            j = 0
            changed = False
            n = len(w)
            new_w: List[int] = []
            while j < n:
                if j + 1 < n and w[j] == a and w[j + 1] == b:
                    new_w.append(new_id)
                    j += 2
                    changed = True
                else:
                    new_w.append(w[j])
                    j += 1
            if changed:
                words[i] = new_w
    return words


def train_tokenizer(
    texts: Iterable[str],
    config: TokenizerConfig,
    minfreq: int = 2,
    num_merges: Optional[int] = None,
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
    resume_merges: Optional[Sequence[Pair]] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_name: str = "tokenizer_checkpoint",
    report: Optional[Callable[[int, int], None]] = None,
) -> TrainResult:
    """Learn merges for ``config`` from ``texts`` and return a trained tokenizer.

    Args:
        texts: iterable of document strings (consumed once, lazily).
        config: vocabulary configuration (vocab size, special tokens, ...).
        minfreq: stop merging once the most frequent pair occurs < ``minfreq``.
        num_merges: exact number of merges to learn; defaults to the
            vocab-derived budget ``config.max_merges()``. Must not exceed it.
        max_docs: stop reading after this many documents.
        max_chars: stop reading after this many characters in total.
        resume_merges: already-learned merges (from a checkpoint) to continue on
            top of; training continues to ``num_merges`` in total.
        checkpoint_every / checkpoint_dir / checkpoint_name: periodic JSON
            checkpoint persistence for resumability.
        report: optional ``(done, total)`` progress callback.

    The corpus is consumed **once** into a unique-word frequency table (see
    :func:`build_word_frequency_table`), which is what keeps memory bounded on
    thousands-of-documents corpora.
    """
    t0 = time.perf_counter()
    _validate_train_args(minfreq, num_merges, max_docs, max_chars,
                         len(resume_merges) if resume_merges else 0, config)
    if max_docs is not None or max_chars is not None:
        texts = _limit_texts(texts, max_docs, max_chars)
    words, counts, total_bytes = build_word_frequency_table(
        texts, config.pre_tokenize
    )
    resume = list(resume_merges) if resume_merges else []
    target = config.max_merges() if num_merges is None else num_merges
    num_new = target - len(resume)
    log.info(
        "training: %d unique words (from %d occurrences), %d bytes, "
        "%d merges already present, %d to learn "
        "(vocab_size=%d, special=%d)",
        len(words), sum(counts) if counts else 0, total_bytes,
        len(resume), num_new, config.vocab_size, len(config.special_tokens),
    )

    checkpoint_dir_abs = None
    if checkpoint_every and checkpoint_dir:
        checkpoint_dir_abs = os.path.abspath(checkpoint_dir)
        os.makedirs(checkpoint_dir_abs, exist_ok=True)

    new_merges: List[Pair] = []

    def _checkpoint(k: int) -> None:
        partial = resume + new_merges
        if report is not None:
            report(len(partial), target)
        if checkpoint_dir_abs is not None:
            _write_checkpoint(partial, config, checkpoint_dir_abs, checkpoint_name)

    if num_new > 0:
        words = preapply_merges(words, resume)
        new_merges = train_bpe(
            words, num_new, minfreq=minfreq, word_counts=counts,
            report_every=checkpoint_every,
            report=_checkpoint,
        )
    if checkpoint_dir_abs is not None:
        _write_checkpoint(resume + new_merges, config, checkpoint_dir_abs, checkpoint_name)

    merges = resume + new_merges
    tokenizer = ByteLevelBPETokenizer(config, merges)
    elapsed = time.perf_counter() - t0
    log.info(
        "done: %d merges, vocab=%d, %.2fs",
        len(merges), tokenizer.vocab_size, elapsed,
    )
    return TrainResult(
        tokenizer=tokenizer,
        merges=merges,
        num_words=sum(counts),
        num_bytes=total_bytes,
        requested_vocab_size=config.vocab_size,
        elapsed_s=elapsed,
    )


def _validate_train_args(
    minfreq: int,
    num_merges: Optional[int],
    max_docs: Optional[int],
    max_chars: Optional[int],
    resume_count: int,
    config: TokenizerConfig,
) -> None:
    """Reject nonsensical training arguments with clear errors.

    Called once per :func:`train_tokenizer` invocation before any corpus work,
    so a bad flag fails fast instead of after streaming the whole corpus.
    """
    if minfreq < 1:
        raise ValueError(f"minfreq must be >= 1, got {minfreq}")
    if num_merges is not None:
        if num_merges < 0:
            raise ValueError(f"num_merges must be >= 0, got {num_merges}")
        if num_merges > config.max_merges():
            raise ValueError(
                f"num_merges={num_merges} exceeds vocab budget "
                f"{config.max_merges()} (vocab_size={config.vocab_size}, "
                f"special={len(config.special_tokens)})"
            )
        if num_merges < resume_count:
            raise ValueError(
                f"num_merges={num_merges} < {resume_count} already-learned merges"
            )
    if max_docs is not None and max_docs < 1:
        raise ValueError(f"max_docs must be >= 1, got {max_docs}")
    if max_chars is not None and max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")


def _limit_texts(
    texts: Iterable[str],
    max_docs: Optional[int],
    max_chars: Optional[int],
) -> Iterable[str]:
    """Bounded view of ``texts`` honouring document and character budgets."""
    count = 0
    chars = 0
    for doc in texts:
        if max_docs is not None and count >= max_docs:
            return
        if max_chars is not None and chars >= max_chars:
            return
        yield doc
        count += 1
        chars += len(doc)


def load_merges_from_checkpoint(path: str) -> Tuple[List[Pair], TokenizerConfig]:
    """Read a checkpoint/model file and return ``(merges, config)`` for resume."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    merges = [tuple(p) for p in payload["merges"]]
    config = TokenizerConfig(**payload["config"])
    return merges, config


def _write_checkpoint(
    merges: Sequence[Pair],
    config: TokenizerConfig,
    checkpoint_dir: str,
    name: str,
) -> str:
    path = os.path.join(checkpoint_dir, f"{name}.json")
    tmp = path + ".tmp"
    payload = {
        "version": 1,
        "type": "ByteLevelBPE.checkpoint",
        "config": asdict(config),
        "merges": [list(p) for p in merges],
    }
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokenizer.train",
        description="Train a byte-level BPE tokenizer over a configurable corpus.",
    )
    p.add_argument("--corpus", required=True, action="append",
                   help="corpus path (.txt/.jsonl or a directory); repeatable")
    p.add_argument("--text-field", default="text",
                   help="JSONL field holding the text (default: text)")
    p.add_argument("--vocab-size", type=int, default=32768,
                   help="total vocab size (bytes+specials+merges); e.g. 32768, 65536, 131072")
    p.add_argument("--output", default="tokenizer.json",
                   help="output JSON model file")
    p.add_argument("--pre-tokenize", default=None,
                   help="regex preset ('gpt2'/'simple') or raw regex, or omit for pure byte-level")
    p.add_argument("--minfreq", type=int, default=2,
                   help="minimum pair frequency to keep merging (default 2)")
    p.add_argument("--on-invalid", choices=["skip", "die"], default="skip",
                   help="behaviour on invalid UTF-8 / JSON lines")
    p.add_argument("--bos", default="<|beginoftext|>")
    p.add_argument("--eos", default="<|endoftext|>")
    p.add_argument("--pad", default="<|pad|>")
    p.add_argument("--unk", default="<|unk|>")
    p.add_argument("--extra-special", action="append", nargs="+", default=None,
                   help="extra special token (e.g. <|fim_prefix|>); repeatable, space-separated")
    p.add_argument("--reserved", action="append", nargs="+", default=None,
                   help="reserved token; repeatable, space-separated")
    p.add_argument("--seed", type=int, default=0, help="seed (for reproducibility)")
    p.add_argument("--resume", default=None,
                   help="checkpoint/model file to resume merges from")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="write a checkpoint every N merges")
    p.add_argument("--checkpoint-dir", default=None,
                   help="directory for checkpoints")
    p.add_argument("--max-docs", type=int, default=None,
                   help="stop after this many documents")
    p.add_argument("--max-chars", type=int, default=None,
                   help="stop after this many characters in total")
    p.add_argument("--num-merges", type=int, default=None,
                   help="exact number of merges to learn "
                        "(default: vocab_size-derived budget)")
    p.add_argument("--report", action="store_true",
                   help="print a vocab-size report after training")
    return p


def main(argv: Optional[list] = None) -> None:
    args = make_arg_parser().parse_args(argv)
    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        bos_token=args.bos,
        eos_token=args.eos,
        pad_token=args.pad,
        unk_token=args.unk,
        extra_special_tokens=[t for g in (args.extra_special or []) for t in g],
        reserved_tokens=[t for g in (args.reserved or []) for t in g],
        pre_tokenize=args.pre_tokenize,
    )

    resume_merges: Optional[List[Pair]] = None
    if args.resume:
        resume_merges, _ = load_merges_from_checkpoint(args.resume)
        log.info("resuming from %s with %d merges", args.resume, len(resume_merges))

    texts = iter_text_documents(
        args.corpus,
        jsonl_field=args.text_field,
        on_invalid=args.on_invalid,
        max_docs=args.max_docs,
    )
    result = train_tokenizer(
        texts,
        config,
        minfreq=args.minfreq,
        num_merges=args.num_merges,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
        resume_merges=resume_merges,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        report=(
            (lambda done, total: log.info(
                "merges: %d/%d (%.1f%%)", done, total, 100.0 * done / total if total else 0.0))
            if args.report else None
        ),
    )
    result.tokenizer.save(args.output)
    if args.report:
        print("\nVocab report:")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
