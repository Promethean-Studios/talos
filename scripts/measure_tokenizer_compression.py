"""Chars/token compression table for Talos tokenizer assets on pinned slices.

The 1B tokenizer decision (which vocabulary, trained on how much data) is made
on **measured compression**, so the measurement itself is a committed, re-runnable
tool rather than an ad-hoc shell loop:

    # every committed asset, on the training sample and the held-out slice
    python -m scripts.fetch_tokenizer_sample --num-rows 24000 \\
        --output /tmp/fw_train_24k.jsonl
    python -m scripts.fetch_tokenizer_sample --num-rows 1000 \\
        --start-offset 1000000 --output /tmp/fw_heldout_1k.jsonl
    python -m scripts.measure_tokenizer_compression \\
        --asset talos_1b_32k --asset talos_1b_49k \\
        --slice train_24k=/tmp/fw_train_24k.jsonl \\
        --slice heldout_1k=/tmp/fw_heldout_1k.jsonl

Any tokenizer file can be measured too — in particular a *previous* asset taken
out of git history, which is how the before/after comparison is produced:

    git show <old-rev>:assets/tokenizers/talos_1b_32k.json > /tmp/v1.json
    python -m scripts.measure_tokenizer_compression \\
        --tokenizer v1=/tmp/v1.json --asset talos_1b_32k \\
        --slice heldout_1k=/tmp/fw_heldout_1k.jsonl

``chars/token`` is also the compression factor over the byte tokenizer, which is
1.0 by definition (one token per byte).

Cost note: the exact BPE encoder is pure Python, so a 120 MB slice takes minutes
per tokenizer. ``--max-docs`` / ``--max-chars`` bound the work, and
``--no-legacy`` skips the (much slower, and only diagnostic) legacy-heap pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from tokenizer.assets import get_asset, load_asset
from tokenizer.corpus import iter_text_documents
from tokenizer.tokenizer import ByteLevelBPETokenizer


def count_slice(
    tokenizer: ByteLevelBPETokenizer,
    path: str,
    text_field: str = "text",
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> Dict[str, object]:
    """``docs``/``chars``/``tokens``/``chars_per_token`` for one JSONL slice.

    ``chars`` counts the *text* field of each document (not the JSONL bytes), so
    the number is directly comparable with the manifest's recorded compression.
    """
    chars = 0
    tokens = 0
    docs = 0
    for doc in iter_text_documents([path], jsonl_field=text_field, max_docs=max_docs):
        if max_chars is not None and chars >= max_chars:
            break
        chars += len(doc)
        tokens += len(tokenizer.encode(doc))
        docs += 1
    return {
        "docs": docs,
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": (chars / tokens) if tokens else 0.0,
    }


def measure(
    tokenizers: Iterable[Tuple[str, ByteLevelBPETokenizer]],
    slices: Iterable[Tuple[str, str]],
    text_field: str = "text",
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Measure every ``(tokenizer, slice)`` pair; returns flat result rows."""
    rows: List[Dict[str, object]] = []
    for tok_label, tok in tokenizers:
        for slice_label, path in slices:
            stats = count_slice(tok, path, text_field, max_docs, max_chars)
            rows.append({
                "tokenizer": tok_label,
                "slice": slice_label,
                "vocab_size": tok.vocab_size,
                "merges": tok.merge_count,
                "encoder": "exact_bpe" if tok.exact_bpe else "legacy_heap",
                **stats,
            })
    return rows


def format_markdown(rows: Iterable[Dict[str, object]]) -> str:
    """Render result rows as a GitHub-flavoured markdown table."""
    header = (
        "| tokenizer | slice | vocab | encoder | docs | chars | tokens | chars/token |\n"
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    for r in rows:
        lines.append(
            "| {tokenizer} | {slice} | {vocab_size:,} | {encoder} | {docs:,} | "
            "{chars:,} | {tokens:,} | **{cpt:.4f}** |".format(
                cpt=float(r["chars_per_token"]), **r
            )
        )
    return "\n".join(lines)


def _pair(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=PATH, got {spec!r}")
    label, _, value = spec.partition("=")
    if not label or not value:
        raise ValueError(f"expected LABEL=PATH, got {spec!r}")
    return label, value


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.measure_tokenizer_compression",
        description="chars/token for tokenizer assets over JSONL slices.",
    )
    p.add_argument("--asset", action="append", default=None,
                   help="committed tokenizer asset name (repeatable), e.g. talos_1b_32k")
    p.add_argument("--tokenizer", action="append", default=None, metavar="LABEL=PATH",
                   help="arbitrary tokenizer JSON (repeatable), e.g. v1=/tmp/v1.json")
    p.add_argument("--slice", action="append", default=None, metavar="LABEL=PATH",
                   required=True, help="JSONL slice to measure (repeatable)")
    p.add_argument("--text-field", default="text", help="JSONL text field (default text)")
    p.add_argument("--max-docs", type=int, default=None, help="cap documents per slice")
    p.add_argument("--max-chars", type=int, default=None, help="cap characters per slice")
    p.add_argument("--no-legacy", action="store_true",
                   help="skip the legacy-heap encoder pass (diagnostic only)")
    p.add_argument("--json-out", default=None, help="write the raw rows as JSON here")
    return p


def main(argv: Optional[list] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    if not args.asset and not args.tokenizer:
        raise SystemExit("nothing to measure: pass --asset and/or --tokenizer")

    tokenizers: List[Tuple[str, ByteLevelBPETokenizer]] = []
    for name in args.asset or []:
        tokenizers.append((name, load_asset(name)))
    for spec in args.tokenizer or []:
        label, path = _pair(spec)
        tokenizers.append((label, ByteLevelBPETokenizer.from_file(path)))
    slices = [_pair(s) for s in args.slice]

    rows = measure(tokenizers, slices, args.text_field, args.max_docs, args.max_chars)
    if not args.no_legacy:
        legacy: List[Tuple[str, ByteLevelBPETokenizer]] = []
        for label, tok in tokenizers:
            if tok.exact_bpe:
                legacy.append((
                    f"{label} (legacy encoder)",
                    ByteLevelBPETokenizer(tok.config, tok.vocab.merges, exact_bpe=False),
                ))
        rows.extend(measure(legacy, slices, args.text_field, args.max_docs, args.max_chars))

    print(format_markdown(rows))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
