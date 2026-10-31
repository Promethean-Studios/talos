"""Train the dedicated 32,000-vocab BPE tokenizer for the Talos 1B presets.

``talos_1b`` / ``talos_1b_spec16`` deliberately use a 32,000-vocab tokenizer
track instead of the canonical 1024 one (owner directive). This script produces
that asset with the repo's own byte-level BPE machinery
(:func:`tokenizer.train.train_tokenizer`), pinned so the artifact is
reproducible:

* **vocabulary** — exactly ``32,000`` = 256 base bytes + 4 special tokens +
  **31,740** learned merges. The config comes from the *guarded*
  :func:`configs.presets.talos_1b_tokenizer_config`, so this script cannot train
  an asset whose vocab disagrees with the 1B model it will be paired with.
* **pre-tokenization** — the ``gpt2`` regex preset (the merges then never cross
  word boundaries, matching GPT-2/LLaMA-family practice and the
  ``c/t`` measurements taken on this repo's data).
* **min frequency** — ``--minfreq 1`` by default: the 1B contract requires a
  *full* 31,740-merge vocabulary, so every merge slot must be filled. A run that
  cannot fill the vocabulary fails loudly instead of silently shipping a smaller
  tokenizer.
* **determinism** — BPE merge learning in :mod:`tokenizer.bpe` is deterministic
  by construction (no RNG: ties are broken by pair value, the corpus is consumed
  in file order). ``--seed`` is accepted and recorded for provenance only, so
  two runs on the same sha256-pinned sample produce a **byte-identical**
  ``tokenizer.json`` (verify with ``--expect-output-sha256``).

The training corpus is the small public sample fetched by
:mod:`scripts.fetch_tokenizer_sample` (documented + sha256-pinned in
``assets/tokenizers/MANIFEST.json``). **The owner's packed training corpus is
never read, referenced or used as tokenizer-training data.**

Usage (from the repo root)::

    python -m scripts.fetch_tokenizer_sample --output /tmp/fw_sample_1b.jsonl
    python -m scripts.train_tokenizer_32k \\
        --sample /tmp/fw_sample_1b.jsonl \\
        --heldout /tmp/fw_heldout_1b.jsonl \\
        --output assets/tokenizers/talos_1b_32k.json \\
        --expect-sample-sha256 <pin from the manifest>

Add ``--report`` for the human-readable vocab + compression report; the same
numbers are always printed as one JSON object for the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

from configs.canonical import TALOS_1B_VOCAB_SIZE
from configs.presets import talos_1b_tokenizer_config
from tokenizer.corpus import iter_text_documents
from tokenizer.tokenizer import ByteLevelBPETokenizer, tokenizer_file_sha256
from tokenizer.train import train_tokenizer

#: Pinned training parameters (the 1B tokenizer contract).
DEFAULT_VOCAB_SIZE = TALOS_1B_VOCAB_SIZE  # 32,000
DEFAULT_PRE_TOKENIZE = "gpt2"
DEFAULT_MINFREQ = 1
DEFAULT_SEED = 0  # provenance only: BPE merge learning has no RNG

#: Number of merge slots the 32,000-vocab contract leaves: 32,000 - 256 - 4.
DEFAULT_NUM_MERGES = DEFAULT_VOCAB_SIZE - 256 - len(talos_1b_tokenizer_config().special_tokens)


def count_chars_and_tokens(
    tokenizer: ByteLevelBPETokenizer,
    docs: Iterable[str],
) -> Dict[str, float]:
    """``chars``, ``tokens`` and ``chars_per_token`` for a document stream."""
    chars = 0
    tokens = 0
    n_docs = 0
    for doc in docs:
        chars += len(doc)
        tokens += len(tokenizer.encode(doc))
        n_docs += 1
    return {
        "docs": n_docs,
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": (chars / tokens) if tokens else 0.0,
        "bytes_per_token": (chars / tokens) if tokens else 0.0,  # ASCII-ish text
    }


def compression_report(
    tokenizer: ByteLevelBPETokenizer,
    train_sample: str,
    heldout_sample: Optional[str] = None,
    text_field: str = "text",
) -> Dict[str, object]:
    """Compression measurement on the training sample and a held-out slice.

    The byte-tokenizer baseline is 1.0 chars/token by definition (one token per
    byte), so ``chars_per_token`` is also the compression factor over that
    baseline. Reported for both slices so the report shows in-sample vs
    held-out generalization (they should be close for a 32K vocabulary).
    """
    legacy = ByteLevelBPETokenizer(
        tokenizer.config, tokenizer.vocab.merges, exact_bpe=False
    )
    report: Dict[str, object] = {
        "train_sample": count_chars_and_tokens(
            tokenizer, iter_text_documents([train_sample], jsonl_field=text_field)
        ),
        #: the same vocabulary run through the legacy heap encoder, so the
        #: cost of that encoder is measured rather than assumed.
        "train_sample_legacy_encoder": count_chars_and_tokens(
            legacy, iter_text_documents([train_sample], jsonl_field=text_field)
        ),
        "baseline_byte_tokenizer_chars_per_token": 1.0,
        "encoder": "exact_bpe" if tokenizer.exact_bpe else "legacy_heap",
    }
    if heldout_sample:
        report["heldout_sample"] = count_chars_and_tokens(
            tokenizer, iter_text_documents([heldout_sample], jsonl_field=text_field)
        )
        report["heldout_sample_legacy_encoder"] = count_chars_and_tokens(
            legacy, iter_text_documents([heldout_sample], jsonl_field=text_field)
        )
    train_cpt = float(report["train_sample"]["chars_per_token"])  # type: ignore[index]
    report["train_compression_vs_byte_baseline"] = train_cpt
    if heldout_sample:
        hold_cpt = float(report["heldout_sample"]["chars_per_token"])  # type: ignore[index]
        report["heldout_compression_vs_byte_baseline"] = hold_cpt
    return report


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.train_tokenizer_32k",
        description="Train the dedicated 32,000-vocab BPE tokenizer for talos_1b.",
    )
    p.add_argument("--sample", required=True,
                   help="training sample JSONL (from scripts.fetch_tokenizer_sample)")
    p.add_argument("--output", required=True,
                   help="where to write the tokenizer asset (e.g. "
                        "assets/tokenizers/talos_1b_32k.json)")
    p.add_argument("--heldout", default=None,
                   help="optional held-out JSONL for the compression report")
    p.add_argument("--text-field", default="text", help="JSONL text field (default text)")
    p.add_argument("--num-merges", type=int, default=DEFAULT_NUM_MERGES,
                   help=f"merges to learn (default {DEFAULT_NUM_MERGES} = the full "
                        f"{DEFAULT_VOCAB_SIZE:,}-vocab budget)")
    p.add_argument("--minfreq", type=int, default=DEFAULT_MINFREQ,
                   help=f"minimum pair frequency (default {DEFAULT_MINFREQ}: the full "
                        "vocabulary is mandatory, so the tail merges are learned)")
    p.add_argument("--pre-tokenize", default=DEFAULT_PRE_TOKENIZE,
                   help=f"pre-tokenization preset (default {DEFAULT_PRE_TOKENIZE!r})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="recorded for provenance (BPE learning itself is deterministic)")
    p.add_argument("--max-docs", type=int, default=None, help="cap documents read")
    p.add_argument("--max-chars", type=int, default=None, help="cap characters read")
    p.add_argument("--expect-sample-sha256", default=None,
                   help="fail unless the training sample's sha256 matches this pin")
    p.add_argument("--expect-output-sha256", default=None,
                   help="fail unless the written tokenizer.json matches this pin "
                        "(reproducibility check)")
    p.add_argument("--allow-short-vocab", action="store_true",
                   help="accept a vocabulary smaller than the 32,000 contract "
                        "(NOT valid for the committed 1B asset; diagnostics only)")
    p.add_argument("--report", action="store_true",
                   help="print the human-readable vocab + compression report")
    return p


def main(argv: Optional[list] = None) -> int:
    args = make_arg_parser().parse_args(argv)

    if not os.path.isfile(args.sample):
        raise FileNotFoundError(f"--sample not found: {args.sample}")
    sample_sha = tokenizer_file_sha256(args.sample)
    if args.expect_sample_sha256 and sample_sha != args.expect_sample_sha256:
        raise ValueError(
            f"training sample sha256 {sample_sha} != pinned "
            f"{args.expect_sample_sha256} — the sample changed; refusing to "
            "train a different tokenizer than the one the manifest describes"
        )

    config = talos_1b_tokenizer_config()  # guarded: 32,000 vocab, 1B pairing
    # The guarded 1B config carries the vocab + special tokens; the
    # pre-tokenizer is a *training* choice, so it is set here (and recorded in
    # the manifest). Without this the trainer would silently fall back to pure
    # byte-level splitting, where merges cross word boundaries.
    config.pre_tokenize = args.pre_tokenize
    if config.vocab_size != DEFAULT_VOCAB_SIZE:  # pragma: no cover - config guard
        raise ValueError(
            f"1B tokenizer config vocab_size {config.vocab_size} != "
            f"{DEFAULT_VOCAB_SIZE} — the guarded config drifted"
        )
    if args.num_merges > config.max_merges():
        raise ValueError(
            f"--num-merges {args.num_merges} exceeds the {config.vocab_size:,}-vocab "
            f"budget of {config.max_merges():,} merges"
        )

    texts = iter_text_documents([args.sample], jsonl_field=args.text_field)
    t0 = time.perf_counter()
    result = train_tokenizer(
        texts,
        config,
        minfreq=args.minfreq,
        num_merges=args.num_merges,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
    )
    elapsed = time.perf_counter() - t0
    # The 1B asset uses the CORRECT greedy BPE encoder
    # (tokenizer.bpe.encode_bytes_exact). The legacy heap encoder is retained
    # only so the canonical 1024 tokenizers stay byte-identical; with it a
    # trained 32K vocabulary encodes ~2.7x worse than it should (1.72 vs 4.61
    # chars/token on this sample), which would waste the owner's GPU time.
    tokenizer = ByteLevelBPETokenizer(config, result.merges, exact_bpe=True)

    shortfall = config.vocab_size - tokenizer.vocab_size
    if shortfall and not args.allow_short_vocab:
        raise RuntimeError(
            f"trained vocabulary is {tokenizer.vocab_size:,} "
            f"({tokenizer.merge_count:,} merges) but the 1B contract requires "
            f"exactly {config.vocab_size:,} ({config.max_merges():,} merges): "
            f"{shortfall:,} slots short. The sample ran out of pairs above "
            f"minfreq={args.minfreq}; lower --minfreq or use a larger sample. "
            f"(--allow-short-vocab exists for diagnostics only.)"
        )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(args.output)
    output_sha = tokenizer_file_sha256(args.output)

    report: Dict[str, object] = {
        "output": args.output,
        "sha256": output_sha,
        "vocab_size": tokenizer.vocab_size,
        "configured_vocab_size": config.vocab_size,
        "merges": tokenizer.merge_count,
        "max_merges": config.max_merges(),
        "special_ids": tokenizer.special_id_map(),
        "pre_tokenize": args.pre_tokenize,
        "minfreq": args.minfreq,
        "seed": args.seed,
        "sample": {
            "path": args.sample,
            "sha256": sample_sha,
            "bytes": os.path.getsize(args.sample),
        },
        "num_docs": result.num_words,
        "sample_bytes": result.num_bytes,
        "train_elapsed_s": round(elapsed, 2),
    }
    report["compression"] = compression_report(
        tokenizer, args.sample, args.heldout, text_field=args.text_field
    )

    if args.report:
        print("\nVocab report:")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        comp = report["compression"]  # type: ignore[assignment]
        print("\nCompression (chars/token; byte baseline = 1.0):")
        for name in ("train_sample", "heldout_sample",
                     "train_sample_legacy_encoder",
                     "heldout_sample_legacy_encoder"):
            if name in comp:  # type: ignore[operator]
                row = comp[name]  # type: ignore[index]
                print(f"  {name:14s}: {row['chars_per_token']:.4f} c/t "
                      f"({row['tokens']:,} tokens over {row['chars']:,} chars)")
    print(json.dumps(report, indent=2))

    if args.expect_output_sha256 and output_sha != args.expect_output_sha256:
        print(
            f"ERROR: tokenizer sha256 {output_sha} != pinned "
            f"{args.expect_output_sha256} — training is not reproducible",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
