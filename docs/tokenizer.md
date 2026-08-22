# Tokenizer (Phase 2) — design notes

This document records the design decisions behind the byte-level BPE tokenizer
(`tokenizer/`). It is written for people who will extend or integrate the
tokenizer in later phases (training, inference, model grafting).

## What it is

A **byte-level BPE** tokenizer in the GPT-2 / LLaMA tradition, implemented in
pure Python + stdlib (no tokenizer library dependency). Every Unicode string is
first UTF-8 encoded to bytes; BPE merges are learned and applied over *bytes*,
not characters or `\p{...}` script classes. The design gives:

- **guaranteed lossless round-trip** — `decode(encode(text)) == text` for
  *arbitrary* Unicode (all world languages, emoji, math symbols, binary-ish
  data) because base tokens are the 256 byte values and every merge is a
  reversible byte-pair composition. There is no OOV / `UNK` path for ordinary
  bytes;
- **no pretrained-weight dependency** — the vocab is *trained* from a corpus
  the user points at (see `tokenizer/train.py`).

### Why byte-level BPE (over char or subword-token)
- Character-level is too granular for efficient model FLOPS and gives poor
  code/math packing.
- Pre-tokenised word-level BPE (a regex splits text into "words" first) is
  fast but bakes in script/whitespace assumptions. Byte-level BPE needs no such
  assumptions, so it is the only choice that trains cleanly on a *multilingual*
  corpus mixing Latin, CJK, Arabic, Devanagari, Cyrillic, code and math without
  a hand-maintained script table. This matches the Forge plan's requirement for
  "strong reasoning/coding/multilingual capability" from one code path.

## Vocabulary layout

`tokenizer/vocab.py` fixes a deterministic id layout:

| range            | content                                  |
|------------------|------------------------------------------|
| `0 .. 255`       | raw bytes (id == byte value)             |
| `256 .. 256+n-1` | learned merges, in merge-rank order       |
| `top-n_special..top-1` | special tokens (BOS/EOS/PAD/UNK + extras + reserved), LLaMA-style at the **top** |

`TokenizerConfig.vocab_size` is the *total* vocab (base + specials + merges);
the number of merge slots is `vocab_size - 256 - n_special`. Because merge id
== `256 + rank` both during training (`tokenizer.bpe`) and in the saved model,
merge *pairs* stored on disk reference the same ids with no remapping on load.

## Pre-tokenization trade-off

`pre_tokenize` is **off by default** (pure byte-level). When enabled it is a
user-supplied regex (presets `gpt2` / `simple`, or a raw string) that splits the
corpus into words; merges are then learned *within* words and never cross a
word boundary (GPT-2/LLaMA style). The trade-offs:

- *On:* cheaper training and GPT-2-like token boundaries, but a space or symbol
  inside a token becomes impossible and — with a regex that does not perfectly
  partition the text — the round-trip guarantee can be lost for some inputs.
- *Off (default):* fully general byte boundaries, slower convergence, but the
  exact round-trip guarantee holds for every input.

**We keep the default off and treat round-trip as the contract of the default
path.** Pre-tokenization is an opt-in experimental mode; tests cover the default
path. Note also that the `simple` preset uses `\p{...}` and therefore requires
the optional `regex` package (Python's stdlib `re` raises otherwise).

## Special tokens

BOS/EOS/PAD/UNK plus any `extra_special_tokens` and `reserved_tokens` occupy the
top ids. By default they are **not** split out of plain text during `encode`
(only added via the `bos`/`eos` flags), which keeps the round-trip guarantee
exact — even for text that literally contains a special-token string. Set
`split_special=True` to opt into recognising special-token strings inside input
(e.g. FIM / tool-call markers).

## Vocabulary-size choice

- Default `vocab_size = 32768`; the CLI accepts any `--vocab-size` (e.g.
  32768, 65536, 131072) and the model presets go up to **131072 (128K tokens)**.
- Byte-level BPE is why a ~128K vocab is *enough*: multi-byte characters are
  packed into a few merges, so the vocab is spent on reusable sub-word units
  instead of whole-word entries. The `Vocabulary` stores merges (pairs) and
  derives all id tables O(1), so a 128K config is not memory-prohibitive.

## Multilingual / code / math handling

Because everything operates on bytes, multilingual scripts, source-code
whitespace/indentation, and math symbols (`∑ ∫ ² ≈ ≤` …) need no special casing.
Merges are learned from whatever bytes co-occur, so code idioms and math token
shapes emerge from the corpus rather than from hard-coded rules. `model_compat`
mapping and the test-suite cover this explicitly.

## Scaling to ~128K vocab and training/resume

`tokenizer/bpe.train_bpe` is an incremental implementation (global pair counts +
a lazy max-heap + a pair→words index) that touches only the words containing the
chosen pair per round, so it reaches 100K+ merges without re-scanning the corpus
each round. `tokenizer/train.train_tokenizer` streams/refines merges up to
`config.max_merges()` and optionally writes a JSON **checkpoint every N merges**
so a large run can be **resumed** with `--resume`. The CLI:

```
python -m tokenizer.train --corpus <path|dir> --vocab-size 131072 \
    --checkpoint-every 10000 --checkpoint-dir ckpts --output tokenizer.json --report
```

The corpus loader (`tokenizer/corpus.py`) reads `.txt` and `.jsonl` (with a
`--text-field`) and validates UTF-8, so you point it at *your own* legally
obtained corpus. The repo ships only synthetic fixtures.

## Integration with the model vocab (`model_compat`)

`tokenizer/model_compat.py` defines the contract between the tokenizer's actual
vocab and `model.config.ModelConfig.vocab_size` (the max embedding width):

- `tokenizer_vocab_size <= model_vocab_size` must hold.
- the tokenizer's first `vocab_size` ids (bytes, then specials, then merges)
  map 1:1 onto the first `tokenizer_vocab_size` embedding rows;
- `model_vocab_size - tokenizer_vocab_size` extra rows are **padding / reserved
  for future vocab growth** (adding specials or merges without rebuilding the
  model).

`map_to_model_vocab(tokenizer, model_vocab_size)` returns a `VocabMapping`
(monotone special-id map + `.fits` + `.padding`); `resize_embedding_plan(...)`
returns the classic "grow the embedding" dimensions for the phase that owns the
tensors. It is deliberately torch-free.

## Limitations / notes

- `encode` of a Python string containing a **lone surrogate** (no UTF-8
  encoding) raises `UnicodeEncodeError`; normal surrogate pairs (emoji) work.
- The greedy byte-level encoder is fast for typical text but scales
  super-linearly on very large *highly-repeated* inputs; use `chunk_encode` /
  `encode_ids_chunks` for long-context workloads (see tests).
- Pre-tokenization round-trip is only guaranteed for the default (off) path.
