# Tokenizer — shipped assets and design notes

This document has two halves. **§0 is the current state** — the tokenizers that
actually ship, which model they pair with, and the rules for using them. The
rest (§1 onward) records the original design decisions behind the byte-level BPE
implementation in `tokenizer/`, for people extending or integrating it.

## 0. Shipped tokenizers (current as of 2026-11-07)

There are **two independent tokenizer tracks** in Talos, and they must not be
mixed.

### 0.1 The canonical track — vocab 1,024, trained inside each run

`tiny` / `tiny_1m` / `tiny_10m` / `tiny_100m` use a vocab-1024 BPE tokenizer
(256 byte tokens + 4 specials + 764 merges) that the trainer builds from the
run's own training split (`scripts/train_oasst1.py`), or loads via
`--tokenizer-json`. No vocab-1024 asset is committed to the repo. These presets
keep the **legacy heap-based encoder** (`tokenizer.bpe.encode_bytes`) as their
default so that existing tokenizers, checkpoints and fingerprints stay
byte-identical.

### 0.2 The 1B track — committed, sha256-pinned assets

The 1B presets do **not** train a tokenizer. They load a committed asset from
`assets/tokenizers/`, whose `MANIFEST.json` is the source of truth:

| asset | vocab | merges | paired with | status |
|---|---:|---:|---|---|
| `talos_1b_49k` | 49,152 | 48,892 | `talos_1b` | **primary 1B tokenizer** (owner decision 2026-11-07) |
| `talos_1b_32k` (**version 2**) | 32,000 | 31,740 | `talos_1b_21l`, `talos_1b_spec16` | paired with the two 32K siblings |

Both use the **exact BPE encoder** (`tokenizer.bpe.encode_bytes_exact`), which
replaces the legacy heap-based greedy encoder for these assets. The legacy
encoder is retained, opt-in and unchanged, for the canonical presets — that
divergence is deliberate (owner decision 2026-11-05), because re-encoding the
canonical tokenizers with a different encoder would change every existing
fingerprint and checkpoint.

Special-token ids are recorded per asset in the manifest, at the **top** of the
vocab: `<|beginoftext|>` / `<|endoftext|>` / `<|pad|>` / `<|unk|>` are
31,996–31,999 for the 32K asset and 49,148–49,151 for the 49K asset.

#### Training provenance

`talos_1b_32k` v2 was retrained on a 16,000-row / 79,444,384-byte sample of
`HuggingFaceFW/fineweb-edu` (`default`, `train`, rows 0–15,999; ODC-By 1.0).
The sample is **not committed** — the manifest pins its sha256
(`2bafe0fb…815033`) and `scripts/fetch_tokenizer_sample.py` re-fetches it, so the
repo carries no third-party text while the artifact stays reproducible.
`talos_1b_49k` uses the same sample and the same trainer
(`scripts/train_tokenizer_32k.py --vocab-size 49152`).

#### Measured compression (held-out, identical text)

`chars/token` on FineWeb-Edu held-out slices, measured with
`scripts/measure_tokenizer_compression.py`; **higher is better** (fewer tokens
per character):

| tokenizer | 200-row slice | 1,000-row slice | tokens vs v1 |
|---|---:|---:|---:|
| `talos_1b_32k` **v2** | 4.5039 | 4.5195 | **−1.83% / −2.04%** |
| `talos_1b_32k` v1 (preserved in git history) | 4.4216 | 4.4273 | baseline |
| `talos_1b_49k` | 4.6411 | 4.6525 | −4.73% / −4.84% |
| byte baseline | 1.0 | 1.0 | — |

Read the direction carefully: **more characters per token is better**. An
earlier team verdict that "more training data makes compression worse" was a
metric-direction inversion and is retracted; compression improves
monotonically with both sample size and vocab size in the measured sweep.
`talos_1b_49k` is ~3% better than the 32K v2 on the same slices.

#### Guards (all fail loudly, none are advisory)

- **sha256 verification at load time.** A modified or truncated asset raises
  instead of tokenizing silently (`tokenizer/assets.py`,
  `TokenizerAsset.verify`).
- **Asset ↔ preset pairing.** An asset may only be loaded for a preset listed
  in its manifest `presets` field. Both 1B assets refuse `tiny_*`; a
  vocab-1024 tokenizer refuses every 1B preset; `talos_1b_49k` is refused for
  `talos_1b_21l` / `talos_1b_spec16` and `talos_1b_32k` is refused for
  `talos_1b`.
- **Exact vocab equality for the 1B track.** The canonical track keeps the
  historical `tokenizer_vocab <= model_vocab` contract; the 1B track requires
  exact equality (49,152 for `talos_1b`, 32,000 for the two siblings), which is
  what stops a stray 1,024-vocab tokenizer — or the wrong 1B asset — from
  slipping under a model's embedding matrix.
- **Checkpoint tokenizer metadata.** Every checkpoint and `metrics.json`
  records the tokenizer's sha256 fingerprint and is validated on resume; a
  packed corpus's manifest records the fingerprint its rows were built with.

#### What you must regenerate

**Any dataset tokenized with ANY other tokenizer must be regenerated before
training the preset it is meant for.** Token ids are not interchangeable across
assets: every asset learned its own merge set, so the same id means different
bytes in different assets. Concretely, before training **`talos_1b`** a corpus
must be tokenized with **`talos_1b_49k`** — datasets built with the old
1,024-vocab tokenizer, with 32K v1, *or with 32K v2* are all invalid for it.
Before training `talos_1b_21l` / `talos_1b_spec16`, the corpus must be tokenized
with `talos_1b_32k` (v2). The pairing guards refuse a mismatched tokenizer
rather than silently training on it, and a packed corpus whose manifest
fingerprint differs from the tokenizer you pass is rejected.

#### The 49K asset is the decided primary tokenizer

The owner **decided the 1B vocabulary on 2026-11-07**: `talos_1b` moves to
49,152 (`talos_1b_49k`) for **1,036,060,672 params**, superseding the
2026-11-04 figure of 1,000,933,376 at vocab 32,000. `talos_1b_21l`
(995,973,120) and `talos_1b_spec16` (774,440,960) keep vocab 32,000 and the
32K v2 asset. Both assets are sha256-pinned and were not modified by the
decision — only the manifest's `presets` lists and the model config moved.

### 0.3 Rebuilding the assets

```bash
# fetch the documented public sample (not committed; verified against the pinned sha256)
python -m scripts.fetch_tokenizer_sample --num-rows 2000 --start-offset 0 \
    --output /tmp/tokscale/part_0.jsonl
# ... repeat at offsets 2000,4000,...,14000 and concatenate in order (8 x 2,000 = 16,000 rows)

# train (32K by default; --vocab-size 49152 builds the 49K primary-1B asset)
python -m scripts.train_tokenizer_32k --sample /tmp/tokscale/train.jsonl \
    --output assets/tokenizers/talos_1b_32k.json --report

# measure held-out compression
python -m scripts.fetch_tokenizer_sample --start-offset 1000000 --num-rows 1000 \
    --output /tmp/tokscale/fw_heldout_1k.jsonl
python -m scripts.measure_tokenizer_compression --asset talos_1b_32k \
    --slice heldout=/tmp/tokscale/fw_heldout_1k.jsonl
```

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
  a hand-maintained script table. This matches the Talos plan's requirement for
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

- Default `vocab_size = 32768`; `tokenizer/train.py` accepts any `--vocab-size`
  (e.g. 32768, 65536, 131072). The **shipped** assets use 32,000 and 49,152
  (§0.2) — note that `scripts/train_tokenizer_32k.py` deliberately restricts
  `--vocab-size` to those two choices.
- Of the *model* presets, only the design-only `100b`/`400b` MoE configs use a
  131,072 (128K) vocab; every shipped preset uses 1,024, 32,000 or 49,152.
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

### Two encoders, and why

`tokenizer/bpe.py` contains **two** greedy encoders:

- `encode_bytes` — the original **legacy heap-based** encoder. It is retained
  because it is what every vocab-1024 tokenizer was trained, saved and
  fingerprinted with; changing it would silently change existing checkpoints'
  token streams.
- `encode_bytes_exact` — the correct greedy BPE encoder, selected by
  `ByteLevelBPETokenizer(exact_bpe=True)`. Every **new** tokenizer (the 32K and
  49K 1B assets) sets it, and the flag is persisted inside the saved
  `tokenizer.json`, so a reload cannot silently switch encoders.

The legacy encoder's defect was diagnosed during the Styx generation-quality
investigation: it does not version its heap entries, so a stale entry can be
applied to a node whose token has already changed, which makes it **drop merge
chains** — it returns a longer token sequence than the true greedy BPE
segmentation. The exact encoder versions every entry and discards stale ones,
so it always pops the global minimum-rank pair. The measured gap is large: on
the 49K asset's own training sample, the exact encoder yields **4.7155
chars/token** and the legacy encoder **1.7595** on the same text (both recorded
in the manifest), which is why the 1B assets are pinned to the exact path.

## Integration with the model vocab (`model_compat`)

`tokenizer/model_compat.py` defines the contract between the tokenizer's actual
vocab and `model.config.ModelConfig.vocab_size` (the max embedding width):

- `tokenizer_vocab_size <= model_vocab_size` must hold (the canonical track's
  historical contract; the 1B track additionally requires exact equality — see
  §0.2).
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
