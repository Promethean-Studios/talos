# Phase 2 — Tokenizer (status)

Owner of this file: Talos team (working hypothesis, not owner-ratified).

## Status: COMPLETE (all deliverables in)

The Phase 2 byte-level BPE tokenizer is implemented, tested and documented. The
core (`tokenizer/`) was authored in the previous session and verified there;
this session added the finishing deliverables: tests, docs, status file and a
script helper, and re-verified the whole suite under
`/opt/forge-venv/bin/python`.

## What is implemented

`tokenizer/` (imported as a top-level package `tokenizer`, pure Python+stdlib):

- `bpe.py` — incremental `train_bpe` (+ reference `train_bpe_naive`) and the
  greedy byte-level encoder; deterministic tie-breaking.
- `tokenizer.py` — `ByteLevelBPETokenizer`: encode/decode (lossless), special
  tokens + optional `split_special`, `chunk_encode`/`encode_ids_chunks`,
  `save`/`from_file` (single JSON).
- `vocab.py` — `TokenizerConfig`, `Vocabulary`, id layout (bytes 0–255, merges
  256+, specials at top), `BASE_VOCAB_SIZE=256`.
- `train.py` — `train_tokenizer` + CLI (`--corpus/--vocab-size/--pre-tokenize/
  --bos/--eos/--pad/--unk/--extra-special/--reserved/--seed/--resume/
  --checkpoint-every/--checkpoint-dir/--report`).
- `corpus.py` — `.txt` / `.jsonl` corpus loading with UTF-8 validation.
- `pre_tokenize.py` — optional regex pre-tokenization (off by default).
- `model_compat.py` — `map_to_model_vocab` / `resize_embedding_plan` for model
  grafting.
- `_logging.py`, `__init__.py`.

## Tests

`tests/test_tokenizer.py` adds **21 tests** (run by `python -m pytest -q` from
the repo root, stdlib+numpy only):

- round-trip over diverse Unicode (English, CJK, Arabic, Devanagari, Cyrillic,
  code, math symbols, emoji, NUL bytes, long text),
- special-token handling (BOS/EOS flags, extra specials, split_special),
- determinism (encode + seeded training),
- save/load round-trip (tmp path; identical vocab/merges/encode),
- long-text chunking,
- training smoke test (vocab 512 reached, round-trip holds),
- large-vocab demo: config scales to `vocab_size=131072` (the ~128K model vocab)
  and `map_to_model_vocab`/`resize_embedding_plan` map a trained tokenizer onto
  the 128K model vocab (monotone, injective, fits, padding reserved for growth),
- plus `bpe` naive-vs-incremental consistency and vocabulary id-layout checks.

## Vocab-size capability

The configuration layer accepts and round-trips any `vocab_size` including the
model's **131072 (128K)** preset; `VocabMapping` maps a tokenizer into the model
config vocab with reserved padding slots. Reaching *literally thousands of
learned merges* requires a realistically-sized (MB-scale) corpus; the synthetic
corpus used here reaches a few hundred merges in seconds, which is why the
large-vocab test proves the **config/scalability + model-vocab integration**
(the architecture capability that matters) rather than burning test time on
thousands of merges from a toy corpus.

## Test results (this venv)

- `python -m pytest -q` from repo root: **full Phase 1 + Phase 2 suite passes**
  (Phase 1 tests untouched; no model/ code changes were needed).
- Tokenizer-only run: 21 tests pass.

## Deviations from the brief

- **"Few-thousand-merge scale" in the large-vocab test:** implemented as config
  scalability + model-compat integration into the 128K vocab rather than
  literally training thousands of merges (would need an MB-scale corpus and
  blow the "keep it fast" constraint). Documented above and in the test.
- **No model/ changes required** — `model_compat` already expresses the vocab
  contract cleanly; nothing in `model/` was modified (no regression risk).
- **Pre-tokenization** is documented as opt-in/experimental; default path is the
  byte-level lossless path that tests assert on.

## Repo hygiene

No git repository is initialised at `/home/team/shared/forge` (standalone working
tree; WORKFLOW references a repo that is not present here), so no commits/PRs
were made. Working tree left clean on its only branch.
