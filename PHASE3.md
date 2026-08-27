# Phase 3 — Data pipeline (status)

Owner: Talos team (working hypothesis, not owner-ratified).

## Status: COMPLETE

The Phase 3 data pipeline is fully implemented, tested and documented under
`data/`. It replaces the `data/PLAN.md` placeholder (kept, updated) with real,
research-grade, executable code.

## What is implemented

`data/` (pure stdlib + numpy; optional extras guarded):

- **Readers** (`readers.py`) — `DatasetReader` interface + `JSONLReader`
  (incl. `.gz`), `TextReader` (whole-file or paragraph-split), `ParquetReader`
  (pyarrow, optional), `HuggingFaceReader` (datasets, optional). All streaming;
  records are `dict{"text": ..., **metadata}`. `reader_from_config` maps a
  declarative dict to an instance.
- **Processors** (`processors.py`, `dedup.py`, `contamination.py`, `langid.py`)
  — one `Processor.process(record) -> record|None` contract:
  length, quality-heuristic, code-fence, blacklist-regex, url-blacklist,
  language filter (pluggable `LanguageIdentifier`, default dependency-free
  heuristic, optional langdetect backend), exact sha256 dedup, MinHash
  near-dedup (real shingled MinHash + banded LSH, seeded), contamination
  framework (substring/doc-id + optional word fuzzy-ngram, drop/flag), and a
  tokenizer-aware token counter wired to the Phase-2 `tokenizer/`.
- **Mixer** (`mixer.py`) — `WeightedMixer`: deterministic smooth weighted
  round-robin; supports `seed`, per-source weights, epoch (`n>1`) and streaming
  (`num_epochs=None`) modes; `num_epochs=1` default is the safe finite mode.
- **Writers** (`writers.py`) — `ShardedWriter`: JSONL (and optional Parquet)
  shards with per-shard record/byte limits, atomic temp→rename writes, and a
  `manifest.json` recording shard name/count/per-source distribution.
- **Pipeline + CLI** (`pipeline.py`) — `python -m data.pipeline --config X
  --out DIR [--max-docs N] [--seed N]`; `run_pipeline()` is the same logic
  callable from Python; `--dry-run` prints the config without running.
- **Download/prep** (`setup/`) — declarative registry of legally-usable public
  datasets (The Pile, RedPajama, FineWeb, StarCoder, CodeAlpaca, NuminaMath,
  Wikipedia, C4) with licensing notes + a `python -m data.setup.download
  --dataset NAME --out DIR` CLI (`--list` / `--dry-run` need no network). No
  data is vendored; nothing is fetched at build/test time.
- **Example config** (`configs/example.json`) and full docs (`data/README.md`).

## Tests

`tests/test_data.py` adds **29 tests** (28 run + 1 skipped), all CPU/stdlib+
numpy, no network:

- Reader round-trips: JSONL (round-trip, malformed-skip), text, Parquet
  (`importorskip pyarrow`).
- Each processor: filters drop bad / keep good; exact + MinHash near-dedup;
  contamination drops/flags configured substrings + doc-IDs + fuzzy-ngram;
  language heuristic classifies en/ja/ru/ar; token counter (bytes fallback and
  real tokenizer).
- Mixer: same seed ⇒ identical ordering; different seeds differ; prefix
  weights ≈ 4:1; epoch mode.
- Sharded writer: correct shard counts under size limits (records + bytes),
  valid manifest, atomicity (no leftover `.tmp`), re-readable shards.
- **Integration**: a tiny multilingual+code+math+noise synthetic corpus runs
  the full pipeline (read → process → mix → shard) via `run_pipeline` with
  `--max-docs`, and the output shards + manifest re-read cleanly.
- **Determinism**: two full pipeline runs with the same seed produce identical
  output text.
- Setup registry + config validation.

## Test results (this venv)

`/opt/forge-venv/bin/python -m pytest -q` from the repo root: **the full
Phase 1 + Phase 2 + Phase 3 suite passes** (EXIT=0). Phase 1/2 tests were not
modified; `model/` and `tokenizer/` are untouched (no regression).

## Deviations from the brief

- **Parquet / HF / langdetect tests** are skipped on this machine because the
  optional deps are not installed (by design they are optional); the code paths
  are implemented and guarded with lazy imports + `importorskip`.
- **NuminaMath / OpenWebMath**: prefer documented lightweight samples (e.g.
  NuminaMath-CoT) rather than the heavy full torrents; noted in the registry.
- **MinHash `threshold`** is surfaced as a documented tuning knob driving the
  band geometry choice rather than a hard runtime filter; the working default
  (64 perms / 8 bands × 8 rows) reliably catches the near-dup test case.

## Repo hygiene

As with Phases 1–2, `/home/team/shared/forge` is a standalone working tree with
no git repository initialised (WORKFLOW references a repo that is not present
here), so no commits/PRs were made. Working tree left clean.
