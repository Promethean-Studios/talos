# Talos data pipeline (Phase 3)

A modular, configurable, **streaming** data-processing pipeline for building
mixed training corpora: read → process → mix → shard. It is pure stdlib+numpy;
everything heavier (pyarrow, HuggingFace `datasets`, langdetect) is optional and
guarded with lazy imports so the core runs anywhere.

Quick start:

```bash
# run a pipeline described by a JSON config, capping output for a smoke run
/opt/forge-venv/bin/python -m data.pipeline \
    --config data/configs/example.json --out /tmp/mycorp \
    --max-docs 10000 --seed 0

# or build a corpus from a known public dataset (dry-run first = no network)
/opt/forge-venv/bin/python -m data.setup.download --list
/opt/forge-venv/bin/python -m data.setup.download --dataset fineweb --out /tmp/raw --dry-run
```

---

## Architecture

The pipeline is a linear graph of four independent, composable pieces, each
owned by one module:

| Stage | Module | Responsibility |
|-------|--------|----------------|
| **Read** | `data/readers.py` | turn raw files / external corpora into a stream of `Record` dicts |
| **Process** | `data/processors.py` (+ `dedup.py`, `contamination.py`, `langid.py`) | filter / annotate records |
| **Mix** | `data/mixer.py` | deterministically blend weighted sources |
| **Write** | `data/writers.py` | shard records + write a manifest |

A `Record` (`data/types.py`) is a plain `dict` with at least a `text` field
plus arbitrary metadata (`url`, `source`, `lang`, `quality`, `num_tokens`, ...).
Using `dict` keeps it trivially JSON/Parquet serialisable and lets processors
attach derived fields without changing types.

```
[JSONL] [Parquet] [HF] [text]          readers  (streaming)
   \       |        |     /               |
    ->  ProcessorChain  <-               |  Exactly one interface each:
        length, quality, lang,            |    Reader.__iter__ -> Record
        dedup, minhash, contam,           |    Processor.process(rec) -> rec|None
        token_count, ...                 v
                     WeightedMixer  ----+  Mixer(seed) -> deterministic stream
                            |            v
                    ShardedWriter  ----->  writer.write(rec); close() -> manifest
```

### Reader interface
Every reader subclasses `DatasetReader` and implements `_iter_records()` which
yields `dict`s with a `text` key. `__iter__` validates each record and stamps a
`source` field. Adding a format is one small subclass + one entry in
`reader_from_config`:

```python
from data.readers import DatasetReader, reader_from_config

class MyReader(DatasetReader):
    source = "myfmt"
    def _iter_records(self):
        for chunk in ...:
            yield {"text": chunk, "url": ..., "extra_meta": ...}

cfg = {"type": "myfmt", "path": "...", "source": "mine"}
reader = reader_from_config(cfg)   # now registered
```

Built-in readers: `jsonl` (incl. `.gz`), `text` (whole file or
paragraph-split), `parquet` (pyarrow), `huggingface` (datasets lib).

### Processor interface
Every stage follows one contract — `process(record) -> record | None`, where
`None` **drops** the document. Processors are stateful when they need to be
(dedup, contamination) and are composed via `ProcessorChain` or listed directly
in the config. Built-ins:

* **`length`** — drop docs outside `[min_chars, max_chars]`.
* **`quality_heuristic`** — dependency-free heuristics (most-common-char ratio,
  punctuation ratio, newline ratio, bullet/nav-line ratio, min sentence count).
  Documented in the class docstring.
* **`code_fence`** — keep/drop code vs. prose (`mode="keep"|"drop"`,
  `min_code_ratio`), using keyword/indent/brace/assignment line heuristics.
* **`blacklist_regex`** / **`regex_filter`** / **`url_blacklist`** — drop by
  regex or blocked domains.
* **`language`** — pluggable `LanguageIdentifier`; default heuristic (byte /
  Unicode-block + stopword + bigram) is dependency-free; optional `langdetect`
  backend. `allow=[...]` restrict to a language set; `drop_unknown=True` drops
  untagged docs.
* **`exact_dedup`** — sha256 of normalised text (case/whitespace folded).
* **`minhash_dedup`** — near-duplicates; real shingled MinHash + banded LSH,
  seeded and deterministic (64 permutations / 8 bands × 8 rows by default).
* **`contamination`** — drop or flag (`action="flag"`) docs matching a
  caller-supplied list of contaminated substrings / doc-IDs; optional word
  `fuzzy_ngram` catches lightly-paraphrased copies. The framework is shipped;
  the actual benchmark answer lists are the caller's (never hard-coded).
* **`token_count`** — integrates the Phase-2 `tokenizer/` to add `num_tokens`
  per doc; bytes-fallback when no tokenizer supplied.
* **`field_remap`** — map e.g. `content` → `text`.

### Mixer
`WeightedMixer(sources, seed, num_epochs=1)` blends `(iterable, weight)` pairs
into one deterministic stream. Smooth weighted round-robin gives each source its
weight share of the ordering, and the result is fully reproducible from `seed`
(same seed ⇒ byte-identical output). `num_epochs=1` (default) is the safe finite
mode; `num_epochs=None` enables streaming (restart exhausted sources — only for
truly infinite readers); `num_epochs=n` gives n-pass epoch mixing.

### Sharded writer
`ShardedWriter(out_dir, shard_size, size_by="records"|"bytes", format="jsonl"|"parquet", prefix)`
buffers records, flushes numbered shards (temp-file + atomic rename), then
writes a `manifest.json` listing each shard's file, record count and
per-`source` distribution. Interrupted runs never leave a half-written shard
under its real name.

---

## Configuring a mixed multilingual + code + math corpus

A pipeline config is a JSON object with `sources`, `processors`, `output` and
(optionally) `num_epochs`. See `data/configs/example.json`. Each source has a
`reader` (format + path) and a `weight`. Order of `processors` matters.

```json
{
  "sources": [
    {"name": "web", "weight": 0.45,
     "reader": {"type": "jsonl", "path": "/data/web.jsonl", "source": "web"}},
    {"name": "code", "weight": 0.30,
     "reader": {"type": "jsonl", "path": "/data/code.jsonl", "source": "code"}},
    {"name": "math", "weight": 0.15,
     "reader": {"type": "jsonl", "path": "/data/math.jsonl", "source": "math"}},
    {"name": "multilingual", "weight": 0.10,
     "reader": {"type": "jsonl", "path": "/data/raw/multilingual.jsonl", "source": "multilingual"}}
  ],
  "processors": [
    {"type": "length", "min_chars": 200, "max_chars": 10000000},
    {"type": "quality_heuristic"},
    {"type": "language", "allow": ["en","zh","fr","de","es","ja","ko","ru"]},
    {"type": "exact_dedup", "normalize": true},
    {"type": "minhash_dedup"},
    {"type": "contamination", "contaminated_texts": ["..."],
     "contaminated_ids": ["..."], "fuzzy_ngram": 8, "action": "drop"},
    {"type": "token_count"}
  ],
  "num_epochs": 1
}
```

Run it:
```bash
/opt/forge-venv/bin/python -m data.pipeline --config data/configs/example.json --out /tmp/out --max-docs 50000 --seed 0
```
The `--max-docs` cap makes smoke runs cheap; drop it for a full pass. The writer
emits `shard-0000.jsonl`, ..., and `manifest.json`.

`data.setup.download.py` prepares each raw source (web / code / math / polyglot)
from public datasets via `--dataset`. Wire each prepared directory into the
`reader.path` fields above.

---

## Adding a new dataset / format

1. **New file format:** subclass `DatasetReader`, implement `_iter_records`,
   register in `reader_from_config` (`data/readers.py`). Done.
2. **New public dataset:** add a `DatasetSpec` to `data/setup/datasets.py`
   (name, license + URL, HF id / urls, text field, notes) — no data is
   vendored. `python -m data.setup.download --dataset NAME --out DIR` fetches
   and shards it; `--dry-run`/`--list` describe it without network.

---

## Licensing notes

Talos never ships copyrighted training data in the repo. `data/setup/datasets.py`
only records *where* legally-usable public datasets live and their licenses.
Review the per-dataset license before use — several (Pile, StarCoder, C4,
RedPajama, FineWeb, Wikipedia) have per-source or share-alike terms. OpenWebMath
/ NuminaMath full torrents are heavy; prefer the documented lighter samples.

## Determinism notes

* Every random source in the pipeline is seeded: the mixer (`seed`), MinHash
  permutations (`seed`), and language heuristics (deterministic functions).
  Processing order is deterministic given the same config and seed.
* Hash-based processors (dedup, contamination) use `hashlib` only.
* Writes are atomic (temp + rename); shard order follows processing order, so an
  identical input+seed yields identical shards.
* Run-to-run identical outputs require identical *input ordering*; the reader
  order is the on-disk file order (sort your globs if you want stability).

## Tests

`tests/test_data.py` covers reader round-trips (JSONL/text/Parquet-skip),
every processor, mixer determinism + weights, sharded writing + manifest, a
full end-to-end pipeline over a tiny synthetic multilingual+code+math corpus,
and pipeline determinism. Parquet tests use `pytest.importorskip("pyarrow")`.
No network is used anywhere.
