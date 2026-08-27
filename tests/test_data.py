"""Tests for the Phase 3 data pipeline.

All fixtures are synthetic / in-memory / small temp files — no network access.
Optional dependencies (pyarrow, datasets, langdetect) are guarded with
``pytest.importorskip`` so the suite runs on stdlib + numpy alone.
"""
from __future__ import annotations

import json
import os

import pytest

from data import (
    BlacklistRegexFilter,
    CodeFenceFilter,
    ExactDedupFilter,
    FieldRemap,
    HeuristicLanguageIdentifier,
    JSONLReader,
    LanguageFilter,
    LengthFilter,
    MinHashNearDupFilter,
    QualityHeuristicFilter,
    RegexFilter,
    ShardedWriter,
    TextReader,
    TokenCounter,
    URLBlacklistFilter,
    WeightedMixer,
)
from data.contamination import ContaminationFilter
from data.pipeline import _load_config, run_pipeline
from data.setup.datasets import get_dataset

GOOD_ENGLISH = (
    "The quick brown fox jumps over the lazy dog near the river bank while the "
    "sun sets behind the distant hills and the birds return to their nests at "
    "the end of another calm and peaceful day in the countryside. Everyone was "
    "delighted to hear the news about the upcoming festival they had waited "
    "for all year long."
)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def test_jsonl_reader_roundtrip(tmp_path):
    p = tmp_path / "a.jsonl"
    rows = [
        {"text": "hello world", "url": "http://x", "n": 1},
        {"text": "second document here", "url": "http://y", "n": 2},
    ]
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    recs = list(JSONLReader(str(p), source="s1"))
    assert len(recs) == 2
    assert recs[0]["text"] == "hello world"
    assert recs[0]["url"] == "http://x"
    assert recs[0]["source"] == "s1"
    assert recs[1]["n"] == 2


def test_jsonl_reader_skips_malformed(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"text": "ok"}\nnot-json\n{"n": 1}\n', encoding="utf-8")
    recs = list(JSONLReader(str(p)))
    assert [r["text"] for r in recs] == ["ok"]  # non-json + missing-text dropped


def test_text_reader_split(tmp_path):
    a = tmp_path / "doc.txt"
    a.write_text("para one content here.\n\npara two content here.", encoding="utf-8")
    whole = list(TextReader(str(a)))
    assert len(whole) == 1 and "para one" in whole[0]["text"]
    paras = list(TextReader(str(a), split_paragraphs=True))
    assert len(paras) == 2


def test_parquet_reader_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = tmp_path / "d.parquet"
    pq.write_table(
        pa.table(
            {"text": ["alpha doc", "beta doc"], "url": ["u1", "u2"]}
        ),
        str(p),
    )
    from data.readers import ParquetReader

    recs = list(ParquetReader(str(p), source="pq"))
    assert len(recs) == 2
    assert recs[0]["text"] == "alpha doc"
    assert recs[0]["url"] == "u1"


# --------------------------------------------------------------------------
# Processors
# --------------------------------------------------------------------------
def test_length_filter():
    f = LengthFilter(min_chars=10, max_chars=1000)
    assert f.process({"text": "short"}) is None
    assert f.process({"text": "x" * 2000}) is None
    assert f.process({"text": "a" * 50}) is not None


def test_quality_heuristic_filter():
    f = QualityHeuristicFilter()
    assert f.process({"text": GOOD_ENGLISH}) is not None
    assert f.process({"text": "aaaaaaaaaaaaaaaaaaaaaaaaa"}) is None  # repeated char
    assert f.process({"text": "!!!###$$$%%%&&&((()))__"}) is None  # punctuation spam
    assert f.process({"text": "no punctuation at all"}) is None  # < min_sentences


def test_blacklist_regex_filter():
    f = BlacklistRegexFilter([r"\bsell\b", r"\bspam"])
    assert f.process({"text": "buy this now please thank you."}) is not None
    assert f.process({"text": "we sell products here ok."}) is None


def test_regex_filter_keep_and_invert():
    assert RegexFilter(r"\d{4}").process({"text": "year 2024 ok"}) is not None
    assert RegexFilter(r"\d{4}").process({"text": "no digits here"}) is None
    inv = RegexFilter(r"\d{4}", invert=True)
    assert inv.process({"text": "no digits"}) is not None
    assert inv.process({"text": "year 2024"}) is None


def test_url_blacklist_filter():
    f = URLBlacklistFilter(["ads.example", "tracker.net"])
    assert f.process({"text": "x", "url": "http://good.example/page"}) is not None
    assert f.process({"text": "x", "url": "http://ads.example/page"}) is None


def test_code_fence_filter_keep():
    f = CodeFenceFilter(mode="keep", min_code_ratio=0.2)
    code = "def f():\n    return 1\n\nx = f()\nprint(x)\n" * 20
    assert f.process({"text": code}) is not None
    prose = ("Here is some natural human language text about the weather.") * 20
    assert f.process({"text": prose}) is None


def test_field_remap():
    f = FieldRemap(text_field="content")
    rec = {"content": "the actual body", "meta": 1}
    out = f.process(rec)
    assert out["text"] == "the actual body"
    assert f.process({"other": "no content"}) is None


def test_exact_dedup():
    f = ExactDedupFilter()
    assert f.process({"text": "Hello  World"}) is not None
    assert f.process({"text": "hello world"}) is None  # normalize -> dup
    assert f.process({"text": "different text here"}) is not None


def test_minhash_near_dedup():
    f = MinHashNearDupFilter()
    base = GOOD_ENGLISH * 3
    variant = GOOD_ENGLISH.replace("quick brown fox", "swift brown fox") * 3
    distinct = (GOOD_ENGLISH.replace("fox", "elephant") + " Completely different "
                "content and topics and words throughout the entire rest of this "
                "document.") * 3
    assert f.process({"text": base}) is not None
    assert f.process({"text": variant}) is None  # near duplicate
    assert f.process({"text": distinct}) is not None


def test_contamination_filter():
    cont = ["The secret benchmark question asks what capital city is shown in the map"]
    f_drop = ContaminationFilter(contaminated_texts=cont, contaminated_ids=["id-7"])
    assert f_drop.process({"text": "unrelated ordinary text here.", "id": "id-1"}) is not None
    assert f_drop.process({"text": "the secret benchmark question asks what capital "
                                    "city is shown in the map today.", "id": "id-2"}) is None
    assert f_drop.process({"text": "fine text.", "id": "id-7"}) is None  # by doc id
    f_flag = ContaminationFilter(contaminated_texts=cont, action="flag")
    out = f_flag.process({"text": "the secret benchmark question asks what capital "
                                  "city is shown in the map.", "id": "id-3"})
    assert out is not None and out["contaminated"] is True


def test_contamination_fuzzy_ngram():
    cont = ["The capital of France is Paris and the river Seine flows through it yes"]
    f = ContaminationFilter(contaminated_texts=cont, fuzzy_ngram=6)
    # shares a 6-word run with the contaminated text
    hit = "The capital of France is Paris and the river Seine flows through it great"
    miss = "Paris is a lovely city with many cafes and the weather is usually fine"
    assert f.process({"text": hit, "id": "a"}) is None
    assert f.process({"text": miss, "id": "b"}) is not None


def test_language_filter_heuristic():
    f = LanguageFilter(allow=["en"])
    assert f.process({"text": GOOD_ENGLISH}) is not None
    ru = "Привет, это русский текст для проверки определения языка в системе."
    assert f.process({"text": ru}) is None  # not in allow set
    f2 = LanguageFilter(allow=["ru"])
    assert f2.process({"text": ru}) is not None
    assert f2.process({"text": ru})["lang"] == "ru"


def test_language_identifier_direct():
    li = HeuristicLanguageIdentifier()
    assert li.identify(GOOD_ENGLISH) == "en"
    assert li.identify("これは日本語のテキストです") == "ja"
    assert li.identify("Привет мир") == "ru"
    assert li.identify("السلام عليكم") == "ar"


def test_token_counter_bytes_fallback():
    f = TokenCounter()
    rec = f.process({"text": "hello"})
    assert rec["num_tokens"] == 5  # utf-8 byte count


def test_token_counter_with_tokenizer(tmp_path):
    from tokenizer.vocab import TokenizerConfig
    from tokenizer.tokenizer import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer(TokenizerConfig())
    # empty vocab -> each byte is one token; "hello" -> 5 bytes -> 5 tokens
    f = TokenCounter(tokenizer=tok)
    rec = f.process({"text": "hello"})
    assert rec["num_tokens"] == 5
    assert f.process({"text": ""})["num_tokens"] == 0


# --------------------------------------------------------------------------
# Mixer
# --------------------------------------------------------------------------
def test_mixer_determinism():
    s0 = [{"text": f"a{i}", "source": "s0"} for i in range(20)]
    s1 = [{"text": f"b{i}", "source": "s1"} for i in range(20)]
    sources = [(s0, 3.0), (s1, 1.0)]

    def run(seed):
        return [r["text"] for r in WeightedMixer(sources, seed=seed)]

    assert run(42) == run(42)
    assert run(42) != run(7)  # different seeds -> different ordering


def test_mixer_weights_roughly_respected():
    s0 = [{"text": f"a{i}", "source": "s0"} for i in range(1000)]
    s1 = [{"text": f"b{i}", "source": "s1"} for i in range(1000)]
    out = list(WeightedMixer([(s0, 4.0), (s1, 1.0)], seed=0))
    # With equal-length sources, every record of each is consumed (order,
    # not count, is weighted) — so check the ratio over a prefix that runs
    # before the heavier source is exhausted.
    prefix = out[:800]
    n0 = sum(1 for r in prefix if r["source"] == "s0")
    n1 = sum(1 for r in prefix if r["source"] == "s1")
    assert 3.0 < n0 / n1 < 5.5


def test_mixer_epoch_mode():
    s0 = [{"text": "x", "source": "s0"}]
    out = list(WeightedMixer([(s0, 1.0)], seed=0, num_epochs=3))
    assert len(out) == 3


# --------------------------------------------------------------------------
# Sharded writer
# --------------------------------------------------------------------------
def test_sharded_writer(tmp_path):
    out = tmp_path / "shards"
    with ShardedWriter(str(out), shard_size=3, format="jsonl", prefix="s") as w:
        for i in range(10):
            w.write({"text": f"doc {i}", "source": "src"})
    manifest = json.load(open(out / "manifest.json"))
    assert manifest["num_shards"] >= 3
    assert manifest["total_records"] == 10
    # re-read all records from shards, sorted by shard then order
    from data.readers import JSONLReader

    all_recs = []
    for entry in manifest["shards"]:
        recs = list(JSONLReader(str(out / entry["shard"])))
        all_recs.extend(recs)
        assert len(recs) <= 3
        assert list(entry["sources"].values()) == [len(recs)]
    assert len(all_recs) == 10


def test_sharded_writer_bytes_mode(tmp_path):
    out = tmp_path / "bytestest"
    with ShardedWriter(str(out), shard_size=30, size_by="bytes") as w:
        for i in range(10):
            w.write({"text": "x" * 8, "source": "a"})
    manifest = json.load(open(out / "manifest.json"))
    assert manifest["total_records"] == 10
    assert manifest["num_shards"] >= 3


def test_sharded_writer_atomic_no_leftover_tmp(tmp_path):
    out = tmp_path / "atomic"
    with ShardedWriter(str(out), shard_size=2) as w:
        w.write({"text": "a"})
        w.write({"text": "b"})
        w.write({"text": "c"})
    leftovers = [f for f in os.listdir(out) if f.endswith(".tmp")]
    assert leftovers == []


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------
def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_pipeline_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_jsonl(raw / "web.jsonl", [{"text": GOOD_ENGLISH, "url": "u"} for _ in range(5)])
    _write_jsonl(raw / "code.jsonl", [{"text": "def f():\n    return 1\n\n" * 10,
                                       "url": "c"} for _ in range(4)])
    _write_jsonl(raw / "math.jsonl", [{"text": "Solve the equation x^2 = 16 for x and "
                                                "show your reasoning carefully here.",
                                       "url": "m"} for _ in range(3)])
    # include a short doc that should be filtered, and a duplicate
    _write_jsonl(raw / "noise.jsonl", [
        {"text": "bleh", "url": "n"},  # too short -> dropped
        {"text": GOOD_ENGLISH, "url": "dup"},  # exact dup of web[0]
    ])

    config = {
        "sources": [
            {"name": "web", "weight": 4.0,
             "reader": {"type": "jsonl", "path": str(raw / "web.jsonl"), "source": "web"}},
            {"name": "code", "weight": 3.0,
             "reader": {"type": "jsonl", "path": str(raw / "code.jsonl"), "source": "code"}},
            {"name": "math", "weight": 2.0,
             "reader": {"type": "jsonl", "path": str(raw / "math.jsonl"), "source": "math"}},
            {"name": "noise", "weight": 1.0,
             "reader": {"type": "jsonl", "path": str(raw / "noise.jsonl"), "source": "noise"}},
        ],
        "processors": [
            {"type": "length", "min_chars": 20},
            {"type": "exact_dedup", "normalize": True},
            {"type": "token_count"},
        ],
        "output": {"shard_size": 3},
        "num_epochs": 1,
    }
    out = tmp_path / "out"
    stats = run_pipeline(config, str(out), max_docs=50, seed=0)
    # input = 5+4+3+2 = 14; noise has a short doc dropped (bleh) and a dup dropped
    assert stats.total_dropped >= 2
    assert stats.total_kept == stats.total_input - stats.total_dropped

    manifest = json.load(open(out / "manifest.json"))
    assert manifest["total_records"] > 0
    # every kept record has num_tokens set and is valid re-readable
    from data.readers import JSONLReader

    seen = []
    for ent in manifest["shards"]:
        seen.extend(JSONLReader(str(out / ent["shard"])))
    assert len(seen) == manifest["total_records"]
    assert all("text" in r and "num_tokens" in r for r in seen)


def test_pipeline_determinism_two_runs(tmp_path):
    raw = tmp_path / "raw2"
    raw.mkdir()
    _write_jsonl(raw / "a.jsonl", [{"text": f"document number {i} with text body",
                                    "source": "a"} for i in range(30)])
    _write_jsonl(raw / "b.jsonl", [{"text": f"b arbitration {i} content here words",
                                    "source": "b"} for i in range(30)])
    config = {
        "sources": [
            {"name": "a", "weight": 1.0,
             "reader": {"type": "jsonl", "path": str(raw / "a.jsonl"), "source": "a"}},
            {"name": "b", "weight": 1.0,
             "reader": {"type": "jsonl", "path": str(raw / "b.jsonl"), "source": "b"}},
        ],
        "processors": [{"type": "length", "min_chars": 1}],
        "output": {"shard_size": 100},
        "num_epochs": 1,
    }

    def collect(out):
        run_pipeline(config, str(out), max_docs=60, seed=123)
        texts = []
        for ent in json.load(open(os.path.join(out, "manifest.json")))["shards"]:
            with open(os.path.join(out, ent["shard"]), encoding="utf-8") as fh:
                for line in fh:
                    texts.append(json.loads(line)["text"])
        return texts

    t1 = collect(str(tmp_path / "o1"))
    t2 = collect(str(tmp_path / "o2"))
    assert t1 == t2


def test_setup_registry():
    spec = get_dataset("fineweb")
    assert spec.license  # documented
    assert spec.hf_id == "HuggingFaceFW/fineweb"
    get_dataset("wiki_multilingual")
    from data.setup.datasets import known_datasets

    assert "numina_math" in known_datasets()


def test_load_config_validates(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"sources": []}', encoding="utf-8")
    cfg = _load_config(str(p))
    assert cfg["sources"] == []
