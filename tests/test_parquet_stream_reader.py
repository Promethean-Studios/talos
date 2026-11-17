"""Offline unit tests for the bounded parquet read path in :mod:`data.readers`.

The bounded path (:class:`ParquetStreamReader`, :class:`HuggingFaceParquetReader`,
``_import_pyarrow``) shipped in PR #46 without unit tests, and two real bugs got
through as a result:

* ``import pyarrow`` alone does **not** load the ``pyarrow.parquet`` submodule
  (nothing else in the bounded path imports ``datasets``, which used to load it
  as a side effect), so ``pa.parquet.ParquetFile`` raised ``AttributeError`` at
  corpus scale;
* a test monkeypatch was dead because the packer's default reader had become
  the parquet path.

Every test here builds its parquet fixtures locally with ``pyarrow`` and touches
no network: local files only, and the one HF-resolution test injects the file
list explicitly (``files=``) so :func:`resolve_hf_parquet_files` is never called.

Coverage:

* doc/batch ordering through :class:`ParquetStreamReader` (across files, across
  row groups, and across ``batch_size`` values);
* text-field handling (custom column, null rows, non-string rows, and the fact
  that only the text column is materialised);
* file-listing resolution via explicit ``files=`` plus the laziness/caching of
  resolution in :class:`HuggingFaceParquetReader`;
* ``_import_pyarrow()`` returning a module whose ``.parquet.ParquetFile`` works,
  asserted in a *fresh interpreter* (the exact shape of the shipped bug);
* doc-order parity against an independent full-table read, and against the
  ``datasets`` path when ``datasets`` is installed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from data.readers import (  # noqa: E402  (after the pyarrow guard)
    DEFAULT_PARQUET_BATCH_SIZE,
    HuggingFaceParquetReader,
    ParquetStreamReader,
    _import_pyarrow,
    _select_parquet_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Documents per fixture file. Distinct prefixes make cross-file ordering
#: violations (a later file leaking before an earlier one) obvious.
FILE_TEXTS: Dict[str, List[str]] = {
    "part-00000.parquet": [f"a{i} document text" for i in range(5)],
    "part-00001.parquet": [f"b{i} document text" for i in range(4)],
    "part-00002.parquet": [f"c{i} document text" for i in range(3)],
}


def _write_parquet(
    path: Path,
    texts: Sequence[Optional[str]],
    *,
    row_group_size: Optional[int] = None,
    extra: Optional[Dict[str, Sequence[object]]] = None,
    column: str = "text",
) -> str:
    """Write one parquet file with ``texts`` under ``column`` and return its path."""
    columns: Dict[str, object] = {column: list(texts)}
    if extra:
        columns.update({k: list(v) for k, v in extra.items()})
    table = pa.table(columns)
    kwargs: Dict[str, object] = {}
    if row_group_size is not None:
        kwargs["row_group_size"] = row_group_size
    pq.write_table(table, str(path), **kwargs)
    return str(path)


@pytest.fixture()
def shard_dir(tmp_path: Path) -> Path:
    """Three local parquet shards, each split into 2-row row groups."""
    out = tmp_path / "shards"
    out.mkdir()
    for name, texts in FILE_TEXTS.items():
        _write_parquet(out / name, texts, row_group_size=2, extra={"id": range(len(texts))})
    return out


@pytest.fixture()
def shard_paths(shard_dir: Path) -> List[str]:
    """The three shards in file-name (= resolution) order."""
    return [str(shard_dir / name) for name in sorted(FILE_TEXTS)]


# ---------------------------------------------------------------------------
# ParquetStreamReader: ordering
# ---------------------------------------------------------------------------
def test_stream_reader_doc_order_across_files(shard_paths: List[str]) -> None:
    """Documents come out in file order, rows in file order, files in given order."""
    reader = ParquetStreamReader(shard_paths, source="fixture")
    records = list(reader)
    expected = [t for texts in FILE_TEXTS.values() for t in texts]
    assert [r["text"] for r in records] == expected
    assert {r["source"] for r in records} == {"fixture"}
    assert reader.files_done == len(shard_paths)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, DEFAULT_PARQUET_BATCH_SIZE])
def test_batch_size_does_not_change_doc_order(
    shard_paths: List[str], batch_size: int
) -> None:
    """The transient-memory unit (batch_size) never reorders or drops documents."""
    records = list(ParquetStreamReader(shard_paths, batch_size=batch_size))
    expected = [t for texts in FILE_TEXTS.values() for t in texts]
    assert [r["text"] for r in records] == expected


def test_explicit_path_order_is_honoured(shard_dir: Path) -> None:
    """A reversed file list yields reversed file blocks (order is not re-sorted)."""
    names = sorted(FILE_TEXTS)
    paths = [str(shard_dir / n) for n in reversed(names)]
    texts = [r["text"] for r in ParquetStreamReader(paths)]
    expected = [t for n in reversed(names) for t in FILE_TEXTS[n]]
    assert texts == expected


def test_num_records_sums_files_and_defaults(shard_paths: List[str]) -> None:
    """``num_records`` is the metadata sum across files; source defaults."""
    reader = ParquetStreamReader(shard_paths)
    assert reader.num_records() == sum(len(v) for v in FILE_TEXTS.values())
    assert reader.source == "hf-parquet"
    assert reader.pre_buffer is False  # bounded by default (no ReadRangeCache)


def test_stream_reader_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ParquetStreamReader([])
    with pytest.raises(ValueError):
        ParquetStreamReader([str(tmp_path / "x.parquet")], batch_size=0)


# ---------------------------------------------------------------------------
# ParquetStreamReader: text-field handling
# ---------------------------------------------------------------------------
def test_custom_text_field_is_used(tmp_path: Path) -> None:
    path = _write_parquet(
        tmp_path / "custom.parquet",
        ["content one", "content two"],
        extra={"text": ["ignored one", "ignored two"], "id": [1, 2]},
        column="content",
    )
    records = list(ParquetStreamReader([path], text_field="content"))
    assert [r["text"] for r in records] == ["content one", "content two"]


def test_only_text_and_source_are_materialised(tmp_path: Path) -> None:
    """Non-text columns never reach the record (the bounded path decodes text only)."""
    path = _write_parquet(
        tmp_path / "wide.parquet",
        ["only doc"],
        extra={"url": ["u1"], "id": [7]},
    )
    records = list(ParquetStreamReader([path]))
    assert records == [{"text": "only doc", "source": "hf-parquet"}]


def test_null_and_non_string_rows_are_skipped(tmp_path: Path) -> None:
    """Null text cells are skipped, not yielded as ``None`` or as the string 'None'."""
    nulls = _write_parquet(
        tmp_path / "nulls.parquet", ["keep one", None, "keep two", None, ""]
    )
    assert [r["text"] for r in ParquetStreamReader([nulls])] == [
        "keep one",
        "keep two",
        "",
    ]


def test_non_string_text_column_yields_nothing(tmp_path: Path) -> None:
    """Pointing ``text_field`` at a non-string column drops every row (no crash)."""
    ints = _write_parquet(
        tmp_path / "ints.parquet", [1, 2, 3], column="text"  # type: ignore[arg-type]
    )
    assert list(ParquetStreamReader([ints])) == []


def test_empty_parquet_file_yields_no_records(tmp_path: Path) -> None:
    empty = _write_parquet(tmp_path / "empty.parquet", [])
    reader = ParquetStreamReader([empty])
    assert list(reader) == []
    assert reader.num_records() == 0


# ---------------------------------------------------------------------------
# File-listing resolution via explicit files (no network)
# ---------------------------------------------------------------------------
def test_explicit_files_skip_resolution(
    shard_paths: List[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``files=`` is used verbatim: the resolver is never called, order preserved."""
    import data.readers as readers_mod

    def _boom(*args: object, **kwargs: object) -> List[str]:
        raise AssertionError("resolve_hf_parquet_files must not run for explicit files")

    monkeypatch.setattr(readers_mod, "resolve_hf_parquet_files", _boom)
    reader = HuggingFaceParquetReader("local/fixture", files=shard_paths, batch_size=2)
    assert reader.files == shard_paths
    records = list(reader)
    expected = [t for texts in FILE_TEXTS.values() for t in texts]
    assert [r["text"] for r in records] == expected
    # ``source`` is the dataset id, as for the ``datasets``-based reader.
    assert {r["source"] for r in records} == {"local/fixture"}
    assert reader.bytes_fetched == 0  # local files: nothing fetched
    assert reader.num_records() is None


def test_resolution_is_lazy_and_cached(
    shard_paths: List[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing never resolves; the first ``.files`` access resolves once."""
    import data.readers as readers_mod

    calls: List[Dict[str, object]] = []

    def _fake_resolve(dataset: str, **kwargs: object) -> List[str]:
        calls.append({"dataset": dataset, **kwargs})
        return list(shard_paths)

    monkeypatch.setattr(readers_mod, "resolve_hf_parquet_files", _fake_resolve)
    reader = HuggingFaceParquetReader(
        "local/fixture", split="validation", config="sample-10BT", batch_size=4
    )
    assert calls == []  # lazy: construction touches nothing
    assert reader.files == shard_paths
    assert reader.files == shard_paths
    assert len(calls) == 1  # cached after the first access
    assert calls[0] == {
        "dataset": "local/fixture",
        "config": "sample-10BT",
        "split": "validation",
        "revision": None,
    }
    # Iteration reuses the resolved list rather than resolving again.
    assert len(list(reader)) == sum(len(v) for v in FILE_TEXTS.values())
    assert len(calls) == 1


def test_reader_close_releases_stream_reader(
    shard_paths: List[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``close()`` on the HF reader closes the underlying bounded reader."""
    import data.readers as readers_mod

    monkeypatch.setattr(
        readers_mod, "resolve_hf_parquet_files", lambda *a, **k: list(shard_paths)
    )
    reader = HuggingFaceParquetReader("local/fixture", batch_size=2)
    gen = reader._iter_records()
    assert next(gen)["text"].startswith("a0")
    inner = reader._reader
    assert inner is not None
    reader.close()
    assert inner._closed is True
    assert reader._reader is None


def test_select_parquet_paths_orders_like_datasets() -> None:
    """The listing filter keeps only ``.parquet``, filters config/split, and sorts."""
    listing = [
        "sample/10BT/002.parquet",
        "sample/10BT/001.parquet",
        "sample/100BT/000.parquet",
        "README.md",
        "sample/10BT/train-00000.parquet",
    ]
    # Config filter accepts the slash form the Hub uses (``sample-10BT``).
    assert _select_parquet_paths(listing, config="sample-10BT", split="train") == [
        "sample/10BT/train-00000.parquet"
    ]
    # No split match keeps the whole config, still sorted.
    assert _select_parquet_paths(listing, config="sample-10BT", split="validation") == [
        "sample/10BT/001.parquet",
        "sample/10BT/002.parquet",
        "sample/10BT/train-00000.parquet",
    ]
    # No config: every parquet file, sorted, non-parquet dropped.
    assert _select_parquet_paths(listing, config=None, split="") == [
        "sample/100BT/000.parquet",
        "sample/10BT/001.parquet",
        "sample/10BT/002.parquet",
        "sample/10BT/train-00000.parquet",
    ]


# ---------------------------------------------------------------------------
# _import_pyarrow: the ``pyarrow.parquet`` submodule requirement
# ---------------------------------------------------------------------------
def test_import_pyarrow_exposes_parquet_submodule() -> None:
    """``_import_pyarrow()`` returns a module with a usable ``.parquet.ParquetFile``."""
    module = _import_pyarrow()
    assert hasattr(module, "parquet")
    assert module.parquet.ParquetFile is pq.ParquetFile


def test_import_pyarrow_loads_parquet_in_fresh_interpreter() -> None:
    """Regression for the shipped bug: a bare ``import pyarrow`` is not enough.

    Runs in a *fresh* interpreter (so no earlier ``import pyarrow.parquet`` can
    mask the failure) and asserts ``_import_pyarrow()`` leaves the submodule in
    ``sys.modules`` and that its ``ParquetFile`` is constructible on a local file.
    """
    script = "\n".join(
        [
            "import sys",
            "from data.readers import _import_pyarrow",
            "before = 'pyarrow.parquet' in sys.modules",
            "mod = _import_pyarrow()",
            "loaded = 'pyarrow.parquet' in sys.modules",
            "cls = mod.parquet.ParquetFile",
            "print(f'{before} {loaded} {cls.__name__}')",
        ]
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    before, loaded, name = proc.stdout.split()
    assert before == "False", "pyarrow.parquet was already imported; test is not fresh"
    assert loaded == "True", "_import_pyarrow() did not import pyarrow.parquet"
    assert name == "ParquetFile"


# ---------------------------------------------------------------------------
# Doc-order parity: bounded path vs. an independent read, and vs. ``datasets``
# ---------------------------------------------------------------------------
def test_doc_order_parity_with_full_table_read(shard_paths: List[str]) -> None:
    """Streamed order equals the order of a whole-table read (rows in file order)."""
    expected: List[str] = []
    for path in shard_paths:
        column = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        expected.extend(t for t in column if isinstance(t, str))
    streamed = [r["text"] for r in ParquetStreamReader(shard_paths, batch_size=2)]
    assert streamed == expected


def test_doc_order_parity_with_datasets(
    shard_paths: List[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same documents in the same order as the ``datasets`` path (when installed).

    ``datasets`` is an optional dependency, so this skips when it is absent; the
    full-table parity test above is the always-on guard. The load is local
    (``data_files`` on disk), so no network is touched.
    """
    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    ds = datasets.load_dataset(
        "parquet",
        data_files=shard_paths,
        split="train",
        cache_dir=str(tmp_path / "hf-cache"),
        download_mode="force_redownload",
    )
    from_datasets = [row["text"] for row in ds if isinstance(row["text"], str)]
    from_stream = [r["text"] for r in ParquetStreamReader(shard_paths, batch_size=2)]
    assert from_stream == from_datasets
