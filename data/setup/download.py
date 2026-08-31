"""Download + prepare public datasets the way a real project does.

This is a configuration-driven downloader: it maps a ``--dataset NAME`` to a
spec in :mod:`data.setup.datasets`, streams the source (HuggingFace via the
optional ``datasets`` lib, or HTTP for plain URLs), applies a light
prepare step, and writes per-source JSONL shards into ``--out DIR`` that the
pipeline can read directly.

Safety / reproducibility:
* ``--dry-run`` prints exactly what would be fetched and prepared without
  touching the network (safe to run in CI / tests).
* Nothing is fetched at build time; this CLI only runs on demand.
* Licensing per dataset is documented in :mod:`data.setup.datasets`; review it
  before downloading commercial/restricted sources.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

from data._logging import get_logger
from data.setup.datasets import DatasetSpec, get_dataset, known_datasets
from data.writers import ShardedWriter

log = get_logger("setup")


def _fetch_hf(spec: DatasetSpec, out_dir: str, max_docs: Optional[int]) -> None:
    try:
        import datasets  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "HuggingFace datasets require 'pip install datasets'"
        ) from exc

    kwargs: Dict[str, Any] = {}
    if spec.subset:
        kwargs["name"] = spec.subset
    stream = datasets.load_dataset(
        spec.hf_id, split=spec.split, streaming=True, **kwargs
    )
    writer = ShardedWriter(
        os.path.join(out_dir, spec.name), shard_size=10_000, prefix="shard"
    )
    written = 0
    with writer:
        for row in stream:
            text = row.get(spec.text_field)
            if not isinstance(text, str):
                continue
            rec: Dict[str, Any] = {"text": text, "source": spec.name}
            for k in spec.keep_fields:
                if k in row and row[k] is not None:
                    rec[k] = row[k]
            writer.write(rec)
            written += 1
            if max_docs is not None and written >= max_docs:
                break
    log.info("prepared %d docs for %s", written, spec.name)


def _iter_url_lines(url: str) -> Iterator[str]:
    """Stream decoded lines from a (possibly gzipped) URL.

    Memory is O(line): the response is wrapped in :class:`gzip.GzipFile` and
    iterated lazily instead of ``resp.read() -> gzip.decompress -> decode ->
    splitlines()``, which held ~3 full copies of the whole file in RAM at once.
    """
    import gzip  # pylint: disable=import-outside-toplevel
    import urllib.request  # pylint: disable=import-outside-toplevel

    with urllib.request.urlopen(url) as resp:
        if url.endswith(".gz"):
            with gzip.GzipFile(fileobj=resp) as gz:
                for raw in gz:
                    yield raw.decode("utf-8", errors="replace")
        else:
            for raw in resp:
                yield raw.decode("utf-8", errors="replace")


def _fetch_urls(spec: DatasetSpec, out_dir: str, max_docs: Optional[int]) -> None:
    """Fetch plain-URL sources (e.g. gzipped JSONL) into shards."""
    writer = ShardedWriter(
        os.path.join(out_dir, spec.name), shard_size=10_000, prefix="shard"
    )
    written = 0
    with writer:
        for url in spec.urls:
            for line in _iter_url_lines(url):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and isinstance(obj.get(spec.text_field), str):
                    obj["source"] = spec.name
                    writer.write(obj)
                    written += 1
                    if max_docs is not None and written >= max_docs:
                        return
    log.info("prepared %d docs for %s", written, spec.name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download + prepare a public dataset for the Talos pipeline."
    )
    parser.add_argument("--dataset", default=None, help="registry name, see --list")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--max-docs", type=int, default=None, help="cap docs")
    parser.add_argument("--dry-run", action="store_true", help="describe only, no network")
    parser.add_argument("--list", action="store_true", help="list known datasets")
    args = parser.parse_args(argv)

    if args.list:
        for name in known_datasets():
            print(get_dataset(name).describe())
        return 0

    # --list requires neither --dataset nor --out; everything else does.
    if not args.dataset or not args.out:
        parser.error("--dataset and --out are required unless --list is given")

    spec = get_dataset(args.dataset)
    print(spec.describe())
    os.makedirs(args.out, exist_ok=True)
    if args.dry_run:
        print(f"[dry-run] would fetch {spec.source} source for {spec.name}")
        return 0

    if spec.source == "hf":
        _fetch_hf(spec, args.out, args.max_docs)
    elif spec.source in ("http", "torrent"):
        if spec.source == "torrent":
            raise NotImplementedError(
                "torrent sources need a torrent client; use --dry-run and fetch "
                "the .torrent manually, or switch to the http/hf variant."
            )
        _fetch_urls(spec, args.out, args.max_docs)
    else:
        raise ValueError(f"unsupported source type {spec.source!r} for {spec.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
