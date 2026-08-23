"""End-to-end data pipeline: readers -> processors -> mixer -> sharded writer.

This module is also the CLI entry point::

    python -m data.pipeline --config data/configs/example.json --out /tmp/out \\
        --max-docs 1000 --seed 0

The ``--config`` file is a declarative JSON document (see
``data/configs/example.json`` and ``data/README.md``) describing sources,
processors and output. ``run_pipeline`` is the same logic callable from Python.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

from data._logging import get_logger
from data.mixer import WeightedMixer
from data.processors import Processor, ProcessorChain, processor_from_config
from data.readers import DatasetReader, reader_from_config
from data.types import PipelineStats, Record
from data.writers import ShardedWriter

log = get_logger("pipeline")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("pipeline config must be a JSON object")
    return cfg


def build_reader(config: Dict[str, Any]) -> DatasetReader:
    return reader_from_config(config)


def build_processors(config: List[Dict[str, Any]]) -> Processor:
    procs = [processor_from_config(p) for p in config]
    if len(procs) == 1:
        return procs[0]
    return ProcessorChain(procs)


def run_pipeline(
    config: Dict[str, Any],
    out_dir: str,
    max_docs: Optional[int] = None,
    seed: int = 0,
) -> PipelineStats:
    """Run a full pipeline described by ``config``.

    Returns a :class:`PipelineStats` with input/kept/dropped counts.
    """
    sources_spec = config.get("sources")
    if not isinstance(sources_spec, list) or not sources_spec:
        raise ValueError("config requires a non-empty 'sources' list")
    processors = build_processors(
        config.get("processors", [{"type": "length", "min_chars": 1}])
    )
    output = config.get("output", {})
    out_dir = output.get("dir", out_dir)
    shard_size = output.get("shard_size", 100_000)
    size_by = output.get("size_by", "records")
    fmt = output.get("format", "jsonl")

    stats = PipelineStats()
    build_sources: List[tuple] = []
    names: List[str] = []
    for spec in sources_spec:
        reader = build_reader(spec.get("reader", {}))
        weight = float(spec.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"source weight must be >= 0, got {weight}")
        names.append(reader.source)
        build_sources.append((_process_stream(reader, processors, stats, max_docs), weight))

    # Mixing: default one pass over each source (finite). Pass num_epochs=None
    # in the config for streaming mode with infinite readers.
    num_epochs = config.get("num_epochs")
    mixer = WeightedMixer(build_sources, seed=seed, num_epochs=num_epochs)
    with ShardedWriter(
        out_dir,
        shard_size=shard_size,
        size_by=size_by,
        format=fmt,
        prefix=output.get("prefix", "shard"),
    ) as writer:
        written = 0
        for record in mixer:
            writer.write(record)
            written += 1
            if max_docs is not None and written >= max_docs:
                break

    log.info(
        "pipeline done: input=%d kept=%d dropped=%d written=%d",
        stats.total_input,
        stats.total_kept,
        stats.total_dropped,
        written,
    )
    return stats


def _process_stream(
    reader: DatasetReader,
    chain: Processor,
    stats: PipelineStats,
    cap: Optional[int],
) -> Iterable[Record]:
    """Yield processed records from ``reader``, accounting pipeline stats."""
    for record in reader:
        stats.record_input()
        res = chain.process(record)
        # first processor to drop controls the "stage" label (best-effort)
        if res is None:
            stats.record_dropped(chain.name)
        else:
            stats.record_kept()
            yield res


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Talos data pipeline (read -> process -> mix -> shard)."
    )
    parser.add_argument("--config", required=True, help="path to pipeline JSON config")
    parser.add_argument("--out", default=None, help="output directory (overrides config)")
    parser.add_argument("--max-docs", type=int, default=None, help="cap on docs to write")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--dry-run", action="store_true", help="build and report only")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    out_dir = args.out or cfg.get("output", {}).get("dir")
    if not out_dir:
        raise ValueError("no --out given and config has no output.dir")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sources": cfg.get("sources"),
                    "processors": cfg.get("processors"),
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        return 0
    stats = run_pipeline(cfg, out_dir, max_docs=args.max_docs, seed=args.seed)
    print(json.dumps(stats.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
