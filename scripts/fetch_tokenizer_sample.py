"""Fetch the small PUBLIC sample used to train the Talos 1B 32K BPE tokenizer.

The Talos 1B track (``talos_1b`` / ``talos_1b_spec16``) needs a dedicated
32,000-vocab tokenizer asset. Per the project's corpus policy the tokenizer is
**not** trained on the owner's packed training corpus (that corpus is never
downloaded, reprocessed or used as tokenizer-training data) — it is trained on a
small, publicly licensed, freshly fetched sample documented here and pinned by
sha256 in ``assets/tokenizers/MANIFEST.json``.

Source
------
``HuggingFaceFW/fineweb-edu`` (ODC-By 1.0 — openly licensed educational web
text; the same corpus family the project's data layer already lists as a
supported public source in :mod:`data.setup.datasets`). Rows are read through
the HuggingFace **datasets-server** HTTP rows API, which needs no extra
dependency (plain ``urllib``) and no authentication for public datasets::

    GET https://datasets-server.huggingface.co/rows
        ?dataset=HuggingFaceFW%2Ffineweb-edu&config=default&split=train
        &offset=<offset>&length=<page>

What is taken
-------------
A **deterministic, offset-pinned slice**: pages of ``--page-size`` rows at
offsets ``start_offset + i*page_size`` for ``i in range(n_pages)``, where
``n_pages = ceil(num_rows / page_size)``. The default (1000 rows, page 100 →
10 pages at offsets 0, 100, …, 900) is ~3 MB of text — plenty for 31,740 BPE
merges and small enough to train on a 2-core CPU box. The held-out slice used
for the compression report is fetched the same way from a *disjoint* offset
region (``--start-offset 1000000``), so no held-out text can leak into the
training sample.

Output
------
JSONL, one document per line, fields ``id``, ``url`` and ``text``, written with
sorted keys and ``ensure_ascii=False`` so the file is byte-deterministic for a
given set of upstream rows. The script prints the row count, character count,
file size and **sha256** of the produced file; pass ``--expect-sha256`` to make
the run fail loudly if the upstream data has drifted from the pinned digest.

Reproduce (from the repo root)::

    python -m scripts.fetch_tokenizer_sample \
        --output /tmp/fineweb_edu_1b_tokenizer_sample.jsonl

    python -m scripts.fetch_tokenizer_sample \
        --start-offset 1000000 --num-rows 100 \
        --output /tmp/fineweb_edu_1b_tokenizer_heldout.jsonl

Nothing is downloaded into the repository: point ``--output`` at a scratch
directory. The committed artifact is the *trained* tokenizer plus the sha256
pins in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
ROWS_API = "https://datasets-server.huggingface.co/rows"

#: Fields kept from each upstream row (the rest — scores, dumps, crawl
#: metadata — is irrelevant to tokenizer training and would bloat the sample).
KEEP_FIELDS = ("id", "url", "text")


def build_rows_url(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> str:
    """The datasets-server rows-API URL for one page of documents."""
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split,
         "offset": offset, "length": length}
    )
    return f"{ROWS_API}?{query}"


def _get_json(url: str, retries: int = 5, timeout: float = 60.0) -> dict:
    """GET ``url`` and parse JSON, retrying transient failures with backoff."""
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(min(2.0 ** attempt, 30.0))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last}")


def fetch_rows(
    num_rows: int,
    start_offset: int = 0,
    page_size: int = 100,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Fetch ``num_rows`` rows from a deterministic, offset-pinned slice.

    The offsets are a pure function of ``start_offset``, ``page_size`` and
    ``num_rows`` (no randomness, no shuffling), so two runs fetch the same rows
    and the produced JSONL is byte-identical — which is what makes the sha256
    pin in the manifest meaningful.
    """
    if num_rows < 1:
        raise ValueError(f"num_rows must be >= 1, got {num_rows}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    if start_offset < 0:
        raise ValueError(f"start_offset must be >= 0, got {start_offset}")

    rows: List[Dict[str, object]] = []
    remaining = num_rows
    offset = start_offset
    while remaining > 0:
        length = min(page_size, remaining)
        url = build_rows_url(dataset, config, split, offset, length)
        payload = _get_json(url)
        page = payload.get("rows") or []
        if not page:
            raise RuntimeError(
                f"datasets-server returned no rows at offset {offset} "
                f"(dataset={dataset}, split={split}) — refusing to silently "
                "shorten the sample"
            )
        for entry in page:
            row = entry.get("row", entry)
            kept = {k: row[k] for k in KEEP_FIELDS if k in row}
            if "text" not in kept:
                raise RuntimeError(
                    f"row at offset {offset} has no 'text' field "
                    f"(fields: {sorted(row)})"
                )
            rows.append(kept)
        got = len(page)
        remaining -= got
        offset += got
        if verbose:
            print(f"  fetched {len(rows)}/{num_rows} rows (next offset {offset})",
                  file=sys.stderr)
    return rows


def write_jsonl(rows: Iterable[Dict[str, object]], path: str) -> Dict[str, object]:
    """Write ``rows`` as deterministic JSONL and return its fingerprint.

    Returns ``{"path", "rows", "chars", "bytes", "sha256"}``.
    """
    n_rows = 0
    chars = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n_rows += 1
            chars += len(str(row.get("text", "")))
    digest, n_bytes = file_sha256(path)
    return {"path": path, "rows": n_rows, "chars": chars,
            "bytes": n_bytes, "sha256": digest}


def file_sha256(path: str) -> tuple:
    """``(sha256_hex, size_in_bytes)`` of a file's raw bytes."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.fetch_tokenizer_sample",
        description="Fetch the small public sample for the Talos 1B 32K tokenizer.",
    )
    p.add_argument("--output", required=True,
                   help="where to write the JSONL sample (a scratch path — the "
                        "sample is not committed to the repo)")
    p.add_argument("--num-rows", type=int, default=1000,
                   help="documents to fetch (default 1000 ≈ 3 MB of text)")
    p.add_argument("--page-size", type=int, default=100,
                   help="rows per API request (default 100, the server's page size)")
    p.add_argument("--start-offset", type=int, default=0,
                   help="first row offset of the slice (default 0; use a "
                        "disjoint region, e.g. 1000000, for a held-out slice)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--expect-sha256", default=None,
                   help="fail if the produced file's sha256 differs from this pin")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    rows = fetch_rows(
        num_rows=args.num_rows,
        start_offset=args.start_offset,
        page_size=args.page_size,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        verbose=not args.quiet,
    )
    info = write_jsonl(rows, args.output)
    print(json.dumps({
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "start_offset": args.start_offset,
        "num_rows": args.num_rows,
        "page_size": args.page_size,
        **info,
    }, indent=2))
    if args.expect_sha256 and info["sha256"] != args.expect_sha256:
        print(
            f"ERROR: sample sha256 {info['sha256']} != pinned {args.expect_sha256} "
            "— the upstream slice changed; re-pin the manifest deliberately",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
