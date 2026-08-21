"""Corpus loading for tokenizer training: plain text and JSONL.

Handles UTF-8 validation and reports/skips invalid lines according to a
policy so a noisy corpus never silently corrupts the learned merges.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, List, Optional

from tokenizer._logging import get_logger

log = get_logger("corpus")


class CorpusError(ValueError):
    """Raised when a corpus cannot be consumed (e.g. corrupt input)."""


def iter_text_documents(
    corpus_paths: Iterable[str],
    jsonl_field: str = "text",
    on_invalid: str = "skip",
    max_docs: Optional[int] = None,
) -> Iterator[str]:
    """Yield decoded text documents from ``.txt`` / ``.jsonl`` files.

    Args:
        corpus_paths: paths to text or JSONL files (directories are expanded to
            their ``.txt``/``.jsonl`` children).
        jsonl_field: which JSON field holds the text for ``.jsonl`` inputs.
        on_invalid: ``"skip"`` (default) or ``"die"`` when a line is not valid
            UTF-8 or a JSONL record is malformed / missing the text field.
        max_docs: stop after this many documents (``None`` = unlimited).
    """
    count = 0
    for path in _expand_paths(corpus_paths):
        if path.endswith(".jsonl") or path.endswith(".json"):
            docs = _iter_jsonl(path, jsonl_field, on_invalid)
        else:
            docs = _iter_text(path, on_invalid)
        for doc in docs:
            if doc is None:
                if on_invalid == "die":
                    raise CorpusError(f"invalid content in corpus file: {path}")
                continue
            yield doc
            count += 1
            if max_docs is not None and count >= max_docs:
                return


def _expand_paths(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith((".txt", ".jsonl", ".json")):
                        out.append(os.path.join(root, f))
        else:
            out.append(p)
    return out


def _iter_text(path: str, on_invalid: str) -> Iterator[Optional[str]]:
    with open(path, "rb") as fh:
        for raw in fh:
            yield _decode_line(raw, on_invalid, path)


def _iter_jsonl(path: str, field: str, on_invalid: str) -> Iterator[Optional[str]]:
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            text = _decode_line(raw, on_invalid, f"{path}:{lineno}")
            if text is None:
                yield None
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                if on_invalid == "die":
                    raise CorpusError(f"invalid JSON at {path}:{lineno}") from None
                log.warning("skipping non-JSON line at %s:%d", path, lineno)
                continue
            value = record.get(field) if isinstance(record, dict) else None
            if not isinstance(value, str):
                if on_invalid == "die":
                    raise CorpusError(
                        f"missing text field {field!r} at {path}:{lineno}"
                    )
                continue
            yield value


def _decode_line(raw: bytes, on_invalid: str, where: str) -> Optional[str]:
    try:
        return raw.decode("utf-8").rstrip("\n").rstrip("\r")
    except UnicodeDecodeError:
        if on_invalid == "die":
            raise CorpusError(f"invalid UTF-8 in corpus file: {where}") from None
        log.warning("skipping invalid UTF-8 line in %s", where)
        return None
