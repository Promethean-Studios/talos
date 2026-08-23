"""Benchmark/test-set contamination detection framework.

The goal here is *infrastructure*, not a curated benchmark answer pool: we
provide a :class:`ContaminationFilter` that drops (or flags) documents that
fuzzily overlap a caller-supplied list of **contaminated substrings /
document IDs**. The caller supplies the list (from the actual eval sets they
care about); Talos never hard-codes benchmark answers into the repo.

A record is considered contaminated when:
* its ``doc_id``/``id``/``url`` matches a contaminated document id verbatim, or
* any contaminated substring appears in the text, either exactly or after
  light normalisation (case/whitespace folding), with an optional contiguous
  n-gram fuzzy margin so tampered copies (e.g. a single changed word) are also
  caught.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

from data._logging import get_logger
from data.processors import Processor
from data.types import Record

log = get_logger("contamination")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _word_ngrams(text: str, n: int) -> Set[str]:
    """Word-level n-grams (contiguous ``n``-token runs) of a normalised text."""
    tokens = _normalize(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class ContaminationFilter(Processor):
    """Flag/drop documents overlapping a supplied contaminated list.

    Args:
        contaminated_texts: list of benchmark/etc. substrings to look for.
        contaminated_ids: list of document IDs / URLs considered contaminated.
        fuzzy_ngram: if > 1, treat a doc as contaminated when it shares a
            contiguous ``fuzzy_ngram``-*word* run with any contaminated text
            (robust to light paraphrasing of surrounding words).
        action: ``"drop"`` removes the doc; ``"flag"`` keeps it and sets
            ``record["contaminated"] = True``.
    """

    def __init__(
        self,
        contaminated_texts: Optional[Iterable[str]] = None,
        contaminated_ids: Optional[Iterable[str]] = None,
        fuzzy_ngram: int = 0,
        action: str = "drop",
    ) -> None:
        if action not in ("drop", "flag"):
            raise ValueError("action must be 'drop' or 'flag'")
        self.texts = sorted({_normalize(t) for t in (contaminated_texts or [])})
        self.ids: Set[str] = {str(i).strip() for i in (contaminated_ids or [])}
        self.fuzzy_ngram = fuzzy_ngram
        self.action = action
        # Precompute word n-gram sets for the fuzzy path.
        self._text_ngrams = (
            {g for t in self.texts for g in _word_ngrams(t, fuzzy_ngram)}
            if fuzzy_ngram > 1
            else set()
        )

    @property
    def name(self) -> str:
        return f"contamination:{self.action}"

    def _is_contaminated(self, record: Record) -> bool:
        # document-id match
        for key in ("doc_id", "id", "url", "sha256"):
            val = record.get(key)
            if isinstance(val, str) and val.strip() in self.ids:
                return True
        text = _normalize(record.get("text", ""))
        # exact/normalised substring match
        for t in self.texts:
            if t and t in text:
                return True
        # fuzzy word n-gram overlap
        if self.fuzzy_ngram > 1 and text:
            if self._text_ngrams & _word_ngrams(text, self.fuzzy_ngram):
                return True
        return False

    def process(self, record: Record) -> Optional[Record]:
        # Nothing configured -> pass-through (identity).
        if not self.texts and not self.ids and self.fuzzy_ngram <= 1:
            return record
        contaminated = self._is_contaminated(record)
        if contaminated:
            if self.action == "drop":
                log.debug("dropped contaminated doc id=%s", record.get("id"))
                return None
            record["contaminated"] = True
            return record
        return record
