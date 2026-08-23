"""Composable, configurable pipeline stages (processors).

Every stage follows one :class:`Processor` interface::

    def process(self, record: Record) -> Optional[Record]

Returning ``None`` drops the document; returning a (possibly mutated) ``dict``
keeps it. This lets us chain filters, annotators and translaters into a single
:class:`ProcessorChain` whose output feeds the mixer.

The config factory :func:`processor_from_config` maps a declarative dict
(``{"type": ..., **options}``) to an instance, so a pipeline can be described
entirely in a JSON/YAML file.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

from data._logging import get_logger
from data.langid import LanguageIdentifier, make_language_identifier
from data.types import Record

log = get_logger("processor")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class Processor(ABC):
    """A single pipeline stage operating on one record at a time."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in stats/drop reporting."""

    @abstractmethod
    def process(self, record: Record) -> Optional[Record]:
        """Return the (possibly modified) record or ``None`` to drop it."""


class ProcessorChain(Processor):
    """Run processors in order; a ``None`` from any stage drops the record."""

    def __init__(self, processors: Sequence[Processor]) -> None:
        if not processors:
            raise ValueError("ProcessorChain needs at least one processor")
        self.processors = list(processors)

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.processors)

    def __iter__(self):
        return iter(self.processors)

    def process(self, record: Record) -> Optional[Record]:
        for proc in self.processors:
            if record is None:
                return None
            record = proc.process(record)
        return record


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
class LengthFilter(Processor):
    """Drop documents shorter than ``min_chars`` or longer than ``max_chars``."""

    def __init__(self, min_chars: int = 50, max_chars: int = 10_000_000) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return "length"

    def process(self, record: Record) -> Optional[Record]:
        n = len(record.get("text", ""))
        if n < self.min_chars or n > self.max_chars:
            return None
        return record


class QualityHeuristicFilter(Processor):
    """Heuristic quality filters (documented, dependency-free).

    Drops documents whose text looks like machine-generated / boilerplate /
    navigation junk based on cheap statistics. None of these is a perfect
    detector; together they catch the common failure modes in web crawls.

    Criteria (each threshold is configurable):
    * ``max_repeated_ratio`` — fraction of characters that are the single most
      common character (all-``a`` spam, long runs of one char).
    * ``max_punct_ratio`` — fraction of characters that are punctuation/symbols
      (``!@#$%^&*()`` spam).
    * ``max_newline_ratio`` — fraction of characters that are newlines
      (a chunk of nothing but empty lines).
    * ``max_bullet_ratio`` — fraction of *lines* beginning with a list/nav
      bullet (``-``/``*``/``#``/``1.``), a crude boilerplate/nav signal.
    * ``min_sentences`` — minimum number of sentence-ending punctuation marks
      required so pure header/URL dumps are rejected.
    """

    def __init__(
        self,
        max_repeated_ratio: float = 0.2,
        max_punct_ratio: float = 0.35,
        max_newline_ratio: float = 0.25,
        max_bullet_ratio: float = 0.5,
        min_sentences: int = 1,
    ) -> None:
        self.max_repeated_ratio = max_repeated_ratio
        self.max_punct_ratio = max_punct_ratio
        self.max_newline_ratio = max_newline_ratio
        self.max_bullet_ratio = max_bullet_ratio
        self.min_sentences = min_sentences
        self._punct = set("!@#$%^&*()_+=[]{}|;:'\",.<>/?~`")

    @property
    def name(self) -> str:
        return "quality_heuristic"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        if not text:
            return None
        n = len(text)
        # most-common-character ratio
        counts: Dict[str, int] = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        top = max(counts.values())
        if top / n > self.max_repeated_ratio:
            return None
        punct = sum(1 for ch in text if ch in self._punct)
        if punct / n > self.max_punct_ratio:
            return None
        newlines = text.count("\n")
        if newlines / n > self.max_newline_ratio:
            return None
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            bullets = sum(
                1
                for ln in lines
                if re.match(r"^[\s]*([-*#>]|\d+[.)])", ln)
            )
            if bullets / len(lines) > self.max_bullet_ratio:
                return None
        sentences = len(re.findall(r"[.!?。！？]", text))
        if sentences < self.min_sentences:
            return None
        return record


# Common programming-language keywords used by the code detector.
_CODE_KEYWORDS = re.compile(
    r"^\s*(?:def|class|import|from|return|if|elif|else|for|while|try|except|"
    r"finally|function|const|let|var|public|private|protected|static|int|void|"
    r"float|char|double|bool|#include|struct|using|namespace|package|interface|"
    r"enum|print|echo|yield|lambda)\b"
)
_ASSIGN_RE = re.compile(r"[^=<>!]=[^=<>!]|->|=>")


class CodeFenceFilter(Processor):
    """Route documents by how code-like they are.

    ``mode="keep"`` keeps code-like docs and drops prose-like ones; ``mode=
    "drop"`` does the opposite. A line counts as code when it:
      * starts with a common programming keyword (``def``, ``class``, ...), or
      * is indented (leading spaces/tabs) and not a prose list bullet, or
      * ends with ``{``/``}``/``;``, or
      * contains an assignment/arrow (``=``, ``->``, ``=>``), or
      * is a Markdown code fence (````` ``` ```` / ``~~~``).

    The document is code-like when at least ``min_code_ratio`` of its non-empty
    lines qualify. This is a documented heuristic used to *separate* code from
    natural language; it is not a parser.
    """

    def __init__(self, mode: str = "keep", min_code_ratio: float = 0.3) -> None:
        if mode not in ("keep", "drop"):
            raise ValueError("mode must be 'keep' or 'drop'")
        self.mode = mode
        self.min_code_ratio = min_code_ratio

    @property
    def name(self) -> str:
        return f"code_fence:{self.mode}"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None if self.mode == "keep" else record
        code_lines = 0
        for ln in lines:
            stripped = ln.strip()
            if _CODE_KEYWORDS.match(stripped):
                code_lines += 1
            elif stripped.startswith(("  ", "\t")) and not re.match(
                r"^\s*[-*#+>]", stripped
            ):
                code_lines += 1
            elif stripped.endswith(("{", "}", ";")):
                code_lines += 1
            elif _ASSIGN_RE.search(stripped):
                code_lines += 1
            elif re.match(r"^`{3}|^~{3}", stripped):
                code_lines += 1
        ratio = code_lines / len(lines)
        is_code = ratio >= self.min_code_ratio
        if self.mode == "keep":
            return record if is_code else None
        return None if is_code else record


class BlacklistRegexFilter(Processor):
    """Drop documents matching any of a list of regular expressions."""

    def __init__(self, patterns: List[str]) -> None:
        self.patterns = [re.compile(p) for p in patterns]

    @property
    def name(self) -> str:
        return "blacklist_regex"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        for pat in self.patterns:
            if pat.search(text):
                return None
        return record


class URLBlacklistFilter(Processor):
    """Drop documents whose ``url``/``source`` is on a block list of domains."""

    def __init__(self, domains: List[str]) -> None:
        self.domains = set(domains)

    @property
    def name(self) -> str:
        return "url_blacklist"

    def process(self, record: Record) -> Optional[Record]:
        url = record.get("url") or record.get("source") or ""
        if not url:
            return record
        for dom in self.domains:
            if dom in url:
                return None
        return record


class LanguageFilter(Processor):
    """Keep/drop documents by predicted language code(s).

    Uses a :class:`LanguageIdentifier` (default the heuristic one). With
    ``allow=("en",)`` and ``keep=...`` semantics you can restrict a corpus to
    English, or with ``drop_unknown`` drop docs that the identifier can't tag.
    """

    def __init__(
        self,
        allow: Optional[List[str]] = None,
        drop_unknown: bool = False,
        backend: str = "heuristic",
        identifier: Optional[LanguageIdentifier] = None,
    ) -> None:
        self.allow = set(allow or [])
        self.drop_unknown = drop_unknown
        self.identifier = identifier or make_language_identifier(backend)

    @property
    def name(self) -> str:
        return f"language:{self.backend_label()}"

    def backend_label(self) -> str:
        return type(self.identifier).__name__.lower().replace("language", "").replace(
            "identifier", ""
        ) or "id"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        lang = self.identifier.identify(text)
        if lang is not None:
            record["lang"] = lang
        elif self.drop_unknown:
            return None
        if self.allow and (lang is None or lang not in self.allow):
            return None
        return record


# ---------------------------------------------------------------------------
# Tokenizer-aware counting
# ---------------------------------------------------------------------------
class TokenCounter(Processor):
    """Add ``num_tokens`` to each record.

    Uses the Phase-2 :class:`~tokenizer.tokenizer.ByteLevelBPETokenizer` when a
    tokenizer instance or path is provided. If no tokenizer is given,
    ``fallback`` controls the estimate: ``"bytes"`` uses UTF-8 byte length
    (a documented, cheap upper-bound proxy for a byte-level BPE).
    """

    def __init__(
        self,
        tokenizer=None,
        fallback: str = "bytes",
        add_special: bool = False,
    ) -> None:
        if tokenizer is None:
            from tokenizer.tokenizer import ByteLevelBPETokenizer as _B
            self._has_tok = False
        else:
            if isinstance(tokenizer, str):
                from tokenizer.tokenizer import ByteLevelBPETokenizer as _B
                tokenizer = _B.from_file(tokenizer)
            if not hasattr(tokenizer, "encode"):
                raise TypeError("tokenizer must expose an encode() method")
            self._tokenizer = tokenizer
            self._has_tok = True
        self.fallback = fallback
        self.add_special = add_special
        if not self._has_tok and fallback not in ("bytes",):
            raise ValueError(f"unknown token fallback {fallback!r}")

    @property
    def name(self) -> str:
        return "token_count"

    def process(self, record: Record) -> Optional[Record]:
        text = record.get("text", "")
        if self._has_tok:
            ids = self._tokenizer.encode(text, bos=self.add_special, eos=self.add_special)
            record["num_tokens"] = len(ids)
        elif self.fallback == "bytes":
            record["num_tokens"] = len(text.encode("utf-8", errors="replace"))
        return record


class FieldRemap(Processor):
    """Rename/select record fields (e.g. map ``content`` -> ``text``)."""

    def __init__(
        self,
        text_field: str = "text",
        keep_other_fields: bool = True,
    ) -> None:
        self.text_field = text_field
        self.keep_other_fields = keep_other_fields

    @property
    def name(self) -> str:
        return "field_remap"

    def process(self, record: Record) -> Optional[Record]:
        if self.text_field != "text":
            if self.text_field not in record:
                return None
            record["text"] = record[self.text_field]
        if not self.keep_other_fields and self.text_field != "text":
            record.pop(self.text_field, None)
        return record


class RegexFilter(Processor):
    """Keep a document only if it matches (or, with ``invert``, fails) a regex."""

    def __init__(self, pattern: str, invert: bool = False) -> None:
        self.pattern = re.compile(pattern)
        self.invert = invert

    @property
    def name(self) -> str:
        return "regex_filter"

    def process(self, record: Record) -> Optional[Record]:
        matched = self.pattern.search(record.get("text", "")) is not None
        ok = (not matched) if self.invert else matched
        return record if ok else None


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------
def processor_from_config(config: Dict[str, Any]) -> Processor:
    """Build a processor from a declarative dict ``{"type": ..., **options}``.

    A processor of ``type="chain"`` has ``options.processors`` which is a list
    of nested processor dicts, letting you compose arbitrarily deep graphs.
    """
    cfg = dict(config)
    ptype = cfg.pop("type", None)
    if ptype == "chain":
        subs = cfg.pop("processors", [])
        return ProcessorChain([processor_from_config(s) for s in subs])
    factories: Dict[str, Callable[..., Processor]] = {
        "length": LengthFilter,
        "quality_heuristic": QualityHeuristicFilter,
        "code_fence": CodeFenceFilter,
        "blacklist_regex": BlacklistRegexFilter,
        "url_blacklist": URLBlacklistFilter,
        "language": LanguageFilter,
        "token_count": TokenCounter,
        "field_remap": FieldRemap,
        "regex_filter": RegexFilter,
    }
    if ptype not in factories:
        # Heavy filters live in dedicated modules; import lazily to avoid cycles.
        from data import dedup  # noqa: PLC0415
        factories["exact_dedup"] = dedup.ExactDedupFilter
        factories["minhash_dedup"] = dedup.MinHashNearDupFilter
        from data import contamination  # noqa: PLC0415
        factories["contamination"] = contamination.ContaminationFilter
    if ptype not in factories:
        raise ValueError(
            f"unknown processor type {ptype!r}; expected one of {sorted(factories)}"
        )
    return factories[ptype](**cfg)
