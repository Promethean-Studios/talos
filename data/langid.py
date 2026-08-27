"""Pluggable language identification for the Talos data pipeline.

Language filtering is a first-class, *optional* stage. The design goal is a
clean :class:`LanguageIdentifier` interface with a fast, dependency-free
default (byte/character-profile heuristics) so the pipeline never requires
``fasttext`` / ``langdetect`` / ``langid``. Those heavier backends can be
plugged in by subclassing or by passing a backend name to
:func:`make_language_identifier`.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class LanguageIdentifier(ABC):
    """Predict the ISO-639-1 language tag of a text (e.g. ``"en"``, ``"zh"``)."""

    @abstractmethod
    def identify(self, text: str) -> Optional[str]:
        """Return a lower-case language code, or ``None`` if unknown."""


# Unicode block ranges used by the heuristic identifier.
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_JAPANESE_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_KOREAN_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_GREEK_RE = re.compile(r"[\u0370-\u03ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
_VIETNAMESE_RE = re.compile(r"[\u1e00-\u1eff]")
_LATIN_RE = re.compile(r"[a-zA-Z\u00c0-\u024f]")

# High-scoring English character bigrams (from shared western scripts).
_EN_BIGRAMS = {
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es",
    "or", "te", "of", "ed", "is", "it", "al", "ar", "st", "to", "nt",
}
# German / French / Spanish / Portuguese discriminating bigrams handled below.

#: Common English stopwords used to disambiguate Latin-script languages.
_EN_STOPWORDS = {
    "the", "and", "that", "with", "this", "from", "have", "not", "are",
    "you", "for", "but", "was", "his", "her", "all", "she", "they", "were",
    "will", "what", "when", "which", "their", "there", "would", "about",
    "because", "school", "people", "water", "after",
}
_FR_STOPWORDS = {"le", "la", "les", "de", "des", "un", "une", "et", "est",
                 "vous", "nous", "que", "qui", "dans", "pour", "avec"}
_ES_STOPWORDS = {"el", "la", "los", "las", "de", "del", "un", "una", "y", "es",
                 "que", "en", "para", "por", "con", "se"}
_DE_STOPWORDS = {"der", "die", "das", "und", "ich", "nicht", "ein", "eine",
                 "ist", "den", "von", "mit", "sie", "auch"}
_PT_STOPWORDS = {"o", "a", "os", "as", "de", "do", "da", "que", "em", "um",
                 "uma", "para", "com", "e", "não", "nao"}


class HeuristicLanguageIdentifier(LanguageIdentifier):
    """Fast, dependency-free language guesser.

    Strategy (documented, deterministic):
    1. If the text contains significant non-Latin script, classify by the
       dominant Unicode block (CJK -> zh/ja/ko by kana/hangul; cyrillic -> ru;
       arabic -> ar; greek -> el; hebrew -> he; devanagari -> hi; thai -> th;
       extended-latin-with-combining -> vi).
    2. Otherwise it is Latin script: choose among en/fr/es/de/pt by counting
       language-specific stopwords and English character bigrams.
    3. Default to ``"en"`` when the Latin sample is too short to judge.

    This is deliberately a rough heuristic (a real data pipeline would swap in
    fastText/multilingual model); it is *not* a panacea and is documented as
    such. It is fully deterministic given the text.
    """

    def identify(self, text: str) -> Optional[str]:
        t = text[:20000]  # cap cost on huge docs
        if not t.strip():
            return None

        for pattern, code in (
            (_JAPANESE_KANA_RE, "ja"),
            (_KOREAN_RE, "ko"),
            (_CJK_RE, "zh"),
            (_CYRILLIC_RE, "ru"),
            (_GREEK_RE, "el"),
            (_ARABIC_RE, "ar"),
            (_HEBREW_RE, "he"),
            (_DEVANAGARI_RE, "hi"),
            (_THAI_RE, "th"),
            (_VIETNAMESE_RE, "vi"),
        ):
            if pattern.search(t):
                return code

        return self._classify_latin(t)

    def _classify_latin(self, text: str) -> Optional[str]:
        words = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
        scores: Dict[str, int] = {}
        for w in words:
            for lang, stops in (
                ("en", _EN_STOPWORDS),
                ("fr", _FR_STOPWORDS),
                ("es", _ES_STOPWORDS),
                ("de", _DE_STOPWORDS),
                ("pt", _PT_STOPWORDS),
            ):
                if w in stops:
                    scores[lang] = scores.get(lang, 0) + 1
        if any(scores.values()):
            best = max(scores, key=lambda k: scores[k])
            return best

        # No stopwords matched; guess by character bigram profile.
        seq = "".join(re.findall(r"[a-zA-Z]", text.lower()))
        if len(seq) < 200:
            return "en"
        bigrams = [seq[i : i + 2] for i in range(len(seq) - 1)]
        hits = sum(1 for b in bigrams if b in _EN_BIGRAMS)
        # Heuristic: a Latin text dominated by typical English bigrams.
        return "en" if hits / len(bigrams) > 0.06 else None


class LangDetectIdentifier(LanguageIdentifier):
    """Optional backend wrapping the third-party ``langdetect`` library."""

    def __init__(self) -> None:
        try:
            from langdetect import DetectorFactory  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langdetect backend requires 'pip install langdetect'"
            ) from exc
        DetectorFactory.seed = 0  # deterministic detection

    def identify(self, text: str) -> Optional[str]:
        from langdetect import detect  # type: ignore

        try:
            return detect(text)
        except Exception:
            return None


BACKENDS = {"heuristic": HeuristicLanguageIdentifier, "langdetect": LangDetectIdentifier}


def make_language_identifier(backend: str = "heuristic") -> LanguageIdentifier:
    """Instantiate a :class:`LanguageIdentifier` by backend name."""
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown language backend {backend!r}; expected one of {sorted(BACKENDS)}"
        )
    return BACKENDS[backend]()
