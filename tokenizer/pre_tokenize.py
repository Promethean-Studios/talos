"""Optional pre-tokenization for BPE training.

Byte-level BPE trains on *byte* sequences. Before counting pairs we may split
the corpus into short "words" via a regex; merges are then learned **within**
each word and never cross a word boundary. This is the GPT-2 / LLaMA family's
approach: it concentrates merge capacity on common sub-words and keeps training
fast, because each word is small and independent.

``None`` (the default) disables pre-tokenization entirely: the corpus is treated
as raw bytes and merges may span any byte boundary (including whitespace). This
is the most faithful "pure byte-level" behaviour and works losslessly for every
script, but it is slower to converge and typically needs a larger vocab to
represent frequent multi-byte tokens (space-prefixed words are not shared the
way the GPT-2 pattern shares them).

Trade-off summary (see docs/tokenizer.md):
  * Regex pre-tokenization → GPT-2/LLaMA-like token boundaries, faster training,
    but a space-or-symbol inside a token is impossible (e.g. "hello world" is
    two words, so no merge can span the space).
  * No pre-tokenization → fully general byte boundaries, slower convergence,
    larger vocab needed.

Both modes round-trip losslessly: pre-tokenization only constrains *which*
merges are legal; it never discards bytes.
"""
from __future__ import annotations

import re
from typing import List, Optional, Pattern

# GPT-2's HSBM pattern, written with explicit Unicode ranges so it compiles
# under the stdlib ``re`` (Python's re has no ``\\p{...}``). It is a good
# approximation for Latin/Greek/Cyrillic scripts; CJK and other scripts fall
# through to the generic "any non-space non-ASCII-letter run" clause, so they
# are still preserved exactly.
_GPT2_ASCII = (
    r"A-Za-z\xaa\xb5\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4"
    r"\u02ec\u02ee\u0370-\u0374\u0376-\u0377\u037a-\u037d\u037f\u0386"
    r"\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u048a-\u052f"
    r"\u0531-\u0556\u0559\u0561-\u0587\u05d0-\u05ea\u05ef-\u05f2"
    r"\u0620-\u064a\u066e-\u066f\u0671-\u06d3\u06d5\u06e5-\u06e6"
    r"\u06ee-\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5"
    r"\u07b1\u07ca-\u07ea\u07f4-\u07f5\u07fa\u0800-\u0815\u081a\u0824\u0828"
    r"\u0840-\u0858\u0860-\u086a\u08a0-\u08b4\u08b6-\u08bd\u0904-\u0939"
    r"\u093d\u0950\u0958-\u0961\u0971-\u0980\u0985-\u098c\u098f-\u0990"
    r"\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc-\u09dd"
    r"\u09df-\u09e1\u09f0-\u09f1\u09fc\u0a05-\u0a0a\u0a0f-\u0a10"
    r"\u0a13-\u0a28\u0a2a-\u0a30\u0a32-\u0a33\u0a35-\u0a36\u0a38-\u0a39"
    r"\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91"
    r"\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2-\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0"
    r"\u0ae0-\u0ae1\u0af9\u0b05-\u0b0c\u0b0f-\u0b10\u0b13-\u0b28"
    r"\u0b2a-\u0b30\u0b32-\u0b33\u0b35-\u0b39\u0b3d\u0b5c-\u0b5d\u0b5f-\u0b61"
    r"\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99-\u0b9a"
    r"\u0b9c\u0b9e-\u0b9f\u0ba3-\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0"
    r"\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c39\u0c3d\u0c58-\u0c5a"
    r"\u0c60-\u0c61\u0c80\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8"
    r"\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0-\u0ce1\u0cf1-\u0cf2"
    r"\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d54-\u0d56"
    r"\u0d5f-\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb"
    r"\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32-\u0e33\u0e40-\u0e46"
    r"\u0e81-\u0e82\u0e84\u0e87-\u0e88\u0e8a\u0e8d\u0e94-\u0e97"
    r"\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa-\u0eab\u0ead-\u0eb0"
    r"\u0eb2-\u0eb3\u0ebd\u0ec0-\u0ec4\u0ec6\u0edc-\u0edf\u0f00"
    r"\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f"
    r"\u1050-\u1055\u105a-\u105d\u1061\u1065-\u1066\u106e-\u1070"
    r"\u1075-\u1081\u108e\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa"
    r"\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d"
    r"\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be"
    r"\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315"
    r"\u1318-\u135a\u1380-\u138f\u13a0-\u13f5\u13f8-\u13fd"
    r"\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f8"
    r"\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c"
    r"\u176e-\u1770\u1780-\u17b3\u17d7\u17dc\u1820-\u1877\u1880-\u1884"
    r"\u1887-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191e\u1950-\u196d"
    r"\u1970-\u1974\u1980-\u19ab\u19b0-\u19c9\u1a00-\u1a16\u1a20-\u1a54"
    r"\u1aa7\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae-\u1baf"
    r"\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c7d"
    r"\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5-\u1cf6\u1d00-\u1dbf"
    r"\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d"
    r"\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4"
    r"\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3"
    r"\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2071"
    r"\u207f\u2090-\u209c\u2102\u2107\u210a-\u2113\u2115\u2119-\u211d"
    r"\u2124\u2126\u2128\u212a-\u212d\u212f-\u2139\u213c-\u213f"
    r"\u2145-\u2149\u214e\u2183-\u2184\u2c00-\u2c2e\u2c30-\u2c5e"
    r"\u2c60-\u2ce4\u2ceb-\u2cee\u2cf2-\u2cf3\u2d00-\u2d25\u2d27\u2d2d"
    r"\u2d30-\u2d67\u2d6f\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae"
    r"\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6"
    r"\u2dd8-\u2dde\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303c"
    r"\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff"
    r"\u3105-\u312d\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff"
    r"\u3400-\u4db5\u4e00-\u9fd5\ua000-\ua48c\ua4d0-\ua4fd"
    r"\ua500-\ua60c\ua610-\ua61f\ua62a-\ua62b\ua640-\ua66e"
    r"\ua67f-\ua69d\ua6a0-\ua6e5\ua717-\ua71f\ua722-\ua788"
    r"\ua78b-\ua7ae\ua7b0-\ua7b7\ua7f7-\ua801\ua803-\ua805"
    r"\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3"
    r"\ua8f2-\ua8f7\ua8fb\ua8fd\ua90a-\ua925\ua930-\ua946"
    r"\ua960-\ua97c\ua984-\ua9b2\ua9cf\ua9e0-\ua9e4\ua9e6-\ua9ef"
    r"\ua9fa-\ua9fe\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b"
    r"\uaa60-\uaa76\uaa7a\uaa7e-\uaaaf\uaab1\uaab5-\uaab6\uaab9-\uaabd"
    r"\uaac0\uaac2\uaadb-\uaadd\uaae0-\uaaea\uaaf2-\uaaf4"
    r"\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26"
    r"\uab28-\uab2e\uab30-\uab5a\uab5c-\uab65\uab70-\uabe2"
    r"\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d"
    r"\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d-\ufb28"
    r"\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44"
    r"\ufb46-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7"
    r"\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\uff21-\uff3a"
    r"\uff41-\uff5a\uff66-\uffbe\uffc2-\uffc7\uffca-\uffcf"
    r"\uffd2-\uffd7\uffda-\uffdc"
)


def gpt2_pattern() -> str:
    """The GPT-2 re-token pattern, ``re``-compatible (no ``\\p{...}``)."""
    return (
        r"'s|'t|'re|'ve|'m|'ll|'d"
        r"| ?" + _GPT2_ASCII + "+"
        r"| ?[0-9]+"
        r"| ?[^\sA-Za-z0-9" + _GPT2_ASCII + "]+"
        r"|\s+(?!\S)|\s+"
    )


PRESETS: dict = {
    "gpt2": gpt2_pattern(),
    # A minimal safety-net pattern: a letter/digit run optionally preceded by a
    # space, symbol runs, whitespace runs. Equivalent to GPT-2 minus the giant
    # Unicode letter table — purely an example of a user-supplied pattern.
    "simple": r" ?[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
}

# Names of known presets vs. raw regex strings.
_KNOWN_PRESETS = {"gpt2", "simple"}


def resolve_pattern(spec: Optional[str]) -> Optional[Pattern[str]]:
    """Turn a preset name or raw regex string into a compiled ``re`` Pattern.

    Returns ``None`` when ``spec`` is falsy (pre-tokenization disabled). A
    preset that requires ``regex``'s ``\\p{...}`` support is reported so callers
    can decide whether to depend on the (optional) ``regex`` package.
    """
    if not spec:
        return None
    pattern = None
    if spec in _KNOWN_PRESETS:
        pattern = PRESETS[spec]
    else:
        pattern = spec  # treat as a raw regex string
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid pre-tokenize pattern {spec!r}: {exc}") from exc


def has_unicode_classes(pattern: str) -> bool:
    """Whether ``pattern`` needs the optional ``regex`` package (``\\p{...}``)."""
    return "\\p{" in pattern


def tokenize_bytes(text: bytes, pattern: Optional[Pattern[str]]) -> List[bytes]:
    """Split ``text`` bytes into a list of byte "words" using ``pattern``.

    When ``pattern`` is ``None`` the whole input is returned as a single word
    (raw byte-level). The regex runs on the *decoded* string so it can match
    Unicode classes, then each matched piece is re-encoded to bytes.
    """
    if pattern is None:
        return [text]
    decoded = text.decode("utf-8", errors="replace")
    return [m.group(0).encode("utf-8") for m in pattern.finditer(decoded)]


def tokenize_text(text: str, pattern: Optional[Pattern[str]]) -> List[bytes]:
    """Split a decoded ``text`` string into byte words (helper for tests)."""
    if pattern is None:
        return [text.encode("utf-8")]
    return [m.group(0).encode("utf-8") for m in pattern.finditer(text)]
