"""The byte-level BPE tokenizer: encode/decode, save/load, chunking.

Byte-level means every Unicode string is first UTF-8 encoded to bytes, then
tokenized independently of script. Because there is one base token per byte and
merges are lossless, ``decode(encode(text)) == text`` holds for *arbitrary*
Unicode (all world languages, emoji, math symbols, source code) with no UNK and
no pre-tokenisation assumptions baked in.

Special tokens (BOS/EOS/PAD/UNK + extras + reserved) occupy ids above the base
byte vocabulary. By default they are *not* split out of plain text during
``encode`` (only added via the ``bos``/``eos`` flags), which keeps the
round-trip guarantee exact for all inputs. ``split_special`` opts into
recognising special-token strings inside the input instead.
"""
from __future__ import annotations

import json
import re
import unicodedata  # noqa: F401  (kept for documented byte/char invariants)
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tokenizer._logging import get_logger
from tokenizer.bpe import encode_bytes, ranks_from_merges
from tokenizer.pre_tokenize import iter_words_with_gaps, resolve_pattern
from tokenizer.vocab import BASE_VOCAB_SIZE, TokenizerConfig, Vocabulary

log = get_logger("tokenizer")


class ByteLevelBPETokenizer:
    """A GPT-2/LLaMA-style byte-level BPE tokenizer.

    Args:
        config: :class:`TokenizerConfig` (vocab size, special tokens, ...).
        merges: ordered list of byte pairs (``[left_byte, right_byte]``) in merge
            rank order. Omit to build an empty-vocab tokenizer (just the 256 base
            bytes + special tokens), e.g. when encoding text before training.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        merges: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        self.config = config
        self._vocab = Vocabulary(config, merges or [])
        self._pre_pattern = resolve_pattern(config.pre_tokenize)
        self._special_re: Optional[re.Pattern] = None
        if config.split_special and config.special_tokens:
            self._special_re = re.compile(
                "|".join(re.escape(t) for t in config.special_tokens)
            )
        self._ranks = ranks_from_merges(self._vocab.merges)
        self._merge_id_map = self._vocab.pair_to_merge_id

    # -- properties ---------------------------------------------------------
    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab.vocab_size()

    @property
    def merge_count(self) -> int:
        return len(self._vocab.merges)

    @property
    def bos_id(self) -> int:
        return self._vocab.bos_id()

    @property
    def eos_id(self) -> int:
        return self._vocab.eos_id()

    @property
    def pad_id(self) -> int:
        return self._vocab.pad_id()

    @property
    def unk_id(self) -> int:
        return self._vocab.unk_id()

    # -- encoding -----------------------------------------------------------
    def encode(
        self,
        text: str,
        bos: bool = False,
        eos: bool = False,
        split_special: Optional[bool] = None,
    ) -> List[int]:
        """Encode ``text`` to a list of token ids.

        Args:
            text: any Python string (arbitrary Unicode).
            bos: prepend the BOS token id.
            eos: append the EOS token id.
            split_special: override ``config.split_special`` for this call.
        """
        split = self.config.split_special if split_special is None else split_special
        if split:
            ids = self._encode_with_special(text)
        else:
            ids = self._encode_plain(text)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def _encode_plain(self, text: str) -> List[int]:
        data = text.encode("utf-8")
        # Without a pre-tokenizer, the whole text is one byte sequence; with one,
        # each matched word is encoded independently (merges never cross words).
        # Unmatched spans are kept as their own words (see iter_words_with_gaps):
        # a pattern that does not cover the text must never silently drop bytes.
        if self._pre_pattern is None:
            byte_ids = list(data)
            return encode_bytes(byte_ids, self._ranks, self._merge_id_map)
        out: List[int] = []
        for word in iter_words_with_gaps(text, self._pre_pattern):
            if word:
                out.extend(
                    encode_bytes(list(word.encode("utf-8")), self._ranks, self._merge_id_map)
                )
        return out

    def _encode_with_special(self, text: str) -> List[int]:
        """Encode text, splitting literal special-token strings into ids."""
        if self._special_re is None:
            return self._encode_plain(text)
        ids: List[int] = []
        last = 0
        for m in self._special_re.finditer(text):
            if m.start() > last:
                ids.extend(self._encode_plain(text[last : m.start()]))
            ids.append(self._vocab.special_to_id[m.group(0)])
            last = m.end()
        if last < len(text):
            ids.extend(self._encode_plain(text[last:]))
        return ids

    def encode_with_special(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text, always splitting literal special-token strings. With
        ``add_special_tokens`` (default) BOS/EOS are added, matching the common
        HuggingFace ``__call__`` convention."""
        ids = self._encode_with_special(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    # -- chunked encoding ---------------------------------------------------
    def chunk_encode(
        self,
        text: str,
        chunk_size: int = 8192,
        bos: bool = False,
        eos: bool = False,
    ) -> List[int]:
        """Encode long ``text`` in ``chunk_size``-token chunks.

        ``chunk_size`` refers to *characters* per chunk (a conservative upper
        bound on tokens, since a multi-byte char is at least one byte/token).
        Chunk boundaries fall on character boundaries, so round-tripping is
        exact. Returns the concatenated id list; use :meth:`encode_ids_chunks`
        for a per-chunk list (e.g. long-context training/inference).
        """
        ids: List[int] = []
        for chunk in self.iter_chunks(text, chunk_size):
            ids.extend(self.encode(chunk, bos=bos, eos=eos))
        return ids

    def iter_chunks(self, text: str, chunk_size: int) -> Iterable[str]:
        """Yield ``text`` split into ≤ ``chunk_size``-character pieces."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    def encode_ids_chunks(
        self,
        text: str,
        chunk_size: int = 8192,
        bos: bool = True,
        eos: bool = True,
    ) -> List[List[int]]:
        """Encode ``text`` into a list of per-chunk id lists (padding to a fixed
        length is left to the caller so tokenizer stays dependency-free)."""
        return [
            self.encode(c, bos=bos, eos=eos)
            for c in self.iter_chunks(text, chunk_size)
        ]

    # -- decoding -----------------------------------------------------------
    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids back to text.

        Args:
            ids: token ids to decode.
            skip_special_tokens: if True (default), special-token ids (incl. any
                BOS/EOS added by ``encode``) are dropped so the output is the
                original text. If False, special tokens decode to their literal
                token strings (so ``decode(encode(x), False) == x`` for all x).
        """
        data = bytearray()
        for token_id in ids:
            special = self._vocab.id_to_special.get(int(token_id))
            if special is not None:
                if not skip_special_tokens:
                    data.extend(special.encode("utf-8"))
                continue
            data.extend(self._vocab.id_to_bytes[int(token_id)])
        return bytes(data).decode("utf-8", errors="replace")

    # -- save / load --------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the tokenizer to a single JSON file.

        Format::

            {
              "version": 1,
              "type": "ByteLevelBPE",
              "config": {...TokenizerConfig...},
              "merges": [[left, right], ...]   # in merge-rank order
            }

        The base byte vocabulary is implied (ids 0..255 = bytes), so merges +
        special tokens + config fully reconstruct the model.
        """
        payload = {
            "version": 1,
            "type": "ByteLevelBPE",
            "config": asdict(self.config),
            "merges": [list(p) for p in self._vocab.merges],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        log.info("saved tokenizer (vocab=%d, merges=%d) to %s", self.vocab_size, self.merge_count, path)

    @classmethod
    def from_file(cls, path: str) -> "ByteLevelBPETokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("type") != "ByteLevelBPE":
            raise ValueError(f"not a ByteLevelBPE file: {path}")
        if payload.get("version") != 1:
            raise ValueError(f"unsupported tokenizer file version: {payload.get('version')}")
        config = TokenizerConfig(**payload["config"])
        merges = [tuple(p) for p in payload["merges"]]
        tok = cls(config, merges)
        return tok

    # -- reporting ----------------------------------------------------------
    def vocab_report(self) -> Dict[str, object]:
        """A summary of the vocabulary for the ``--vocab-size`` report."""
        unused = self.config.vocab_size - self.vocab_size
        return {
            "vocab_size": self.vocab_size,
            "configured_vocab_size": self.config.vocab_size,
            "base_bytes": BASE_VOCAB_SIZE,
            "special_tokens": len(self._vocab.special_to_id),
            "merges": len(self._vocab.merges),
            "unused_slots": unused,
            "first_merge_id": self.config.merge_base_id(),
            "last_merge_id": self.config.merge_base_id() + self.merge_count - 1,
        }

    # -- helper for model integration --------------------------------------
    def special_id_map(self) -> Dict[str, int]:
        """Return ``{token_string: id}`` for every special token."""
        return dict(self._vocab.special_to_id)

    def byte_sequence_of(self, token_id: int) -> bytes:
        """Raw byte expansion of a token id (for low-level debugging)."""
        return self._vocab.id_to_bytes[int(token_id)]
