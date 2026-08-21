"""Vocabulary construction and id assignment for the byte-level tokenizer.

Id assignment is deterministic and stable:

* raw bytes ``0..255`` → ids ``0..255`` (a byte's id equals its value),
* learned BPE merges → ids ``256 .. 256+n_merges-1`` in merge-rank order. This
  id numbering is identical to the one used internally during training
  (``bpe.train_bpe``), so merge *pairs* stored in the model file reference the
  same ids here — no remapping on load.
* special tokens (BOS/EOS/PAD/UNK + extras + reserved) → ids at the **top** of
  the vocabulary (LLaMA-style), ``vocab_size-n_special .. vocab_size-1`` in
  configuration order.

This layout keeps the base byte vocabulary fixed and the learned merges
immediately above it, which pairs cleanly with the model config
(``model.config.ModelConfig.vocab_size`` is the *maximum*; the tokenizer defines
the actual used vocab — see ``tokenizer/model_compat.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

BASE_VOCAB_SIZE = 256  # one token per byte value

DEFAULT_SPECIAL_TOKENS: List[str] = [
    "<|beginoftext|>",  # BOS
    "<|endoftext|>",    # EOS
    "<|pad|>",          # PAD
    "<|unk|>",          # UNK (kept for interface parity; byte-level needs no OOV)
]


@dataclass
class TokenizerConfig:
    """Configuration for :class:`~tokenizer.tokenizer.ByteLevelBPETokenizer`.

    ``vocab_size`` is the *total* vocabulary size (base bytes + special +
    merges). It is a train-time and extension-time parameter — never hard-coded.
    """

    vocab_size: int = 32768
    bos_token: str = "<|beginoftext|>"
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    #: Extra special tokens beyond the big-four (e.g. FIM / tool markers).
    extra_special_tokens: List[str] = field(default_factory=list)
    #: Reserved tokens (reserved ids never produced by encoding; kept for
    #: future model-grafting needs). They occupy ids like any special token.
    reserved_tokens: List[str] = field(default_factory=list)
    #: Pre-tokenization pattern preset name (``"gpt2"``/``"simple"``), raw regex
    #: string, or ``None`` (default) for pure byte-level. See pre_tokenize.py.
    pre_tokenize: Optional[str] = None
    #: Whether special tokens found literally inside input text are split out
    #: during ``encode``. OFF by default so ``decode(encode(x)) == x`` holds for
    #: *all* Unicode, including the literal special strings themselves.
    split_special: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE} (base bytes), got {self.vocab_size}"
            )
        if self.vocab_size % 1 != 0:
            raise ValueError("vocab_size must be an integer")
        dedup: List[str] = []
        seen: set = set()
        for t in (
            [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
            + list(self.extra_special_tokens)
            + list(self.reserved_tokens)
        ):
            if t not in seen:
                seen.add(t)
                dedup.append(t)
            elif t in (self.bos_token, self.eos_token, self.pad_token, self.unk_token):
                # The big-four must remain unique; a later identical override is
                # dropped rather than duplicated.
                continue
        self.special_tokens: List[str] = dedup
        if self.vocab_size < BASE_VOCAB_SIZE + len(self.special_tokens):
            raise ValueError(
                "vocab_size too small to hold base bytes plus special tokens "
                f"({BASE_VOCAB_SIZE} + {len(self.special_tokens)} > {self.vocab_size})"
            )
        if len(set(dedup)) != len(dedup):
            raise ValueError("special tokens must be unique")

    # -- derived accessors --------------------------------------------------
    def max_merges(self) -> int:
        """Maximum number of learned BPE merges this vocabulary can hold."""
        return self.vocab_size - BASE_VOCAB_SIZE - len(self.special_tokens)

    def merge_base_id(self) -> int:
        """First id assigned to a learned merge token (always 256)."""
        return BASE_VOCAB_SIZE


class Vocabulary:
    """Immutable id↔token tables for a trained byte-level tokenizer.

    ``merges`` is an ordered list of byte pairs (index = merge rank). The
    class derives all id mappings deterministically from it.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        merges: Sequence[Sequence[int]],
    ) -> None:
        self.config = config
        self.merges: List[tuple] = [tuple(m) for m in merges]
        n_merges = len(self.merges)
        if n_merges > config.max_merges():
            raise ValueError(
                f"{n_merges} merges exceed capacity {config.max_merges()} "
                f"(vocab_size={config.vocab_size}, special={len(config.special_tokens)})"
            )

        # Merge pairs -> ids and their byte expansions. Merge ids occupy
        # ``256 .. 256+n_merges-1`` in rank order (== the training numbering).
        self.pair_to_merge_id: Dict[tuple, int] = {}
        self.pair_to_rank: Dict[tuple, int] = {}
        for rank, pair in enumerate(self.merges):
            self.pair_to_merge_id[pair] = BASE_VOCAB_SIZE + rank
            self.pair_to_rank[pair] = rank

        # Special tokens sit at the TOP of the vocabulary (LLaMA-style).
        n_special = len(self.config.special_tokens)
        top = config.vocab_size
        self.special_to_id: Dict[str, int] = {
            t: top - n_special + i for i, t in enumerate(self.config.special_tokens)
        }
        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special_to_id.items()}

        # Full token id -> raw byte sequence, for O(1) decode. Byte ids map to
        # themselves; merge tokens expand recursively (their components may be
        # earlier merge tokens, not just raw bytes).
        self.id_to_bytes: Dict[int, bytes] = {b: bytes([b]) for b in range(BASE_VOCAB_SIZE)}
        self._expand_merge_bytes()

    def _expand_merge_bytes(self) -> None:
        """Compute the byte expansion of every merge token, in rank order.

        A merge token whose components are themselves merge tokens resolves
        against the already-computed byte maps (ranks are processed in order, so
        lower ranks are always available first).
        """
        base = self.config.merge_base_id()
        for rank, pair in enumerate(self.merges):
            token_id = base + rank
            left = self.id_to_bytes[pair[0]]
            right = self.id_to_bytes[pair[1]]
            self.id_to_bytes[token_id] = left + right

    # -- accessors ----------------------------------------------------------
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.special_to_id) + len(self.merges)

    def bos_id(self) -> int:
        return self.special_to_id[self.config.bos_token]

    def eos_id(self) -> int:
        return self.special_to_id[self.config.eos_token]

    def pad_id(self) -> int:
        return self.special_to_id[self.config.pad_token]

    def unk_id(self) -> int:
        return self.special_to_id[self.config.unk_token]

    def id_to_special_token(self, token_id: int) -> Optional[str]:
        return self.id_to_special.get(token_id)
