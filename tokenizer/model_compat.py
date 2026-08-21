"""Vocab ↔ model-config integration helper (Phase 2 → later phases).

The model config (``model.config.ModelConfig.vocab_size``) is the **maximum**
embedding width a model is built with. The tokenizer's ``vocab_size`` is the
**actual** number of tokens that will be used. This module documents and
implements that contract so a later phase can graft the tokenizer onto the
model's embedding / LM head cleanly.

Rules:
* ``tokenizer_vocab_size <= model_vocab_size`` must hold.
* The tokenizer's first ``vocab_size`` ids (bytes, then specials, then merges)
  map 1:1 onto the first ``tokenizer_vocab_size`` embedding rows.
* The model may keep ``model_vocab_size - tokenizer_vocab_size`` extra
  rows as *padding* — either ignored, or reserved for future vocab extension
  (e.g. growing merges / adding special tokens without rebuilding the model).

This module deliberately has **no torch dependency**: ``build_embeddings`` is
left to the training/inference phase (they own the tensor types). We provide the
pure-Python mapping contract plus a classic resize plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from tokenizer.tokenizer import ByteLevelBPETokenizer


@dataclass
class VocabMapping:
    """Description of how a tokenizer maps onto a model's vocabulary.

    Attributes:
        tokenizer_vocab_size: number of token ids the tokenizer actually uses.
        model_vocab_size: the model config's (maximum) vocab size.
        padding: how many extra (unused) rows the model keeps.
        fits: whether ``tokenizer_vocab_size <= model_vocab_size``.
        special_ids: ``{token_string: id}`` for the tokenizer's special tokens.
    """

    tokenizer_vocab_size: int
    model_vocab_size: int
    special_ids: Dict[str, int]

    @property
    def padding(self) -> int:
        return self.model_vocab_size - self.tokenizer_vocab_size

    @property
    def fits(self) -> bool:
        return self.tokenizer_vocab_size <= self.model_vocab_size

    def to_dict(self) -> dict:
        return {
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "model_vocab_size": self.model_vocab_size,
            "padding_slots": self.padding,
            "fits": self.fits,
            "special_ids": self.special_ids,
        }


def map_to_model_vocab(
    tokenizer: ByteLevelBPETokenizer,
    model_vocab_size: int,
) -> VocabMapping:
    """Compute the mapping between a trained tokenizer and a model config.

    If ``model_vocab_size`` is smaller than the tokenizer vocab, a
    :class:`VocabMapping` with ``fits=False`` is still returned (so diagnostics
    are possible); callers that require a hard failure should check ``.fits``.

    The tokenizer must not exceed the model vocab for a valid model: the first
    ``tokenizer_vocab_size`` ids are used verbatim and the remaining model slots
    are padding/reserved.
    """
    tok_vocab = tokenizer.vocab_size
    if tok_vocab > model_vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tok_vocab}) exceeds model vocab ({model_vocab_size}). "
            "Increase the model config vocab_size or reduce the tokenizer vocab."
        )
    return VocabMapping(
        tokenizer_vocab_size=tok_vocab,
        model_vocab_size=model_vocab_size,
        special_ids=tokenizer.special_id_map(),
    )


def resize_embedding_plan(
    tokenizer_vocab_size: int,
    model_vocab_size: int,
    hidden_size: int,
) -> Dict[str, int]:
    """Return a documented plan for building/extending an embedding matrix.

    This is the classic "grow the embedding" recipe used by later phases; it
    returns *dimensions* only (no torch). The implementing phase converts it to
    tensor ops: allocate ``(model_vocab_size, hidden_size)``, copy the first
    ``tokenizer_vocab_size`` rows from any existing (smaller) embedding, and
    re-initialise/leave the ``padding`` extra rows as unused/reserved.

    Args:
        tokenizer_vocab_size: actual tokenizer vocab.
        model_vocab_size: model config maximum vocab.
        hidden_size: embedding dimension.

    Returns:
        Dict with the rows to copy, the padding rows, and total rows, so a
        later phase can build the matrix directly.
    """
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer_vocab_size} > model vocab {model_vocab_size}"
        )
    return {
        "used_rows": tokenizer_vocab_size,
        "padding_rows": model_vocab_size - tokenizer_vocab_size,
        "total_rows": model_vocab_size,
        "embedding_size": hidden_size,
    }
