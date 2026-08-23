"""Talos tokenizer (Phase 2): byte-level BPE tokenizer + training.

A modern, research-grade byte-level BPE tokenizer (GPT-2/LLaMA-style) that can
represent arbitrary Unicode losslessly, a configurable vocabulary size up to
~128K, special/reserved tokens, and a scalable training CLI with checkpoint and
resume. Pure Python + stdlib (numpy not even required for the core).

See ``docs/tokenizer.md`` for design decisions and ``tokenizer/PHASE2.md`` for
the Phase 2 status.

Note: heavy submodules (``tokenizer.train``) are loaded lazily via
``__getattr__`` so that ``python -m tokenizer.train`` does not trigger a
"module already in sys.modules" runtime warning.
"""
from __future__ import annotations

from typing import Any

from tokenizer.bpe import train_bpe, train_bpe_naive
from tokenizer.corpus import CorpusError, iter_text_documents
from tokenizer.model_compat import VocabMapping, map_to_model_vocab, resize_embedding_plan
from tokenizer.tokenizer import ByteLevelBPETokenizer
from tokenizer.vocab import BASE_VOCAB_SIZE, TokenizerConfig, Vocabulary

__all__ = [
    "BASE_VOCAB_SIZE",
    "ByteLevelBPETokenizer",
    "CorpusError",
    "TokenizerConfig",
    "TrainResult",
    "VocabMapping",
    "Vocabulary",
    "iter_text_documents",
    "map_to_model_vocab",
    "resize_embedding_plan",
    "train_bpe",
    "train_bpe_naive",
    "train_tokenizer",
]


def __getattr__(name: str) -> Any:
    """Lazily import ``train_tokenizer`` / ``TrainResult`` from train.py."""
    if name in ("train_tokenizer", "TrainResult"):
        from tokenizer.train import TrainResult, train_tokenizer  # noqa: PLC0415

        return {"train_tokenizer": train_tokenizer, "TrainResult": TrainResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
