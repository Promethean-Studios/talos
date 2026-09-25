"""Shared utilities: deterministic seeding, token-id guards, structured logging."""
from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

LOGGER_NAME = "talos"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the shared ``talos`` namespace.

    A single handler is installed once on the root ``talos`` logger so that all
    module loggers share the same console format without duplicating handlers.
    """
    logger = logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(_log_level())
    logger.propagate = False
    return logger


def _log_level() -> int:
    level = os.environ.get("FORGE_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def set_seed(seed: int) -> None:
    """Seed all random sources for reproducible runs.

    Seeds Python's ``random``, NumPy and every torch generator (CPU+GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Give a per-primitive deterministic ordering on CUDA.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_token_ids(
    input_ids: "torch.Tensor",
    vocab_size: int,
    *,
    where: str = "input_ids",
) -> None:
    """Range-check token ids against ``[0, vocab_size)`` — cheap and always-on.

    This is the CUDA device-side-assert defense: ``nn.Embedding`` and
    ``CrossEntropyLoss`` index ops fail out-of-range ids with a raw
    ``IndexError`` on CPU and an **uncatchable device-side assert** on CUDA
    (which permanently poisons the context and kills a T4 session with no
    Python traceback). Running this check *before* any index op launches turns
    that into a clear, actionable ``ValueError`` naming the offending id and
    the vocab bound.

    Cost: two reductions (``min``/``max``) and no mask materialisation — the
    bounds comparisons short-circuit on 0-dim tensors, so training throughput
    is not measurably affected (a few microseconds per batch vs. tens of
    milliseconds per step).

    Args:
        input_ids: ``(batch, seq)`` token ids (any int dtype).
        vocab_size: the model's embedding width — every id must be in
            ``[0, vocab_size)``.
        where: label for the error message (e.g. ``"train batch"``,
            ``"generation prompt"``) so a failure names its source seam.

    Raises:
        ValueError: when any id is negative or ``>= vocab_size``, naming the
            first offending id, the vocab bound, and (when recoverable) the
            min/max over the whole tensor.
    """
    if input_ids.numel() == 0:
        return
    ids_min = input_ids.min()
    ids_max = input_ids.max()
    if ids_min < 0 or ids_max >= vocab_size:
        bad = input_ids[(input_ids < 0) | (input_ids >= vocab_size)]
        raise ValueError(
            f"token id {int(bad[0])} out of range [0, {vocab_size}) in {where} "
            f"(min={int(ids_min)}, max={int(ids_max)}) — the tokenizer/model "
            f"vocab contract is violated: this id cannot be embedded. Check "
            f"for a swapped tokenizer, a config drift, or a vocab change."
        )


def human_bytes(n: float) -> str:
    """Format a byte count in a human readable way (``1.23 GiB``)."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"  # pragma: no cover - unreachable


def human_count(n: float) -> str:
    """Format a parameter/FLOP count (``1.23B``)."""
    value = float(n)
    for unit in ("", "K", "M", "B", "T", "P"):
        if abs(value) < 1000.0 or unit == "P":
            return f"{value:.3g}{unit}"
        value /= 1000.0
    return f"{value:.3g}P"  # pragma: no cover - unreachable
