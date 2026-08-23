"""Shared utilities: deterministic seeding and structured logging."""
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
