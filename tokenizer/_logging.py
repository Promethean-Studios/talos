"""Forge-style structured logging for the tokenizer (no torch dependency).

The main :mod:`model.utils` logger imports torch, which Phase 2 explicitly
does *not* want. This module provides a same-format logger using only the
standard library so the tokenizer package stays importable with numpy +
stdlib alone.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

LOGGER_NAME = "forge.tokenizer"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the shared ``forge.tokenizer`` namespace.

    Mirrors :func:`model.utils.get_logger`'s formatting but intentionally
    avoids importing torch.
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
