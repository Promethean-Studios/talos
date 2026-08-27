"""Logging helper for the ``data`` package.

Mirrors :mod:`tokenizer._logging` (which itself avoids importing torch) so the
data pipeline — which is pure stdlib + numpy — can log with the same style the
rest of Talos uses without pulling in any framework dependency.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

LOGGER_NAME = "talos.data"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return (and install, once) a logger under ``talos.data``.

    Args:
        name: sub-logger name, e.g. ``"reader"`` -> ``talos.data.reader``.
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
    level = os.environ.get("TALOS_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)
