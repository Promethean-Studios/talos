"""Promethean: reusable, dependency-free branding helpers.

The only public API lives in :mod:`promethean.loader`. Importing this package
must be side-effect free: it should never print, configure logging, or touch
any terminal/stream state.
"""

from __future__ import annotations

__all__ = ["loader"]

__version__ = "0.1.0"
