"""Pytest bootstrap: make the repo root importable as ``model`` / ``configs``.

This lets ``pytest`` run from the repo root with only ``torch`` + ``pytest``
installed, even when Forge hasn't been ``pip install -e``'d.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.utils import set_seed  # noqa: E402

set_seed(0)
