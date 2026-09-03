"""Dependency-free ASCII banner loader for Promethean CLI tools.

This module prints a short branding banner when a CLI starts, and it is
engineered around one hard rule: **it must never corrupt terminal output,
logging, errors, or piped scripts.** Every safety gate below exists to make
sure the banner only appears on a real, interactive terminal that can actually
fit it, and that any failure degrades to *silently printing nothing*.

Safety rules
------------
1. **TTY only** — the banner is written only when ``stream.isatty()`` is
   truthy. Piped, redirected, and captured output (``pytest``, ``subprocess``,
   ``io.StringIO``, ``> file``) never receives art.
2. **Environment opt-out** — if ``PROMETHEAN_NO_BANNER`` or
   ``TALOS_NO_BANNER`` is set to a truthy value, the banner is skipped. Both
   names are supported so the generic Promethean tooling and Talos-specific
   tooling share one switch. Truthy values are (case-insensitive)
   ``1``, ``true``, ``yes``, ``on``, ``y``, ``t``.
3. **Narrow terminal** — if the terminal width (via
   :func:`shutil.get_terminal_size`, preferring the stream's own file
   descriptor) is narrower than the art's widest line, the banner is skipped
   so nothing wraps into garbage.
4. **Never raise** — all work is wrapped in ``try/except`` and fails silent.
   This module never configures or touches ``logging`` handlers, never writes
   to ``sys.stderr``, and never sleeps or animates in the default path.

Art storage
-----------
Art is stored as plain UTF-8 text files under ``promethean/art/`` and loaded
with :func:`load_art`. The Talos graphic lives at ``promethean/art/talos.txt``
and is copied byte-for-byte from the owner's spec (never retyped/reformatted).
Swap the graphic for another project by adding a new file under ``art/`` and
passing its name to :func:`show` — no code change required.
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, TextIO

#: Environment variables that suppress the banner (checked in order).
DEFAULT_ENV_VARS: tuple[str, ...] = ("PROMETHEAN_NO_BANNER", "TALOS_NO_BANNER")

#: Values treated as "truthy" when interpreting the opt-out variables.
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})

#: Default art name to load when :func:`show` is called without ``art``.
DEFAULT_ART_NAME = "talos"


def _truthy(value: Optional[str]) -> bool:
    """Interpret an environment value as a boolean opt-out flag."""
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY_VALUES


def _display_width(line: str) -> int:
    """Visual width of ``line`` (wide CJK block chars count as 2 columns)."""
    width = 0
    for ch in line:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _max_display_width(text: str) -> int:
    """Width of the widest line in ``text``."""
    return max((_display_width(line) for line in text.split("\n")), default=0)


def load_art(name: str = DEFAULT_ART_NAME) -> str:
    """Load bundled art text for ``name`` (``promethean/art/<name>.txt``).

    Uses :mod:`importlib.resources` when available so the file is found even
    inside a zipped/installed package, and falls back to ``Path(__file__)`` for
    plain source-tree runs. Returns the file contents decoded as UTF-8,
    preserving the art byte-for-byte.
    """
    try:
        from importlib import resources

        ref = resources.files("promethean").joinpath("art", f"{name}.txt")
        return ref.read_bytes().decode("utf-8")
    except Exception:
        path = Path(__file__).resolve().parent / "art" / f"{name}.txt"
        return path.read_bytes().decode("utf-8")


def _terminal_width(stream: TextIO) -> Optional[int]:
    """Return the terminal column count for ``stream``, or ``None`` if unknown."""
    # Prefer the stream's own fd so the measured width matches the real target.
    try:
        fd = stream.fileno()  # type: ignore[attr-defined]
    except Exception:
        fd = None
    if fd is not None:
        try:
            columns = os.get_terminal_size(fd).columns
            if columns > 0:
                return columns
        except Exception:
            pass
    try:
        columns = shutil.get_terminal_size().columns
        if columns > 0:
            return columns
    except Exception:
        pass
    return None


def show(
    name: str,
    art: Optional[str] = None,
    stream: Optional[TextIO] = None,
    enabled: Optional[bool] = None,
    env_vars: Optional[Iterable[str]] = None,
) -> bool:
    """Print the banner ``name`` to ``stream`` if (and only if) it is safe.

    Parameters
    ----------
    name:
        Banner label. When ``art`` is ``None`` this is also the bundled art file
        to load (``promethean/art/<name>.txt``).
    art:
        Exact banner text. If ``None``, :func:`load_art` loads ``name``.
    stream:
        Destination stream. Defaults to ``sys.stdout``.
    enabled:
        ``False`` disables the banner unconditionally (e.g. a CLI
        ``--no-banner`` flag). ``True`` is equivalent to the default ``None``:
        the banner is *attempted*, but the TTY/env/width safety gates below
        still apply and can still suppress it.
    env_vars:
        Environment variables to treat as opt-out switches. Defaults to
        :data:`DEFAULT_ENV_VARS`.

    Returns
    -------
    bool
        ``True`` if the banner was written, ``False`` otherwise. Never raises.
    """
    try:
        if enabled is False:
            return False

        stream = stream if stream is not None else sys.stdout

        # 1. Environment opt-out (generic + Talos-specific names).
        for var in env_vars if env_vars is not None else DEFAULT_ENV_VARS:
            if _truthy(os.environ.get(var)):
                return False

        # 2. Only ever write to a real interactive terminal.
        try:
            is_tty = bool(stream.isatty())  # type: ignore[attr-defined]
        except Exception:
            is_tty = False
        if not is_tty:
            return False

        # 3. Load the art (byte-for-byte from the bundled data file).
        text = art if art is not None else load_art(name)

        # 4. Never let a too-narrow terminal wrap the graphic.
        columns = _terminal_width(stream)
        if columns is None or _max_display_width(text) > columns:
            return False

        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        try:
            stream.flush()
        except Exception:
            pass
        return True
    except Exception:
        # Fail silent: a broken banner must never break the CLI.
        return False
