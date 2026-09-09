"""Tests for the Promethean loader: banner safety and byte-exact art.

The loader's one hard requirement is that it can never corrupt terminal output,
logging, errors, or piped scripts. These tests pin that contract down:

* the bundled art is byte-identical to the owner's spec (and a known SHA-256);
* the banner prints only on a real TTY that is wide enough to fit it;
* ``PROMETHEAN_NO_BANNER`` / ``TALOS_NO_BANNER`` suppress it;
* any failure (missing art, broken stream, weird terminal) fails silent;
* the ``logging`` subsystem and ``stderr`` are never touched.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path

import pytest

from promethean import loader

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_ART = REPO_ROOT / "tests" / "data" / "promethean-loader-art.txt"
BUNDLED_ART = REPO_ROOT / "promethean" / "art" / "talos.txt"

# SHA-256 of the owner's spec art file (embedded so any accidental reflow,
# retyping, or trailing-space change is caught even without the spec copy).
SPEC_SHA256 = "ad9faa1e2c57702fd5d85ebbf02e53c46f78bb73f6e36a5a4039cf0a9810a713"


class FakeTTY(io.StringIO):
    """A stream that reports itself as a TTY but has no real file descriptor."""

    def isatty(self) -> bool:
        return True


class NoIsattyStream:
    """A stream whose ``isatty`` raises — must be handled without raising."""

    def isatty(self):  # noqa: D401 - deliberately broken
        raise OSError("isatty exploded")

    def write(self, _data):
        raise OSError("write exploded")

    def flush(self):
        raise OSError("flush exploded")


def _wide(columns: int = 200):
    """Monkeypatch terminal width to ``columns`` (used via pytest monkeypatch)."""

    def patch(monkeypatch):
        class Size:
            def __init__(self):
                self.columns = columns

        monkeypatch.setattr(loader.shutil, "get_terminal_size", lambda *a, **k: Size())
        monkeypatch.setattr(loader.os, "get_terminal_size", lambda *a, **k: Size())

    return patch


@pytest.fixture
def spec_bytes() -> bytes:
    return SPEC_ART.read_bytes()


# ---------------------------------------------------------------------------
# Art byte-exactness
# ---------------------------------------------------------------------------

def test_bundled_art_matches_spec_byte_for_byte(spec_bytes):
    assert BUNDLED_ART.read_bytes() == spec_bytes


def test_bundled_art_sha256_matches_embedded_hash(spec_bytes):
    assert hashlib.sha256(BUNDLED_ART.read_bytes()).hexdigest() == SPEC_SHA256
    assert hashlib.sha256(spec_bytes).hexdigest() == SPEC_SHA256


def test_art_ends_with_newline_and_is_utf8(spec_bytes):
    text = spec_bytes.decode("utf-8")
    assert text.endswith("\n")
    # Only spaces and the U+2588 full-block glyph may appear (no reflow junk).
    assert set(text) <= {"\n", " ", "\u2588"}


# ---------------------------------------------------------------------------
# TTY gating
# ---------------------------------------------------------------------------

def test_no_output_when_stream_is_not_a_tty(monkeypatch):
    _wide(200)(monkeypatch)
    stream = io.StringIO()  # StringIO.isatty() -> False
    assert loader.show("talos", stream=stream) is False
    assert stream.getvalue() == ""


def test_output_when_tty_and_wide(monkeypatch, spec_bytes):
    _wide(200)(monkeypatch)
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is True
    assert stream.getvalue() == spec_bytes.decode("utf-8")


def test_custom_art_is_written_exactly(monkeypatch):
    _wide(200)(monkeypatch)
    stream = FakeTTY()
    assert loader.show("talos", art="ABC\n", stream=stream) is True
    assert stream.getvalue() == "ABC\n"


def test_missing_trailing_newline_is_added(monkeypatch):
    _wide(200)(monkeypatch)
    stream = FakeTTY()
    assert loader.show("talos", art="ABC", stream=stream) is True
    assert stream.getvalue() == "ABC\n"


# ---------------------------------------------------------------------------
# Environment opt-out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("var", ["PROMETHEAN_NO_BANNER", "TALOS_NO_BANNER"])
def test_env_opt_out_suppresses_banner(monkeypatch, var):
    _wide(200)(monkeypatch)
    monkeypatch.setenv(var, "1")
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is False
    assert stream.getvalue() == ""


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "Y", "t"])
def test_env_truthy_values_suppress(monkeypatch, value):
    _wide(200)(monkeypatch)
    monkeypatch.setenv("PROMETHEAN_NO_BANNER", value)
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is False


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "   "])
def test_env_falsy_values_do_not_suppress(monkeypatch, value):
    _wide(200)(monkeypatch)
    monkeypatch.setenv("PROMETHEAN_NO_BANNER", value)
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is True
    assert "\u2588" in stream.getvalue()


def test_enabled_false_disables_unconditionally(monkeypatch):
    _wide(200)(monkeypatch)
    stream = FakeTTY()
    assert loader.show("talos", stream=stream, enabled=False) is False
    assert stream.getvalue() == ""


def test_custom_env_vars_override_defaults(monkeypatch):
    _wide(200)(monkeypatch)
    monkeypatch.setenv("MY_FLAG", "yes")
    stream = FakeTTY()
    assert loader.show("talos", stream=stream, env_vars=["MY_FLAG"]) is False
    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# Narrow-terminal skip
# ---------------------------------------------------------------------------

def test_narrow_terminal_skips_banner(monkeypatch):
    _wide(101)(monkeypatch)  # art's widest line is 102 columns
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is False
    assert stream.getvalue() == ""


def test_exact_width_fits(monkeypatch):
    _wide(102)(monkeypatch)  # exactly the art's widest line — must not wrap
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is True
    assert "\u2588" in stream.getvalue()


# ---------------------------------------------------------------------------
# Exception safety / never raise
# ---------------------------------------------------------------------------

def test_missing_art_fails_silent(monkeypatch):
    _wide(200)(monkeypatch)
    stream = FakeTTY()
    assert loader.show("does-not-exist", stream=stream) is False
    assert stream.getvalue() == ""


def test_broken_isatty_does_not_raise():
    assert loader.show("talos", stream=NoIsattyStream()) is False


def test_broken_write_does_not_raise(monkeypatch):
    _wide(200)(monkeypatch)

    class BadWriteTTY(FakeTTY):
        def write(self, _data):
            raise OSError("write exploded")

    assert loader.show("talos", stream=BadWriteTTY()) is False


def test_load_art_exception_is_caught(monkeypatch):
    _wide(200)(monkeypatch)

    def boom(_name):
        raise RuntimeError("art load failed")

    monkeypatch.setattr(loader, "load_art", boom)
    stream = FakeTTY()
    assert loader.show("talos", stream=stream) is False
    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# Logging untouched
# ---------------------------------------------------------------------------

def test_show_never_touches_logging_handlers(monkeypatch, capsys):
    root = logging.getLogger()
    before_root = len(root.handlers)
    talos_logger = logging.getLogger("talos")
    before_talos = len(talos_logger.handlers)

    _wide(200)(monkeypatch)
    stream = FakeTTY()
    loader.show("talos", stream=stream)

    assert len(root.handlers) == before_root
    assert len(talos_logger.handlers) == before_talos
    # Nothing may be written to stderr.
    assert capsys.readouterr().err == ""


def test_import_is_side_effect_free(capsys):
    """Importing the loader must not print or configure logging."""
    import importlib

    root = logging.getLogger()
    before = len(root.handlers)
    mod = importlib.import_module("promethean.loader")
    assert hasattr(mod, "show")
    assert len(root.handlers) == before
    assert capsys.readouterr().err == ""
