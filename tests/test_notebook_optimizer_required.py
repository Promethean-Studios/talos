"""Regression: 8-bit AdamW requested means 8-bit AdamW required.

Owner directive (2026-11-25): the notebook must never silently downgrade to
plain AdamW when the hardware plan asked for bitsandbytes. Before this fix
``build_optimizer`` caught every bitsandbytes failure and returned
``torch.optim.AdamW`` with a log line — an fp32 AdamW needs ~8.3 GB of optimizer
state for the 1B model, which does not fit a 16/24 GB Colab GPU, so the run
"worked" right up to the point where it was unusable (and its optimizer state
meant something different from what the checkpoint claimed).

The contract these tests pin down:

* 8-bit requested + bitsandbytes unavailable => ``RuntimeError`` naming the
  dependency and the fix, and **no** optimizer is constructed at all;
* 8-bit requested + ``AdamW8bit`` construction fails => same, with the
  construction failure in the message;
* the plan itself saying plain AdamW (CPU smoke run, CI, >48 GiB GPU) still
  builds plain AdamW — that is not a fallback;
* the built optimizer's identity string is exact, because fix 2 persists it in
  the checkpoint and compares it on resume.

Torch is required (skipped where it is absent, like the other notebook tests);
the ``plan_for_gpu`` half of the fix is torch-free and lives in
``tests/test_notebook_hardware.py``.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import talos_torch_utils as tu  # noqa: E402


def _params():
    return torch.nn.Linear(4, 4).parameters()


def _forbid_plain_adamw(monkeypatch):
    """Make any plain-AdamW construction fail loudly instead of silently."""
    original = torch.optim.AdamW

    def _explode(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError(
            "plain torch.optim.AdamW was constructed although 8-bit was required")

    monkeypatch.setattr(torch.optim, "AdamW", _explode)
    return original


# ---------------------------------------------------------------------------
# 8-bit required: import failure raises and never falls back
# ---------------------------------------------------------------------------
def test_missing_bitsandbytes_raises_and_never_builds_plain_adamw(monkeypatch):
    # `None` in sys.modules makes `import bitsandbytes` raise ImportError.
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    _forbid_plain_adamw(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        tu.build_optimizer(_params(), lr=1e-3, use_8bit_adam=True)

    message = str(excinfo.value)
    assert "bitsandbytes" in message
    assert "pip install bitsandbytes" in message
    assert "NOT downgraded to plain AdamW" in message
    assert "USE_8BIT_ADAM = False" in message


def test_a_broken_8bit_optimizer_construction_raises(monkeypatch):
    """Importable bitsandbytes whose AdamW8bit cannot be built (CUDA mismatch)."""
    fake = types.ModuleType("bitsandbytes")
    fake.optim = types.SimpleNamespace(
        AdamW8bit=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("no kernel image is available for execution on the device")))
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    _forbid_plain_adamw(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        tu.build_optimizer(_params(), lr=1e-3, use_8bit_adam=True)

    message = str(excinfo.value)
    assert "AdamW8bit construction failed" in message
    assert "no kernel image" in message
    assert "pip install bitsandbytes" in message


def test_a_working_bitsandbytes_yields_the_8bit_identity(monkeypatch):
    """When it works, the identity is exactly the string fix 2 persists."""
    built = {}

    def _adamw8bit(params, **kwargs):
        built["kwargs"] = kwargs
        return torch.optim.AdamW(list(params), lr=kwargs["lr"])

    fake = types.ModuleType("bitsandbytes")
    fake.optim = types.SimpleNamespace(AdamW8bit=_adamw8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)

    optimizer, identity = tu.build_optimizer(_params(), lr=1e-3, use_8bit_adam=True)

    assert identity == tu.OPTIMIZER_8BIT == "AdamW8bit (bitsandbytes)"
    assert isinstance(optimizer, torch.optim.AdamW)     # the stand-in class
    assert built["kwargs"]["lr"] == 1e-3
    assert built["kwargs"]["weight_decay"] == 0.1


# ---------------------------------------------------------------------------
# The plan says plain: that is the plan, not a fallback
# ---------------------------------------------------------------------------
def test_plain_adamw_when_the_plan_says_plain(monkeypatch):
    # bitsandbytes being present must not change the answer either way.
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    optimizer, identity = tu.build_optimizer(_params(), lr=1e-3, use_8bit_adam=False)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert identity == tu.OPTIMIZER_PLAIN == "AdamW"


def test_the_two_identities_are_distinct():
    """Resume validation keys on these strings, so they must never collide."""
    assert tu.OPTIMIZER_8BIT != tu.OPTIMIZER_PLAIN
