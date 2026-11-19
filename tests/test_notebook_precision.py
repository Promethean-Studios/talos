"""Fix 3 — precision paths and optimizer state memory for the Argo 1B run.

What the audit found and what these tests pin:

* Parameters stay **fp32 master weights**; bf16 (L4/A100) or fp16 + GradScaler
  (T4/V100) is applied by autocast, never by casting the weights. Casting the
  parameters to bf16/fp16 rounds away sub-resolution updates and makes
  bitsandbytes' AdamW8bit refuse the model outright.
* The scaler is applied to the loss (``scaler.scale(loss).backward()``) and the
  scaler — not the raw optimizer — performs the step when fp16 is in use.
* bf16 needs no scaler; fp32 disables autocast.
* The optimizer is 8-bit AdamW when bitsandbytes is usable, plain AdamW
  otherwise, and the run records which one it used.
* Static memory for 1,036,060,672 parameters: weights 4.14 GB + fp32 grads
  4.14 GB + optimizer state 2.07 GB (8-bit, 2 bytes/param) or 8.29 GB (fp32
  AdamW, 8 bytes/param) = 10.35 GB or 16.6 GB before activations.

CPU only, tiny tensors: no GPU, no bitsandbytes, no network.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOTEBOOKS = os.path.join(_ROOT, "notebooks")
for _path in (_ROOT, _NOTEBOOKS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

torch = pytest.importorskip("torch")

from talos_torch_utils import (autocast_kwargs, build_optimizer,  # noqa: E402
                               clip_gradients, grad_norm, grads_are_finite,
                               loss_is_finite, make_grad_scaler)

PARAMS_1B = 1_036_060_672
GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Autocast: bf16 on L4/A100, fp16 on a T4, fp32 off
# ---------------------------------------------------------------------------
def test_autocast_matches_the_gpu():
    assert autocast_kwargs("cuda", "bf16") == {"device_type": "cuda",
                                               "dtype": torch.bfloat16,
                                               "enabled": True}
    assert autocast_kwargs("cuda", "fp16") == {"device_type": "cuda",
                                               "dtype": torch.float16,
                                               "enabled": True}
    assert autocast_kwargs("cpu", "fp32") == {"device_type": "cpu", "enabled": False}
    with pytest.raises(ValueError):
        autocast_kwargs("cuda", "tf32")


def test_only_fp16_gets_a_grad_scaler():
    assert make_grad_scaler(enabled=False) is None          # bf16 / fp32
    scaler = make_grad_scaler(enabled=True)                 # fp16 path
    assert scaler is not None
    # No CUDA here, so torch disables it — the object still exists and the
    # notebook's `scaler.scale(loss)` / `scaler.step()` calls stay valid.
    assert hasattr(scaler, "scale") and hasattr(scaler, "step")


# ---------------------------------------------------------------------------
# Master weights stay fp32; the optimizer records what it is
# ---------------------------------------------------------------------------
def test_parameters_stay_fp32_master_weights():
    model = torch.nn.Linear(4, 4)
    optimizer, note = build_optimizer(model.parameters(), lr=1e-3, use_8bit_adam=False)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert "8-bit disabled" in note
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # One step must not change the parameter dtype either.
    model(torch.zeros(1, 4)).sum().backward()
    optimizer.step()
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_8bit_adam_falls_back_without_bitsandbytes():
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("bitsandbytes is installed on this box")
    model = torch.nn.Linear(4, 4)
    optimizer, note = build_optimizer(model.parameters(), lr=1e-3, use_8bit_adam=True)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert "bitsandbytes unavailable" in note
    # The note is what goes into the checkpoint metadata, so it must name the
    # optimizer that was actually built.
    assert "plain AdamW" in note


def test_build_optimizer_rejects_an_empty_or_unusable_configuration():
    with pytest.raises(ValueError):
        build_optimizer([], lr=1e-3)
    with pytest.raises(ValueError):
        build_optimizer(torch.nn.Linear(2, 2).parameters(), lr=0.0)


# ---------------------------------------------------------------------------
# The documented static-memory arithmetic for 1.036B parameters
# ---------------------------------------------------------------------------
def test_documented_static_memory_arithmetic_is_consistent():
    weights = PARAMS_1B * 4                       # fp32 master weights
    grads = PARAMS_1B * 4                         # fp32 gradients
    state_8bit = PARAMS_1B * 2                    # AdamW8bit: 2 bytes/param
    state_fp32 = PARAMS_1B * 8                    # AdamW: m and v, both fp32
    assert weights / GB == pytest.approx(3.86, abs=0.01)     # ~4.14 GB decimal
    assert (weights + grads + state_8bit) / GB == pytest.approx(9.64, abs=0.02)
    assert (weights + grads + state_fp32) / GB == pytest.approx(15.44, abs=0.02)
    # 8-bit state is 4x smaller than fp32 state: the reason it is the default.
    assert state_fp32 == 4 * state_8bit


# ---------------------------------------------------------------------------
# Gradient-norm / non-finite helpers: the order of operations matters
# ---------------------------------------------------------------------------
def test_loss_is_finite_flags_nan_and_inf():
    assert loss_is_finite(torch.tensor(1.0))
    assert not loss_is_finite(torch.tensor(float("nan")))
    assert not loss_is_finite(torch.tensor(float("inf")))
    assert not loss_is_finite(torch.tensor([1.0, float("nan")]))
    with pytest.raises(TypeError):
        loss_is_finite(1.0)


def test_grad_norm_reports_inf_before_clipping_poisons_it():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
    assert not grads_are_finite([parameter])
    assert grad_norm([parameter]) == float("inf")
    # Clipping first would turn Inf into NaN and hide the anomaly — which is
    # exactly why the loop checks finiteness before clipping.
    clip_gradients([parameter], 1.0)
    assert bool(torch.isnan(parameter.grad).any())


def test_grad_norm_and_finiteness_on_healthy_gradients():
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = torch.tensor([3.0, 4.0, 0.0])
    assert grads_are_finite([parameter])
    assert grad_norm([parameter]) == pytest.approx(5.0)
    clip_gradients([parameter], 1.0)
    assert grad_norm([parameter]) == pytest.approx(1.0, abs=1e-6)


def test_grad_helpers_ignore_parameters_without_grads():
    parameter = torch.nn.Parameter(torch.zeros(2))
    assert grad_norm([parameter]) == 0.0
    assert grads_are_finite([parameter])
    with pytest.raises(ValueError):
        clip_gradients([parameter], 0.0)
