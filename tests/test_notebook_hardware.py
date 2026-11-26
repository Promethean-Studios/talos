"""Tests for the notebook's Colab hardware detection and plan (``talos_hardware``).

Fake torch modules stand in for a GPU, so every branch — A100, L4, T4 (no bf16),
a CUDA build with no device, a CPU-only wheel — is exercised with no GPU here.
"""
from __future__ import annotations

import os
import sys

import pytest

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import talos_hardware as hw  # noqa: E402

SEQ = 2048


class _FakeProps:
    def __init__(self, total_memory):
        self.total_memory = total_memory


class _FakeCuda:
    def __init__(self, name, vram_gib, capability, available=True, count=1):
        self._name = name
        self._vram = vram_gib
        self._cap = capability
        self._available = available
        self._count = count

    def is_available(self):
        return self._available

    def get_device_name(self, index=0):
        return self._name

    def get_device_properties(self, index=0):
        return _FakeProps(self._vram * 2 ** 30)

    def get_device_capability(self, index=0):
        return self._cap

    def device_count(self):
        return self._count


class _FakeTorch:
    def __init__(self, cuda, version="2.9.0+cu124"):
        self.cuda = cuda
        self.__version__ = version
        self.version = type("V", (), {"cuda": "12.4"})()
        self.bfloat16 = "bfloat16"


def _torch(name, vram, capability, **kwargs):
    return _FakeTorch(_FakeCuda(name, vram, capability, **kwargs))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_detects_an_l4_and_enables_bf16():
    info = hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9)))
    assert info.is_gpu and info.bf16_supported
    assert info.vram_gib == pytest.approx(24.0, abs=0.1)
    assert info.capability == (8, 9)
    assert "L4" in info.label and "sm89" in info.label


def test_t4_has_no_bf16():
    info = hw.detect_gpu(_torch("Tesla T4", 16.0, (7, 5)))
    assert info.is_gpu and info.bf16_supported is False


def test_cuda_build_without_a_device_reports_cpu():
    info = hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9), available=False, count=0))
    assert not info.is_gpu and info.label == "CPU"


def test_a_missing_torch_does_not_raise():
    class _Broken:
        def __getattr__(self, item):
            raise RuntimeError("no torch here")

    info = hw.detect_gpu(_Broken())
    assert info.is_gpu is False
    assert info.torch_version in ("unknown", "unreadable")


def test_check_cuda_torch_explains_how_to_fix_a_cpu_runtime():
    with pytest.raises(RuntimeError) as excinfo:
        hw.check_cuda_torch(_torch("NVIDIA L4", 24.0, (8, 9), available=False, count=0))
    message = str(excinfo.value)
    assert "GPU" in message
    assert "pip install torch" in message          # names the trap explicitly
    assert "factory reset" in message.lower() or "delete runtime" in message.lower()


def test_check_cuda_torch_passes_on_a_real_gpu():
    assert hw.check_cuda_torch(_torch("NVIDIA A100-SXM4-40GB", 40.0, (8, 0)))["cuda_available"]


def test_colab_detection_uses_environment_markers():
    assert hw.is_colab({"COLAB_RELEASE_TAG": "2026-11"})
    assert hw.is_colab({"COLAB_GPU": "1"})
    assert not hw.is_colab({})


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,vram,cap,precision,scaler", [
    ("NVIDIA A100-SXM4-40GB", 40.0, (8, 0), "bf16", False),
    ("NVIDIA L4", 24.0, (8, 9), "bf16", False),
    ("Tesla T4", 16.0, (7, 5), "fp16", True),
    ("Tesla V100-SXM2-16GB", 16.0, (7, 0), "fp16", True),
])
def test_plan_picks_precision_by_capability(name, vram, cap, precision, scaler):
    plan = hw.plan_for_gpu(hw.detect_gpu(_torch(name, vram, cap)), seq_len=SEQ)
    assert plan.precision == precision
    assert plan.use_grad_scaler is scaler
    assert plan.gradient_checkpointing is True


@pytest.mark.parametrize("name,vram,cap", [
    ("NVIDIA A100-SXM4-40GB", 40.0, (8, 0)),
    ("NVIDIA L4", 24.0, (8, 9)),
    ("Tesla T4", 16.0, (7, 5)),
])
def test_every_gpu_gets_the_same_tokens_per_step(name, vram, cap):
    plan = hw.plan_for_gpu(hw.detect_gpu(_torch(name, vram, cap)), seq_len=SEQ)
    assert plan.tokens_per_step == hw.ROWS_PER_STEP * SEQ
    assert plan.micro_batch * plan.grad_accum == hw.ROWS_PER_STEP


def test_bigger_vram_gets_a_bigger_micro_batch():
    assert hw.batch_shape_for_vram(80) == (8, 8)
    assert hw.batch_shape_for_vram(40) == (8, 8)
    assert hw.batch_shape_for_vram(24) == (4, 16)
    assert hw.batch_shape_for_vram(16) == (2, 32)
    assert hw.batch_shape_for_vram(8) == (1, 64)


def test_plan_for_cpu_is_refused():
    with pytest.raises(RuntimeError, match="GPU"):
        hw.plan_for_gpu(hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9), available=False, count=0)))


def test_throughput_estimate_scales_with_the_device():
    a100 = hw.estimate_tokens_per_sec(hw.TALOS_1B_PARAMS, peak_tflops=312.0, mfu=0.30)
    l4 = hw.estimate_tokens_per_sec(hw.TALOS_1B_PARAMS, peak_tflops=121.0, mfu=0.30)
    t4 = hw.estimate_tokens_per_sec(hw.TALOS_1B_PARAMS, peak_tflops=65.0, mfu=0.22)
    assert a100 > l4 > t4 > 0
    # 6ND + recomputation = 8ND: 312e12*0.30 / (8 * 1.036e9) ~= 11.3k tokens/sec
    assert 10_000 < a100 < 13_000
    with pytest.raises(ValueError):
        hw.estimate_tokens_per_sec(hw.TALOS_1B_PARAMS, peak_tflops=0, mfu=0.3)
    with pytest.raises(ValueError):
        hw.estimate_tokens_per_sec(hw.TALOS_1B_PARAMS, peak_tflops=100, mfu=0)


def test_plan_converts_hours_to_tokens():
    plan = hw.plan_for_gpu(hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9))), seq_len=SEQ)
    tokens = plan.tokens_in_hours(10.0)
    assert tokens == int(10 * 3600 * plan.est_tokens_per_sec)
    assert 100_000_000 < tokens < 300_000_000        # ~157M for an L4 at 10 h
    with pytest.raises(ValueError):
        plan.tokens_in_hours(0)


def test_unknown_gpu_falls_back_to_a_conservative_profile():
    assert hw.profile_for("Some Future GPU").match == hw.DEFAULT_PROFILE.match
    plan = hw.plan_for_gpu(hw.detect_gpu(_torch("Some Future GPU", 32.0, (9, 0))), seq_len=SEQ)
    assert plan.est_tokens_per_sec > 0


# ---------------------------------------------------------------------------
# Guidance text
# ---------------------------------------------------------------------------
def test_oom_guidance_names_the_levers_in_order():
    plan = hw.plan_for_gpu(hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9))), seq_len=SEQ)
    text = hw.oom_guidance(plan)
    assert "MICRO_BATCH = 2" in text and "GRAD_ACCUM = 32" in text
    assert "8BIT" in text and "checkpoint" in text.lower()
    # 8-bit is required on this tier, so the OOM list must not tell the reader
    # to "set USE_8BIT_ADAM = True" as if it were an optional lever.
    assert "USE_8BIT_ADAM stays True" in text


def test_the_l4_plan_forces_8bit_adam():
    """8-bit is required on the L4 tier, not left to the reader (fix 1)."""
    plan = hw.plan_for_gpu(hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9))), seq_len=SEQ)
    assert plan.use_8bit_adam is True
    assert plan.as_dict()["use_8bit_adam"] is True
    assert any("required" in note and "bitsandbytes" in note for note in plan.notes)
    # The report the notebook prints states it too.
    report = hw.format_hardware_report(
        hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9))), plan)
    assert "8-bit AdamW    : True" in report


def test_install_guidance_forbids_reinstalling_torch():
    text = hw.install_guidance()
    assert "NOT installed" in text
    assert "CPU-only" in text


def test_hardware_report_lists_the_plan():
    info = hw.detect_gpu(_torch("NVIDIA L4", 24.0, (8, 9)))
    plan = hw.plan_for_gpu(info, seq_len=SEQ)
    report = hw.format_hardware_report(info, plan)
    assert "L4" in report
    assert f"{plan.tokens_per_step:,} tokens/step" in report
    assert "estimate" in report
