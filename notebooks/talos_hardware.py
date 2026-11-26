"""Colab hardware detection and the memory/precision plan — pure logic.

The notebook targets **Colab GPUs, auto-detected**, not any particular local
card. This module answers three questions without needing a GPU to answer them
in tests (``tests/test_notebook_hardware.py`` injects fake torch modules):

1. *What am I running on?* — :func:`detect_gpu` -> :class:`GpuInfo` (name, VRAM,
   compute capability, bf16 support, torch/CUDA versions, Colab or not).
2. *How should the run be configured?* — :func:`plan_for_gpu` ->
   :class:`TrainPlan` (micro-batch, gradient accumulation, precision, 8-bit
   AdamW, gradient checkpointing, tokens per step, estimated tokens/sec).
3. *Is this a CUDA PyTorch at all?* — :func:`check_cuda_torch` raises with an
   actionable message instead of silently training on the CPU, and
   :func:`install_guidance` says what to do about it (never "pip install torch":
   in Colab that replaces the CUDA build with a CPU-only wheel, which is exactly
   the silent-CPU-fallback failure the owner called out).

Throughput numbers are **estimates from FLOPs arithmetic**, not measurements:
``tokens/sec ≈ (peak_TFLOPS × 1e12 × MFU) / flops_per_token`` with
``flops_per_token = 8 × parameters`` under gradient checkpointing (6 for
fwd+bwd, +2 for the recomputed forward). The MFU assumptions are documented in
:data:`GPU_PROFILES` and the docs table states the ±40 % spread explicitly —
this team has no GPU, so nothing here is a measured claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

#: Talos 1B's pinned parameter count (``configs.presets``); used for the FLOPs
#: estimate so the plan does not have to import torch to be tested.
TALOS_1B_PARAMS = 1_036_060_672

#: Colab's Drive mount point.
DRIVE_MOUNT = "/content/drive"


# ---------------------------------------------------------------------------
# GPU profiles (peak dense tensor-core TFLOPS, bf16 unless noted)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuProfile:
    """One known accelerator's peak throughput and the MFU we assume for it."""

    match: str                 # substring matched against the device name (upper-case)
    peak_tflops: float         # dense bf16/fp16 tensor-core peak, no sparsity
    mfu: float                 # assumed model-FLOPs utilisation for a 1B run
    bf16: bool                 # does this part support bf16?
    vram_gib: float            # nominal VRAM
    note: str = ""


#: Known Colab / common GPUs. ``mfu`` is deliberately conservative (0.22-0.30):
#: a 1B model with gradient checkpointing and a 131k-token step is far from the
#: efficiency of a 70B run on the same silicon.
GPU_PROFILES: Tuple[GpuProfile, ...] = (
    GpuProfile("H100", 990.0, 0.30, True, 80.0, "Colab Pro+ / pay-as-you-go"),
    GpuProfile("A100", 312.0, 0.30, True, 40.0, "Colab Pro / Pro+ (40 GB SXM)"),
    GpuProfile("L40S", 362.0, 0.30, True, 48.0, ""),
    GpuProfile("L4", 121.0, 0.30, True, 24.0, "Colab free / Pro (24 GB, Ada)"),
    GpuProfile("A10G", 125.0, 0.30, True, 24.0, ""),
    GpuProfile("V100", 125.0, 0.22, False, 16.0, "fp16 only — no bf16 on sm70"),
    GpuProfile("T4", 65.0, 0.22, False, 16.0, "fp16 only — no bf16 on sm75"),
    GpuProfile("P100", 19.0, 0.20, False, 16.0, "fp16 only, no tensor cores"),
)

#: Fallback profile for an unrecognised GPU: assume L4-class throughput.
DEFAULT_PROFILE = GpuProfile("UNKNOWN", 60.0, 0.25, True, 16.0,
                             "unrecognised GPU — using a conservative L4-class estimate")


def profile_for(name: str) -> GpuProfile:
    """Best-effort profile lookup by device-name substring (never raises)."""
    upper = str(name or "").upper()
    for profile in GPU_PROFILES:
        if profile.match in upper:
            return profile
    return DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuInfo:
    """What the notebook is actually running on."""

    name: str
    vram_gib: float
    capability: Optional[Tuple[int, int]]
    bf16_supported: bool
    cuda_available: bool
    device_count: int
    torch_version: str
    cuda_version: Optional[str]
    is_colab: bool

    @property
    def is_gpu(self) -> bool:
        return self.cuda_available and self.device_count > 0

    @property
    def label(self) -> str:
        """One-line GPU label for the tracker/metrics rows."""
        if not self.is_gpu:
            return "CPU"
        cap = "" if self.capability is None else f" sm{self.capability[0]}{self.capability[1]}"
        return f"{self.name} ({self.vram_gib:.0f} GiB{cap})"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vram_gib": self.vram_gib,
            "capability": None if self.capability is None else list(self.capability),
            "bf16_supported": self.bf16_supported,
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "is_colab": self.is_colab,
        }


def is_colab(environ: Optional[Dict[str, str]] = None) -> bool:
    """True when running inside Google Colab (no ``google.colab`` import needed)."""
    import os

    env = os.environ if environ is None else environ
    return bool(env.get("COLAB_GPU") or env.get("COLAB_RELEASE_TAG") or env.get("COLAB_TPU_ADDR"))


def detect_gpu(torch_module: Any = None, *, environ: Optional[Dict[str, str]] = None) -> GpuInfo:
    """Inspect the runtime and return a :class:`GpuInfo` — never raises.

    ``torch_module`` is injectable so CI can exercise every branch (CUDA build
    with a GPU, CUDA build without one, a CPU-only wheel, a broken import) with
    no GPU present.
    """
    colab = is_colab(environ)
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:  # noqa: BLE001 — a missing torch is reported, not raised
            return GpuInfo(name="CPU", vram_gib=0.0, capability=None,
                           bf16_supported=False, cuda_available=False,
                           device_count=0, torch_version="not installed",
                           cuda_version=None, is_colab=colab)
    try:
        torch_version = str(getattr(torch_module, "__version__", "unknown"))
        cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        cuda_version = None if cuda_version is None else str(cuda_version)
    except Exception:  # noqa: BLE001 - a partially imported torch is reported, not raised
        torch_version, cuda_version = "unreadable", None
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_available = False
    if not cuda_available:
        return GpuInfo(name="CPU", vram_gib=0.0, capability=None,
                       bf16_supported=False, cuda_available=False, device_count=0,
                       torch_version=torch_version, cuda_version=cuda_version,
                       is_colab=colab)
    try:
        name = str(torch_module.cuda.get_device_name(0))
        props = torch_module.cuda.get_device_properties(0)
        vram = float(props.total_memory) / 2 ** 30
        raw_cap = torch_module.cuda.get_device_capability(0)
        capability: Optional[Tuple[int, int]] = (int(raw_cap[0]), int(raw_cap[1]))
        device_count = int(torch_module.cuda.device_count())
    except Exception:  # noqa: BLE001 — a partially working CUDA build
        return GpuInfo(name="CUDA (unreadable)", vram_gib=0.0, capability=None,
                       bf16_supported=False, cuda_available=True, device_count=1,
                       torch_version=torch_version, cuda_version=cuda_version,
                       is_colab=colab)
    # bf16 needs both the silicon (sm80+) and a torch build that knows the dtype.
    bf16 = bool(capability is not None and capability[0] >= 8
                and hasattr(torch_module, "bfloat16"))
    return GpuInfo(name=name, vram_gib=round(vram, 1), capability=capability,
                   bf16_supported=bf16, cuda_available=True,
                   device_count=device_count, torch_version=torch_version,
                   cuda_version=cuda_version, is_colab=colab)


def check_cuda_torch(torch_module: Any = None) -> Dict[str, Any]:
    """Assert that torch can actually see a GPU; raise with an actionable message.

    This is the guard against the failure mode the owner flagged: a working CUDA
    PyTorch replaced by a CPU-only wheel, after which a 1B run "works" at ~100×
    slower. The notebook calls this before building the model, so a CPU-only
    environment stops there with instructions instead of starting a run that
    cannot finish.
    """
    info = detect_gpu(torch_module)
    if not info.is_gpu:
        raise RuntimeError(
            "No CUDA GPU is visible to PyTorch "
            f"(torch {info.torch_version}, CUDA build: {info.cuda_version}).\n"
            "This notebook trains a 1,036,060,672-parameter model and cannot run "
            "on a CPU.\n"
            "In Colab: Runtime -> Change runtime type -> Hardware accelerator: "
            "GPU (T4/L4/A100), then re-run from the first cell.\n"
            "If the runtime *has* a GPU but torch still reports none, the torch "
            "install is CPU-only: use Runtime -> Disconnect and delete runtime "
            "(a factory reset restores Colab's CUDA torch). Do NOT run "
            "`pip install torch` — that is what replaces the CUDA build with a "
            "CPU wheel."
        )
    return info.as_dict()


# ---------------------------------------------------------------------------
# Throughput estimate
# ---------------------------------------------------------------------------
def flops_per_token(parameters: int, *, gradient_checkpointing: bool = True) -> float:
    """Training FLOPs per token: ``6N`` (fwd+bwd), ``8N`` with recomputation."""
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    return float(parameters) * (8.0 if gradient_checkpointing else 6.0)


def estimate_tokens_per_sec(
    parameters: int,
    *,
    peak_tflops: float,
    mfu: float,
    gradient_checkpointing: bool = True,
) -> float:
    """Estimated training throughput: ``peak × MFU / flops_per_token``."""
    if peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive")
    if not 0.0 < float(mfu) <= 1.0:
        raise ValueError("mfu must be in (0, 1]")
    per_token = flops_per_token(parameters, gradient_checkpointing=gradient_checkpointing)
    return float(peak_tflops) * 1e12 * float(mfu) / per_token


def estimate_wall_clock_s(token_budget: int, tokens_per_sec: float) -> Optional[float]:
    """Seconds to spend ``token_budget`` at ``tokens_per_sec`` (``None`` if unknown)."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if tokens_per_sec is None or tokens_per_sec <= 0:
        return None
    return float(token_budget) / float(tokens_per_sec)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainPlan:
    """How this GPU runs the 1B: batch shape, precision, memory tactics."""

    micro_batch: int
    grad_accum: int
    precision: str                 # "bf16" | "fp16"
    use_grad_scaler: bool          # fp16 needs one; bf16 does not
    gradient_checkpointing: bool
    use_8bit_adam: bool
    seq_len: int
    tokens_per_step: int
    est_tokens_per_sec: float
    est_effective_tflops: float
    vram_gib: float
    gpu: str
    notes: Tuple[str, ...] = ()

    def tokens_in_hours(self, hours: float) -> int:
        """Estimated tokens trainable in ``hours`` wall-clock hours."""
        if hours <= 0:
            raise ValueError("hours must be positive")
        return int(float(hours) * 3600.0 * float(self.est_tokens_per_sec))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "precision": self.precision,
            "use_grad_scaler": self.use_grad_scaler,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_8bit_adam": self.use_8bit_adam,
            "seq_len": self.seq_len,
            "tokens_per_step": self.tokens_per_step,
            "est_tokens_per_sec": round(self.est_tokens_per_sec, 1),
            "est_effective_tflops": round(self.est_effective_tflops, 1),
            "vram_gib": self.vram_gib,
            "gpu": self.gpu,
            "notes": list(self.notes),
        }


#: Effective rows per optimizer step (micro-batch × accumulation). 64 rows ×
#: 2048 tokens = 131,072 tokens/step on every GPU, so the schedule (steps,
#: warmup, LR curve) is identical across hardware and only wall-clock differs.
ROWS_PER_STEP = 64


def batch_shape_for_vram(vram_gib: float) -> Tuple[int, int]:
    """``(micro_batch, grad_accum)`` sized for ``vram_gib`` — 64 rows per step.

    Static memory for 1B bf16 + 8-bit AdamW is ~6.2 GB (params 2.07 + grads
    2.07 + moments 2.07); the rest is activations and the logits/cross-entropy
    working set, which scale with micro-batch. Sizing down micro-batch and up
    accumulation keeps the *tokens per step* constant.
    """
    if vram_gib >= 38:
        micro = 8
    elif vram_gib >= 20:
        micro = 4
    elif vram_gib >= 14:
        micro = 2
    else:
        micro = 1
    micro = max(1, min(micro, ROWS_PER_STEP))
    return micro, max(1, ROWS_PER_STEP // micro)


def plan_for_gpu(
    info: GpuInfo,
    *,
    seq_len: int = 2048,
    parameters: int = TALOS_1B_PARAMS,
) -> TrainPlan:
    """Build the :class:`TrainPlan` for the detected hardware.

    Raises ``RuntimeError`` on a CPU-only runtime: there is no honest plan for a
    1B model without an accelerator, and pretending otherwise is how a run ends
    up "training" at 40 tokens/sec for 30 days.
    """
    if not info.is_gpu:
        raise RuntimeError(
            "plan_for_gpu needs a CUDA GPU; this runtime has none. "
            "Select a GPU runtime in Colab (Runtime -> Change runtime type)."
        )
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    profile = profile_for(info.name)
    bf16 = bool(info.bf16_supported and profile.bf16)
    precision = "bf16" if bf16 else "fp16"
    micro, accum = batch_shape_for_vram(info.vram_gib)
    tokens_per_step = micro * accum * seq_len
    # fp16 halves the mantissa of the accumulation path; keep the same MFU
    # assumption (tensor cores run either dtype at the same peak on T4/V100).
    rate = estimate_tokens_per_sec(parameters, peak_tflops=profile.peak_tflops,
                                   mfu=profile.mfu, gradient_checkpointing=True)
    notes = [
        f"profile: {profile.match} ({profile.note or 'no note'})",
        f"precision: {precision}"
        + ("" if bf16 else " (this GPU has no bf16 — GradScaler enabled)"),
        f"~{profile.mfu:.0%} MFU assumed; throughput is an estimate (±40%), not a measurement",
    ]
    # 8-bit AdamW is REQUIRED on the small-VRAM tiers, not a suggestion: fp32
    # AdamW needs ~8.3 GB of state for the 1B model on top of 4.14 GB of weights
    # and 4.14 GB of gradients, which does not fit 16/24 GB cards. An L4 is
    # always one of those tiers (24 GB nominal, ~15 GB free after the display
    # driver), so the plan forces it there explicitly rather than leaving the
    # reader to set USE_8BIT_ADAM by hand; `build_optimizer` then raises instead
    # of silently falling back to plain AdamW.
    forced_8bit = profile.match == "L4" or info.vram_gib < 48.0
    use_8bit_adam = bool(forced_8bit)
    if use_8bit_adam:
        notes.append(
            "8-bit AdamW (bitsandbytes) is required on this tier — it is not "
            "optional and there is no plain-AdamW fallback: a bitsandbytes "
            "failure stops the run with instructions instead"
        )
    if info.vram_gib and info.vram_gib < 16:
        notes.append("under 16 GiB: expect to drop MICRO_BATCH by hand if the first step OOMs")
    return TrainPlan(
        micro_batch=micro,
        grad_accum=accum,
        precision=precision,
        use_grad_scaler=precision == "fp16",
        gradient_checkpointing=True,
        use_8bit_adam=use_8bit_adam,
        seq_len=int(seq_len),
        tokens_per_step=tokens_per_step,
        est_tokens_per_sec=rate,
        est_effective_tflops=profile.peak_tflops * profile.mfu,
        vram_gib=float(info.vram_gib),
        gpu=info.label,
        notes=tuple(notes),
    )


def oom_guidance(plan: Optional[TrainPlan] = None) -> str:
    """What to do about a CUDA OOM, in order — printed by the training cell."""
    micro = 2 if plan is None else max(1, plan.micro_batch // 2)
    accum = 32 if plan is None else max(1, plan.grad_accum * 2)
    return (
        "CUDA out of memory. In this order:\n"
        f"  1. set MICRO_BATCH = {micro} and GRAD_ACCUM = {accum} in the config "
        "cell (tokens/step stays the same, so the schedule does not change);\n"
        "  2. keep GRADIENT_CHECKPOINTING = True (it is on by default and is what "
        "makes a 1B model fit at all);\n"
        "  3. USE_8BIT_ADAM stays True (bitsandbytes): it is required on this "
        "tier, not a knob — it is already what keeps optimizer state at ~2.1 GB "
        "instead of ~8.3 GB, so do not disable it to save memory; if bitsandbytes "
        "itself fails, fix that (the error names the fix) rather than downgrading;\n"
        "  4. lower SEQ_LEN only as a last resort — the 1B preset is trained at "
        "2,048 and a shorter context changes the objective;\n"
        "  5. if it still OOMs, the GPU is too small: switch to a larger Colab "
        "accelerator. Resume from the last checkpoint — nothing is lost."
    )


def install_guidance() -> str:
    """The dependency policy, as text the notebook prints.

    Deliberately never installs torch: in Colab, ``pip install torch`` resolves
    to a CPU-only wheel on the default index and silently destroys GPU training.
    """
    return (
        "Dependency policy: torch is NOT installed by this notebook. Colab ships "
        "a CUDA torch build; `pip install torch` would replace it with a CPU-only "
        "wheel and the run would silently fall back to the CPU. Only the pure-"
        "Python extras (datasets, huggingface_hub, bitsandbytes) are installed."
    )


def format_hardware_report(info: GpuInfo, plan: TrainPlan) -> str:
    """Multi-line human summary of the detection + plan (notebook output)."""
    lines = [
        f"GPU            : {info.label}",
        f"torch          : {info.torch_version}   CUDA build: {info.cuda_version}",
        f"colab          : {info.is_colab}",
        f"precision      : {plan.precision}"
        + ("  (GradScaler on)" if plan.use_grad_scaler else "  (no GradScaler needed)"),
        f"micro/accum    : {plan.micro_batch} x {plan.grad_accum} x {plan.seq_len} "
        f"= {plan.tokens_per_step:,} tokens/step",
        f"est throughput : {plan.est_tokens_per_sec:,.0f} tokens/sec "
        f"(~{plan.est_effective_tflops:,.0f} effective TFLOPS)",
        f"8-bit AdamW    : {plan.use_8bit_adam}    grad checkpointing: "
        f"{plan.gradient_checkpointing}",
    ]
    lines.extend(f"note           : {note}" for note in plan.notes)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PROFILE",
    "DRIVE_MOUNT",
    "GPU_PROFILES",
    "GpuInfo",
    "GpuProfile",
    "ROWS_PER_STEP",
    "TALOS_1B_PARAMS",
    "TrainPlan",
    "batch_shape_for_vram",
    "check_cuda_torch",
    "detect_gpu",
    "estimate_tokens_per_sec",
    "estimate_wall_clock_s",
    "flops_per_token",
    "format_hardware_report",
    "install_guidance",
    "is_colab",
    "oom_guidance",
    "plan_for_gpu",
    "profile_for",
]
