#!/usr/bin/env python3
"""W4 — Talos 1B on-device benchmark harness (owner's RTX 3060 12 GB).

One command, run by the OWNER on their own GPU. The team has no GPU access, so
this module is written to be *fully structured and CPU-verifiable*: every
GPU-specific measurement is isolated behind a probe and reports
``{"skipped": true, "reason": ...}`` when CUDA is absent, instead of crashing.
``--device cpu`` therefore exercises the complete wiring (arg parsing, guards,
the OOM ladder, the activation matrix, the save->kill->resume harness, the
projections and the JSON contract) on tiny dimensions.

What it measures (GPU) / validates (CPU)
----------------------------------------
1. **Measured metrics** — VRAM peak (``torch.cuda.max_memory_allocated`` /
   ``reserved`` *and* ``nvidia-smi`` used-MiB), GPU utilisation %, power draw
   (W) and temperature, train tokens/sec and inference tokens/sec (prefill and
   KV-cache decode separately).
2. **OOM microbatch ladder** at ``--seq`` (default 2048): start at
   ``--microbatch-start`` (default 8) and halve down to 1, catching
   ``torch.cuda.OutOfMemoryError`` cleanly (never a crash), reporting every
   rung and the winning microbatch.
3. **Activation matrix** — bf16 / fp16 / none x SDPA / plain x
   ``--grad-checkpoint`` on/off x ``--grad-accum`` values. Every *attempted*
   combination is reported with its outcome (ok / OOM / skipped + why).
   CPU-offload is reported as *not implemented in the engine* rather than
   faked.
4. **save -> kill -> resume harness** — launches the real trainer
   (``scripts/train_oasst1.py``) as a subprocess, waits for a checkpoint, then
   ``SIGKILL``s it mid-run, resumes from the checkpoint, and verifies step /
   token accounting against an uninterrupted reference run plus loss
   continuity. Prints and records a PASS/FAIL verdict.
5. **Runtime projections** — from the MEASURED tokens/sec, wall-clock for
   200M / 600M / 1B / 10B tokens, each entry labelled measured-vs-estimated.
6. **Checkpoint storage + retention** — per-checkpoint size arithmetic and a
   retention policy that keeps the latest AND the known-good checkpoint and
   never auto-deletes a known-good one.
7. **Guards** — ``--max-steps`` defaults small (10); exceeding it needs
   ``--confirm-full-run``; an existing results JSON or work directory is
   refused without ``--force``; results files are timestamped.

Import-vs-subprocess decision
-----------------------------
* The **measured benchmarks** (ladder, matrix, tok/sec, VRAM) *import* the
  engine (``scripts.train_oasst1``) and drive it in-process. Rationale: the
  VRAM/utilisation/power windows must bracket exactly the code region being
  measured, per-microbatch OOM has to be caught without paying process
  start-up + model-build (which for 1B is minutes and ~4 GB of host RAM) on
  every rung, and the benchmark must reproduce the trainer's *exact* step
  semantics (autocast placement, ``scaler.scale`` -> ``unscale_`` -> ``step``,
  ``set_gradient_checkpointing``, ``loss_fn`` over ``x[:, :-1]`` / ``x[:, 1:]``).
  It reuses the trainer's own helpers (``build_preset_model``,
  ``build_optimizer``, ``make_amp_autocast``, ``make_grad_scaler``,
  ``save_checkpoint``, ``load_checkpoint``, ``validate_resume_checkpoint``)
  rather than reimplementing them.
* The **resume harness** *subprocesses* the real trainer entry point, because
  the whole point is to kill a real process mid-run (SIGKILL) and prove the
  on-disk checkpoint survives it. An in-process version cannot test that.

Data
----
Nothing here reads the owner's packed corpus. The measured benchmarks feed
**synthetic** token ids (a seeded generator) — throughput and VRAM do not
depend on token *content*. The resume harness writes its own small synthetic
JSONL corpus (generic English filler words generated deterministically) into
its work directory. The 1B presets use the committed, sha256-verified
``talos_1b_32k`` tokenizer asset from ``assets/tokenizers/MANIFEST.json``; no
data fetch is required.

Safety
------
No full-scale training is ever launched: ``--max-steps`` caps the measured
work at 10 optimizer steps by default, and the resume harness runs a handful
of steps at a short sequence length. Nothing in this module deletes a
checkpoint.

Usage (owner, on the RTX 3060)
------------------------------
    python -m scripts.benchmark_1b --preset talos_1b --device cuda

Usage (team, CPU wiring validation)
-----------------------------------
    python -m scripts.benchmark_1b --device cpu --dry-run
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from configs.canonical import CANONICAL_PRESETS
from configs.presets import ALL_PRESETS
from inference.generate import decode_step, prefill
from model.gpt import TalosGPT
from scripts.train_oasst1 import (
    AMP_CHOICES,
    ATTENTION_BACKEND_CHOICES,
    build_optimizer,
    build_preset_model,
    load_checkpoint,
    make_amp_autocast,
    make_grad_scaler,
    save_checkpoint,
    validate_resume_checkpoint,
)

#: Schema version of the results JSON. Bump on any breaking key change.
BENCHMARK_VERSION = "talos-1b-benchmark-v1"

#: Default cap on measured optimizer steps per benchmark. Anything above this
#: is a "long run" and requires ``--confirm-full-run``.
DEFAULT_MAX_STEPS = 10
DEFAULT_WARMUP_STEPS = 2
DEFAULT_MEASURE_STEPS = 5

#: Sequence length for the measured ladder / matrix (the 1B track's context).
DEFAULT_SEQ = 2048
DEFAULT_MICROBATCH_START = 8
LADDER_FLOOR = 1

#: Training-run projections the owner asked for.
TOKEN_BUDGETS: Tuple[int, ...] = (
    200_000_000,
    600_000_000,
    1_000_000_000,
    10_000_000_000,
)

#: Gradient-accumulation values exercised by the default activation matrix.
DEFAULT_MATRIX_ACCUM: Tuple[int, ...] = (1, 4)

#: The 1B presets that require the dedicated 32K tokenizer asset.
TOKENIZER_ASSET_PRESETS = ("talos_1b", "talos_1b_spec16")
TOKENIZER_ASSET_NAME = "talos_1b_32k"

#: Exit codes.
EXIT_OK = 0
EXIT_GUARD = 2
EXIT_RESUME_FAIL = 3


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """UTC timestamp, second resolution, filesystem-safe."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    """UTC timestamp for filenames (no characters a shell would quote)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error (both torch's class and the text).

    ``torch.cuda.OutOfMemoryError`` only exists in torch >= 2.0 and is only
    raised for real CUDA OOMs, but a few allocator paths still surface a plain
    ``RuntimeError("CUDA out of memory...")``; both must be handled cleanly so
    the ladder never crashes the benchmark.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _sync(device: torch.device) -> None:
    """Wait for queued CUDA work so timers bracket real execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def deep_get(obj: Any, key: str, default: Any = None) -> Any:
    """Breadth-first search for ``key`` in nested dicts (top level first).

    The trainer's ``metrics.json`` layout is nested and has grown across
    passes; a search keeps the harness working across schema additions
    without pinning an exact path.
    """
    queue: List[Any] = [obj]
    seen = 0
    while queue and seen < 10_000:
        cur = queue.pop(0)
        seen += 1
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(v for v in cur if isinstance(v, (dict, list)))
    return default


def git_state() -> Dict[str, Any]:
    """Best-effort repo commit + dirty flag (never raises)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=10, check=False,
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, timeout=10, check=False,
        ).stdout
        out["dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dep
        pass
    return out


def _system_ram_bytes() -> Optional[int]:
    """Total system RAM from /proc/meminfo (None when unavailable)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):  # pragma: no cover - non-Linux
        pass
    return None


# ---------------------------------------------------------------------------
# Environment probe
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentReport:
    """Everything the owner's box reports about itself, GPU fields optional."""

    benchmark_version: str
    utc_time: str
    host: str
    platform: str
    python: str
    torch_version: str
    device_requested: str
    device_resolved: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_total_mem_bytes: Optional[int]
    gpu_total_mem_mib: Optional[int]
    gpu_sm: Optional[str]
    gpu_bf16_supported: Optional[bool]
    nvidia_smi_available: bool
    nvidia_smi_version: Optional[str]
    driver_version: Optional[str]
    system_ram_bytes: Optional[int]
    cpu_count: int
    git: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _nvidia_smi(args: Sequence[str]) -> Optional[str]:
    """Run ``nvidia-smi <args>`` and return stdout, or None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dep
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def nvidia_smi_info() -> Dict[str, Optional[str]]:
    """Driver / CUDA / product name as reported by nvidia-smi."""
    out: Dict[str, Optional[str]] = {
        "available": None, "driver_version": None, "name": None, "memory_total": None,
    }
    text = _nvidia_smi([
        "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader",
    ])
    out["available"] = text is not None
    if text:
        first = text.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            out["driver_version"] = parts[0]
            out["name"] = parts[1]
            out["memory_total"] = parts[2]
    return out


def probe_environment(device_requested: str, device: torch.device) -> EnvironmentReport:
    """Collect the environment report; every GPU field degrades to None."""
    notes: List[str] = []
    cuda_available = torch.cuda.is_available()
    smi = nvidia_smi_info()

    gpu_name = gpu_total = gpu_sm = gpu_bf16 = None
    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_total = int(props.total_memory)
            gpu_sm = f"SM{props.major}{props.minor}"
            try:
                gpu_bf16 = bool(torch.cuda.is_bf16_supported())
            except (RuntimeError, AttributeError):  # pragma: no cover - old torch
                gpu_bf16 = None
            if gpu_bf16 is False:
                notes.append(
                    f"{gpu_sm} has no bf16 tensor-core path — use --amp fp16 "
                    "or none on this GPU (bf16 matrix rows will be skipped)."
                )
        except (RuntimeError, AssertionError):  # pragma: no cover - env dep
            notes.append("torch.cuda is available but device properties failed.")
    else:
        notes.append(
            "No CUDA device: GPU-specific measurements are skipped and the "
            "functional sections run on tiny dimensions (dry-run wiring check)."
        )
    if cuda_available and not smi["available"]:
        notes.append(
            "nvidia-smi not found: VRAM/utilisation/power come from torch only."
        )

    return EnvironmentReport(
        benchmark_version=BENCHMARK_VERSION,
        utc_time=utc_now(),
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch_version=torch.__version__,
        device_requested=device_requested,
        device_resolved=str(device),
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        gpu_name=gpu_name,
        gpu_total_mem_bytes=gpu_total,
        gpu_total_mem_mib=None if gpu_total is None else int(gpu_total // (1024 * 1024)),
        gpu_sm=gpu_sm,
        gpu_bf16_supported=gpu_bf16,
        nvidia_smi_available=bool(smi["available"]),
        nvidia_smi_version=smi["driver_version"],
        driver_version=smi["driver_version"],
        system_ram_bytes=_system_ram_bytes(),
        cpu_count=os.cpu_count() or 1,
        git=git_state(),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# nvidia-smi sampler (utilisation / power / temperature / used VRAM)
# ---------------------------------------------------------------------------
_SMI_FIELDS = "utilization.gpu,memory.used,power.draw,temperature.gpu"


def _parse_smi_value(text: str) -> Optional[float]:
    """Parse one nvidia-smi CSV cell; ``[N/A]`` / ``[Not Supported]`` -> None."""
    text = text.strip()
    if not text or text.startswith("["):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sample_nvidia_smi() -> Optional[Dict[str, Optional[float]]]:
    """One nvidia-smi reading, or None when the tool is unavailable."""
    text = _nvidia_smi([f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"])
    if not text:
        return None
    line = text.strip().splitlines()[0]
    cells = line.split(",")
    if len(cells) < 4:
        return None
    util, mem, power, temp = (_parse_smi_value(c) for c in cells[:4])
    return {
        "gpu_util_pct": util,
        "memory_used_mib": mem,
        "power_w": power,
        "temperature_c": temp,
    }


@dataclass
class GpuSamples:
    """Aggregate of the nvidia-smi readings taken during one measurement window."""

    ok: bool
    samples: int
    gpu_util_pct_peak: Optional[float] = None
    gpu_util_pct_mean: Optional[float] = None
    memory_used_mib_peak: Optional[float] = None
    power_w_peak: Optional[float] = None
    power_w_mean: Optional[float] = None
    temperature_c_peak: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NvidiaSmiSampler:
    """Background nvidia-smi poller used as a context manager.

    ``nvidia-smi`` costs ~30-80 ms per call, so the default interval is 0.2 s:
    fine for a multi-second measurement window, cheap enough not to perturb a
    short one. Every failure mode (no binary, driver hiccup, ``[N/A]``) is
    recorded, never raised.
    """

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = max(0.05, float(interval_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rows: List[Dict[str, Optional[float]]] = []
        self._reason: Optional[str] = None

    def __enter__(self) -> "NvidiaSmiSampler":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def start(self) -> None:
        probe = sample_nvidia_smi()
        if probe is None:
            self._reason = (
                "nvidia-smi unavailable or returned no parsable row "
                "(GPU utilisation / power / used-VRAM unavailable)"
            )
            return
        self._rows.append(probe)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            row = sample_nvidia_smi()
            if row is not None:
                self._rows.append(row)

    def stop(self) -> GpuSamples:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return self.aggregate()

    def aggregate(self) -> GpuSamples:
        """Peak/mean over the collected rows (None when nothing was collected)."""
        rows = self._rows
        if not rows:
            return GpuSamples(ok=False, samples=0, reason=self._reason)

        def _peak(key: str) -> Optional[float]:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return max(vals) if vals else None

        def _mean(key: str) -> Optional[float]:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        return GpuSamples(
            ok=True,
            samples=len(rows),
            gpu_util_pct_peak=_peak("gpu_util_pct"),
            gpu_util_pct_mean=_mean("gpu_util_pct"),
            memory_used_mib_peak=_peak("memory_used_mib"),
            power_w_peak=_peak("power_w"),
            power_w_mean=_mean("power_w"),
            temperature_c_peak=_peak("temperature_c"),
            reason=None,
        )


# ---------------------------------------------------------------------------
# Torch VRAM accounting
# ---------------------------------------------------------------------------
@dataclass
class VramSnapshot:
    """Torch allocator peak for one measurement window (MiB)."""

    peak_allocated_mib: int
    peak_reserved_mib: int
    total_mib: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def reset_vram_peak(device: torch.device) -> None:
    """Reset the CUDA allocator peak counters (no-op off CUDA)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def vram_snapshot(device: torch.device) -> Optional[VramSnapshot]:
    """Peak allocated/reserved since the last reset, or None off CUDA."""
    if device.type != "cuda":
        return None
    mib = 1024 * 1024
    total = None
    try:
        total = int(torch.cuda.get_device_properties(0).total_memory // mib)
    except (RuntimeError, AssertionError):  # pragma: no cover - env dep
        total = None
    return VramSnapshot(
        peak_allocated_mib=int(torch.cuda.max_memory_allocated(device) // mib),
        peak_reserved_mib=int(torch.cuda.max_memory_reserved(device) // mib),
        total_mib=total,
    )


# ---------------------------------------------------------------------------
# Synthetic batches (no corpus is ever read)
# ---------------------------------------------------------------------------
def synthetic_batch(
    vocab_size: int,
    batch: int,
    seq: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Deterministic ``(batch, seq)`` token ids in ``[0, vocab_size)``.

    Throughput and peak VRAM do not depend on token content, so the measured
    benchmarks never touch a corpus. The generator is CPU-side and seeded, so
    two runs of the same benchmark see identical data.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    ids = torch.randint(
        0, int(vocab_size), (int(batch), int(seq)), generator=gen, dtype=torch.long
    )
    return ids.to(device)


# ---------------------------------------------------------------------------
# Train-step benchmark
# ---------------------------------------------------------------------------
@dataclass
class TrainBenchConfig:
    """One point in the activation matrix."""

    amp: str = "none"
    attention_backend: str = "sdpa"
    grad_checkpoint: bool = False
    grad_accum: int = 1
    microbatch: int = 1
    seq: int = DEFAULT_SEQ

    def label(self) -> str:
        """Compact stable identifier used in reports and test assertions."""
        return (
            f"amp={self.amp}/backend={self.attention_backend}"
            f"/ckpt={'on' if self.grad_checkpoint else 'off'}"
            f"/accum={self.grad_accum}/mb={self.microbatch}/seq={self.seq}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainBenchResult:
    """Outcome of one train-step benchmark attempt."""

    label: str
    config: Dict[str, Any]
    ok: bool
    oom: bool = False
    skipped: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None
    tokens: int = 0
    optimizer_steps: int = 0
    wall_s: float = 0.0
    tokens_per_sec: Optional[float] = None
    ms_per_optimizer_step: Optional[float] = None
    peak_allocated_mib: Optional[int] = None
    peak_reserved_mib: Optional[int] = None
    gpu: Optional[Dict[str, Any]] = None
    final_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _skipped_result(cfg: TrainBenchConfig, reason: str) -> TrainBenchResult:
    return TrainBenchResult(
        label=cfg.label(), config=cfg.to_dict(), ok=False, skipped=True, reason=reason
    )


def _failed_result(
    cfg: TrainBenchConfig, exc: BaseException
) -> TrainBenchResult:
    oom = _is_oom(exc)
    return TrainBenchResult(
        label=cfg.label(),
        config=cfg.to_dict(),
        ok=False,
        oom=oom,
        reason="CUDA out of memory" if oom else f"{type(exc).__name__}: {exc}",
        error=str(exc),
    )


def run_train_bench(
    model: TalosGPT,
    cfg: TrainBenchConfig,
    device: torch.device,
    *,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measure_steps: int = DEFAULT_MEASURE_STEPS,
    seed: int = 0,
    with_optimizer: bool = True,
    lr: float = 1e-4,
    sample_gpu: bool = True,
    smi_interval_s: float = 0.2,
) -> TrainBenchResult:
    """Measure train throughput + peak VRAM for one activation-matrix point.

    Mirrors the trainer's step semantics exactly (see ``train_epochs``):
    autocast wraps forward+loss, ``logits, _ = model(x[:, :-1])`` with
    ``CrossEntropyLoss`` against ``x[:, 1:]``, fp16 goes through
    ``scaler.scale(loss).backward()`` -> ``unscale_`` -> ``step()``, and
    ``--grad-checkpoint`` is applied with ``model.set_gradient_checkpointing``.

    A CUDA OOM is caught and returned as ``ok=False, oom=True`` — the ladder
    depends on that, so this function never propagates an OOM.
    """
    vocab = model.config.vocab_size
    loss_fn = torch.nn.CrossEntropyLoss()
    opt: Optional[torch.optim.Optimizer] = None
    amp_autocast = make_amp_autocast(cfg.amp, device)
    scaler = make_grad_scaler(cfg.amp, device)
    amp_fp16 = scaler.is_enabled()
    total_steps = max(0, int(warmup_steps)) + max(0, int(measure_steps))

    try:
        model.train()
        model.set_gradient_checkpointing(bool(cfg.grad_checkpoint))
        if with_optimizer:
            opt = build_optimizer(model, lr, fused=False, device=device)

        _sync(device)
        _empty_cache(device)
        reset_vram_peak(device)
        sampler = NvidiaSmiSampler(smi_interval_s) if sample_gpu else None
        if sampler is not None:
            sampler.start()

        measure_s = 0.0
        measure_steps_done = 0
        last_loss: Optional[float] = None

        for step_idx in range(total_steps):
            measuring = step_idx >= int(warmup_steps)
            if measuring:
                _sync(device)
                t0 = time.perf_counter()
            for _micro in range(max(1, int(cfg.grad_accum))):
                x = synthetic_batch(
                    vocab, cfg.microbatch, cfg.seq, device,
                    seed=seed * 1_000_003 + step_idx * 7919 + _micro,
                )
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                with amp_autocast:
                    logits, _ = model(x[:, :-1])
                    loss = loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
                if not torch.isfinite(loss):
                    # A non-finite loss on synthetic data is a numerical warning,
                    # not an OOM: keep going so the throughput number stays valid.
                    last_loss = float(loss.detach())
                    continue
                last_loss = float(loss.detach())
                scaled = loss / max(1, int(cfg.grad_accum))
                if opt is None:
                    if scaled.requires_grad:
                        scaled.backward()
                elif amp_fp16:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                del logits, loss, scaled
            if opt is not None:
                if amp_fp16:
                    scaler.unscale_(opt)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if measuring:
                _sync(device)
                measure_s += time.perf_counter() - t0
                measure_steps_done += 1

        gpu = sampler.stop() if sampler is not None else None
        snap = vram_snapshot(device)
        tokens = cfg.microbatch * cfg.seq * max(1, cfg.grad_accum) * measure_steps_done
        return TrainBenchResult(
            label=cfg.label(),
            config=cfg.to_dict(),
            ok=measure_steps_done > 0,
            tokens=tokens,
            optimizer_steps=measure_steps_done,
            wall_s=round(measure_s, 4),
            tokens_per_sec=(
                round(tokens / measure_s, 2) if measure_s > 0 and tokens else None
            ),
            ms_per_optimizer_step=(
                round(1000.0 * measure_s / measure_steps_done, 2)
                if measure_steps_done else None
            ),
            peak_allocated_mib=None if snap is None else snap.peak_allocated_mib,
            peak_reserved_mib=None if snap is None else snap.peak_reserved_mib,
            gpu=None if gpu is None else gpu.to_dict(),
            final_loss=None if last_loss is None else round(last_loss, 6),
        )
    except BaseException as exc:  # noqa: BLE001 - OOM must never escape
        if not isinstance(exc, (Exception, KeyboardInterrupt)):
            raise
        if isinstance(exc, KeyboardInterrupt):
            raise
        return _failed_result(cfg, exc)
    finally:
        try:
            model.set_gradient_checkpointing(False)
        except (AttributeError, RuntimeError):  # pragma: no cover - defensive
            pass
        del opt
        _empty_cache(device)


# ---------------------------------------------------------------------------
# OOM microbatch ladder
# ---------------------------------------------------------------------------
@dataclass
class LadderRung:
    """One microbatch size tried by the ladder."""

    microbatch: int
    status: str  # "ok" | "oom" | "error" | "skipped"
    tokens_per_sec: Optional[float] = None
    peak_allocated_mib: Optional[int] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LadderResult:
    """The full ladder plus the winning microbatch."""

    start_microbatch: int
    seq: int
    rungs: List[LadderRung] = field(default_factory=list)
    winner_microbatch: Optional[int] = None
    winner_tokens_per_sec: Optional[float] = None
    winner_peak_allocated_mib: Optional[int] = None
    max_tokens_per_optimizer_step: Optional[int] = None
    diagnosis: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_microbatch": self.start_microbatch,
            "seq": self.seq,
            "rungs": [r.to_dict() for r in self.rungs],
            "winner_microbatch": self.winner_microbatch,
            "winner_tokens_per_sec": self.winner_tokens_per_sec,
            "winner_peak_allocated_mib": self.winner_peak_allocated_mib,
            "max_tokens_per_optimizer_step": self.max_tokens_per_optimizer_step,
            "diagnosis": self.diagnosis,
            "note": self.note,
        }


def ladder_microbatches(
    start: int, floor: int = LADDER_FLOOR
) -> List[int]:
    """``start``, ``start // 2``, ... down to ``floor`` (deduplicated, sorted desc).

    Halving from 8 gives ``[8, 4, 2, 1]``. A non-power-of-two start (e.g. 12)
    gives ``[12, 6, 3, 1]`` — the floor is always included so the ladder can
    always terminate on the smallest possible microbatch.
    """
    start = max(int(start), int(floor))
    floor = max(1, int(floor))
    sizes: List[int] = []
    cur = start
    while cur > floor:
        sizes.append(cur)
        cur = max(floor, cur // 2)
        if cur == sizes[-1]:  # pragma: no cover - defensive
            break
    sizes.append(floor)
    seen: List[int] = []
    for s in sizes:
        if s not in seen:
            seen.append(s)
    return seen


def run_oom_ladder(
    attempt: Callable[[int], TrainBenchResult],
    *,
    start: int,
    seq: int,
    floor: int = LADDER_FLOOR,
    stop_at_first_ok: bool = True,
) -> LadderResult:
    """Halve the microbatch on OOM until one fits; report every rung.

    ``attempt`` maps a microbatch size to a :class:`TrainBenchResult` and is
    responsible for having already caught its own OOM (``run_train_bench``
    does). Keeping the loop here, parameterised by ``attempt``, is what makes
    the ladder logic testable on CPU with a fake that OOMs above a threshold.
    """
    result = LadderResult(start_microbatch=int(start), seq=int(seq))
    for mb in ladder_microbatches(start, floor):
        res = attempt(mb)
        if res.skipped:
            result.rungs.append(
                LadderRung(microbatch=mb, status="skipped", reason=res.reason)
            )
            continue
        if res.oom:
            result.rungs.append(
                LadderRung(microbatch=mb, status="oom", reason=res.reason)
            )
            continue
        if not res.ok:
            result.rungs.append(
                LadderRung(
                    microbatch=mb, status="error",
                    reason=res.reason or "unknown error",
                )
            )
            continue
        result.rungs.append(
            LadderRung(
                microbatch=mb,
                status="ok",
                tokens_per_sec=res.tokens_per_sec,
                peak_allocated_mib=res.peak_allocated_mib,
            )
        )
        if result.winner_microbatch is None:
            result.winner_microbatch = mb
            result.winner_tokens_per_sec = res.tokens_per_sec
            result.winner_peak_allocated_mib = res.peak_allocated_mib
            result.max_tokens_per_optimizer_step = mb * int(seq)
        if stop_at_first_ok:
            break
    return result


# ---------------------------------------------------------------------------
# Activation matrix
# ---------------------------------------------------------------------------
@dataclass
class MatrixEntry:
    """One attempted activation-matrix combination and its outcome."""

    label: str
    config: Dict[str, Any]
    status: str  # "ok" | "oom" | "skipped" | "error"
    tokens_per_sec: Optional[float] = None
    ms_per_optimizer_step: Optional[float] = None
    peak_allocated_mib: Optional[int] = None
    peak_reserved_mib: Optional[int] = None
    gpu_util_pct_mean: Optional[float] = None
    power_w_mean: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_matrix_plan(
    *,
    microbatches: Sequence[int],
    seq: int,
    amps: Sequence[str] = ("none", "bf16", "fp16"),
    backends: Sequence[str] = ("sdpa", "plain"),
    grad_checkpoints: Sequence[bool] = (False, True),
    accums: Sequence[int] = DEFAULT_MATRIX_ACCUM,
    full: bool = False,
    max_combos: int = 12,
) -> List[TrainBenchConfig]:
    """The activation combinations to attempt, in a deliberate order.

    ``full=False`` (default) is a **curated** plan that touches every axis at
    least once without exploding into a 24-point cross product that would take
    tens of minutes on a 1B model:

    1. baseline (amp=none, sdpa, no checkpointing)
    2. bf16 (the 3060's fast path)
    3. fp16 (the T4 path, kept for comparison)
    4. bf16 + grad-checkpoint (the VRAM lever)
    5. plain attention (SDPA off)
    6. bf16 + plain attention
    7. bf16 + grad-accum
    8. bf16 + grad-checkpoint + grad-accum

    ``full=True`` returns the complete cross product (capped by
    ``max_combos``). Microbatch defaults to the ladder winner (largest first);
    ``grad-accum > 1`` divides the microbatch so the effective batch — and thus
    the activation footprint — stays comparable across rows.
    """
    base_mb = int(microbatches[0]) if microbatches else 1

    def _mk(amp: str, backend: str, ckpt: bool, accum: int) -> TrainBenchConfig:
        return TrainBenchConfig(
            amp=amp,
            attention_backend=backend,
            grad_checkpoint=ckpt,
            grad_accum=int(accum),
            microbatch=max(1, base_mb // max(1, int(accum))),
            seq=int(seq),
        )

    if full:
        plan: List[TrainBenchConfig] = []
        for amp in amps:
            for backend in backends:
                for ckpt in grad_checkpoints:
                    for accum in accums:
                        plan.append(_mk(amp, backend, ckpt, accum))
        return plan[: max(0, int(max_combos))]

    curated: List[TrainBenchConfig] = [
        _mk("none", "sdpa", False, 1),
        _mk("bf16", "sdpa", False, 1),
        _mk("fp16", "sdpa", False, 1),
        _mk("bf16", "sdpa", True, 1),
        _mk("none", "plain", False, 1),
        _mk("bf16", "plain", False, 1),
        _mk("bf16", "sdpa", False, max(accums) if accums else 2),
        _mk("bf16", "sdpa", True, max(accums) if accums else 2),
    ]
    # Keep only axes the caller actually enabled, preserving order.
    kept: List[TrainBenchConfig] = []
    for cfg in curated:
        if cfg.amp not in amps or cfg.attention_backend not in backends:
            continue
        if cfg.grad_checkpoint and not any(grad_checkpoints):
            continue
        if cfg.grad_accum > 1 and not any(a > 1 for a in accums):
            continue
        kept.append(cfg)
    return kept[: max(0, int(max_combos))]


def cpu_offload_note() -> Dict[str, Any]:
    """Honest statement about CPU offload (the engine has no offload path)."""
    return {
        "supported": False,
        "attempted": False,
        "note": (
            "The Talos training engine has no optimizer/activation CPU-offload "
            "path (no ZeRO-Offload, no DeepSpeed). Nothing is reported as "
            "measured for offload because nothing was run. The VRAM levers "
            "that DO exist are --grad-checkpoint (activations recomputed "
            "instead of stored), --grad-accum (smaller microbatch per step) "
            "and --amp bf16 (halves activation bytes)."
        ),
    }


def run_activation_matrix(
    model: TalosGPT,
    plan: Sequence[TrainBenchConfig],
    device: torch.device,
    *,
    warmup_steps: int,
    measure_steps: int,
    seed: int = 0,
) -> List[MatrixEntry]:
    """Run every planned combination, recording ok / OOM / skipped per row."""
    entries: List[MatrixEntry] = []
    bf16_ok = True
    if device.type == "cuda":
        try:
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        except (RuntimeError, AttributeError):  # pragma: no cover - old torch
            bf16_ok = True
    for cfg in plan:
        if cfg.amp == "bf16" and not bf16_ok:
            entries.append(
                MatrixEntry(
                    label=cfg.label(), config=cfg.to_dict(), status="skipped",
                    reason="bf16 requires Ampere+ (SM80+); this GPU has no bf16 path",
                )
            )
            continue
        res = run_train_bench(
            model, cfg, device,
            warmup_steps=warmup_steps, measure_steps=measure_steps, seed=seed,
        )
        gpu = res.gpu or {}
        if res.ok:
            status = "ok"
        elif res.oom:
            status = "oom"
        elif res.skipped:
            status = "skipped"
        else:
            status = "error"
        entries.append(
            MatrixEntry(
                label=cfg.label(),
                config=cfg.to_dict(),
                status=status,
                tokens_per_sec=res.tokens_per_sec,
                ms_per_optimizer_step=res.ms_per_optimizer_step,
                peak_allocated_mib=res.peak_allocated_mib,
                peak_reserved_mib=res.peak_reserved_mib,
                gpu_util_pct_mean=gpu.get("gpu_util_pct_mean"),
                power_w_mean=gpu.get("power_w_mean"),
                reason=res.reason,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Inference benchmark (prefill + KV-cache decode)
# ---------------------------------------------------------------------------
@dataclass
class InferenceBenchResult:
    """Inference throughput for one configuration."""

    ok: bool
    amp: str
    batch: int
    prompt_len: int
    decode_steps: int
    prefill_tokens: int = 0
    prefill_s: float = 0.0
    prefill_tokens_per_sec: Optional[float] = None
    decode_tokens: int = 0
    decode_s: float = 0.0
    decode_tokens_per_sec: Optional[float] = None
    peak_allocated_mib: Optional[int] = None
    peak_reserved_mib: Optional[int] = None
    gpu: Optional[Dict[str, Any]] = None
    oom: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_inference_bench(
    model: TalosGPT,
    device: torch.device,
    *,
    amp: str = "none",
    batch: int = 1,
    prompt_len: int = 256,
    decode_steps: int = 32,
    seed: int = 0,
    smi_interval_s: float = 0.2,
) -> InferenceBenchResult:
    """Measure prefill tokens/sec and KV-cache decode tokens/sec.

    Prefill = one ``inference.generate.prefill`` over ``batch x prompt_len``.
    Decode = ``decode_steps`` single-token ``decode_step`` calls against the
    growing cache, each timed; tokens/sec here is *generated* tokens/sec (the
    number the owner actually feels), and the per-token latency is what a
    12 GB card is judged on.
    """
    vocab = model.config.vocab_size
    result = InferenceBenchResult(
        ok=False, amp=amp, batch=batch, prompt_len=prompt_len,
        decode_steps=decode_steps,
    )
    sampler = NvidiaSmiSampler(smi_interval_s)
    try:
        model.eval()
        model.set_gradient_checkpointing(False)
        _sync(device)
        _empty_cache(device)
        reset_vram_peak(device)
        sampler.start()
        with torch.inference_mode():
            prompt = synthetic_batch(vocab, batch, prompt_len, device, seed=seed)
            _sync(device)
            t0 = time.perf_counter()
            logits, cache = prefill(model, prompt)
            _sync(device)
            prefill_s = time.perf_counter() - t0

            tokens = (
                logits[:, -1:, :].argmax(dim=-1)
                if logits.dim() == 3
                else logits.argmax(dim=-1, keepdim=True)
            )
            position = prompt_len
            decode_s = 0.0
            done = 0
            for _ in range(int(decode_steps)):
                _sync(device)
                t1 = time.perf_counter()
                step_logits = decode_step(model, cache, tokens, position)
                _sync(device)
                decode_s += time.perf_counter() - t1
                tokens = (
                    step_logits[:, -1:, :].argmax(dim=-1)
                    if step_logits.dim() == 3
                    else step_logits.argmax(dim=-1, keepdim=True)
                )
                position += 1
                done += 1
            del logits, cache
        gpu = sampler.stop()
        snap = vram_snapshot(device)
        prefill_tokens = batch * prompt_len
        decode_tokens = batch * done
        result.ok = True
        result.prefill_tokens = prefill_tokens
        result.prefill_s = round(prefill_s, 5)
        result.prefill_tokens_per_sec = (
            round(prefill_tokens / prefill_s, 2) if prefill_s > 0 else None
        )
        result.decode_tokens = decode_tokens
        result.decode_s = round(decode_s, 5)
        result.decode_tokens_per_sec = (
            round(decode_tokens / decode_s, 2) if decode_s > 0 else None
        )
        result.peak_allocated_mib = None if snap is None else snap.peak_allocated_mib
        result.peak_reserved_mib = None if snap is None else snap.peak_reserved_mib
        result.gpu = gpu.to_dict()
    except BaseException as exc:  # noqa: BLE001 - OOM must never escape
        if isinstance(exc, KeyboardInterrupt):
            raise
        sampler.stop()
        result.ok = False
        result.oom = _is_oom(exc)
        result.reason = (
            "CUDA out of memory" if result.oom else f"{type(exc).__name__}: {exc}"
        )
    finally:
        _empty_cache(device)
    return result


# ---------------------------------------------------------------------------
# Runtime projections
# ---------------------------------------------------------------------------
@dataclass
class Projection:
    """Wall-clock projection for one token budget."""

    token_budget: int
    hours: float
    days: float
    label: str
    rate_is_measured: bool
    source_tokens_per_sec: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def project_runtime(
    tokens_per_sec: Optional[float],
    *,
    budgets: Sequence[int] = TOKEN_BUDGETS,
    rate_is_measured: bool,
    source: str,
) -> List[Projection]:
    """Project wall-clock for each budget from a measured (or estimated) rate.

    Every entry carries ``rate_is_measured`` and a human ``label`` so a reader
    can never mistake an extrapolation for a measurement: the *rate* may be
    measured on the owner's GPU, but the 200M/600M/1B/10B totals are always
    extrapolations and are labelled ``"measured-rate extrapolation"`` or
    ``"estimated (no measured rate)"`` accordingly.
    """
    out: List[Projection] = []
    if not tokens_per_sec or tokens_per_sec <= 0:
        for budget in budgets:
            out.append(
                Projection(
                    token_budget=int(budget),
                    hours=float("nan"),
                    days=float("nan"),
                    label="estimated (no measured rate)",
                    rate_is_measured=False,
                    source_tokens_per_sec=0.0,
                    source=source,
                )
            )
        return out
    label = (
        "measured-rate extrapolation"
        if rate_is_measured
        else "estimated (rate not measured on GPU)"
    )
    for budget in budgets:
        hours = float(budget) / float(tokens_per_sec) / 3600.0
        out.append(
            Projection(
                token_budget=int(budget),
                hours=round(hours, 3),
                days=round(hours / 24.0, 3),
                label=label,
                rate_is_measured=bool(rate_is_measured),
                source_tokens_per_sec=float(tokens_per_sec),
                source=source,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Checkpoint storage + retention
# ---------------------------------------------------------------------------
def checkpoint_storage_report(
    n_params: int, *, amp: str = "none", include_optimizer: bool = True
) -> Dict[str, Any]:
    """Bytes per checkpoint, by component, for AdamW full fine-tuning.

    ``--amp bf16`` does not shrink the *stored* weights: autocast only changes
    compute dtype, so the master weights stay fp32 and AdamW keeps two fp32
    moment buffers. That is why the optimizer state dominates.
    """
    weights_bytes = 4 if amp == "none" else 4  # fp32 master weights in both cases
    params = int(n_params)
    weights = params * weights_bytes
    moments = params * 4 * 2 if include_optimizer else 0
    misc = 1024 * 1024  # config, RNG, tokenizer fingerprint, losses
    total = weights + moments + misc
    return {
        "params": params,
        "amp": amp,
        "compute_dtype_note": (
            "autocast compute dtype does not change the stored/optimizer "
            "dtype: master weights stay fp32 and AdamW keeps two fp32 moments"
        ),
        "weights_bytes": weights,
        "optimizer_bytes": moments,
        "misc_bytes": misc,
        "total_bytes": total,
        "total_gib": round(total / (1024 ** 3), 3),
        "weights_only_gib": round(weights / (1024 ** 3), 3),
        "note": (
            "Includes optimizer state so the number matches what resume needs. "
            "Weights-only checkpoints (inference export) are ~"
            f"{round(weights / (1024 ** 3), 3)} GiB."
        ),
    }


@dataclass
class RetentionPolicy:
    """Checkpoint retention the benchmark reports and *never* violates.

    ``keep`` always contains ``latest`` and ``known-good``; deleting a
    known-good checkpoint is not offered as an option at all.
    """

    keep: Tuple[str, ...] = ("latest", "known-good")
    auto_delete_known_good: bool = False
    auto_delete_latest: bool = False
    delete_anything_automatically: bool = False
    note: str = (
        "Retention keeps the latest checkpoint AND every known-good "
        "checkpoint. This harness never deletes a checkpoint; known-good "
        "checkpoints can only be removed by hand."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def retention_decide(
    entries: Sequence[Tuple[str, int]],
    *,
    known_good: Sequence[str] = (),
    policy: Optional[RetentionPolicy] = None,
) -> Dict[str, Any]:
    """Classify checkpoint files under the retention policy.

    ``entries`` is ``[(path, step), ...]``. Returns ``keep`` (latest +
    known-good), ``protected`` (known-good, never deletable automatically) and
    ``would_delete`` — which is **always empty** while
    ``policy.delete_anything_automatically`` is False, i.e. always in this
    harness. Pure and side-effect free, so the guarantee is unit-testable.
    """
    policy = policy or RetentionPolicy()
    if not entries:
        return {
            "keep": [],
            "protected": list(known_good),
            "would_delete": [],
            "policy": policy.to_dict(),
        }
    latest_path = max(entries, key=lambda e: (e[1], e[0]))[0]
    keep: List[str] = []
    if "latest" in policy.keep:
        keep.append(latest_path)
    protected = [p for p in known_good]
    for path in protected:
        if path not in keep:
            keep.append(path)
    candidates = [p for p, _ in entries if p not in keep]
    would_delete: List[str] = []
    if policy.delete_anything_automatically:
        would_delete = [
            p for p in candidates
            if not (p in protected and not policy.auto_delete_known_good)
        ]
    return {
        "keep": keep,
        "protected": protected,
        "would_delete": would_delete,
        "latest": latest_path,
        "policy": policy.to_dict(),
    }


def list_checkpoints(directory: str) -> List[Tuple[str, int]]:
    """``[(path, step), ...]`` for ``step-<N>.pt`` files under ``directory``."""
    out: List[Tuple[str, int]] = []
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if not (name.startswith("step-") and name.endswith(".pt")):
            continue
        stem = name[len("step-"):-len(".pt")]
        if stem.isdigit():
            out.append((os.path.join(directory, name), int(stem)))
    return sorted(out, key=lambda e: e[1])


# ---------------------------------------------------------------------------
# Resume harness: save -> kill -> resume
# ---------------------------------------------------------------------------
#: Generic English filler words — no third-party text is written or shipped.
_FILLER_WORDS: Tuple[str, ...] = (
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and",
    "then", "runs", "into", "forest", "where", "river", "flows", "past",
    "old", "stone", "bridge", "under", "blue", "sky", "above", "green",
    "field", "beside", "village", "road", "small", "house", "with", "red",
    "roof", "warm", "light", "inside", "window", "morning", "birds", "sing",
    "while", "children", "play", "outside", "garden", "near", "tall", "trees",
    "sway", "in", "wind", "quiet", "evening", "arrives", "slowly", "stars",
    "appear", "over", "the", "hills", "far", "away", "from", "here",
)


def write_synthetic_corpus(path: str, n_docs: int = 200, seed: int = 0) -> int:
    """Write a deterministic JSONL corpus of generated filler text.

    Returns the number of documents written. This is *generated* text — no
    third-party corpus is read, copied or shipped, and the owner's packed
    corpus is never touched.
    """
    rng = random.Random(int(seed))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(int(n_docs)):
            n_words = rng.randint(40, 120)
            doc = " ".join(rng.choice(_FILLER_WORDS) for _ in range(n_words))
            fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            written += 1
    return written


@dataclass
class ResumeCheck:
    """One named assertion in the resume harness."""

    name: str
    ok: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResumeHarnessResult:
    """Verdict + evidence for the save -> kill -> resume cycle."""

    ok: bool
    verdict: str  # "PASS" | "FAIL"
    preset: str
    reason: str = ""
    checkpoint: Optional[str] = None
    killed_returncode: Optional[int] = None
    resume_returncode: Optional[int] = None
    reference_returncode: Optional[int] = None
    steps: Dict[str, Any] = field(default_factory=dict)
    tokens: Dict[str, Any] = field(default_factory=dict)
    losses: Dict[str, Any] = field(default_factory=dict)
    checks: List[ResumeCheck] = field(default_factory=list)
    retention: Dict[str, Any] = field(default_factory=dict)
    work_dir: Optional[str] = None
    logs: Dict[str, Optional[str]] = field(default_factory=dict)
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "preset": self.preset,
            "reason": self.reason,
            "checkpoint": self.checkpoint,
            "killed_returncode": self.killed_returncode,
            "resume_returncode": self.resume_returncode,
            "reference_returncode": self.reference_returncode,
            "steps": self.steps,
            "tokens": self.tokens,
            "losses": self.losses,
            "checks": [c.to_dict() for c in self.checks],
            "retention": self.retention,
            "work_dir": self.work_dir,
            "logs": self.logs,
            "note": self.note,
        }


def _read_metrics(path: str) -> Dict[str, Any]:
    """Load a trainer ``metrics.json`` (empty dict if absent/unreadable)."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return {}


def build_resume_cmd(
    *,
    out_dir: str,
    data: str,
    preset: str,
    device: str,
    seq: int,
    batch: int,
    epochs: int,
    steps_per_epoch: int,
    seed: int,
    resume: Optional[str] = None,
    tokenizer_asset: Optional[str] = None,
    extra: Sequence[str] = (),
) -> List[str]:
    """The trainer argv used by the resume harness (explicit and reproducible)."""
    cmd = [
        sys.executable, "-u", "-m", "scripts.train_oasst1",
        "--data", data,
        "--out-dir", out_dir,
        "--preset", preset,
        "--device", device,
        "--seq", str(seq),
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--max-steps-per-epoch", str(steps_per_epoch),
        "--val-max-steps", "1",
        "--log-every-steps", "0",
        "--seed", str(seed),
    ]
    if tokenizer_asset:
        cmd += ["--tokenizer-asset", tokenizer_asset]
    if resume:
        cmd += ["--resume", resume]
    cmd += list(extra)
    return cmd


def _evaluate_resume(
    *,
    preset: str,
    ckpt_path: str,
    ckpt: Dict[str, Any],
    ckpt_step: int,
    batch: int,
    seq: int,
    killed_rc: Optional[int],
    resume_rc: Optional[int],
    ref_rc: Optional[int],
    resumed_metrics: Dict[str, Any],
    ref_metrics: Dict[str, Any],
    loss_rtol: float,
) -> Tuple[bool, str, List[ResumeCheck], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Pure verdict logic for the resume harness (no I/O) — unit-testable."""
    checks: List[ResumeCheck] = []

    # 1. The checkpoint exists, loads, and passes the trainer's own validator.
    try:
        validate_resume_checkpoint(ckpt, preset)
        checks.append(ResumeCheck(
            "checkpoint_valid", True,
            "loads and passes scripts.train_oasst1.validate_resume_checkpoint",
        ))
    except (ValueError, KeyError, RuntimeError) as exc:
        checks.append(ResumeCheck(
            "checkpoint_valid", False, f"{type(exc).__name__}: {exc}"
        ))

    ckpt_step_rec = ckpt.get("step")
    ckpt_tokens = ckpt.get("tokens_consumed")
    ckpt_loss = ckpt.get("train_loss")
    # The trainer credits the *predicted* tokens of each window: it forwards
    # ``x[:, :-1]`` and scores ``x[:, 1:]``, so one optimizer step consumes
    # ``batch * (seq - 1)`` tokens, not ``batch * seq`` (measured: batch 2 /
    # seq 64 -> 126 tokens/step, not 128).
    tokens_per_step = int(batch) * (int(seq) - 1)
    expected_tokens = int(ckpt_step) * tokens_per_step
    checks.append(ResumeCheck(
        "checkpoint_accounting",
        ckpt_step_rec == ckpt_step and ckpt_tokens == expected_tokens,
        f"ckpt step={ckpt_step_rec} tokens={ckpt_tokens} "
        f"(expected step={ckpt_step} tokens={expected_tokens} = "
        f"{ckpt_step} x batch {batch} x (seq {seq} - 1))",
    ))

    # 2. The process was actually killed (the point of the harness).
    killed_ok = killed_rc is not None and killed_rc != 0
    checks.append(ResumeCheck(
        "process_killed",
        killed_ok,
        f"killed process returncode={killed_rc} (non-zero means SIGKILL landed "
        "before the run finished)",
    ))

    # 3. The resumed run completed.
    checks.append(ResumeCheck(
        "resume_completed", resume_rc == 0,
        f"resume returncode={resume_rc}",
    ))
    checks.append(ResumeCheck(
        "reference_completed", ref_rc == 0,
        f"reference returncode={ref_rc}",
    ))

    ref_steps = deep_get(ref_metrics, "steps")
    res_steps = deep_get(resumed_metrics, "steps")
    ref_tokens = deep_get(ref_metrics, "tokens_consumed")
    res_tokens = deep_get(resumed_metrics, "tokens_consumed")

    # The trainer's two accounting fields have DIFFERENT scopes, which is
    # exactly what a resume test has to pin down:
    #   * ``steps``           = ``history.attempts`` for THIS invocation only;
    #   * ``tokens_consumed`` = ``history.tokens_processed``, CUMULATIVE from
    #     the resumed checkpoint.
    # So the resumed run's total step count is ``ckpt_step + its own steps``.
    cumulative_steps = (
        None if res_steps is None else int(ckpt_step) + int(res_steps)
    )

    # 4. Step accounting matches the uninterrupted reference run.
    checks.append(ResumeCheck(
        "step_accounting",
        ref_steps is not None and cumulative_steps == int(ref_steps),
        f"checkpoint step {ckpt_step} + resumed-run steps {res_steps} = "
        f"{cumulative_steps} vs reference steps {ref_steps} "
        "(metrics 'steps' is per-invocation; the checkpoint carries the rest)",
    ))
    # 5. Token accounting matches, and is self-consistent with the step count.
    tokens_match = (
        ref_tokens is not None and res_tokens is not None and ref_tokens == res_tokens
    )
    self_consistent = (
        cumulative_steps is not None
        and res_tokens == int(cumulative_steps) * tokens_per_step
    )
    checks.append(ResumeCheck(
        "token_accounting",
        bool(tokens_match and self_consistent),
        f"resumed tokens={res_tokens} vs reference tokens={ref_tokens}; "
        f"self-consistent with {cumulative_steps} steps x {tokens_per_step} "
        f"tokens/step = "
        f"{None if cumulative_steps is None else cumulative_steps * tokens_per_step}",
    ))

    # 6. Loss continuity: resume must not restart or diverge the objective.
    ref_loss = deep_get(ref_metrics, "final_train_loss")
    res_loss = deep_get(resumed_metrics, "final_train_loss")
    if ref_loss is not None and res_loss is not None:
        denom = max(abs(float(ref_loss)), 1e-9)
        rel = abs(float(res_loss) - float(ref_loss)) / denom
        loss_ok = rel <= float(loss_rtol)
        detail = (
            f"resumed final_train_loss={float(res_loss):.6f} vs reference "
            f"{float(ref_loss):.6f} (relative diff {rel:.3e} <= {loss_rtol})"
        )
    else:
        loss_ok = False
        rel = None
        detail = f"loss unavailable (resumed={res_loss}, reference={ref_loss})"
    checks.append(ResumeCheck("loss_continuity", loss_ok, detail))

    # 7. Tokens must never go backwards across the resume boundary.
    monotonic = (
        res_tokens is not None and ckpt_tokens is not None
        and int(res_tokens) >= int(ckpt_tokens)
    )
    checks.append(ResumeCheck(
        "tokens_monotonic", bool(monotonic),
        f"resumed tokens={res_tokens} >= checkpoint tokens={ckpt_tokens}",
    ))

    ok = all(c.ok for c in checks)
    failed = [c.name for c in checks if not c.ok]
    reason = "all checks passed" if ok else "failed checks: " + ", ".join(failed)
    steps = {
        "checkpoint": ckpt_step,
        "resumed_run": res_steps,
        "cumulative": cumulative_steps,
        "reference": ref_steps,
        "tokens_per_step": tokens_per_step,
    }
    tokens = {
        "checkpoint": ckpt_tokens, "resumed": res_tokens, "reference": ref_tokens,
    }
    losses = {
        "checkpoint_train_loss": ckpt_loss,
        "resumed_final_train_loss": res_loss,
        "reference_final_train_loss": ref_loss,
        "relative_diff": rel,
        "rtol": loss_rtol,
    }
    return ok, reason, checks, steps, tokens, losses


def run_resume_harness(
    *,
    preset: str,
    device: torch.device,
    work_dir: str,
    epochs: int = 3,
    steps_per_epoch: int = 4,
    seq: int = 64,
    batch: int = 2,
    n_docs: int = 80,
    seed: int = 0,
    kill_timeout_s: float = 900.0,
    run_timeout_s: float = 3600.0,
    loss_rtol: float = 0.05,
    tokenizer_asset: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> ResumeHarnessResult:
    """Run the real trainer three ways and verify checkpoint resume.

    ``reference`` — an uninterrupted run, the accounting ground truth.
    ``killed``    — the same run, ``SIGKILL``ed as soon as the first
                    ``step-<steps_per_epoch>.pt`` appears.
    ``resumed``   — the killed run's directory resumed from that checkpoint.

    Every subprocess is launched from ``repo_root`` so ``-m scripts.train_oasst1``
    and the committed tokenizer asset resolve identically to a manual run.
    """
    repo_root = repo_root or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    device_str = str(device)
    if device.type != "cuda" and preset in TOKENIZER_ASSET_PRESETS:
        # Belt and braces: the guard in main() should have caught this, but the
        # harness itself must never launch a ~1B fp32 trainer on a CPU box.
        return ResumeHarnessResult(
            ok=False,
            verdict="SKIPPED",
            preset=preset,
            reason=(
                f"refused to launch a {preset} trainer on {device_str}: the "
                "resume harness needs a CUDA device for the 1B presets. Pass "
                "--resume-preset tiny for the CPU wiring check."
            ),
            work_dir=work_dir,
        )
    os.makedirs(work_dir, exist_ok=True)
    corpus = os.path.join(work_dir, "synthetic_corpus.jsonl")
    write_synthetic_corpus(corpus, n_docs=n_docs, seed=seed)
    ref_dir = os.path.join(work_dir, "reference")
    killed_dir = os.path.join(work_dir, "killed")
    for d in (ref_dir, killed_dir):
        os.makedirs(d, exist_ok=True)

    def _cmd(out_dir: str, resume: Optional[str] = None) -> List[str]:
        return build_resume_cmd(
            out_dir=out_dir, data=corpus, preset=preset, device=device_str,
            seq=seq, batch=batch, epochs=epochs, steps_per_epoch=steps_per_epoch,
            seed=seed, resume=resume, tokenizer_asset=tokenizer_asset,
        )

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", repo_root)
    # Unbuffered: stdout redirected to a file is block-buffered, so a child
    # that is SIGKILLed (OOM) would otherwise leave an EMPTY log and no
    # post-mortem at all.
    env["PYTHONUNBUFFERED"] = "1"

    def _run(cmd: List[str], timeout: float, log_path: str) -> Tuple[int, str]:
        """Run one trainer subprocess, streaming its output to a log file.

        A log *file* rather than a pipe: a pipe would deadlock if the child
        ever outran the 64 KiB kernel buffer, and the file survives a crash so
        the post-mortem has something to read.
        """
        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                proc = subprocess.run(
                    cmd, cwd=repo_root, env=env, stdout=log_fh,
                    stderr=subprocess.STDOUT, text=True, timeout=timeout,
                    check=False,
                )
            return proc.returncode, _tail(log_path)
        except subprocess.TimeoutExpired as exc:
            return -1, f"timeout after {timeout}s: {exc}"
        except OSError as exc:  # pragma: no cover - env dep
            return -1, f"failed to launch: {exc}"

    def _tail(path: str, limit: int = 2000) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()[-limit:]
        except OSError:  # pragma: no cover - defensive
            return ""

    # ---- 1. reference run -------------------------------------------------
    ref_log_path = os.path.join(work_dir, "reference.log")
    ref_rc, ref_log = _run(_cmd(ref_dir), run_timeout_s, ref_log_path)
    ref_metrics = _read_metrics(os.path.join(ref_dir, "metrics.json"))
    if ref_rc != 0:
        # Nothing downstream is meaningful without the accounting ground truth.
        return ResumeHarnessResult(
            ok=False,
            verdict="FAIL",
            preset=preset,
            reason=(
                f"the uninterrupted reference run failed (returncode={ref_rc}"
                + (", SIGKILL — the OS out-of-memory killer, not a code path"
                   if ref_rc == -9 else "")
                + ") — the resume comparison needs it. Child log tail: "
                + (ref_log[-600:] or "(empty)")
            ),
            reference_returncode=ref_rc,
            work_dir=work_dir,
            note=ref_log[-2000:] or None,
        )

    # ---- 2. killed run ----------------------------------------------------
    ckpt_target = os.path.join(killed_dir, f"step-{steps_per_epoch}.pt")
    killed_rc: Optional[int] = None
    kill_log = ""
    killed_metrics: Dict[str, Any] = {}
    kill_log_path = os.path.join(work_dir, "killed.log")
    kill_fh = open(kill_log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            _cmd(killed_dir), cwd=repo_root, env=env,
            stdout=kill_fh, stderr=subprocess.STDOUT, text=True,
        )
    except OSError as exc:  # pragma: no cover - env dep
        kill_fh.close()
        return ResumeHarnessResult(
            ok=False, verdict="FAIL", preset=preset,
            reason=f"could not launch the trainer: {exc}", work_dir=work_dir,
        )
    deadline = time.monotonic() + kill_timeout_s
    killed_at_checkpoint = False
    while time.monotonic() < deadline:
        if os.path.isfile(ckpt_target):
            killed_at_checkpoint = True
            proc.send_signal(signal.SIGKILL)
            break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    if proc.poll() is None:
        proc.send_signal(signal.SIGKILL)
    try:
        proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()
    kill_fh.close()
    killed_rc = proc.returncode
    kill_log = _tail(kill_log_path)
    killed_metrics = _read_metrics(os.path.join(killed_dir, "metrics.json"))
    if killed_metrics:
        # Preserve the killed run's metrics: the resume overwrites metrics.json.
        try:
            with open(os.path.join(work_dir, "killed_metrics.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(killed_metrics, fh, indent=2)
        except OSError:  # pragma: no cover - defensive
            pass

    if not killed_at_checkpoint or not os.path.isfile(ckpt_target):
        ckpts = list_checkpoints(killed_dir)
        if killed_rc == -9:
            why = (
                "the killed run was SIGKILLed before it reached "
                f"{os.path.basename(ckpt_target)} and wrote no checkpoint "
                f"({len(ckpts)} seen). Exit code -9 with no log output is the "
                "OS out-of-memory killer; free memory on this machine and "
                "re-run, or lower --resume-docs / --resume-seq."
            )
        else:
            why = (
                "the kill window was missed: "
                f"{os.path.basename(ckpt_target)} never appeared before the run "
                f"ended (returncode={killed_rc}, checkpoints seen={len(ckpts)}). "
                "This is a harness timing issue, not an engine defect — increase "
                "--resume-epochs or --resume-steps-per-epoch and re-run."
            )
        return ResumeHarnessResult(
            ok=False,
            verdict="FAIL",
            preset=preset,
            reason=why + " Child log tail: " + (kill_log[-600:] or "(empty)"),
            killed_returncode=killed_rc,
            reference_returncode=ref_rc,
            work_dir=work_dir,
            note=(kill_log[-2000:] if kill_log else None),
        )

    # ---- 3. resume from the checkpoint ------------------------------------
    resume_log_path = os.path.join(work_dir, "resumed.log")
    resume_rc, resume_log = _run(
        _cmd(killed_dir, resume=ckpt_target), run_timeout_s, resume_log_path
    )
    resumed_metrics = _read_metrics(os.path.join(killed_dir, "metrics.json"))

    try:
        ckpt = load_checkpoint(ckpt_target)
    except (OSError, RuntimeError, ValueError) as exc:
        return ResumeHarnessResult(
            ok=False, verdict="FAIL", preset=preset,
            reason=f"checkpoint {ckpt_target} failed to load: {exc}",
            checkpoint=ckpt_target, killed_returncode=killed_rc,
            resume_returncode=resume_rc, reference_returncode=ref_rc,
            work_dir=work_dir,
        )

    ok, reason, checks, steps, tokens, losses = _evaluate_resume(
        preset=preset,
        ckpt_path=ckpt_target,
        ckpt=ckpt,
        ckpt_step=int(steps_per_epoch),
        batch=batch,
        seq=seq,
        killed_rc=killed_rc,
        resume_rc=resume_rc,
        ref_rc=ref_rc,
        resumed_metrics=resumed_metrics,
        ref_metrics=ref_metrics,
        loss_rtol=loss_rtol,
    )

    entries = list_checkpoints(killed_dir)
    retention = retention_decide(entries, known_good=[ckpt_target])

    return ResumeHarnessResult(
        ok=ok,
        verdict="PASS" if ok else "FAIL",
        preset=preset,
        # (SKIPPED results are constructed directly and never reach here.)
        reason=reason,
        checkpoint=ckpt_target,
        killed_returncode=killed_rc,
        resume_returncode=resume_rc,
        reference_returncode=ref_rc,
        steps=steps,
        tokens=tokens,
        losses=losses,
        checks=checks,
        retention=retention,
        work_dir=work_dir,
        logs={
            "reference": ref_log_path,
            "killed": kill_log_path,
            "resumed": resume_log_path,
        },
        note=None if ok else (resume_log[-2000:] or None),
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
class GuardError(ValueError):
    """A guard refused the requested run (exit code 2)."""


def validate_guards(args: argparse.Namespace) -> None:
    """Refuse unsafe/overwriting runs before anything is built or launched."""
    if args.max_steps < 1:
        raise GuardError("--max-steps must be >= 1")
    if args.warmup_steps < 0:
        raise GuardError("--warmup-steps must be >= 0")
    if args.measure_steps < 1:
        raise GuardError("--measure-steps must be >= 1")
    total = args.warmup_steps + args.measure_steps
    if total > args.max_steps:
        raise GuardError(
            f"--warmup-steps ({args.warmup_steps}) + --measure-steps "
            f"({args.measure_steps}) = {total} exceeds --max-steps "
            f"({args.max_steps}). Lower them or raise --max-steps."
        )
    if args.max_steps > DEFAULT_MAX_STEPS and not args.confirm_full_run:
        raise GuardError(
            f"--max-steps {args.max_steps} is above the safe default of "
            f"{DEFAULT_MAX_STEPS} optimizer steps. This benchmark is not a "
            "training launcher; pass --confirm-full-run to acknowledge you "
            "want the longer measurement."
        )
    if args.seq < 2:
        raise GuardError("--seq must be >= 2")
    if args.microbatch_start < 1:
        raise GuardError("--microbatch-start must be >= 1")
    if args.grad_accum < 1:
        raise GuardError("--grad-accum must be >= 1")
    if args.preset not in ALL_PRESETS:
        raise GuardError(
            f"--preset {args.preset!r} is not a registered preset "
            f"(known: {', '.join(sorted(ALL_PRESETS))})"
        )
    if args.resume_preset is not None and args.resume_preset not in ALL_PRESETS:
        raise GuardError(
            f"--resume-preset {args.resume_preset!r} is not a registered preset"
        )
    if (
        args.device == "cpu"
        and args.resume_preset in TOKENIZER_ASSET_PRESETS
        and not args.allow_big_cpu
    ):
        raise GuardError(
            f"--device cpu with --resume-preset {args.resume_preset} would make "
            "the resume harness launch a ~1B-parameter fp32 trainer on this "
            "machine (several GB of host RAM). Use the default (the dry-run "
            "preset) or pass --allow-big-cpu."
        )
    if args.device == "cpu" and args.preset in TOKENIZER_ASSET_PRESETS:
        # Never instantiate a ~1B fp32 model on a small CPU box.
        if not args.dry_run and not args.allow_big_cpu:
            raise GuardError(
                f"--device cpu with --preset {args.preset} would instantiate a "
                "~1B-parameter fp32 model in host RAM (multi-GB). Pass "
                "--dry-run to validate the wiring on tiny dimensions instead, "
                "or --allow-big-cpu if this machine really has the RAM."
            )
    if args.force and not args.confirm_full_run and args.max_steps > DEFAULT_MAX_STEPS:
        raise GuardError("--force does not bypass --confirm-full-run")


def resolve_results_path(args: argparse.Namespace, stamp: str) -> str:
    """Timestamped results path; refuses to clobber without ``--force``."""
    if args.results_file:
        path = args.results_file
    else:
        path = os.path.join(
            args.results_dir, f"benchmark_1b_{args.preset}_{stamp}.json"
        )
    if os.path.exists(path) and not args.force:
        raise GuardError(
            f"results file already exists: {path} — refusing to overwrite. "
            "Pass --force to replace it, or let the timestamped default name "
            "be used."
        )
    return path


def prepare_work_dir(args: argparse.Namespace, stamp: str) -> str:
    """Create a fresh work directory; refuses to reuse one without ``--force``."""
    work_dir = args.work_dir or os.path.join(
        args.results_dir, f"work_{args.preset}_{stamp}"
    )
    if os.path.isdir(work_dir) and os.listdir(work_dir):
        if not args.force:
            raise GuardError(
                f"work directory already has contents: {work_dir} — refusing "
                "to write checkpoints over it. Pass --force to reuse it."
            )
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _primary_rate(
    ladder: Optional[LadderResult], matrix: Optional[List[MatrixEntry]]
) -> Tuple[Optional[float], str, bool]:
    """Pick the tokens/sec that feeds the projections, and say where it came from."""
    if ladder is not None and ladder.winner_tokens_per_sec:
        return (
            float(ladder.winner_tokens_per_sec),
            f"OOM ladder winner (microbatch={ladder.winner_microbatch}, "
            f"seq={ladder.seq})",
            True,
        )
    if matrix:
        best = [e for e in matrix if e.status == "ok" and e.tokens_per_sec]
        if best:
            top = max(best, key=lambda e: e.tokens_per_sec or 0.0)
            return (
                float(top.tokens_per_sec or 0.0),
                f"activation matrix best row ({top.label})",
                True,
            )
    return None, "no measured training throughput", False


def build_report(
    *,
    args: argparse.Namespace,
    env: EnvironmentReport,
    model_info: Dict[str, Any],
    ladder: Optional[LadderResult],
    matrix: Optional[List[MatrixEntry]],
    inference: Optional[InferenceBenchResult],
    resume: Optional[ResumeHarnessResult],
    dry_run: bool,
) -> Dict[str, Any]:
    """Assemble the results JSON (the contract the owner reads)."""
    rate, rate_source, rate_measured = _primary_rate(ladder, matrix)
    projections = [
        p.to_dict()
        for p in project_runtime(
            rate,
            rate_is_measured=rate_measured,
            source=rate_source,
        )
    ]
    storage = checkpoint_storage_report(
        int(model_info.get("params") or 0),
        amp=args.amp,
        include_optimizer=True,
    )
    retention = RetentionPolicy()

    verdicts = {
        "resume_harness": None if resume is None else resume.verdict,
    }
    # SKIPPED (e.g. a 1B resume harness refused off CUDA) is not a failure.
    overall_ok = all(
        v != "FAIL" for v in verdicts.values() if v is not None
    ) and (resume is None or resume.ok)

    report: Dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_utc": utc_now(),
        "preset": args.preset,
        "validity": {
            "gpu_measurements": bool(env.cuda_available and not dry_run),
            "dry_run": bool(dry_run),
            "reason": (
                "measured on a CUDA device"
                if env.cuda_available and not dry_run
                else "no CUDA device (or --dry-run): functional sections ran on "
                     "tiny dimensions; throughput/VRAM/power numbers are NOT 1B "
                     "measurements"
            ),
        },
        "command": " ".join([sys.executable, "-m", "scripts.benchmark_1b", *sys.argv[1:]]),
        "args": {
            k: v for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, type(None), list, tuple))
        },
        "environment": env.to_dict(),
        "model": model_info,
        "measured": {
            "ladder": None if ladder is None else ladder.to_dict(),
            "activation_matrix": None if matrix is None else [e.to_dict() for e in matrix],
            "cpu_offload": cpu_offload_note(),
            "inference": None if inference is None else inference.to_dict(),
        },
        "resume_harness": None if resume is None else resume.to_dict(),
        "projections": projections,
        "projection_basis": {
            "tokens_per_sec": rate,
            "source": rate_source,
            "rate_is_measured": rate_measured,
        },
        "checkpoint_storage": storage,
        "retention": retention.to_dict(),
        "verdicts": verdicts,
        "overall_ok": bool(overall_ok),
        "skipped": [],
        "notes": list(env.notes),
    }
    if dry_run or not env.cuda_available:
        report["skipped"] = [
            "VRAM peak (torch.cuda.max_memory_allocated/reserved)",
            "nvidia-smi GPU utilisation / power draw / temperature",
            "1B-scale throughput at seq 2048",
        ]
    return report


def format_summary(report: Dict[str, Any]) -> str:
    """Human-readable console summary (the owner reads this first)."""
    lines: List[str] = []
    add = lines.append
    env = report.get("environment", {})
    validity = report.get("validity", {})
    add("=" * 72)
    add(f"Talos 1B benchmark — {report.get('preset')}  ({report.get('benchmark_version')})")
    add("=" * 72)
    add(f"generated      : {report.get('generated_utc')}")
    add(f"device         : {env.get('device_resolved')}  cuda={env.get('cuda_available')}")
    if env.get("gpu_name"):
        add(f"gpu            : {env['gpu_name']} ({env.get('gpu_sm')}, "
            f"{env.get('gpu_total_mem_mib')} MiB, bf16={env.get('gpu_bf16_supported')})")
    add(f"torch          : {env.get('torch_version')} (cuda {env.get('cuda_version')})")
    add(f"gpu measurements: {validity.get('gpu_measurements')} — {validity.get('reason')}")

    model = report.get("model", {})
    add("")
    add(f"model          : {model.get('params'):,} params, vocab "
        f"{model.get('vocab_size')}, {model.get('layers')} layers, "
        f"{model.get('hidden')} hidden" if model.get("params") else "model: n/a")

    ladder = report.get("measured", {}).get("ladder")
    if ladder:
        add("")
        add(f"OOM microbatch ladder (seq {ladder['seq']}, start "
            f"{ladder['start_microbatch']}):")
        for rung in ladder["rungs"]:
            extra = ""
            if rung.get("tokens_per_sec"):
                extra = f"  {rung['tokens_per_sec']:.0f} tok/s"
            if rung.get("peak_allocated_mib"):
                extra += f"  {rung['peak_allocated_mib']} MiB peak"
            if rung.get("reason"):
                extra += f"  ({rung['reason']})"
            add(f"  microbatch {rung['microbatch']:>3} : {rung['status']:<8}{extra}")
        add(f"  winner: microbatch {ladder['winner_microbatch']} "
            f"({ladder['max_tokens_per_optimizer_step']} tokens/step)")
        if ladder.get("diagnosis"):
            add(f"  diagnosis: {ladder['diagnosis']}")

    matrix = report.get("measured", {}).get("activation_matrix")
    if matrix:
        add("")
        add("activation matrix:")
        add(f"  {'combination':<58} {'status':<8} {'tok/s':>10} {'peak MiB':>9}")
        for row in matrix:
            tps = row.get("tokens_per_sec")
            peak = row.get("peak_allocated_mib")
            add(f"  {row['label']:<58} {row['status']:<8} "
                f"{(tps if tps is not None else float('nan')):>10.1f} "
                f"{('n/a' if peak is None else peak):>9}")
        offload = report["measured"].get("cpu_offload", {})
        add(f"  CPU offload: not supported — {offload.get('note', '')[:120]}")

    inference = report.get("measured", {}).get("inference")
    if inference:
        add("")
        if inference.get("ok"):
            add(f"inference: prefill {inference['prefill_tokens_per_sec']} tok/s "
                f"({inference['prefill_tokens']} tok), decode "
                f"{inference['decode_tokens_per_sec']} tok/s "
                f"({inference['decode_steps']} steps, batch {inference['batch']})")
        else:
            add(f"inference: FAILED — {inference.get('reason')}")

    resume = report.get("resume_harness")
    if resume:
        add("")
        add(f"resume harness : {resume['verdict']} — {resume['reason']}")
        if resume["verdict"] == "SKIPPED":
            add("  (no checks ran)")
        for chk in resume.get("checks", []):
            add(f"  [{'PASS' if chk['ok'] else 'FAIL'}] {chk['name']}: {chk['detail']}")
        add(f"  steps: {resume.get('steps')}")
        add(f"  tokens: {resume.get('tokens')}")

    proj = report.get("projections", [])
    if proj:
        add("")
        add(f"runtime projections (rate: {report['projection_basis']['tokens_per_sec']}"
            f" tok/s, measured={report['projection_basis']['rate_is_measured']}):")
        for p in proj:
            budget = p["token_budget"]
            label = f"{budget/1e9:g}B tokens" if budget >= 1e9 else f"{budget/1e6:g}M tokens"
            if p["hours"] != p["hours"]:  # NaN
                add(f"  {label:<14} n/a ({p['label']})")
            else:
                add(f"  {label:<14} {p['hours']:>9.2f} h  ({p['days']:.2f} days)"
                    f"  [{p['label']}]")

    storage = report.get("checkpoint_storage", {})
    if storage:
        add("")
        add(f"checkpoint size: {storage.get('total_gib')} GiB each "
            f"(weights {storage.get('weights_only_gib')} GiB + AdamW state)")
    retention = report.get("retention", {})
    if retention:
        add(f"retention      : keep {retention.get('keep')}; auto-delete "
            f"known-good = {retention.get('auto_delete_known_good')} "
            f"(nothing is ever auto-deleted)")

    if report.get("skipped"):
        add("")
        add("skipped (no GPU): " + ", ".join(report["skipped"]))
    for note in report.get("notes", []):
        add(f"note: {note}")

    add("")
    verdicts = report.get("verdicts", {})
    add(f"RESUME HARNESS: {verdicts.get('resume_harness') or 'not run'}")
    add(f"OVERALL       : {'PASS' if report.get('overall_ok') else 'FAIL'}")
    add("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _host_can_hold(n_params: Optional[int], bytes_per_param: int = 4) -> bool:
    """Whether ``n_params`` fp32 weights fit comfortably in this host's RAM.

    This is the guard against instantiating a 1B fp32 model on a small CPU
    box: the weights must be under ~40 % of total RAM, leaving room for
    activations, the optimizer state and the interpreter.
    """
    if not n_params:
        return False
    ram = _system_ram_bytes()
    if not ram:
        return False
    return n_params * bytes_per_param <= 0.4 * ram


def build_bench_model(
    preset: str, device: torch.device, attention_backend: str
) -> Tuple[TalosGPT, Dict[str, Any]]:
    """Build the preset model and describe it for the report.

    Off CUDA a preset too large for host RAM is built on the **meta** device:
    parameter count and shapes are still verified, but no multi-GB fp32 tensor
    is allocated (the "never instantiate 1B fp32 on a small CPU box" rule).
    A preset that *does* fit (e.g. the ``tiny`` dry-run preset) is built on the
    real CPU device, so the functional sections actually run instead of failing
    on meta tensors.
    """
    canonical = CANONICAL_PRESETS.get(preset, (None, None))[0]
    meta = device.type != "cuda" and not _host_can_hold(canonical)
    ctx = torch.device("meta") if meta else device
    with torch.device(ctx):
        model = build_preset_model(preset, attention_backend=attention_backend)
    n_params = int(model.num_parameters())
    expected = CANONICAL_PRESETS.get(preset, (None, None))[0]
    info = {
        "preset": preset,
        "params": n_params,
        "canonical_params": expected,
        "params_match_canonical": (expected is None or n_params == expected),
        "vocab_size": int(model.config.vocab_size),
        "layers": int(model.config.num_layers),
        "hidden": int(model.config.hidden_size),
        "built_on": str(ctx),
        "meta_only": meta,
        "note": (
            "built on the meta device — parameter count/shapes verified, no "
            "weights allocated (CPU dry-run)"
            if meta else "built on the CUDA device"
        ),
    }
    return model, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def make_arg_parser() -> argparse.ArgumentParser:
    """The benchmark CLI (every knob documented in ``--help``)."""
    p = argparse.ArgumentParser(
        prog="python -m scripts.benchmark_1b",
        description=(
            "One-command Talos 1B benchmark for the owner's RTX 3060 12 GB: "
            "VRAM/utilisation/power, OOM microbatch ladder, activation matrix, "
            "save->kill->resume verification, runtime projections."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", default="talos_1b",
                   help="model preset to benchmark (talos_1b | talos_1b_spec16 | "
                        "any registered preset)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                   help="compute device; 'auto' picks cuda when available")
    p.add_argument("--seq", type=int, default=DEFAULT_SEQ,
                   help="sequence length for the ladder/matrix")
    p.add_argument("--microbatch-start", type=int, default=DEFAULT_MICROBATCH_START,
                   help="largest microbatch tried by the OOM ladder (halved to 1)")
    p.add_argument("--amp", default="bf16", choices=AMP_CHOICES,
                   help="AMP mode for the headline measured run "
                        "(bf16 is the 3060's fast path)")
    p.add_argument("--attention-backend", default="sdpa",
                   choices=ATTENTION_BACKEND_CHOICES,
                   help="attention backend for the headline measured run")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="enable activation checkpointing for the headline run")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation for the headline run")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   help="hard cap on measured optimizer steps per benchmark; "
                        f"above {DEFAULT_MAX_STEPS} requires --confirm-full-run")
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
                   help="untimed warmup optimizer steps")
    p.add_argument("--measure-steps", type=int, default=DEFAULT_MEASURE_STEPS,
                   help="timed optimizer steps")
    p.add_argument("--confirm-full-run", action="store_true",
                   help="acknowledge a longer-than-default measurement")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting an existing results JSON / work dir")
    p.add_argument("--results-dir", default=os.path.join("benchmarks", "1b"),
                   help="directory for the timestamped results JSON")
    p.add_argument("--results-file", default=None,
                   help="exact results path (still guarded by --force)")
    p.add_argument("--work-dir", default=None,
                   help="checkpoint/work directory (default: timestamped under "
                        "--results-dir)")
    p.add_argument("--dry-run", action="store_true",
                   help="force the tiny-dimension wiring check even on a GPU")
    p.add_argument("--dry-run-preset", default="tiny",
                   help="preset used for the CPU/dry-run functional sections")
    p.add_argument("--dry-run-seq", type=int, default=64,
                   help="sequence length for the CPU/dry-run sections")
    p.add_argument("--allow-big-cpu", action="store_true",
                   help="permit building a 1B preset on CPU (needs several GB "
                        "of host RAM; refused by default)")
    p.add_argument("--skip-ladder", action="store_true", help="skip the OOM ladder")
    p.add_argument("--skip-matrix", action="store_true", help="skip the activation matrix")
    p.add_argument("--skip-inference", action="store_true", help="skip inference timing")
    p.add_argument("--skip-resume", action="store_true", help="skip the resume harness")
    p.add_argument("--no-ladder-optimizer", action="store_true",
                   help="build the ladder without AdamW state (isolates "
                        "activation memory from optimizer memory)")
    p.add_argument("--matrix-full", action="store_true",
                   help="full cross product instead of the curated matrix")
    p.add_argument("--matrix-max-combos", type=int, default=12,
                   help="cap on activation-matrix rows")
    p.add_argument("--inference-batch", type=int, default=1,
                   help="batch size for the inference timing")
    p.add_argument("--inference-prompt-len", type=int, default=256,
                   help="prefill prompt length for the inference timing")
    p.add_argument("--decode-steps", type=int, default=32,
                   help="KV-cache decode steps to time")
    p.add_argument("--resume-preset", default=None,
                   help="preset for the resume harness (default: --preset)")
    p.add_argument("--resume-epochs", type=int, default=3,
                   help="epochs in each resume-harness run")
    p.add_argument("--resume-steps-per-epoch", type=int, default=4,
                   help="optimizer steps per epoch (also the checkpoint step)")
    p.add_argument("--resume-seq", type=int, default=64,
                   help="sequence length for the resume harness")
    p.add_argument("--resume-batch", type=int, default=2,
                   help="batch size for the resume harness")
    p.add_argument("--resume-docs", type=int, default=80,
                   help="synthetic documents for the resume-harness corpus")
    p.add_argument("--resume-loss-rtol", type=float, default=0.05,
                   help="relative loss tolerance for the loss-continuity check")
    p.add_argument("--resume-timeout-s", type=float, default=3600.0,
                   help="per-run timeout for the resume harness")
    p.add_argument("--smi-interval-s", type=float, default=0.2,
                   help="nvidia-smi polling interval")
    p.add_argument("--json", action="store_true",
                   help="print the full results JSON to stdout as well")
    p.add_argument("--quiet", action="store_true", help="suppress the summary")
    return p


def _resolve_device(requested: str) -> torch.device:
    """Resolve ``--device``; 'auto' prefers CUDA, falls back to CPU."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise GuardError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: guards -> probe -> measure -> report."""
    args = make_arg_parser().parse_args(argv)
    stamp = utc_stamp()
    try:
        validate_guards(args)
        device = _resolve_device(args.device)
    except GuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_GUARD

    dry_run = bool(args.dry_run or device.type != "cuda")
    functional_preset = args.dry_run_preset if dry_run else args.preset
    functional_seq = args.dry_run_seq if dry_run else args.seq
    backend = "sdpa" if dry_run and args.attention_backend == "auto" else args.attention_backend
    if args.attention_backend == "auto":
        backend = "auto"

    try:
        results_path = resolve_results_path(args, stamp)
        work_dir = prepare_work_dir(args, stamp)
    except GuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_GUARD

    env = probe_environment(args.device, device)
    print(f"[benchmark] device={device} dry_run={dry_run} preset={args.preset} "
          f"functional={functional_preset} seq={functional_seq}")

    # ---- model -----------------------------------------------------------
    model: Optional[TalosGPT] = None
    model_info: Dict[str, Any] = {}
    try:
        model, model_info = build_bench_model(
            functional_preset if dry_run else args.preset, device, backend
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: could not build preset model: {exc}", file=sys.stderr)
        return EXIT_GUARD

    if args.preset in TOKENIZER_ASSET_PRESETS and not dry_run:
        # Report the sha-verified tokenizer asset the 1B track trains with.
        from tokenizer.assets import get_asset  # noqa: PLC0415 - lazy import

        try:
            asset = get_asset(TOKENIZER_ASSET_NAME)
            asset.verify()  # sha256 against assets/tokenizers/MANIFEST.json
            model_info["tokenizer_asset"] = {
                "name": TOKENIZER_ASSET_NAME,
                "sha256_verified": True,
                "sha256": asset.sha256,
                "vocab_size": asset.vocab_size,
            }
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - env dep
            model_info["tokenizer_asset"] = {
                "name": TOKENIZER_ASSET_NAME,
                "sha256_verified": False,
                "error": str(exc),
            }

    # ---- resume harness --------------------------------------------------
    # Off CUDA the harness resumes the DRY-RUN preset (tiny), never the 1B
    # preset: instantiating ~1B fp32 parameters in host RAM is exactly what the
    # CPU path must not do.
    resume_preset = args.resume_preset or (
        functional_preset if dry_run else args.preset
    )
    # Runs FIRST, before the ladder/matrix/inference sections have allocated
    # and freed optimizer + activation tensors: the trainer subprocesses need
    # the host's headroom, and torch's CPU allocator does not hand pages back
    # to the OS promptly. Ordering here is what keeps this section from being
    # OOM-killed on a memory-tight machine.
    resume: Optional[ResumeHarnessResult] = None
    if not args.skip_resume:
        resume = run_resume_harness(
            preset=resume_preset,
            device=device,
            work_dir=os.path.join(work_dir, "resume"),
            epochs=args.resume_epochs,
            steps_per_epoch=args.resume_steps_per_epoch,
            seq=args.resume_seq if not dry_run else min(args.resume_seq, 64),
            batch=args.resume_batch,
            n_docs=args.resume_docs,
            loss_rtol=args.resume_loss_rtol,
            run_timeout_s=args.resume_timeout_s,
            tokenizer_asset=(
                TOKENIZER_ASSET_NAME
                if resume_preset in TOKENIZER_ASSET_PRESETS else None
            ),
        )
        print(f"[benchmark] resume harness: {resume.verdict} — {resume.reason}")
        gc.collect()

    warmup_steps = args.warmup_steps
    measure_steps = min(args.measure_steps, args.max_steps - warmup_steps)

    # ---- ladder ----------------------------------------------------------
    ladder: Optional[LadderResult] = None
    if not args.skip_ladder:
        with_opt = not args.no_ladder_optimizer
        start_mb = args.microbatch_start if not dry_run else min(args.microbatch_start, 4)

        def _attempt(mb: int) -> TrainBenchResult:
            cfg = TrainBenchConfig(
                amp=args.amp, attention_backend=backend,
                grad_checkpoint=args.grad_checkpoint, grad_accum=1,
                microbatch=mb, seq=functional_seq,
            )
            if cfg.amp == "bf16" and device.type == "cuda":
                try:
                    if not torch.cuda.is_bf16_supported():
                        return _skipped_result(
                            cfg, "bf16 requires Ampere+ (SM80+)"
                        )
                except (RuntimeError, AttributeError):  # pragma: no cover
                    pass
            return run_train_bench(
                model, cfg, device, warmup_steps=warmup_steps,
                measure_steps=measure_steps, with_optimizer=with_opt,
                smi_interval_s=args.smi_interval_s,
            )

        ladder = run_oom_ladder(_attempt, start=start_mb, seq=functional_seq)
        if ladder.winner_microbatch is None:
            # Diagnose: activations alone, without AdamW state?
            diag_cfg = TrainBenchConfig(
                amp=args.amp, attention_backend=backend,
                grad_checkpoint=args.grad_checkpoint, grad_accum=1,
                microbatch=1, seq=functional_seq,
            )
            diag = run_train_bench(
                model, diag_cfg, device, warmup_steps=0,
                measure_steps=max(1, min(2, args.max_steps)),
                with_optimizer=False, smi_interval_s=args.smi_interval_s,
            )
            if diag.ok:
                ladder.diagnosis = (
                    "every ladder rung OOMs WITH AdamW state, but microbatch 1 "
                    "runs fine WITHOUT it — optimizer state (2 fp32 moments per "
                    "parameter) is the binding constraint, not activations. "
                    f"Activation-only peak at microbatch 1: "
                    f"{diag.peak_allocated_mib} MiB."
                )
            else:
                ladder.diagnosis = (
                    "even microbatch 1 without optimizer state does not fit — "
                    f"{diag.reason}. The preset cannot be trained on this GPU "
                    "at this sequence length."
                )
            ladder.note = (
                "Ladder ran with AdamW state included (faithful to the trainer)."
                if with_opt else "Ladder ran WITHOUT optimizer state (diagnostic)."
            )

    # ---- activation matrix ----------------------------------------------
    matrix: Optional[List[MatrixEntry]] = None
    if not args.skip_matrix:
        mb_list = (
            [ladder.winner_microbatch]
            if ladder is not None and ladder.winner_microbatch
            else [1]
        )
        plan = build_matrix_plan(
            microbatches=mb_list,
            seq=functional_seq,
            full=args.matrix_full,
            max_combos=args.matrix_max_combos,
        )
        matrix = run_activation_matrix(
            model, plan, device, warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )

    # ---- inference -------------------------------------------------------
    inference: Optional[InferenceBenchResult] = None
    if not args.skip_inference:
        inference = run_inference_bench(
            model, device, amp=args.amp,
            batch=args.inference_batch,
            prompt_len=min(args.inference_prompt_len, functional_seq),
            decode_steps=args.decode_steps,
            smi_interval_s=args.smi_interval_s,
        )

    # ---- report ----------------------------------------------------------
    del model
    gc.collect()
    report = build_report(
        args=args, env=env, model_info=model_info, ladder=ladder,
        matrix=matrix, inference=inference, resume=resume, dry_run=dry_run,
    )
    report["work_dir"] = work_dir

    try:
        os.makedirs(os.path.dirname(os.path.abspath(results_path)) or ".", exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
            fh.write("\n")
    except OSError as exc:
        print(f"error: could not write results to {results_path}: {exc}", file=sys.stderr)
        return EXIT_GUARD

    if not args.quiet:
        print(format_summary(report))
        print(f"\nresults written to: {results_path}")
    if args.json:
        print(json.dumps(report, indent=2, default=str))

    if resume is not None and not resume.ok:
        return EXIT_RESUME_FAIL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
