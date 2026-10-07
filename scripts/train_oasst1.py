"""Reproducible OASST1-style JSONL -> Talos-native tokenizer -> tiny-model training.

This is the owner-scoped reliability pass for the JSONL training path. It wires
**existing** Talos components end-to-end — no new data pipeline, no new training
framework, no new architecture:

    JSONL (OASST1-shaped, any path)
      -> deterministic train/val split        split_jsonl()  (fixed seed, disjoint files)
      -> Talos-native byte-level BPE          tokenizer.train.train_tokenizer()
         (vocab 1024 shared by every          preset configs.presets.preset_tokenizer_config,
          canonical preset —                     max_docs/max_chars/num_merges/minfreq
          configs.vocab.VOCAB_SIZE,             knobs exposed as CLI flags)
          TRAIN SPLIT ONLY)                   OR LOAD an established tokenizer.json via
                                              --tokenizer-json (the corpus-pretraining
                                              path: no BPE training; the sidecar copy's
                                              sha256 fingerprint is recorded in every
                                              checkpoint + metrics.json)
      -> canonical preset by --preset           configs.presets.<preset>_config()
         (registry-pinned exact params+vocab,   fails fast on any config drift
          any of tiny/tiny_1m/tiny_10m/tiny_100m)
      -> streaming training                   data.tokenized.StreamingTokenizedDataset
         (same objective/components as         + AdamW + CrossEntropyLoss, pack mode,
          examples/tiny_train.py)              x[:, :-1] -> x[:, 1:]
      -> per-epoch checkpoint artifact        weights + optimizer + RNG state + step/epoch
         + tokenizer.json next to it           + tokenizer fingerprint + losses in one .pt
      -> train + validation loss saved        out_dir/metrics.json (losses, wall time,
                                              peak RSS, params, tokenizer vocab + sha256)
      -> resume from any checkpoint           --resume step-<N>.pt continues the run with
         (bit-exact vs uninterrupted runs)     optimizer + RNG state restored

Why the loop lives here instead of examples/tiny_train.py: the existing
``train_stream`` helper is step-based (no epoch boundaries, no evaluation, no
checkpointing). Everything it uses is reused verbatim; this module only adds the
epoch/eval/checkpoint orchestration the task requires. The artifact layout is
flat so a checkpoint and its tokenizer always sit side by side::

    out_dir/
      data/train.jsonl        data/val.jsonl      # disjoint, deterministic split
      tokenizer.json                             # BPE-trained on train split, or a
                                                 # copy of the --tokenizer-json file
      step-<N>.pt                                # checkpoint (weights+opt+RNG+losses)
      metrics.json                               # run report

Usage::

    python -m tools.make_synthetic_oasst1 --docs 300 --seed 0 --output /tmp/oasst1.jsonl
    python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir runs/oasst1-tiny \\
        --epochs 3 --seq 64 --batch 4 --seed 0
    # resume a 100K-step Colab campaign from the last checkpoint:
    python -m scripts.train_oasst1 --data /tmp/oasst1.jsonl --out-dir runs/oasst1-tiny \\
        --resume runs/oasst1-tiny/step-7680.pt --epochs 20000 --seq 64 --batch 4 --seed 0

Pipeline integrity guards (all cheap, all always-on):

* every batch is token-id-range-checked at the data source (per encoded
  document, ``data.tokenized.validate_id_array``), at batch assembly
  (``model.utils.validate_token_ids`` on the train/eval loops), and at the
  embedding seam (``TalosGPT.forward``) — an out-of-range id can never reach a
  CUDA index kernel and poison the T4 context (audit P0 fix 1);
* the checkpoint records the sha256 of the trained ``tokenizer.json`` and
  resumed runs verify it, so a swapped/re-trained sidecar tokenizer fails
  loudly instead of silently mis-tokenizing (audit P0 fix 4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the script
# also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.canonical import CANONICAL_PRESETS, resolve_preset  # noqa: E402
from configs.presets import ALL_PRESETS, preset_tokenizer_config  # noqa: E402
from data.packed import (  # noqa: E402
    PackedTokenDataset,
    load_packed_manifest,
    manifest_identity,
    packed_phase_shard_paths,
)
from data.tokenized import StreamingTokenizedDataset  # noqa: E402
from model import ModelConfig, TalosGPT  # noqa: E402
from model.attention import build_attention_backend  # noqa: E402
from model.utils import get_logger, set_seed, validate_token_ids  # noqa: E402
from tokenizer.corpus import iter_text_documents  # noqa: E402
from tokenizer.tokenizer import (  # noqa: E402
    ByteLevelBPETokenizer,
    tokenizer_file_sha256,
)
from tokenizer.train import train_tokenizer  # noqa: E402

log = get_logger("scripts.train_oasst1")

CHECKPOINT_FORMAT = "talos-training-checkpoint-v1"

RUN_METADATA_FILENAME = "train_run_metadata.json"
RUN_METADATA_SCHEMA = "talos-training-run-metadata-v1"

#: LR-schedule knobs (additive; defaults reproduce the fixed-LR behavior).
LR_DECAY_CHOICES = ("none", "cosine")
#: Cosine decays to this fraction of ``--lr`` over the token budget.
MIN_LR_RATIO = 0.1

ATTENTION_BACKEND_CHOICES = ("auto", "plain", "sdpa", "flash")
AMP_CHOICES = ("none", "fp16")


# ---------------------------------------------------------------------------
# T4 training-engine helpers (P1: throughput; additive, defaults = old path)
# ---------------------------------------------------------------------------
def make_amp_autocast(amp: str, device: torch.device) -> torch.autocast:
    """Autocast context for ``--amp``.

    ``"fp16"`` wraps the train forward/backward in fp16 autocast (CUDA or CPU;
    CPU is supported for logic tests but pointless in production). Validation
    and the loss accumulation stay OUT of the context, so val loss is always
    computed fp32 — loss comparability across AMP modes (P1b/P3).
    """
    if amp not in AMP_CHOICES:
        raise ValueError(f"--amp must be one of {AMP_CHOICES}, got {amp!r}")
    enabled = amp == "fp16" and device.type in ("cuda", "cpu")
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled)


def make_grad_scaler(amp: str, device: torch.device) -> "torch.amp.GradScaler":
    """GradScaler for ``--amp fp16``.

    ``enabled=False`` on every non-AMP path and on CPU (GradScaler requires
    CUDA): all scaler calls become transparent no-ops, which keeps the AMP
    integration code the one code path (and CPU-testable).
    """
    enabled = amp == "fp16" and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)  # type: ignore[attr-defined]


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    *,
    fused: bool,
    device: torch.device,
) -> torch.optim.Optimizer:
    """AdamW with an optional fused CUDA kernel (P1c).

    Default (``fused=False``) constructs exactly the pre-upgrade optimizer
    (``torch.optim.AdamW(model.parameters(), lr=lr)``) — resume bit-exactness
    for existing runs is untouched. ``fused=True`` requests the CUDA-only
    fused AdamW when the installed torch advertises it; a non-CUDA device is a
    loud error (fused is unavailable on CPU), never a silent fallback.
    """
    if not fused:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if device.type != "cuda":
        raise ValueError(
            "--fused-optim requires a CUDA device (fused AdamW is a CUDA-only "
            f"kernel; device is {device})"
        )
    import inspect

    fused_supported = "fused" in inspect.signature(
        torch.optim.AdamW.__init__
    ).parameters
    if not fused_supported:  # pragma: no cover - torch >= 2.0 always has it
        raise ValueError(
            "this torch build's AdamW does not support fused=True — "
            "remove --fused-optim or upgrade torch"
        )
    log.info("using fused AdamW (CUDA multi-tensor kernel).")
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=True, foreach=False)


@dataclass
class IntervalStats:
    """Per-log-interval accumulators (P4 observability)."""

    steps: int = 0
    tokens: int = 0
    data_s: float = 0.0
    fwd_s: float = 0.0
    bwd_s: float = 0.0
    optim_s: float = 0.0
    ckpt_s: float = 0.0
    total_s: float = 0.0
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None

    def asdict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "data_s": round(self.data_s, 4),
            "fwd_s": round(self.fwd_s, 4),
            "bwd_s": round(self.bwd_s, 4),
            "optim_s": round(self.optim_s, 4),
            "ckpt_s": round(self.ckpt_s, 4),
            "total_s": round(self.total_s, 4),
            "tok_s": round(self.tokens / self.total_s, 3) if self.total_s > 0 else None,
            "steps_s": round(self.steps / self.total_s, 4) if self.total_s > 0 else None,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_used_mb": self.vram_used_mb,
        }


class CheckpointStager:
    """Local-then-copy checkpoint persistence (P5, Drive-aware).

    The trainer writes every checkpoint into **local** staging storage first
    (atomic tmp+fsync+rename on the local FS — never Drive's eventual
    consistency), then a single daemon thread copies finished files to the
    final ``out_dir`` (which may be a Colab Drive mount). Training never waits
    for the Drive copy: the thread queue is bounded, and :meth:`drain` (called
    at run end) joins the thread so a clean finish has flushed everything.

    ``staging_dir=None`` (default) preserves the old direct-write behavior —
    the stager is inert and saves go straight to ``out_dir``.
    """

    def __init__(self, staging_dir: Optional[str], out_dir: str) -> None:
        self.staging_dir = staging_dir
        self.out_dir = out_dir
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        if staging_dir:
            os.makedirs(staging_dir, exist_ok=True)
            self._thread = threading.Thread(
                target=self._copier, name="talos-ckpt-copier", daemon=True
            )
            self._thread.start()

    def _copier(self) -> None:
        while True:
            src = self._q.get()
            if src is None:
                self._q.task_done()
                return
            try:
                dst = os.path.join(self.out_dir, os.path.basename(src))
                shutil.copyfile(src, dst)
            except Exception as exc:  # never kill training on a copy failure
                log.error(
                    "checkpoint copy to %s failed: %s — the checkpoint remains "
                    "in staging %s", self.out_dir, exc, src,
                )
            finally:
                self._q.task_done()

    def enqueue(self, final_path: str) -> None:
        """Register ``final_path`` as staged; the thread copies it to out_dir."""
        if self._thread is None:
            return  # staging disabled — nothing to do
        self._q.put(final_path)

    def drain(self, timeout: float = 3600.0) -> bool:
        """Wait for pending Drive copies (run-end flush). True on success."""
        if self._thread is None:
            return True
        self._q.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():  # pragma: no cover - pathological Drive stall
            log.error("checkpoint copier did not finish within %ss", timeout)
            return False
        return self._q.empty()


def expected_state_dict_shapes(cfg: "ModelConfig") -> Dict[str, torch.Size]:
    """The exact ``state_dict`` key->shape map a fresh model of ``cfg`` has.

    Derived from the config arithmetic alone (no model materialization), so
    resume-time tensor-head validation can run on a CPU-only T4 host without
    allocating a second ~386 MiB model. Every parameter a model would create
    must appear here — a key missing on either side fails the check.
    ``tests/test_trainer_engine.py`` pins this map against a real model's
    ``state_dict``, so an architecture drift fails the suite, not a resume.
    """
    d = cfg.derive() if cfg.head_dim is None else cfg
    shapes: Dict[str, torch.Size] = {}
    hs, v = d.hidden_size, d.vocab_size
    nq, nkv, hd = d.num_attention_heads, d.num_kv_heads, d.head_dim
    shapes["embed_tokens.weight"] = torch.Size((v, hs))
    if not d.tie_word_embeddings:
        shapes["lm_head.weight"] = torch.Size((v, hs))
    if d.ffn_type != "dense":
        raise ValueError(
            "expected_state_dict_shapes supports dense FFN configs only "
            f"(got ffn_type={d.ffn_type!r}) — canonical presets are dense"
        )
    for i in range(d.num_layers):
        p = f"layers.{i}."
        shapes[p + "input_layernorm.weight"] = torch.Size((hs,))
        shapes[p + "post_attention_layernorm.weight"] = torch.Size((hs,))
        shapes[p + "attention.q_proj.weight"] = torch.Size((nq * hd, hs))
        shapes[p + "attention.k_proj.weight"] = torch.Size((nkv * hd, hs))
        shapes[p + "attention.v_proj.weight"] = torch.Size((nkv * hd, hs))
        shapes[p + "attention.o_proj.weight"] = torch.Size((hs, nq * hd))
        shapes[p + "ffn.swiglu.gate_proj.weight"] = torch.Size(
            (d.intermediate_size, hs)
        )
        shapes[p + "ffn.swiglu.up_proj.weight"] = torch.Size(
            (d.intermediate_size, hs)
        )
        shapes[p + "ffn.swiglu.down_proj.weight"] = torch.Size(
            (hs, d.intermediate_size)
        )
    shapes["final_norm.weight"] = torch.Size((hs,))
    return shapes


# ---------------------------------------------------------------------------
# Token-budget LR schedule (DELIVERABLE 3 — additive, default = fixed LR)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenSchedule:
    """LR as a function of tokens consumed. ``lr_at(t)`` is monotone.

    * ``warmup_tokens`` > 0: linear ramp ``0 -> lr`` over the first W tokens.
    * ``decay="cosine"``: after warmup, cosine from ``lr`` to
      ``min_lr_ratio * lr`` over the remaining budget (``budget - warmup``).
    * ``decay="none"`` (default): constant ``lr`` after warmup — bit-identical
      to the pre-schedule fixed-LR trainer when warmup is 0.
    """

    lr: float
    warmup_tokens: int = 0
    decay: str = "none"
    budget: Optional[int] = None
    min_lr_ratio: float = MIN_LR_RATIO

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.warmup_tokens < 0:
            raise ValueError(
                f"warmup_tokens must be >= 0, got {self.warmup_tokens}"
            )
        if self.decay not in LR_DECAY_CHOICES:
            raise ValueError(
                f"lr_decay must be one of {LR_DECAY_CHOICES}, got "
                f"{self.decay!r}"
            )
        if self.decay == "cosine" and self.budget is None:
            raise ValueError(
                "--lr-decay cosine needs --token-budget (the decay spans the "
                "token budget); there is no finite schedule to decay over "
                "otherwise"
            )
        if self.budget is not None:
            if self.budget < 1:
                raise ValueError(
                    f"token budget must be >= 1, got {self.budget}"
                )
            if self.warmup_tokens >= self.budget:
                raise ValueError(
                    f"warmup_tokens ({self.warmup_tokens}) must be strictly "
                    f"less than the token budget ({self.budget}) — the decay "
                    "phase would be empty"
                )
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError(
                f"min_lr_ratio must be in (0, 1], got {self.min_lr_ratio}"
            )

    def lr_at(self, tokens: int) -> float:
        """The LR for the step whose *pre-step* token count is ``tokens``."""
        t = max(int(tokens), 0)
        if self.warmup_tokens and t < self.warmup_tokens:
            return self.lr * t / self.warmup_tokens
        if self.decay == "cosine":
            assert self.budget is not None
            span = self.budget - self.warmup_tokens
            x = (t - self.warmup_tokens) / span if span > 0 else 1.0
            x = min(max(x, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * x))
            return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)
        return self.lr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.decay,
            "warmup_tokens": self.warmup_tokens,
            "token_budget": self.budget,
            "min_lr_ratio": self.min_lr_ratio,
        }


# ---------------------------------------------------------------------------
# Checkpoint selection by NUMERIC step (DELIVERABLE 5)
# ---------------------------------------------------------------------------
_STEP_CKPT_RE = re.compile(r"^step-(\d+)\.pt$")


def list_checkpoint_candidates(checkpoint_dir: str) -> List[str]:
    """``step-<N>.pt`` files under ``checkpoint_dir``, NUMERIC step desc.

    Regression-guarded: sorting must be numeric, never lexicographic
    (lexicographic order puts ``step-9.pt`` "newest" ahead of
    ``step-100.pt`` — the exact bug this replaces).
    """
    entries: List[tuple] = []
    for name in os.listdir(checkpoint_dir):
        m = _STEP_CKPT_RE.match(name)
        if m:
            entries.append((int(m.group(1)), os.path.join(checkpoint_dir, name)))
    entries.sort(key=lambda pair: pair[0], reverse=True)
    return [path for _, path in entries]


def validate_resume_checkpoint(ckpt: dict, preset: str) -> None:
    """Sanity checks a checkpoint must pass to resume from (shared by the
    explicit-file and directory-scan paths). Raises ``ValueError`` naming the
    failing contract — reused verbatim by the corrupt-fallback scanner.

    P5 (T4 engine pass): in addition to the pre-existing format/preset/param
    checks, the **tensor head** of the state dict is validated against the
    exact shapes a fresh model of the recorded config would have — a checkpoint
    whose tensors were truncated or length-mismatched (e.g. brain-damaged by an
    interrupted Drive upload) is rejected with a clear error BEFORE any
    ~386 MiB load/compare, and the directory scanner falls back to the previous
    valid checkpoint. Full tensor *values* are still verified implicitly by
    ``load_state_dict`` inside :func:`train_epochs`.
    """
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint has unsupported format {ckpt.get('format')!r}: "
            f"expected {CHECKPOINT_FORMAT!r}"
        )
    ckpt_cfg = ModelConfig(**ckpt["model_config"]).derive()
    ckpt_preset = resolve_preset(ckpt_cfg)
    if ckpt_preset != preset:
        raise ValueError(
            f"checkpoint is the {ckpt_preset} preset but --preset is "
            f"{preset!r} — the preset must match the resumed run"
        )
    exp_params, exp_vocab = CANONICAL_PRESETS[ckpt_preset]
    if int(ckpt["n_params"]) != exp_params:
        raise ValueError(
            f"checkpoint records {ckpt['n_params']:,} params but the canonical "
            f"{ckpt_preset} preset is exactly {exp_params:,} — corrupted or "
            "tampered checkpoint"
        )
    if ckpt.get("optimizer_state_dict") is None:
        raise ValueError(
            f"checkpoint has no optimizer_state_dict — it predates resume "
            "support (old v1 format); use a checkpoint written by the current "
            "script"
        )
    # ---- P5 tensor-head validation (shape-level state-dict sanity) ----
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint model_state_dict is empty or unreadable")
    expected = expected_state_dict_shapes(ckpt_cfg)
    if set(state.keys()) != set(expected.keys()):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            "checkpoint state_dict keys do not match the recorded model "
            f"config: missing={missing} extra={extra} \u2014 corrupted or "
            "config-drifted checkpoint"
        )
    for key, exp_shape in expected.items():
        actual = tuple(state[key].shape)
        if actual != tuple(exp_shape):
            raise ValueError(
                f"checkpoint tensor {key} has shape {actual}, expected "
                f"{tuple(exp_shape)} \u2014 truncated or mismatched state dict "
                "(e.g. an interrupted Drive upload); refusing to resume"
            )


# ---------------------------------------------------------------------------
# Run-metadata sidecar (DELIVERABLE 4)
# ---------------------------------------------------------------------------
def git_repo_state() -> Dict[str, Any]:
    """``{commit, branch, dirty}`` of the running code, best-effort.

    ``None`` when the repo is unavailable (e.g. a Colab checkout without
    ``.git``) — the run must never fail on git introspection.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        branch = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "-C", _REPO_ROOT, "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:  # pragma: no cover - environment dependent
        return {"commit": None, "branch": None, "dirty": None}


def build_run_metadata(
    *,
    preset: str,
    model_cfg: ModelConfig,
    n_params: int,
    training_config: Dict[str, Any],
    data_provenance: Dict[str, Any],
    tokenizer: Dict[str, Any],
    device: str,
    resumed_from: Optional[str] = None,
) -> Dict[str, Any]:
    """The reproducibility record written to ``train_run_metadata.json`` and
    embedded (the *same dict*) in every checkpoint payload."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema": RUN_METADATA_SCHEMA,
        "script": "scripts/train_oasst1.py",
        "preset": preset,
        "model_config": asdict(model_cfg),
        "n_params": int(n_params),
        "training_config": training_config,
        "data_provenance": data_provenance,
        "tokenizer": tokenizer,
        "git": git_repo_state(),
        "timestamps": {"started": now, "finished": None},
        "resumed_from": resumed_from,
        "resume_history": [resumed_from] if resumed_from else [],
    }


def write_run_metadata(out_dir: str, run_metadata: Dict[str, Any]) -> str:
    """Persist the sidecar (atomic) and return its absolute path."""
    path = os.path.join(out_dir, RUN_METADATA_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def merge_run_metadata(
    base: Dict[str, Any], *, resumed_from: str
) -> Dict[str, Any]:
    """Provenance merge for a resume: keep the ORIGINAL run's model/training
    config and data provenance (they are what produced the weights), stamp the
    new session's git state, timestamp, and resume trail."""
    merged = deepcopy(base)
    merged["git"] = git_repo_state()
    merged["timestamps"] = dict(merged.get("timestamps") or {})
    merged["timestamps"]["resumed_at"] = datetime.now(timezone.utc).isoformat()
    merged["resumed_from"] = os.path.abspath(resumed_from)
    history = list(merged.get("resume_history") or [])
    if os.path.abspath(resumed_from) not in history:
        history.append(os.path.abspath(resumed_from))
    merged["resume_history"] = history
    return merged


def capture_rng_state() -> dict:
    """Snapshot every RNG source (torch CPU/GPU, numpy, python random).

    Stored in each checkpoint so a resumed run continues with the exact RNG
    state the uninterrupted run would have had (the repo's determinism
    guarantees make resume bit-exact — asserted by the test suite).
    """
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot from :func:`capture_rng_state` (or its old pair)."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Step 1 — deterministic train/val split (disjoint, no leakage)
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    """Outcome of :func:`split_jsonl`."""

    train_path: str
    val_path: str
    total_docs: int
    train_docs: int
    val_docs: int
    seed: int


def split_jsonl(
    src_path: str,
    out_dir: str,
    ratio: float = 0.9,
    seed: int = 0,
    max_docs: Optional[int] = None,
) -> SplitResult:
    """Deterministically split one JSONL corpus into disjoint train/val files.

    Reads the source **once** into memory (the task scope is a small subset —
    a few hundred documents; the heavy tokenization/training paths afterwards
    stay streaming). Lines are preserved verbatim so the written files are
    byte-identical to the source records. The split is a fixed-seed shuffle of
    document indices, so the same ``(src, ratio, seed)`` always yields the same
    two files, and a document can never appear in both (no leakage).

    Raises ``ValueError`` if the corpus is too small to split (needs at least
    one train and one validation document).
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"corpus JSONL not found: {src_path}")
    docs: List[str] = []
    with open(src_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                docs.append(line)
                if max_docs is not None and len(docs) >= max_docs:
                    break
    if len(docs) < 2:
        raise ValueError(
            f"cannot split {src_path}: need at least 2 documents for a "
            f"train/val split, found {len(docs)}"
        )
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    n_train = max(1, round(len(docs) * ratio))
    if n_train >= len(docs):  # ratio too high: keep exactly one doc for val
        n_train = len(docs) - 1
    train_ids = set(idx[:n_train])
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as ft, open(
        val_path, "w", encoding="utf-8"
    ) as fv:
        for i, line in enumerate(docs):
            (ft if i in train_ids else fv).write(line + "\n")
    return SplitResult(
        train_path=train_path,
        val_path=val_path,
        total_docs=len(docs),
        train_docs=n_train,
        val_docs=len(docs) - n_train,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Step 2 — Talos-native byte-level BPE tokenizer (train split only, vocab 1024)
# ---------------------------------------------------------------------------
def train_tokenizer_for_run(
    train_path: str,
    tokenizer_path: str,
    *,
    preset: str = "tiny",
    num_merges: Optional[int] = None,
    minfreq: int = 2,
    max_docs: Optional[int] = None,
    max_chars: Optional[int] = None,
    text_field: str = "text",
) -> ByteLevelBPETokenizer:
    """Train a byte-level BPE on the train split and persist it.

    Uses the canonical preset's tokenizer config
    (:func:`configs.presets.preset_tokenizer_config` — the vocab-1024 contract
    shared by every canonical preset: 256 base bytes + 4 specials + up to 764
    merges), consumed **one document at a time** via the PR-#16 unique-word
    frequency table, so memory stays bounded. Exposes the ``num_merges`` /
    ``minfreq`` / ``max_docs`` / ``max_chars`` knobs. The artifact is written
    next to the checkpoints so eval/generation can load the exact same
    tokenizer later. The trained vocab is checked against the *preset model's*
    vocab size (``ALL_PRESETS[preset]().vocab_size``) so a tokenizer/model
    mismatch fails here with the model's real bound, regardless of which
    canonical preset is being trained.
    """
    config = preset_tokenizer_config(preset)
    model_vocab = ALL_PRESETS[preset]().vocab_size
    texts: Iterable[str] = iter_text_documents(
        [train_path], jsonl_field=text_field, on_invalid="die"
    )
    result = train_tokenizer(
        texts,
        config,
        minfreq=minfreq,
        num_merges=num_merges,
        max_docs=max_docs,
        max_chars=max_chars,
    )
    tokenizer = result.tokenizer
    if tokenizer.vocab_size > model_vocab:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size} exceeds the "
            f"{preset} model's {model_vocab} — config drift in "
            f"configs/presets.{preset}_tokenizer_config"
        )
    tokenizer.save(tokenizer_path)
    log.info(
        "trained tokenizer: vocab=%d (merges=%d, %d docs, %d chars) -> %s",
        tokenizer.vocab_size,
        tokenizer.merge_count,
        result.num_words,
        result.num_bytes,
        tokenizer_path,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Step 3 — canonical preset model with a hard param-count guard
# ---------------------------------------------------------------------------
def check_preset_compat(preset: str, cfg, n_params: int) -> None:
    """Fail fast if the model is not a registered canonical preset.

    Guards the hard requirement for whichever preset is being trained: its
    exact canonical parameter count at its exact canonical vocab size (from
    :data:`configs.canonical.CANONICAL_PRESETS`). Any drift in
    ``configs/presets`` is caught here *before* training starts. Every
    registered canonical preset is enforced through this same path — ``tiny``
    at exactly 254,272 params / vocab 1024, ``tiny_1m`` at exactly 1,000,320,
    ``tiny_10m`` at exactly 9,952,320, ``tiny_100m`` at exactly 96,482,304 —
    with no per-preset code anywhere.
    """
    expected, expected_vocab = CANONICAL_PRESETS[preset]
    problems: List[str] = []
    if cfg.vocab_size != expected_vocab:
        problems.append(
            f"vocab_size={cfg.vocab_size} (expected {expected_vocab})"
        )
    if n_params != expected:
        problems.append(
            f"param count={n_params:,} (expected exactly {expected:,})"
        )
    if problems:
        raise ValueError(
            f"{preset} preset config drift detected — refusing to train: "
            + "; ".join(problems)
            + f". The canonical {preset} preset is exactly {expected:,} params "
            f"at vocab_size {expected_vocab} (configs/presets.{preset}_config). "
            "Fix the preset before training — do not 'fix' this guard."
        )


def check_tiny_compat(cfg, n_params: int) -> None:
    """Fail fast if the model is not the owner-fixed canonical tiny preset.

    Guards the hard requirement: exactly 254,272 parameters at vocab_size 1024
    (hidden 64, 2 layers, 4 heads, 2 KV heads, dense FFN, seq 512). Any drift
    in ``configs/presets.tiny_config`` is caught here *before* training starts.
    (Registry-enforced via :func:`check_preset_compat` — no per-preset logic.)
    """
    check_preset_compat("tiny", cfg, n_params)


def build_preset_model(preset: str, attention_backend: str = "auto") -> TalosGPT:
    """Build a canonical preset model and assert its exact-param contract.

    ``attention_backend`` (T4 engine pass P1a): "auto" keeps the model factory's
    own routing (flash-attn if installed, else SDPA; the chunked plain path for
    ``attention_chunk_size > 0`` configs); any explicit choice builds that
    backend in-place and passes it to :class:`TalosGPT`.
    """
    if attention_backend not in ATTENTION_BACKEND_CHOICES:
        raise ValueError(
            f"--attention-backend must be one of {ATTENTION_BACKEND_CHOICES}, "
            f"got {attention_backend!r}"
        )
    cfg = ALL_PRESETS[preset]().derive()
    backend = (
        None
        if attention_backend == "auto"
        else build_attention_backend(attention_backend, chunk_size=cfg.attention_chunk_size)
    )
    model = TalosGPT(cfg, attention_backend=backend)
    check_preset_compat(preset, cfg, model.num_parameters())
    return model


def build_tiny_model() -> TalosGPT:
    """Build the canonical tiny model and assert the 254,272-param contract."""
    return build_preset_model("tiny")


# ---------------------------------------------------------------------------
# Step 4 — streaming training with per-epoch validation + checkpoints
# ---------------------------------------------------------------------------
@dataclass
class EpochRow:
    """One epoch's losses and the checkpoint written for it."""

    epoch: int
    steps: int
    global_step: int
    train_loss: float
    val_loss: Optional[float]
    checkpoint: str
    wall_s: float


@dataclass
class TrainingHistory:
    rows: List[EpochRow] = field(default_factory=list)
    tokens_processed: int = 0
    run_wall_s: float = 0.0
    #: token-budget stop state (additive; default = no budget -> False)
    budget_reached: bool = False
    token_budget: Optional[int] = None
    # --- numerical-stability state (hardening pass; always-on detection) ---
    #: structured records of every skipped NaN/Inf step (see train_epochs).
    bad_step_events: List[Dict[str, Any]] = field(default_factory=list)
    #: total bad steps across the whole run (all sessions).
    total_bad_steps: int = 0
    #: bad steps since the last good step (drives the abort policy).
    consecutive_bad_steps: int = 0
    #: pre-clip total grad norm of the last clipped step (None when --grad-clip
    #: is off — the default).
    last_grad_norm: Optional[float] = None
    #: loss of the most recent good step (the epoch-mean counterpart is in the
    #: rows; this is the last single-step value, useful in abort reports).
    last_good_loss: Optional[float] = None
    #: most recent intra-epoch (--save-every-tokens) checkpoint path, if any.
    last_periodic_checkpoint: Optional[str] = None
    #: set when the run aborts after K consecutive bad steps.
    aborted: bool = False
    abort_reason: Optional[str] = None
    #: TOTAL attempted steps across the run (good + bad). Equals the executed-
    #: step count on healthy runs; on aborts it is the honest "how far did the
    #: loop get" number (executed = attempts - total_bad_steps).
    attempts: int = 0
    #: T4 engine pass (P4): per-interval performance records (see
    #: :class:`IntervalStats`), oldest first.
    interval_logs: List[Dict[str, Any]] = field(default_factory=list)
    #: cumulative wall time spent inside save_checkpoint (P4/P5).
    checkpoint_save_s: float = 0.0

    def row(self, epoch: int) -> EpochRow:
        return next(r for r in self.rows if r.epoch == epoch)


class BadStepsAbort(RuntimeError):
    """Raised by :func:`train_epochs` after ``--max-consecutive-bad-steps``
    consecutive NaN/Inf steps.

    Carries the partially-built :class:`TrainingHistory` (bad-step events,
    counters, abort reason) so :func:`train_run` can still write an honest
    ``metrics.json`` + finish-stamped ``train_run_metadata.json`` sidecar
    before re-raising. The last safety checkpoint holds the last-good weights
    and is fully resume-able (``--resume <out-dir>`` picks it as the
    numeric-newest valid checkpoint).
    """

    def __init__(self, message: str, *, history: TrainingHistory) -> None:
        super().__init__(message)
        self.history = history


def evaluate(
    model: TalosGPT,
    dataset: StreamingTokenizedDataset,
    *,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Optional[float]:
    """Mean causal-LM loss over a streamed (never materialized) dataset.

    Returns ``None`` when the stream contains no full batches (e.g. an empty
    validation split). The model is left in ``train()`` mode afterwards. Each
    batch is token-id-range-checked at assembly (before it reaches the model)
    so a drifted tokenizer fails here with batch context, never as an
    ``IndexError``/CUDA assert inside the model.
    """
    was_training = model.training
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    vocab = model.config.vocab_size
    total, count, steps = 0.0, 0, 0
    try:
        with torch.no_grad():
            for batch in dataset:
                if max_steps is not None and steps >= max_steps:
                    break
                x = batch.to(device).long()
                validate_token_ids(x, vocab, where="validation batch")
                logits, _ = model(x[:, :-1])
                n = x[:, 1:].numel()
                total += float(loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)))
                count += n
                steps += 1
    finally:
        if was_training:
            model.train()
    return (total / count) if count else None


def _fsync_file(path: str) -> None:
    """fsync a file, degrading gracefully on filesystems that refuse (Drive).

    Best-effort durability: the pre-existing writer was atomic (tmp+rename)
    but never flushed the page cache — on a Colab Drive mount a power/link
    loss could expose a rename whose data was never flushed. fsync closes
    that gap; an unsupported FS (e.g. Drive FUSE returning EINVAL/ENOTSUP)
    logs once and continues — durability degrades to the old behavior, never
    an error that kills training.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        log.warning(
            "fsync unsupported on this filesystem (%s) — checkpoint "
            "durability falls back to atomic rename only", path,
        )


def save_checkpoint(
    path: str,
    model: TalosGPT,
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    tokenizer_path: str,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    rng_state: Optional[dict] = None,
    epoch: Optional[int] = None,
    tokens_consumed: Optional[int] = None,
    run_metadata: Optional[dict] = None,
    tokenizer_fingerprint: Optional[str] = None,
    last_grad_norm: Optional[float] = None,
    # T4 engine pass (P5): staging + fsync
    stager: Optional["CheckpointStager"] = None,
    fsync: bool = True,
) -> None:
    """One checkpoint artifact: weights + config + step + losses + tokenizer info.

    Format ``talos-training-checkpoint-v1`` stays **backward compatible**: the
    original keys are unchanged and the resume-enabling keys (``optimizer_state_dict``,
    ``rng_state``, ``epoch``, ``tokenizer_fingerprint``) are additive, so
    checkpoints written before this change still load for eval/generation, and
    new checkpoints load in any old reader.

    This is the trainer's SINGLE checkpoint writer — epoch-end checkpoints,
    intra-epoch ``--save-every-tokens`` checkpoints and NaN/Inf safety
    checkpoints all go through it (no duplicated save logic anywhere).

    The write is **atomic** (serialize to ``<path>.tmp``, then ``os.replace``):
    a crash mid-save can never leave a truncated ``step-<N>.pt`` that the
    numeric-newest resume scan would otherwise select. ``last_grad_norm`` is
    the pre-clip total grad norm of the most recent clipped step
    (``--grad-clip`` > 0; ``None`` when clipping is off or before the first
    clipped step).
    """
    cfg = model.config
    if tokenizer_fingerprint is None:
        tokenizer_fingerprint = (
            tokenizer_file_sha256(tokenizer_path) if tokenizer_path else None
        )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "model_config": asdict(cfg),
        "n_params": model.num_parameters(),
        "vocab_size": cfg.vocab_size,
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        #: absolute path so the checkpoint is self-describing from anywhere;
        #: the tokenizer artifact itself lives next to the checkpoint in out_dir.
        "tokenizer_path": os.path.abspath(tokenizer_path) if tokenizer_path else None,
        #: sha256 of the serialized tokenizer.json — content identity, so a
        #: swapped/re-trained sidecar is caught on load (audit P0 fix 4). On
        #: the packed-corpus path without a tokenizer sidecar, this is the
        #: manifest-recorded sha256 of the corpus's tokenizer.
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_state_dict": model.state_dict(),
        # Resume-enabling state (additive, v1 format): optimizer + RNG + counters.
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "rng_state": rng_state,
        "epoch": epoch,
        # Token-budget accounting (additive): tokens consumed SINCE RUN START,
        # including every token consumed by earlier resumed sessions.
        "tokens_consumed": None if tokens_consumed is None else int(tokens_consumed),
        # Run-metadata sidecar (additive; the SAME dict as train_run_metadata.json).
        "run_metadata": run_metadata,
        #: pre-clip total grad norm of the most recent clipped step (D2) —
        #: ``None`` when --grad-clip is off. The *flag* lives in
        #: run_metadata.training_config; this is the dynamic value.
        "last_grad_norm": None if last_grad_norm is None else float(last_grad_norm),
    }
    # P5 (atomic + durable, Drive-aware). When a local staging dir is
    # configured the write goes to the LOCAL filesystem (tmp + fsync +
    # rename), and a background thread copies the finished file to the final
    # out_dir (possibly Drive) — Drive latency never blocks training.
    final_path = path
    if stager is not None and stager.staging_dir:
        path = os.path.join(stager.staging_dir, os.path.basename(path))
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    if fsync:
        _fsync_file(tmp_path)
    os.replace(tmp_path, path)
    if stager is not None and os.path.abspath(path) != os.path.abspath(final_path):
        stager.enqueue(path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (pure dict)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def train_epochs(
    model: TalosGPT,
    train_ds: StreamingTokenizedDataset,
    val_ds: Optional[StreamingTokenizedDataset],
    *,
    out_dir: str,
    tokenizer_path: str,
    lr: float,
    epochs: int,
    device: torch.device,
    seed: int,
    max_steps_per_epoch: Optional[int] = None,
    val_max_steps: Optional[int] = None,
    resume_from: Optional[dict] = None,
    token_budget: Optional[int] = None,
    warmup_tokens: int = 0,
    lr_decay: str = "none",
    run_metadata: Optional[dict] = None,
    save_every_tokens: int = 0,
    grad_clip: float = 0.0,
    max_consecutive_bad_steps: int = 3,
    # --- T4 training-engine pass (additive; defaults reproduce old behavior) ---
    amp: str = "none",
    fused_optim: bool = False,
    log_every_steps: int = 50,
    ckpt_stager: Optional[CheckpointStager] = None,
    ckpt_fsync: bool = True,
) -> TrainingHistory:
    """Train the model on the streamed train split, epoch by epoch.

    (The pre-existing docstring — resume semantics, token budget, LR schedule,
    numerical-stability guards — is unchanged; see the module docstring and
    docs/trainer.md. Added by the T4 engine pass:

    * **AMP** (``amp="fp16"``): the train forward/backward run under fp16
      autocast with a GradScaler; ``scaler.unscale_(opt)`` runs BEFORE the
      gradient-finiteness guard and before ``--grad-clip`` so the pre-clip
      total norm is the **unscaled** norm, and a scaling-overflow step (inf
      surfaced by unscale_) is caught by the SAME guard and counts toward
      ``--max-consecutive-bad-steps``. Validation always runs fp32 (loss
      comparability). With ``amp="none"`` the loop is numerically identical to
      the pre-upgrade trainer (asserted by the AMP tests).
    * **Logging hygiene** (``log_every_steps``): per-step loss is accumulated
      on device and synced only at interval boundaries; the interval line
      reports step, loss, lr, pre-clip grad norm, tokens, tok/s, steps/s,
      elapsed, phase timers (data/fwd/bwd/optim), checkpoint save duration,
      effective batch and VRAM. The only retained per-step syncs are the
      always-on safety guards (loss/grad finiteness), which the hardening pass
      depends on.
    * **Checkpoint staging** (``ckpt_stager``/``ckpt_fsync``): see
      :class:`CheckpointStager` — local tmp+fsync+rename first, background
      copy to the final out_dir (Drive) with a run-end drain.
    """
    set_seed(seed)
    schedule = TokenSchedule(lr=lr, warmup_tokens=warmup_tokens, decay=lr_decay,
                             budget=token_budget)
    opt = build_optimizer(model, lr, fused=fused_optim, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    vocab = model.config.vocab_size
    history = TrainingHistory(token_budget=token_budget)
    # AMP machinery (P1b). make_grad_scaler enables only on CUDA; on CPU every
    # scaler call is a transparent no-op so the integration is one code path.
    amp_autocast = make_amp_autocast(amp, device)
    scaler = make_grad_scaler(amp, device)
    amp_fp16 = scaler.is_enabled()
    start_epoch, global_step = 1, 0
    tokens_consumed = 0
    if resume_from is not None:
        if resume_from.get("optimizer_state_dict") is None:
            raise ValueError(
                "cannot resume: checkpoint has no optimizer_state_dict — it "
                "was written before resume support (old v1 format); retrain "
                "from step 0 or re-save with the current script"
            )
        model.load_state_dict(resume_from["model_state_dict"])
        opt.load_state_dict(resume_from["optimizer_state_dict"])
        restore_rng_state(resume_from["rng_state"])
        # Keep the caller's --lr authoritative (AdamW.state_dict records the
        # lr at save time under the 'lr' group key).
        for group in opt.param_groups:
            group["lr"] = lr
        start_epoch = int(resume_from.get("epoch", 0)) + 1
        global_step = int(resume_from["step"])
        recorded_tokens = resume_from.get("tokens_consumed")
        if recorded_tokens is None:
            if token_budget is not None:
                # Old checkpoint (pre-token-budget): no exact counter. The only
                # deterministic schedule the script has ever produced is
                # batch*(seq-1) tokens per step with constant batch/seq, so the
                # estimate is exact for every legacy run — warn anyway.
                per_step = train_ds.batch_size * (train_ds.seq_len - 1)
                recorded_tokens = global_step * per_step
                log.warning(
                    "resume checkpoint %s has no tokens_consumed counter; "
                    "estimating %d tokens from %d steps at batch*seq-1=%d — "
                    "recorded exactly in the next checkpoint",
                    "step-%d.pt" % global_step, recorded_tokens, global_step,
                    per_step,
                )
            else:
                recorded_tokens = 0
        tokens_consumed = int(recorded_tokens)
        history.tokens_processed = tokens_consumed
        if start_epoch > epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1} "
                f"(>= --epochs {epochs}) — nothing left to train; raise --epochs"
            )
        log.info(
            "resuming from step %d (epoch %d completed, %d tokens consumed) — "
            "continuing to epoch %d",
            global_step, start_epoch - 1, tokens_consumed, epochs,
        )
        if token_budget is not None and tokens_consumed >= token_budget:
            log.warning(
                "token budget %d already consumed at resume (%d) — no further "
                "training needed; the run will finish without new steps",
                token_budget, tokens_consumed,
            )
            history.budget_reached = True
    # Tokenizer identity for checkpoint payloads is fixed for the whole run;
    # compute once so the periodic/safety/epoch-end writers all share it.
    ckpt_fp: Optional[str] = None
    if tokenizer_path:
        ckpt_fp = tokenizer_file_sha256(tokenizer_path)
    elif run_metadata is not None:
        ckpt_fp = (run_metadata.get("tokenizer") or {}).get("sha256")
    # Periodic-checkpoint cadence: absolute multiples of save_every_tokens
    # since run start (restored on resume), so a disconnect loses at most ~N
    # newly-consumed tokens no matter how many sessions the run has had.
    last_periodic_multiple = (
        tokens_consumed // save_every_tokens if save_every_tokens > 0 else 0
    )

    # --- P4 observability state (interval accumulators) ---
    interval = IntervalStats()
    interval.t0 = time.monotonic()
    # device-side loss accumulators: synced only at interval/epoch/checkpoint.
    epoch_acc: Optional[torch.Tensor] = None
    epoch_count: int = 0
    interval_acc: Optional[torch.Tensor] = None
    interval_count: int = 0
    last_eff_batch: int = 0

    def _vram_mb() -> Tuple[Optional[float], Optional[float]]:
        if device.type != "cuda":
            return None, None
        try:
            return (
                torch.cuda.max_memory_allocated(device) / 2**20,
                torch.cuda.memory_allocated(device) / 2**20,
            )
        except Exception:  # pragma: no cover - defensive
            return None, None

    def _save_ckpt(
        ckpt_path: str, step_no: int, ckpt_loss: float,
        ckpt_val: Optional[float], *,
        rng: Optional[dict] = None,
        ckpt_epoch: Optional[int] = None,
        ckpt_tokens: Optional[int] = None,
        ckpt_norm: Optional[float] = None,
    ) -> None:
        """One save through the shared writer with wall-time capture (P4/P5)."""
        nonlocal interval
        t0_ck = time.monotonic()
        save_checkpoint(
            ckpt_path, model, step_no, ckpt_loss, ckpt_val, tokenizer_path,
            optimizer=opt, rng_state=rng, epoch=ckpt_epoch,
            tokens_consumed=ckpt_tokens, run_metadata=run_metadata,
            tokenizer_fingerprint=ckpt_fp, last_grad_norm=ckpt_norm,
            stager=ckpt_stager, fsync=ckpt_fsync,
        )
        dt = time.monotonic() - t0_ck
        interval.ckpt_s += dt
        history.checkpoint_save_s += dt

    def handle_bad_step(
        step_no: int,
        tokens_before: int,
        tensor: str,
        stat: str,
        loss_value: Optional[float],
    ) -> None:
        """One NaN/Inf step: skip the optimizer step entirely, persist the
        last-good state as a safety checkpoint, record the event, and abort
        once ``max_consecutive_bad_steps`` consecutive bad steps accumulate."""
        nonlocal interval
        history.consecutive_bad_steps += 1
        history.total_bad_steps += 1
        ckpt_path = os.path.join(out_dir, f"step-{step_no}.pt")
        ckpt_loss = (
            history.last_good_loss
            if history.last_good_loss is not None
            else float("nan")
        )
        _save_ckpt(
            ckpt_path, step_no, ckpt_loss, None,
            rng=capture_rng_state(), ckpt_epoch=epoch, ckpt_tokens=tokens_before,
            ckpt_norm=history.last_grad_norm,
        )
        event = {
            "step": step_no,
            "tokens_consumed": int(tokens_before),
            "tensor": tensor,
            "stat": stat,
            "loss": None if loss_value is None else float(loss_value),
            "consecutive_bad_steps": history.consecutive_bad_steps,
            "checkpoint": os.path.basename(ckpt_path),
        }
        history.bad_step_events.append(event)
        log.error(
            "BAD STEP epoch=%d step=%d: %s is %s (loss=%s), %d tokens consumed "
            "before it — optimizer step SKIPPED, safety checkpoint %s written "
            "(consecutive bad steps %d; abort at %d)",
            epoch, step_no, tensor, stat,
            "n/a" if loss_value is None else f"{loss_value:.6g}",
            tokens_before, os.path.basename(ckpt_path),
            history.consecutive_bad_steps,
            max_consecutive_bad_steps if max_consecutive_bad_steps > 0 else 0,
        )
        if (
            max_consecutive_bad_steps > 0
            and history.consecutive_bad_steps >= max_consecutive_bad_steps
        ):
            history.aborted = True
            history.abort_reason = (
                f"{history.consecutive_bad_steps} consecutive bad steps (last: "
                f"{tensor} is {stat} at step {step_no}); run aborted — "
                f"recoverable last-good state saved at "
                f"{os.path.basename(ckpt_path)}"
            )
            raise BadStepsAbort(history.abort_reason, history=history)

    def flush_interval(force: bool = False) -> None:
        """Sync the device accumulators once and emit the interval line (P4)."""
        nonlocal interval, interval_acc, interval_count
        if interval.steps <= 0 or not (force or log_every_steps <= 0
                                       or interval.steps >= log_every_steps):
            return
        total_s = time.monotonic() - interval.t0
        interval.total_s = total_s
        if interval_count > 0 and interval_acc is not None:
            interval.loss = float(interval_acc / interval_count)
        else:
            interval.loss = None
        if grad_clip > 0:
            interval.grad_norm = history.last_grad_norm
        interval.vram_peak_mb, interval.vram_used_mb = _vram_mb()
        record = interval.asdict()
        history.interval_logs.append(record)
        tok_s = record["tok_s"]
        steps_s = record["steps_s"]
        log.info(
            "step %d loss %.4f lr %.2e grad_norm %s tokens %d tok/s %s steps/s %s "
            "elapsed %.1fs | data %.1fms fwd %.1fms bwd %.1fms optim %.1fms "
            "ckpt %.1fms | vram_peak %sMiB vram_used %sMiB eff_batch %d",
            global_step, interval.loss if interval.loss is not None else float("nan"),
            opt.param_groups[0]["lr"],
            "n/a" if interval.grad_norm is None else f"{interval.grad_norm:.4g}",
            tokens_consumed,
            "n/a" if tok_s is None else f"{tok_s:.0f}",
            "n/a" if steps_s is None else f"{steps_s:.3f}",
            total_s,
            record["data_s"] * 1e3 / max(1, interval.steps),
            record["fwd_s"] * 1e3 / max(1, interval.steps),
            record["bwd_s"] * 1e3 / max(1, interval.steps),
            record["optim_s"] * 1e3 / max(1, interval.steps),
            record["ckpt_s"] * 1e3 / max(1, interval.steps),
            "n/a" if interval.vram_peak_mb is None else f"{interval.vram_peak_mb:.0f}",
            "n/a" if interval.vram_used_mb is None else f"{interval.vram_used_mb:.0f}",
            last_eff_batch,
        )
        interval = IntervalStats()
        interval.t0 = time.monotonic()
        interval_acc = None
        interval_count = 0

    t_run = time.monotonic()
    for epoch in range(start_epoch, epochs + 1):
        if history.budget_reached:
            break
        model.train()
        steps = 0
        t_epoch = time.monotonic()
        for batch in train_ds:
            t_iter0 = time.monotonic()
            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break
            if history.budget_reached:
                break
            # P1d: pinned H2D handoff (non_blocking) — no-op on CPU paths.
            if device.type == "cuda" and batch.is_pinned():
                x = batch.to(device, non_blocking=True)
                if x.dtype != torch.long:
                    x = x.long()
            else:
                x = batch.to(device).long()
            validate_token_ids(x, vocab, where="train batch")
            # Every ATTEMPT consumes a step number (a bad step is an attempted
            # step: it gets an event + a safety checkpoint but no weight update
            # and no tokens) — checkpoint files stay unique and monotonic.
            global_step += 1
            steps += 1
            history.attempts += 1
            # LR schedule: the step's LR is a function of tokens consumed so
            # far (pre-step). Fixed-LR runs (no warmup, decay "none") always
            # resolve to the caller's lr — bit-identical to the old trainer.
            opt.param_groups[0]["lr"] = schedule.lr_at(tokens_consumed)
            interval.data_s += time.monotonic() - t_iter0
            # ---- forward + loss under the AMP context (P1b) ----
            t_fwd = time.monotonic()
            with amp_autocast:
                logits, _ = model(x[:, :-1])
                loss = loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
            interval.fwd_s += time.monotonic() - t_fwd
            # ---- loss finiteness BEFORE backward (always on) ----
            if not torch.isfinite(loss):
                stat = "nan" if torch.isnan(loss.detach()) else "inf"
                handle_bad_step(
                    global_step, tokens_consumed, "loss", stat,
                    float(loss.detach()),
                )
                continue
            opt.zero_grad(set_to_none=True)
            # ---- backward (scaled under AMP) ----
            t_bwd = time.monotonic()
            if amp_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            interval.bwd_s += time.monotonic() - t_bwd
            # ---- AMP unscale BEFORE the guard and BEFORE clip ----
            # unscale_ surfaces scaling overflows as inf in the grads, so the
            # always-on finiteness guard below also catches overflow steps and
            # counts them toward --max-consecutive-bad-steps (P1b/P3).
            if amp_fp16:
                scaler.unscale_(opt)
            # ---- gradient finiteness AFTER backward (always on) ----
            bad_param: Optional[str] = None
            bad_grad: Optional[torch.Tensor] = None
            for pname, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_param, bad_grad = pname, p.grad
                    break
            if bad_param is not None:
                stat = "nan" if torch.isnan(bad_grad).any() else "inf"
                handle_bad_step(
                    global_step, tokens_consumed, f"grad:{bad_param}", stat, None
                )
                if amp_fp16:
                    scaler.update()  # overflow adjust even though step skipped
                continue
            # ---- gradient clipping (records the PRE-clip total norm) ----
            if grad_clip > 0:
                history.last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                )
            # ---- optimizer step (AMP-aware: skips on found_inf) ----
            t_opt = time.monotonic()
            if amp_fp16:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            interval.optim_s += time.monotonic() - t_opt
            history.consecutive_bad_steps = 0
            # ---- device-side loss accumulation (no per-step .item()) ----
            loss_det = loss.detach()
            epoch_acc = loss_det.clone() if epoch_acc is None else epoch_acc + loss_det
            epoch_count += 1
            interval_acc = (
                loss_det.clone() if interval_acc is None else interval_acc + loss_det
            )
            interval_count += 1
            history.last_good_loss = float(loss_det)
            step_tokens = int(x[:, 1:].numel())
            last_eff_batch = step_tokens
            tokens_consumed += step_tokens
            history.tokens_processed = tokens_consumed
            interval.steps += 1
            interval.tokens += step_tokens
            # ---- intra-epoch periodic checkpoint (--save-every-tokens) ----
            if save_every_tokens > 0:
                multiple = tokens_consumed // save_every_tokens
                if multiple > last_periodic_multiple:
                    last_periodic_multiple = multiple
                    pckpt_path = os.path.join(
                        out_dir, f"step-{global_step}.pt"
                    )
                    mean = (
                        float(epoch_acc / epoch_count)
                        if epoch_acc is not None and epoch_count > 0
                        else float("nan")
                    )
                    _save_ckpt(
                        pckpt_path, global_step, mean, None,
                        rng=capture_rng_state(), ckpt_epoch=epoch,
                        ckpt_tokens=tokens_consumed,
                        ckpt_norm=history.last_grad_norm,
                    )
                    history.last_periodic_checkpoint = pckpt_path
                    log.info(
                        "periodic checkpoint %s (epoch %d step %d, %d tokens "
                        "consumed)",
                        os.path.basename(pckpt_path), epoch, global_step,
                        tokens_consumed,
                    )
            flush_interval()
            if token_budget is not None and tokens_consumed >= token_budget:
                history.budget_reached = True
                break
        flush_interval(force=True)
        if steps == 0 and history.budget_reached:
            # Budget was already exhausted before this epoch (resume case):
            # nothing new was trained — no val pass, no checkpoint overwrite.
            break
        train_loss = (
            float(epoch_acc / epoch_count)
            if epoch_acc is not None and epoch_count > 0
            else float("nan")
        )
        val_loss = (
            evaluate(model, val_ds, device=device, max_steps=val_max_steps)
            if val_ds is not None
            else None
        )
        ckpt_path = os.path.join(out_dir, f"step-{global_step}.pt")
        _save_ckpt(
            ckpt_path, global_step, train_loss, val_loss,
            rng=capture_rng_state(), ckpt_epoch=epoch,
            ckpt_tokens=tokens_consumed, ckpt_norm=history.last_grad_norm,
        )
        row = EpochRow(
            epoch=epoch,
            steps=steps,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint=ckpt_path,
            wall_s=time.monotonic() - t_epoch,
        )
        history.rows.append(row)
        log.info(
            "epoch %d/%d: steps=%d train_loss=%.4f val_loss=%s checkpoint=%s",
            epoch, epochs, steps, train_loss,
            "n/a" if val_loss is None else f"{val_loss:.4f}",
            ckpt_path,
        )
        epoch_acc = None
        epoch_count = 0
    history.run_wall_s = time.monotonic() - t_run
    return history


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------
def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.train_oasst1",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--data", default=None,
                   help="OASST1-shaped JSONL corpus (any path; real or synthetic). "
                        "Exactly one of --data / --packed-dir is required.")
    p.add_argument("--packed-dir", default=None, metavar="DIR",
                   help="train from packed-token .npy shards + manifest.json "
                        "produced by scripts/prepare_corpus.py (rows are used "
                        "verbatim — no re-tokenization). The manifest is "
                        "validated (format, dtype, seq, per-shard row counts, "
                        "tokenizer sha256 vs --tokenizer-json, vocab bounds) "
                        "and val rows come from the val shards. Exactly one of "
                        "--data / --packed-dir is required.")
    p.add_argument("--out-dir", required=True, help="run directory (split, tokenizer, checkpoints, metrics)")
    p.add_argument("--seed", type=int, default=0, help="fixed seed for split + training (default 0)")
    p.add_argument("--preset", default="tiny",
                   help=f"canonical preset to train ({', '.join(sorted(CANONICAL_PRESETS))}; default tiny)")
    p.add_argument("--resume", default=None, metavar="step-<N>.pt | DIR",
                   help="resume training from this checkpoint — a step-<N>.pt "
                        "FILE, or a DIRECTORY of checkpoints (the numerically "
                        "NEWEST valid one wins; corrupt/unloadable candidates "
                        "are skipped with a warning and the next-newest is "
                        "tried). Weights + optimizer + RNG + step/epoch + "
                        "tokens-consumed counters are restored; the tokenizer "
                        "fingerprint must match; --epochs is then the target "
                        "TOTAL. The split + tokenizer are NOT re-trained on "
                        "resume — the checkpoint's sidecar tokenizer is "
                        "verified and reused.")
    # split
    p.add_argument("--split-ratio", type=float, default=0.9, help="train fraction (default 0.9; JSONL path only)")
    p.add_argument("--split-max-docs", type=int, default=None,
                   help="cap how many source docs enter the split (small-subset guard; JSONL path only)")
    # tokenizer (Talos-native BPE trainer, PR #16 knobs)
    p.add_argument("--bpe-num-merges", type=int, default=None,
                   help="exact BPE merges (default: vocab-1024 budget = 764, stopped by minfreq)")
    p.add_argument("--bpe-minfreq", type=int, default=2, help="BPE min pair frequency (default 2)")
    p.add_argument("--bpe-max-docs", type=int, default=None, help="BPE training doc cap")
    p.add_argument("--bpe-max-chars", type=int, default=None, help="BPE training char cap")
    p.add_argument("--tokenizer-json", default=None, metavar="path/to/tokenizer.json",
                   help="load an EXISTING Talos tokenizer.json instead of training "
                        "BPE from --data (the BPE-training path stays the default "
                        "when this is omitted). The file is copied into "
                        "out_dir/tokenizer.json and its sha256 fingerprint is "
                        "recorded in every checkpoint + metrics.json; on --resume "
                        "the provided file's fingerprint must match the "
                        "checkpoint's recorded one. On --packed-dir its sha256 "
                        "must also match the manifest's recorded tokenizer "
                        "sha256 (the rows were packed with that tokenizer). "
                        "When omitted on the packed path, the manifest-recorded "
                        "path is used if it exists on this machine, else the "
                        "run proceeds without a tokenizer sidecar (identity "
                        "still recorded from the manifest).")
    # training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seq", type=int, default=None,
                   help="sequence length per row. Default: 64 for the JSONL "
                        "path; for --packed-dir the manifest's seq_len (an "
                        "explicit --seq that differs from the manifest is an "
                        "error — rows are used verbatim).")
    p.add_argument("--batch", type=int, default=4, help="batch size")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--token-budget", type=int, default=None, metavar="N",
                   help="stop training when the token budget N is consumed "
                        "(tokens since RUN START, including resumed tokens "
                        "restored from checkpoints). The partial epoch still "
                        "gets validation + a checkpoint. Mutually compatible "
                        "with --epochs (whichever limit comes first wins).")
    p.add_argument("--warmup-tokens", type=int, default=0, metavar="W",
                   help="linear LR warmup 0 -> --lr over the first W tokens "
                        "(default 0 = no warmup, fixed LR)")
    p.add_argument("--lr-decay", choices=LR_DECAY_CHOICES, default="none",
                   help="LR decay after warmup: 'none' (default, fixed LR) or "
                        "'cosine' — cosine from --lr down to 10%% of it over "
                        "--token-budget (requires --token-budget).")
    p.add_argument("--save-every-tokens", type=int, default=0, metavar="N",
                   help="write an intra-epoch checkpoint every N consumed "
                        "tokens (0 = the default epoch-end-only cadence). Same "
                        "v1 format + embedded run metadata as epoch-end "
                        "checkpoints, through the same atomic writer; cadence "
                        "is keyed to absolute multiples of N since run start "
                        "(restored on resume), so a disconnect loses at most "
                        "~N newly-consumed tokens. For a T4 run at ~2K "
                        "tok/s, N = one epoch's tokens is the natural value.")
    p.add_argument("--grad-clip", type=float, default=0.0, metavar="N",
                   help="max_grad_norm for torch.nn.utils.clip_grad_norm_ "
                        "after backward (0 = off, the default — the optimizer "
                        "step is byte-identical to the pre-hardening trainer). "
                        "The PRE-clip total grad norm of each clipped step is "
                        "recorded in metrics.json and in every subsequent "
                        "checkpoint; the setting is recorded in the run "
                        "metadata's gradient_clip_type.")
    p.add_argument("--max-consecutive-bad-steps", type=int, default=3,
                   metavar="K",
                   help="abort the run with a recoverable checkpoint after K "
                        "consecutive NaN/Inf steps (default 3; 0 = never "
                        "abort — bad steps are still skipped and safety-"
                        "checkpointed). NaN/Inf loss + gradient detection is "
                        "ALWAYS on: a bad step skips the optimizer step "
                        "entirely, logs a structured event (step, tokens, "
                        "tensor, stat) into metrics.json, and immediately "
                        "writes a safety checkpoint of the last-good state.")
    p.add_argument("--max-steps-per-epoch", type=int, default=None,
                   help="cap steps per epoch (CI / smoke runs)")
    p.add_argument("--val-max-steps", type=int, default=None,
                   help="cap validation batches (CI / smoke runs)")
    p.add_argument("--device", default=None, help="compute device (default: auto)")
    # T4 training-engine flags (P1; additive — defaults reproduce old behavior)
    p.add_argument(
        "--attention-backend", choices=ATTENTION_BACKEND_CHOICES, default="auto",
        help="attention execution backend: 'auto' (default) prefers flash-attn "
             "when installed, else SDPA (torch scaled_dot_product_attention — "
             "the full-causal path materializes NO -inf mask, which removes the "
             "fp16 overflow root cause); 'plain' = the legacy functional path "
             "(the T4 A/B 'old' lane); 'sdpa' / 'flash' force a specific engine.",
    )
    p.add_argument(
        "--amp", choices=AMP_CHOICES, default="none",
        help="mixed precision: 'none' (fp32, default) or 'fp16' (autocast + "
             "GradScaler; requires the SDPA attention backend to avoid the "
             "recorded -inf Half overflow). Validation always runs fp32.",
    )
    p.add_argument(
        "--fused-optim", action="store_true",
        help="use the fused CUDA AdamW kernel (P1c). CUDA-only; errors on CPU.",
    )
    p.add_argument(
        "--pin-memory", action="store_true",
        help="pin the packed-batch pool and hand batches to the device with "
             "non_blocking H2D (P1d).",
    )
    p.add_argument(
        "--prefetch", type=int, default=0, metavar="N",
        help="prefetch N batches on a background thread (P1e); 0 = the classic "
             "synchronous data path. Enables cached shard mmaps + reusable "
             "batch buffers.",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="torch.compile() the model (P1f). Default off; changes numerics "
             "slightly — never combine with bit-exact resume expectations.",
    )
    p.add_argument(
        "--log-every-steps", type=int, default=50, metavar="N",
        help="per-interval observability: accumulate losses/timers on device "
             "and emit one LINE + sidecar record every N steps (P4; 0 = "
             "epoch-end only).",
    )
    p.add_argument(
        "--ckpt-staging-dir", default=None, metavar="DIR",
        help="write checkpoints to this LOCAL directory first (atomic "
             "tmp+fsync+rename on the local FS); a background thread copies "
             "them to --out-dir, so Drive latency never blocks training (P5).",
    )
    p.add_argument(
        "--no-ckpt-fsync", action="store_true",
        help="disable the per-checkpoint fsync (P5). Default: fsync on, with "
             "graceful degradation on filesystems that do not support it.",
    )
    return p


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _verify_and_load_tokenizer(
    path: str, expected_sha: Optional[str], *, label: str
) -> ByteLevelBPETokenizer:
    """Load a tokenizer.json after a sha256 identity check (loud on mismatch)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = tokenizer_file_sha256(path)
    if expected_sha and actual != expected_sha:
        raise ValueError(
            f"{label} identity mismatch: expected sha256 {expected_sha} "
            f"but {path} hashes to {actual} — refusing to use the wrong "
            "tokenizer"
        )
    return ByteLevelBPETokenizer.from_file(path)


def _write_final_artifacts(
    out_dir: str, run_metadata: Dict[str, Any], metrics: dict
) -> None:
    """Stamp the finish time into the sidecar, then persist sidecar + metrics.

    Shared by the clean-finish and :class:`BadStepsAbort` paths so an aborted
    run leaves exactly the same artifacts (plus its bad-step records + abort
    reason) as a clean one. Checkpoints carry the START-time dict — only
    ``timestamps.finished`` is filled in here, after training.
    """
    run_metadata.setdefault("timestamps", {})["finished"] = (
        datetime.now(timezone.utc).isoformat()
    )
    write_run_metadata(out_dir, run_metadata)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")


def _assemble_performance(history: TrainingHistory) -> Dict[str, Any]:
    """Steady-state throughput + phase-time split from the last interval (P4).

    The last full log interval is the best single-lane estimate of maintained
    throughput (warm-up/checkpoint-skewed steps are excluded); run-wide totals
    are reported alongside.
    """
    last = history.interval_logs[-1] if history.interval_logs else None
    run_tokens = history.tokens_processed
    run_s = history.run_wall_s
    intervals = history.interval_logs
    return {
        "last_interval": last,
        "num_intervals": len(intervals),
        "steady_state_tok_s": (
            last["tok_s"] if last is not None and last.get("tok_s") else None
        ),
        "steady_state_steps_s": (
            last["steps_s"] if last is not None and last.get("steps_s") else None
        ),
        "run_wide_tok_s": round(run_tokens / run_s, 3) if run_s > 0 else None,
        "run_wide_steps_s": (
            round(history.attempts / run_s, 4) if run_s > 0 else None
        ),
        "phase_mean_ms_per_step": (
            {
                "data": round(
                    1000 * sum(i["data_s"] for i in intervals)
                    / max(1, sum(i["steps"] for i in intervals)), 3,
                ),
                "fwd": round(
                    1000 * sum(i["fwd_s"] for i in intervals)
                    / max(1, sum(i["steps"] for i in intervals)), 3,
                ),
                "bwd": round(
                    1000 * sum(i["bwd_s"] for i in intervals)
                    / max(1, sum(i["steps"] for i in intervals)), 3,
                ),
                "optim": round(
                    1000 * sum(i["optim_s"] for i in intervals)
                    / max(1, sum(i["steps"] for i in intervals)), 3,
                ),
                "ckpt": round(
                    1000 * sum(i["ckpt_s"] for i in intervals)
                    / max(1, sum(i["steps"] for i in intervals)), 3,
                ),
            }
            if intervals
            else None
        ),
        "checkpoint_save_total_s": round(history.checkpoint_save_s, 3),
    }


def train_run(args: argparse.Namespace) -> dict:
    """Run the full chain; returns the metrics dict (also saved to metrics.json).

    Two data sources (exactly one required):

    * ``--data <jsonl>`` (the original OASST1 path): deterministic train/val
      split, BPE-trained or ``--tokenizer-json`` tokenizer, streamed batches.
    * ``--packed-dir <dir>`` (corpus-pretraining path, DELIVERABLE 1): rows
      from ``prepare_corpus.py`` ``.npy`` shards used verbatim; the manifest
      is validated (format/dtype/seq/row-counts/tokenizer sha256/vocab) and
      val rows come from the val shards.

    With ``--resume`` (a ``step-<N>.pt`` file, or a directory whose
    numerically-newest VALID checkpoint is used — corrupt candidates skipped
    with a warning) the run **continues** an existing run: the tokenizer is
    loaded from the checkpoint's recorded path and verified by sha256
    fingerprint (P0 fix 4), model/optimizer/RNG + step/epoch + tokens-consumed
    counters are restored inside :func:`train_epochs`, and the packed corpus
    is identity-checked against the checkpoint's recorded provenance.
    Tokenizer training and preset build happen only on a fresh run.
    """
    t0 = time.monotonic()
    device = _resolve_device(args.device)
    set_seed(args.seed)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- data-source sanity + early schedule validation ---------------------
    # New flags are read via getattr so legacy callers (tests, notebooks) that
    # build args Namespaces without them keep working unchanged.
    packed_dir_arg = getattr(args, "packed_dir", None)
    token_budget = getattr(args, "token_budget", None)
    warmup_tokens = getattr(args, "warmup_tokens", 0) or 0
    lr_decay = getattr(args, "lr_decay", "none")
    # --- hardening-pass flags (D1/D2/D3; additive, defaults = old behavior) --
    save_every_tokens = getattr(args, "save_every_tokens", 0) or 0
    grad_clip = getattr(args, "grad_clip", 0.0) or 0.0
    max_consecutive_bad_steps = getattr(args, "max_consecutive_bad_steps", 3)
    if max_consecutive_bad_steps is None:
        max_consecutive_bad_steps = 3
    # --- T4 engine-pass flags (P1; additive, defaults reproduce old behavior) --
    attention_backend = getattr(args, "attention_backend", "auto") or "auto"
    amp = getattr(args, "amp", "none") or "none"
    fused_optim = bool(getattr(args, "fused_optim", False))
    pin_memory = bool(getattr(args, "pin_memory", False))
    prefetch = getattr(args, "prefetch", 0) or 0
    compile_enabled = bool(getattr(args, "compile", False))
    log_every_steps = getattr(args, "log_every_steps", 50)
    ckpt_staging_dir = getattr(args, "ckpt_staging_dir", None)
    ckpt_fsync = not bool(getattr(args, "no_ckpt_fsync", False))
    if attention_backend not in ATTENTION_BACKEND_CHOICES:
        raise ValueError(
            f"--attention-backend must be one of {ATTENTION_BACKEND_CHOICES}, "
            f"got {attention_backend!r}"
        )
    if amp not in AMP_CHOICES:
        raise ValueError(f"--amp must be one of {AMP_CHOICES}, got {amp!r}")
    if prefetch < 0:
        raise ValueError(f"--prefetch must be >= 0, got {prefetch}")
    if log_every_steps < 0:
        raise ValueError(f"--log-every-steps must be >= 0, got {log_every_steps}")
    if amp == "fp16" and attention_backend == "plain":
        # The recorded fp16 failure (c10::Half overflow) is root-caused to the
        # plain path's -inf mask materialization; refuse the known-bad combo.
        raise ValueError(
            "--amp fp16 requires the SDPA attention backend "
            "(--attention-backend sdpa or auto without flash-attn): the plain "
            "path's -inf mask materialization overflowed half; SDPA removes it "
            "by construction. Use --attention-backend sdpa for fp16."
        )
    if save_every_tokens < 0:
        raise ValueError(
            f"--save-every-tokens must be >= 0, got {save_every_tokens}"
        )
    if grad_clip < 0:
        raise ValueError(f"--grad-clip must be >= 0, got {grad_clip}")
    if max_consecutive_bad_steps < 0:
        raise ValueError(
            f"--max-consecutive-bad-steps must be >= 0, got "
            f"{max_consecutive_bad_steps} (0 = never abort)"
        )
    if packed_dir_arg and args.data:
        raise ValueError("pass exactly one of --data / --packed-dir, not both")
    data_src = "packed" if packed_dir_arg else ("jsonl" if args.data else None)
    if data_src is None:
        raise ValueError("exactly one of --data / --packed-dir is required")
    # Raises early on invalid combos (e.g. cosine without a token budget).
    TokenSchedule(
        lr=args.lr, warmup_tokens=warmup_tokens, decay=lr_decay,
        budget=token_budget,
    )

    # ---- resume: resolve the actual checkpoint (file, or numeric-desc scan) --
    resume_path: Optional[str] = args.resume
    resume_ckpt: Optional[dict] = None
    if args.resume:
        if os.path.isdir(args.resume):
            candidates = list_checkpoint_candidates(args.resume)
            if not candidates:
                raise FileNotFoundError(
                    f"--resume directory has no step-<N>.pt checkpoints: "
                    f"{args.resume}"
                )
            tried: List[str] = []
            for candidate in candidates:
                try:
                    ckpt = load_checkpoint(candidate)
                    validate_resume_checkpoint(ckpt, args.preset)
                    resume_ckpt, resume_path = ckpt, candidate
                    log.info(
                        "resume: selected %s (numeric-newest VALID checkpoint)",
                        os.path.basename(candidate),
                    )
                    break
                except Exception as exc:  # corrupt/unloadable — skip + warn
                    log.warning(
                        "resume candidate %s unusable (%s: %s) — skipping",
                        os.path.basename(candidate), type(exc).__name__, exc,
                    )
                    tried.append(
                        f"{os.path.basename(candidate)} ({type(exc).__name__}: {exc})"
                    )
            if resume_ckpt is None:
                raise FileNotFoundError(
                    f"no VALID resume checkpoint in {args.resume}: all "
                    f"{len(candidates)} candidate(s) failed — " + "; ".join(tried)
                )
        else:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(
                    f"--resume checkpoint not found: {args.resume}"
                )
            resume_ckpt = load_checkpoint(args.resume)
            validate_resume_checkpoint(resume_ckpt, args.preset)

    # ---- canonical preset config, printed before anything else ------------
    preset = args.preset
    if preset not in CANONICAL_PRESETS:
        raise ValueError(
            f"unknown --preset {preset!r}: choose from "
            f"{', '.join(sorted(CANONICAL_PRESETS))}"
        )
    exp_params, exp_vocab = CANONICAL_PRESETS[preset]
    cfg = ALL_PRESETS[preset]().derive()

    # ---- resume runs must continue the SAME data source ---------------------
    ckpt_run_meta = (resume_ckpt or {}).get("run_metadata") or {}
    ckpt_prov = ckpt_run_meta.get("data_provenance") or {}
    ckpt_source = ckpt_prov.get("source")
    if resume_ckpt is not None and ckpt_source:
        if ckpt_source != data_src:
            raise ValueError(
                f"resume checkpoint is from a {ckpt_source!r} run but the CLI "
                f"args select {data_src!r} — resume must continue the same "
                "data source"
            )

    # ---- 0) data: packed manifest (validated) or deterministic JSONL split --
    packed_manifest: Optional[dict] = None
    split: Optional[SplitResult] = None
    if data_src == "packed":
        packed_dir = os.path.abspath(packed_dir_arg)
        provided_tok_sha = (
            tokenizer_file_sha256(args.tokenizer_json)
            if args.tokenizer_json
            else None
        )
        packed_manifest = load_packed_manifest(
            packed_dir,
            expected_seq=None,  # seq resolved below (manifest wins unless --seq matches)
            expected_vocab=cfg.vocab_size,
            expected_tokenizer_sha256=provided_tok_sha,
        )
        m_seq = packed_manifest["seq_len"]
        if args.seq is None or args.seq == m_seq:
            effective_seq = m_seq
        else:
            raise ValueError(
                f"--seq {args.seq} does not match the packed corpus seq_len "
                f"{m_seq} — packed rows are used verbatim; pass --seq {m_seq} "
                "(or omit --seq)"
            )
        counts = packed_manifest["metadata"]["counts"]
        print(f"  packed corpus : {packed_dir} (seq {m_seq}, dtype "
              f"{packed_manifest['dtype']}, train {counts['train_rows']:,} rows "
              f"/ {counts['train_tokens']:,} tokens, val {counts['val_rows']:,} "
              f"rows, {packed_manifest['num_shards']} shards)")
        if resume_ckpt is not None:
            recorded_identity = ckpt_prov.get("manifest_identity")
            if recorded_identity and recorded_identity != manifest_identity(
                packed_manifest
            ):
                raise ValueError(
                    "resume packed corpus does not match the checkpoint's "
                    "recorded corpus identity (seq/dtype/tokenizer/rows differ)"
                    " — refusing to continue training on different data"
                )
            # Manifest-recovery guard (minimal, additive): the resume counter
            # must be consistent with the corpus the manifest describes.
            # tokens_consumed accrues batch*(seq-1) per step and each step
            # consumes one row-batch, so a single epoch can consume at most
            # train_rows*(seq-1) tokens and the whole run at most --epochs
            # times that. A counter beyond that capacity is impossible for
            # this corpus (corrupted counter, or a manifest for different
            # data) — fail loudly instead of silently continuing. Token
            # accounting semantics are untouched.
            ckpt_tokens = resume_ckpt.get("tokens_consumed")
            if (
                isinstance(ckpt_tokens, int) and ckpt_tokens > 0
                and args.epochs and args.epochs > 0
            ):
                epoch_capacity = counts["train_rows"] * (m_seq - 1)
                if ckpt_tokens > epoch_capacity * args.epochs:
                    raise ValueError(
                        f"resume checkpoint records {ckpt_tokens:,} tokens "
                        f"consumed but the packed corpus (train "
                        f"{counts['train_rows']:,} rows x {m_seq - 1} label "
                        f"tokens) can supply at most "
                        f"{epoch_capacity * args.epochs:,} across {args.epochs} "
                        f"epoch(s) — corrupted counter or a manifest for "
                        "different data; refusing to resume"
                    )
    else:
        effective_seq = args.seq if args.seq is not None else 64
        split = split_jsonl(
            args.data,
            os.path.join(out_dir, "data"),
            ratio=args.split_ratio,
            seed=args.seed,
            max_docs=args.split_max_docs,
        )
        print(f"  split: {split.total_docs} docs -> train {split.train_docs} / "
              f"val {split.val_docs} ({os.path.basename(split.train_path)}, "
              f"{os.path.basename(split.val_path)})")

    print("=" * 72)
    print("Talos OASST1-style training run"
          + (f" (RESUMING from {resume_path})" if resume_ckpt else ""))
    print(f"  model preset  : {preset} ({cfg.ffn_type}) — vocab={cfg.vocab_size} "
          f"hidden={cfg.hidden_size} layers={cfg.num_layers} "
          f"heads={cfg.num_attention_heads} kv={cfg.num_kv_heads} "
          f"ffn={cfg.ffn_type} seq={cfg.max_seq_len}")
    print(f"  expected      : EXACTLY {exp_params:,} params, vocab_size {exp_vocab}")
    print(f"  data          : {args.data or packed_dir_arg} ({data_src})")
    print(f"  seed          : {args.seed} | epochs {args.epochs} | seq "
          f"{effective_seq} | batch {args.batch} | lr {args.lr} | device {device}")
    print(f"  budget        : {token_budget if token_budget is not None else 'unbounded'} "
          f"tokens | warmup {warmup_tokens} | lr-decay {lr_decay}")
    print("=" * 72)

    # ---- 1) tokenizer + data provenance record ------------------------------
    manifest_tok_sha: Optional[str] = None
    if packed_manifest is not None:
        manifest_tok_sha = packed_manifest["metadata"]["tokenizer"]["sha256"]
    tokenizer_origin: str
    tokenizer_json = getattr(args, "tokenizer_json", None)
    tokenizer = None
    tokenizer_path: Optional[str] = None
    if resume_ckpt is None:
        tokenizer_path = os.path.join(out_dir, "tokenizer.json")
        if data_src == "packed":
            tok_source = tokenizer_json
            if tok_source is None:
                recorded = packed_manifest["metadata"]["tokenizer"].get("path")
                if recorded and os.path.isfile(recorded):
                    if (
                        manifest_tok_sha is not None
                        and tokenizer_file_sha256(recorded) == manifest_tok_sha
                    ):
                        tok_source = recorded
            if tok_source:
                tokenizer = _verify_and_load_tokenizer(
                    tok_source, manifest_tok_sha,
                    label="--tokenizer-json (must be the manifest's tokenizer)",
                )
                shutil.copyfile(tok_source, tokenizer_path)
                tokenizer_origin = "loaded"
                print(f"  tokenizer     : loaded from {tok_source} (vocab "
                      f"{tokenizer.vocab_size}, {tokenizer.merge_count} merges) "
                      f"— sha256 matches the manifest; copied to {tokenizer_path}")
            else:
                tokenizer_origin = "packed-manifest"
                tokenizer_path = None
                print(f"  tokenizer     : no sidecar tokenizer.json available — "
                      f"identity recorded from the manifest "
                      f"(sha256 {manifest_tok_sha[:12]}…, vocab "
                      f"{packed_manifest['metadata']['tokenizer']['vocab_size']})")
        elif tokenizer_json:
            if not os.path.isfile(tokenizer_json):
                raise FileNotFoundError(
                    f"--tokenizer-json not found: {tokenizer_json}"
                )
            shutil.copyfile(tokenizer_json, tokenizer_path)
            tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
            tokenizer_origin = "loaded"
            print(f"  tokenizer     : loaded from {tokenizer_json} (vocab "
                  f"{tokenizer.vocab_size}, {tokenizer.merge_count} merges) — BPE "
                  f"training skipped; copied to {tokenizer_path}")
        else:
            tokenizer = train_tokenizer_for_run(
                split.train_path,
                tokenizer_path,
                preset=preset,
                num_merges=args.bpe_num_merges,
                minfreq=args.bpe_minfreq,
                max_docs=args.bpe_max_docs,
                max_chars=args.bpe_max_chars,
            )
            tokenizer_origin = "trained"
    else:
        tokenizer_origin = "resumed"
        tokenizer_path = resume_ckpt.get("tokenizer_path")
        recorded_fp = resume_ckpt.get("tokenizer_fingerprint")
        if tokenizer_path and os.path.isfile(tokenizer_path):
            tokenizer = _verify_and_load_tokenizer(
                tokenizer_path, recorded_fp,
                label="resume checkpoint's sidecar tokenizer.json",
            )
            if tokenizer_json:
                provided_fp = tokenizer_file_sha256(tokenizer_json)
                if provided_fp != recorded_fp:
                    raise ValueError(
                        f"--tokenizer-json {tokenizer_json} does not match the "
                        f"resume checkpoint's recorded tokenizer fingerprint "
                        f"(sha256 {provided_fp[:12]}… vs {recorded_fp[:12]}…) — "
                        f"refusing to resume with a different tokenizer"
                    )
                print(f"  tokenizer     : reused from checkpoint — --tokenizer-json "
                      f"matches the recorded fingerprint (sha256 verified)")
            else:
                print(f"  tokenizer     : reused from checkpoint (vocab "
                      f"{tokenizer.vocab_size}, {tokenizer.merge_count} merges) "
                      f"— sha256 verified")
        else:
            # No sidecar file: only valid for a packed run whose identity the
            # manifest carries (verified below against the checkpoint fp).
            if data_src != "packed" or not recorded_fp:
                raise FileNotFoundError(
                    f"resume checkpoint's tokenizer not found ({tokenizer_path!r}) "
                    f"— expected the sidecar tokenizer.json next to the checkpoint"
                )
            if manifest_tok_sha != recorded_fp:
                raise ValueError(
                    f"resume checkpoint records tokenizer sha256 {recorded_fp} "
                    f"but the packed corpus manifest records {manifest_tok_sha} "
                    f"— resuming on a different tokenizer; refusing"
                )
            print(f"  tokenizer     : reused from packed-corpus manifest "
                  f"(sha256 {manifest_tok_sha[:12]}… verified) — no sidecar file")

    # Uniform vocab contract: a realized tokenizer's vocab must fit inside the
    # preset model's embedding rows. Packed rows are additionally range-checked
    # per batch against the model vocab, and the manifest's vocab_size_bound is
    # verified against it by load_packed_manifest.
    model_vocab = ALL_PRESETS[preset]().vocab_size
    if tokenizer is not None and tokenizer.vocab_size > model_vocab:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size} exceeds the {preset} "
            f"model's {model_vocab} — the tokenizer cannot be embedded; use a "
            f"tokenizer built for the canonical vocab-{model_vocab} contract"
        )

    # ---- 2) canonical preset model + hard param-count guard (fail fast) ---
    model = build_preset_model(preset, attention_backend=attention_backend).to(device)
    n_params = model.num_parameters()
    print(f"  model: {n_params:,} params (vocab {model.config.vocab_size}) — "
          f"config guard OK | attention backend: "
          f"{type(model.backend).__name__} | amp: {amp}")
    if compile_enabled:
        try:
            model = torch.compile(model)
        except Exception as exc:  # pragma: no cover - env dependent
            raise ValueError(f"--compile failed: {exc}") from exc
        print("  model      : torch.compile() enabled")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # ---- 3) streamed token batches -----------------------------------------
    # max_id=cfg.vocab_size: a tokenizer/model vocab mismatch fails at the data
    # source (per-document, with shard/doc context) instead of on the device.
    if packed_manifest is not None:
        train_paths = packed_phase_shard_paths(packed_manifest, packed_dir, "train")
        val_paths = packed_phase_shard_paths(packed_manifest, packed_dir, "val")
        fast_data = dict(
            cache_mmaps=(prefetch > 0) or pin_memory,
            pin_memory=pin_memory,
            prefetch=prefetch,
        )
        train_ds: Iterable = PackedTokenDataset(
            train_paths, seq_len=effective_seq, batch_size=args.batch,
            expected_dtype=packed_manifest["dtype"], max_id=cfg.vocab_size,
            **fast_data,
        )
        val_ds: Optional[Iterable] = (
            PackedTokenDataset(
                val_paths, seq_len=effective_seq, batch_size=args.batch,
                expected_dtype=packed_manifest["dtype"], max_id=cfg.vocab_size,
                **fast_data,
            )
            if val_paths
            else None
        )
    else:
        train_ds = StreamingTokenizedDataset(
            split.train_path, tokenizer, seq_len=effective_seq,
            batch_size=args.batch, mode="pack", eos=True, max_id=cfg.vocab_size,
        )
        val_ds = StreamingTokenizedDataset(
            split.val_path, tokenizer, seq_len=effective_seq,
            batch_size=args.batch, mode="pack", eos=True, max_id=cfg.vocab_size,
        )

    # ---- 4) run-metadata sidecar (DELIVERABLE 4) ---------------------------
    # Build ONCE (fresh or resume-with-provenance-merge); the same dict goes
    # into train_run_metadata.json AND every checkpoint payload.
    training_config = {
        "batch": args.batch,
        "seq": effective_seq,
        "lr": args.lr,
        "lr_decay": lr_decay,
        "warmup_tokens": warmup_tokens,
        "min_lr_ratio": MIN_LR_RATIO,
        "token_budget": token_budget,
        "epochs": args.epochs,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "seed": args.seed,
        #: numerical-stability guards live IN this trainer (hardening pass):
        #: NaN/Inf loss+gradient detection is always on; --grad-clip configures
        #: torch.nn.utils.clip_grad_norm_ after backward (0 = off); a bad step
        #: skips the optimizer step, writes a safety checkpoint, and after
        #: --max-consecutive-bad-steps consecutive failures the run aborts.
        "gradient_clip_type": ("max_grad_norm" if grad_clip > 0 else None),
        "grad_clip_max_norm": (float(grad_clip) if grad_clip > 0 else None),
        "nan_inf_detection": True,
        "max_consecutive_bad_steps": max_consecutive_bad_steps,
        "save_every_tokens": save_every_tokens,
        "device": str(device),
        # T4 engine pass (P1/P4): execution-engine + observability settings.
        "attention_backend": attention_backend,
        "amp": amp,
        "fused_optim": fused_optim,
        "pin_memory": pin_memory,
        "prefetch": prefetch,
        "compile": compile_enabled,
        "log_every_steps": log_every_steps,
        "ckpt_staging_dir": ckpt_staging_dir,
        "ckpt_fsync": ckpt_fsync,
    }
    tokenizer_meta = {
        "sha256": (
            tokenizer_file_sha256(tokenizer_path)
            if tokenizer_path
            else (manifest_tok_sha if manifest_tok_sha is not None else None)
        ),
        "vocab_size": (
            tokenizer.vocab_size
            if tokenizer is not None
            else packed_manifest["metadata"]["tokenizer"]["vocab_size"]
            if packed_manifest is not None
            else None
        ),
        "merges": (
            tokenizer.merge_count
            if tokenizer is not None
            else packed_manifest["metadata"]["tokenizer"]["merge_count"]
            if packed_manifest is not None
            else None
        ),
        "origin": tokenizer_origin,
    }
    if data_src == "packed":
        data_provenance = {
            "source": "packed",
            "packed_dir": packed_dir,
            "manifest_identity": manifest_identity(packed_manifest),
            "manifest_metadata": packed_manifest["metadata"],
        }
    else:
        data_provenance = {
            "source": "jsonl",
            "data": os.path.abspath(args.data),
            "split": {
                "ratio": args.split_ratio,
                "seed": args.seed,
                "max_docs": args.split_max_docs,
                "train_docs": split.train_docs,
                "val_docs": split.val_docs,
                "total_docs": split.total_docs,
            },
        }
    if resume_ckpt is not None:
        sidecar_path = os.path.join(out_dir, RUN_METADATA_FILENAME)
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as fh:
                run_metadata = json.load(fh)
            run_metadata = merge_run_metadata(
                run_metadata, resumed_from=os.path.abspath(resume_path)
            )
        elif resume_ckpt.get("run_metadata"):
            run_metadata = merge_run_metadata(
                resume_ckpt["run_metadata"],
                resumed_from=os.path.abspath(resume_path),
            )
        else:
            log.warning(
                "resuming an old checkpoint without run metadata — building a "
                "fresh sidecar from the current args"
            )
            run_metadata = build_run_metadata(
                preset=preset, model_cfg=cfg, n_params=n_params,
                training_config=training_config, data_provenance=data_provenance,
                tokenizer=tokenizer_meta, device=str(device),
                resumed_from=os.path.abspath(resume_path),
            )
    else:
        run_metadata = build_run_metadata(
            preset=preset, model_cfg=cfg, n_params=n_params,
            training_config=training_config, data_provenance=data_provenance,
            tokenizer=tokenizer_meta, device=str(device),
        )
    metadata_path = write_run_metadata(out_dir, run_metadata)
    print(f"  run metadata  : {metadata_path}")

    # ---- 5) train with per-epoch validation + checkpoints ----------------
    def _assemble_metrics(history: TrainingHistory) -> dict:
        """The run report dict; shared by the clean-finish and abort paths so
        an aborted run records exactly the same fields (plus its bad-step
        events + abort reason)."""
        last = history.rows[-1] if history.rows else None
        return {
            "format": "talos-oasst1-training-metrics-v1",
            "params": n_params,
            "vocab_size": cfg.vocab_size,
            "tokenizer_vocab_size": tokenizer_meta["vocab_size"],
            "tokenizer_merges": tokenizer_meta["merges"],
            #: content identity of the tokenizer that produced/owns the tokens
            #: (sha256 of out_dir/tokenizer.json, or the packed manifest's record).
            "tokenizer_sha256": tokenizer_meta["sha256"],
            "tokenizer_origin": tokenizer_origin,
            "tokenizer_json_arg": tokenizer_json,
            "data_source": data_src,
            "train_docs": split.train_docs if split is not None else None,
            "val_docs": split.val_docs if split is not None else None,
            "split_seed": split.seed if split is not None else None,
            "train_rows": (
                packed_manifest["metadata"]["counts"]["train_rows"]
                if packed_manifest is not None else None
            ),
            "val_rows": (
                packed_manifest["metadata"]["counts"]["val_rows"]
                if packed_manifest is not None else None
            ),
            "resumed_from": os.path.abspath(resume_path) if args.resume else None,
            "epochs": [asdict(r) for r in history.rows],
            "final_train_loss": last.train_loss if last else None,
            "final_val_loss": last.val_loss if last else None,
            "tokens_processed": history.tokens_processed,
            # token-budget accounting (DELIVERABLE 2) — budget, consumed, steps:
            "token_budget": token_budget,
            "tokens_consumed": history.tokens_processed,
            "steps": history.attempts,
            "budget_reached": history.budget_reached,
            "lr_schedule": TokenSchedule(
                lr=args.lr, warmup_tokens=warmup_tokens, decay=lr_decay,
                budget=token_budget,
            ).to_dict(),
            "wall_s": round(time.monotonic() - t0, 3),
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
            ),
            "device": str(device),
            "checkpoint": last.checkpoint if last else None,
            "tokenizer_path": tokenizer_path,
            "run_metadata_file": RUN_METADATA_FILENAME,
            # --- numerical-stability records (hardening pass) ---
            #: the --grad-clip setting actually in force (None = off).
            "grad_clip_max_norm": (float(grad_clip) if grad_clip > 0 else None),
            "nan_inf_detection": True,
            "max_consecutive_bad_steps": max_consecutive_bad_steps,
            "save_every_tokens": save_every_tokens,
            #: pre-clip total grad norm of the most recent clipped step.
            "last_grad_norm": history.last_grad_norm,
            #: loss of the most recent GOOD step (None if no good step yet).
            "last_good_loss": history.last_good_loss,
            #: every skipped NaN/Inf step, structured (step, tokens, tensor,
            #: stat, safety checkpoint), in execution order.
            "bad_steps": [dict(e) for e in history.bad_step_events],
            "total_bad_steps": history.total_bad_steps,
            "consecutive_bad_steps": history.consecutive_bad_steps,
            #: None on a clean finish; {reason, last_checkpoint} on a BadStepsAbort.
            "aborted": (
                {
                    "reason": history.abort_reason,
                    "last_checkpoint": (
                        history.bad_step_events[-1]["checkpoint"]
                        if history.bad_step_events else None
                    ),
                }
                if history.aborted else None
            ),
            #: most recent intra-epoch (--save-every-tokens) checkpoint, if any.
            "periodic_checkpoint": history.last_periodic_checkpoint,
            # --- T4 engine pass (P4/P5): engine config + performance ---
            "attention_backend": attention_backend,
            "engine": {
                "attention_backend": attention_backend,
                "amp": amp,
                "fused_optim": fused_optim,
                "pin_memory": pin_memory,
                "prefetch": prefetch,
                "compile": compile_enabled,
                "log_every_steps": log_every_steps,
                "ckpt_staging_dir": ckpt_staging_dir,
                "ckpt_fsync": ckpt_fsync,
            },
            "checkpoint_save_total_s": round(history.checkpoint_save_s, 3),
            "vram_peak_mb": (
                round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                if device.type == "cuda" else None
            ),
            "interval_logs": [dict(e) for e in history.interval_logs],
            "performance": _assemble_performance(history),
        }

    # P5: local-then-copy checkpoint staging (Drive latency off the hot path).
    ckpt_stager = CheckpointStager(ckpt_staging_dir, out_dir)
    try:
        history = train_epochs(
            model, train_ds, val_ds,
            out_dir=out_dir,
            tokenizer_path=tokenizer_path,
            lr=args.lr,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
            max_steps_per_epoch=args.max_steps_per_epoch,
            val_max_steps=args.val_max_steps,
            resume_from=resume_ckpt,
            token_budget=token_budget,
            warmup_tokens=warmup_tokens,
            lr_decay=lr_decay,
            run_metadata=run_metadata,
            save_every_tokens=save_every_tokens,
            grad_clip=grad_clip,
            max_consecutive_bad_steps=max_consecutive_bad_steps,
            amp=amp,
            fused_optim=fused_optim,
            log_every_steps=log_every_steps,
            ckpt_stager=ckpt_stager,
            ckpt_fsync=ckpt_fsync,
        )
    except BadStepsAbort as exc:
        # The abort is a DELIBERATE stop: write the same artifacts a clean
        # finish would (metrics + finish-stamped sidecar, now including the
        # bad-step events + abort reason), then propagate so the process exits
        # non-zero and the operator sees the loud abort line.
        history = exc.history if exc.history is not None else TrainingHistory()
        metrics = _assemble_metrics(history)
        _write_final_artifacts(out_dir, run_metadata, metrics)
        ckpt_stager.drain()
        print(
            f"\n  ABORTED after {history.consecutive_bad_steps} consecutive "
            f"NaN/Inf steps: {exc}\n  metrics + recoverable checkpoint "
            f"written to {out_dir}",
            file=sys.stderr,
        )
        raise

    # ---- 6) metrics ---------------------------------------------------------
    metrics = _assemble_metrics(history)
    _write_final_artifacts(out_dir, run_metadata, metrics)
    ckpt_stager.drain()
    last = history.rows[-1] if history.rows else None
    print(f"\n  final train loss  : {last.train_loss if last else 'n/a'}")
    print(f"  final val loss    : {last.val_loss if last and last.val_loss is not None else 'n/a'}")
    print(f"  tokens            : {history.tokens_processed:,} consumed"
          + (f" / budget {token_budget:,} (REACHED)" if history.budget_reached else ""))
    if history.total_bad_steps:
        print(f"  bad steps         : {history.total_bad_steps} total "
              f"({history.consecutive_bad_steps} consecutive at finish) — "
              f"see metrics.json 'bad_steps'")
    if grad_clip > 0:
        print(f"  grad clip         : max_norm {grad_clip:g} "
              f"(last pre-clip norm {history.last_grad_norm})")
    print(f"  wall time         : {metrics['wall_s']} s")
    print(f"  peak RSS          : {metrics['peak_rss_mb']} MiB")
    print(f"  checkpoint        : {metrics['checkpoint']}")
    print(f"  metrics           : {os.path.join(out_dir, 'metrics.json')}")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        train_run(args)
    except BadStepsAbort as exc:
        # Deliberate post-K-consecutive-bad-steps stop; metrics.json + the
        # recoverable safety checkpoint were already written by train_run.
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())