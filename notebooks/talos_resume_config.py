"""The checkpoint's training-config snapshot and its resume contract — no torch.

Fix 2 (owner directive, 2026-11-25). A checkpoint used to record *what the model
was* (model config, parameter count, tokenizer fingerprint, stage, dataset
identity) and *how far the run had got* (step, tokens, RNG, dataset position),
but never *how it was being trained*. Resuming across a change in the seed, the
shuffle buffer, the optimizer, the precision, the effective batch, the LR
schedule or the budget silently produces a run that is neither the old run nor a
new one — the worst case being an 8-bit AdamW state dict that a plain AdamW
cannot load at all, or a 2B-token checkpoint continued under a 4B-token
schedule as if nothing had changed.

So every checkpoint now carries a ``training_config`` snapshot
(:func:`build_training_config`) and every resume compares it field by field
(:func:`validate_training_config`):

* **hard rejects** — seed, shuffle buffer, optimizer identity, precision (and
  the fp32-master-weights flag), effective batch, LR schedule, a **shrunk**
  budget, and any change in bitsandbytes availability (an 8-bit optimizer state
  is not loadable by plain AdamW);
* **acknowledgeable rejects** — the repo pin, a torch major/minor change, and a
  **larger** budget. They reject by default and are accepted only when named in
  ``RESUME_ACKNOWLEDGE`` (``RESUME_ACKNOWLEDGE=repo_pin,token_budget``);
* **warnings only** — a torch patch-version change and changes in the other
  dependency versions.

Everything here is pure Python (no torch, no datasets, no network) so the
comparison logic is unit-tested on a CPU: ``tests/test_resume_training_config.py``.
"""

from __future__ import annotations

import importlib
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: Schema tag of the snapshot dict (additive checkpoint metadata field).
TRAINING_CONFIG_SCHEMA = "talos-training-config-v1"

#: The one env var that acknowledges resume-blocking field changes by name,
#: e.g. ``RESUME_ACKNOWLEDGE=repo_pin,token_budget``. Anything **not** named
#: still rejects; there is no "acknowledge everything" value.
ACKNOWLEDGE_ENV = "RESUME_ACKNOWLEDGE"

#: Convenience switch for the deliberate budget extension
#: (``RESUME_ACKNOWLEDGE_BUDGET=1`` is the same as naming ``token_budget``).
ACKNOWLEDGE_BUDGET_ENV = "RESUME_ACKNOWLEDGE_BUDGET"

#: Dependency versions recorded in every snapshot. ``transformers`` is optional
#: (the notebook does not use it); a ``None`` version means "not installed".
DEPENDENCIES = ("torch", "bitsandbytes", "datasets", "tokenizers", "transformers")

#: Fields whose change always rejects, acknowledged or not.
HARD_REJECT_FIELDS = (
    "seed",
    "shuffle_buffer",
    "optimizer",
    "precision",
    "master_weight_dtype",
    "effective_batch",
    "lr_schedule",
)

#: Fields that reject by default but can be acknowledged by name.
ACKNOWLEDGEABLE_FIELDS = ("repo_pin", "torch_version", "token_budget")

_TRUTHY = {"1", "true", "yes", "on"}

_MISSING_SNAPSHOT_WARNING = (
    "WARNING: this checkpoint carries no training_config snapshot (it was written "
    "before 2026-11-25), so seed / shuffle buffer / optimizer / precision / "
    "effective batch / LR schedule / budget cannot be compared — resuming under "
    "the pre-existing checks only (format, preset, stage, dataset, tokenizer, "
    "optimizer-state presence). Re-check the config cell by hand."
)


class ResumeConfigError(ValueError):
    """Raised when a checkpoint's training config is not this run's config."""


# ---------------------------------------------------------------------------
# Building the snapshot
# ---------------------------------------------------------------------------
def dependency_versions(modules: Optional[Mapping[str, Any]] = None) -> Dict[str, Optional[str]]:
    """``{name: __version__}`` for :data:`DEPENDENCIES` (``None`` = absent).

    ``modules`` is injectable so the recording path is testable without
    importing any of them (and so a partially broken install cannot make the
    checkpoint write fail).
    """
    versions: Dict[str, Optional[str]] = {}
    for name in DEPENDENCIES:
        module: Any = None
        if modules is not None:
            module = modules.get(name)
        else:
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - an absent/broken dep is recorded as None
                module = None
        version = getattr(module, "__version__", None) if module is not None else None
        versions[name] = None if version is None else str(version)
    return versions


def build_training_config(
    *,
    repo_pin: Optional[str],
    seed: int,
    shuffle_buffer: int,
    optimizer: str,
    peak_lr: float,
    min_lr_ratio: float,
    warmup_ratio: float,
    grad_clip: float,
    weight_decay: float,
    precision: str,
    master_weight_dtype: str,
    micro_batch: int,
    grad_accum: int,
    seq_len: int,
    token_budget: int,
    total_steps: int,
    versions: Optional[Mapping[str, Optional[str]]] = None,
    world_size: int = 1,
) -> Dict[str, Any]:
    """The full training-config snapshot stored in every checkpoint.

    Everything a resumed run must agree with. ``optimizer`` is the identity
    string :func:`talos_torch_utils.build_optimizer` returns (``"AdamW8bit
    (bitsandbytes)"`` or ``"AdamW"``), never the object. ``repo_pin`` is the
    notebook's ``REPO_PIN_SHA`` (``""``/``None`` = tracking main, which is
    recorded as ``None`` because there is no commit to compare).
    """
    if not str(optimizer or "").strip():
        raise ValueError("optimizer identity must be a non-empty string")
    if int(seed) < 0:
        raise ValueError("seed must be >= 0")
    if int(shuffle_buffer) <= 0:
        raise ValueError("shuffle_buffer must be positive")
    if float(peak_lr) <= 0:
        raise ValueError("peak_lr must be positive")
    if not 0.0 <= float(min_lr_ratio) <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if not 0.0 <= float(warmup_ratio) < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if str(precision) not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}")
    for name, value in (("micro_batch", micro_batch), ("grad_accum", grad_accum),
                        ("seq_len", seq_len), ("world_size", world_size)):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(token_budget) <= 0:
        raise ValueError("token_budget must be positive")
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    rows = int(micro_batch) * int(grad_accum) * int(world_size)
    recorded_versions = {name: (None if versions is None else versions.get(name))
                         for name in DEPENDENCIES}
    return {
        "schema": TRAINING_CONFIG_SCHEMA,
        "repo_pin": (str(repo_pin) or None) if repo_pin else None,
        "seed": int(seed),
        "shuffle_buffer": int(shuffle_buffer),
        "versions": recorded_versions,
        "optimizer": str(optimizer),
        "precision": str(precision),
        "master_weight_dtype": str(master_weight_dtype),
        "effective_batch": {
            "micro_batch": int(micro_batch),
            "grad_accum": int(grad_accum),
            "world_size": int(world_size),
            "rows": rows,
            "tokens": rows * int(seq_len),
        },
        "lr_schedule": {
            "peak_lr": float(peak_lr),
            "min_lr_ratio": float(min_lr_ratio),
            "warmup_ratio": float(warmup_ratio),
        },
        "grad_clip": float(grad_clip),
        "weight_decay": float(weight_decay),
        "token_budget": int(token_budget),
        "total_steps": int(total_steps),
    }


# ---------------------------------------------------------------------------
# Acknowledgements
# ---------------------------------------------------------------------------
def acknowledged_fields(environ: Optional[Mapping[str, str]] = None) -> frozenset:
    """Field names the operator has explicitly acknowledged for this resume.

    ``RESUME_ACKNOWLEDGE=repo_pin, token_budget`` (commas or whitespace), plus
    ``RESUME_ACKNOWLEDGE_BUDGET=1`` as a dedicated switch for the budget
    extension. Nothing else is ever acknowledged implicitly.
    """
    env = os.environ if environ is None else environ
    raw = str(env.get(ACKNOWLEDGE_ENV) or "")
    fields = {part.strip().lower() for part in re.split(r"[,\s]+", raw) if part.strip()}
    if str(env.get(ACKNOWLEDGE_BUDGET_ENV) or "").strip().lower() in _TRUTHY:
        fields.add("token_budget")
    return frozenset(fields)


def _torch_major_minor(version: Any) -> Optional[str]:
    """``"2.9.0+cu124"`` -> ``"2.9"`` (``None`` when unparseable)."""
    if not version:
        return None
    head = str(version).split("+", 1)[0]
    parts = head.split(".")
    if len(parts) < 2:
        return head or None
    return ".".join(parts[:2])


def _fmt(value: Any) -> str:
    return "None" if value is None else repr(value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_training_config(
    saved: Optional[Mapping[str, Any]],
    current: Optional[Mapping[str, Any]],
    *,
    acknowledge: Optional[Iterable[str]] = None,
) -> List[str]:
    """Compare a checkpoint's snapshot with the current run's; raise on a reject.

    Returns the list of warning lines (torch patch-version change, other
    dependency changes, acknowledged changes) for the caller to print — a
    missing snapshot is itself returned as one loud warning so an old checkpoint
    still resumes under the pre-existing checks instead of crashing.

    Raises:
        ResumeConfigError: listing **every** rejected field with both values.
    """
    if not saved:
        return [_MISSING_SNAPSHOT_WARNING]
    saved = dict(saved)
    current = dict(current or {})
    acks = frozenset(str(name).strip().lower() for name in (acknowledge or ()))
    warnings: List[str] = []
    rejections: List[str] = []

    def reject(field: str, detail: str, *, acknowledgeable: bool = False) -> None:
        if acknowledgeable and field in acks:
            warnings.append(f"ACKNOWLEDGED {field} change (RESUME_ACKNOWLEDGE): {detail}")
            return
        hint = ""
        if acknowledgeable:
            hint = (f" [deliberate? re-run with {ACKNOWLEDGE_ENV}={field} "
                    "to acknowledge exactly this field]")
        rejections.append(f"  - {field}: {detail}{hint}")

    # ---- seed / shuffle: the data order itself ----------------------------
    if saved.get("seed") != current.get("seed"):
        reject("seed", f"saved {_fmt(saved.get('seed'))} vs current "
                       f"{_fmt(current.get('seed'))} — a different seed is a "
                       "different run, not a continuation")
    if saved.get("shuffle_buffer") != current.get("shuffle_buffer"):
        reject("shuffle_buffer", f"saved {_fmt(saved.get('shuffle_buffer'))} vs "
                                 f"current {_fmt(current.get('shuffle_buffer'))} — the "
                                 "document order changes, so the resumed stream "
                                 "would not continue the same run")

    # ---- optimizer identity: state dicts are not interchangeable ----------
    saved_optimizer = saved.get("optimizer")
    current_optimizer = current.get("optimizer")
    if saved_optimizer != current_optimizer:
        detail = (f"saved {_fmt(saved_optimizer)} vs current {_fmt(current_optimizer)} "
                  "— an optimizer state dict is only loadable by the class that "
                  "wrote it")
        saved_versions = dict(saved.get("versions") or {})
        current_versions = dict(current.get("versions") or {})
        if bool(saved_versions.get("bitsandbytes")) and not current_versions.get("bitsandbytes"):
            detail += ("; bitsandbytes is no longer available, and 8-bit AdamW state "
                       "(uint8 moments + absmax) cannot be loaded by plain AdamW — "
                       "install bitsandbytes again (this is not acknowledgeable)")
        reject("optimizer", detail)

    # ---- precision / master weights --------------------------------------
    if saved.get("precision") != current.get("precision"):
        reject("precision", f"saved {_fmt(saved.get('precision'))} vs current "
                            f"{_fmt(current.get('precision'))} — autocast dtype changes "
                            "the numerics of every step")
    if saved.get("master_weight_dtype") != current.get("master_weight_dtype"):
        reject("master_weight_dtype", f"saved {_fmt(saved.get('master_weight_dtype'))} vs "
                                      f"current {_fmt(current.get('master_weight_dtype'))} "
                                      "— the parameters were kept in a different dtype")

    # ---- effective batch -------------------------------------------------
    saved_batch = dict(saved.get("effective_batch") or {})
    current_batch = dict(current.get("effective_batch") or {})
    batch_keys = ("micro_batch", "grad_accum", "world_size")
    if any(saved_batch.get(key) != current_batch.get(key) for key in batch_keys):
        reject("effective_batch",
               f"saved {_fmt({k: saved_batch.get(k) for k in batch_keys})} vs current "
               f"{_fmt({k: current_batch.get(k) for k in batch_keys})} — a different "
               "effective batch is a different schedule")

    # ---- LR schedule -----------------------------------------------------
    saved_lr = dict(saved.get("lr_schedule") or {})
    current_lr = dict(current.get("lr_schedule") or {})
    lr_keys = ("peak_lr", "min_lr_ratio", "warmup_ratio")
    if any(saved_lr.get(key) != current_lr.get(key) for key in lr_keys):
        reject("lr_schedule",
               f"saved {_fmt({k: saved_lr.get(k) for k in lr_keys})} vs current "
               f"{_fmt({k: current_lr.get(k) for k in lr_keys})} — resuming would "
               "continue a curve the run never followed")

    # ---- budget ----------------------------------------------------------
    saved_budget = saved.get("token_budget")
    current_budget = current.get("token_budget")
    if saved_budget is not None and current_budget is not None:
        saved_budget, current_budget = int(saved_budget), int(current_budget)
        if current_budget < saved_budget:
            # Never acceptable: the run already spent more than this budget.
            reject("token_budget", f"saved {saved_budget:,} vs current "
                                   f"{current_budget:,} — a SHRUNK budget cannot be "
                                   "acknowledged: the run already consumed "
                                   f"{saved_budget:,} tokens")
        elif current_budget > saved_budget:
            reject("token_budget", f"saved {saved_budget:,} vs current "
                                   f"{current_budget:,} — a budget EXTENSION",
                   acknowledgeable=True)

    # ---- repo pin --------------------------------------------------------
    if saved.get("repo_pin") != current.get("repo_pin"):
        reject("repo_pin", f"saved {_fmt(saved.get('repo_pin'))} vs current "
                           f"{_fmt(current.get('repo_pin'))} — the training code "
                           "changed between sessions", acknowledgeable=True)

    # ---- dependency versions --------------------------------------------
    saved_versions = dict(saved.get("versions") or {})
    current_versions = dict(current.get("versions") or {})
    if bool(saved_versions.get("bitsandbytes")) != bool(current_versions.get("bitsandbytes")):
        rejections.append(
            "  - bitsandbytes availability: saved "
            f"{_fmt(saved_versions.get('bitsandbytes'))} vs current "
            f"{_fmt(current_versions.get('bitsandbytes'))} — 8-bit AdamW state is "
            "not loadable by plain AdamW (or vice versa); this is not "
            "acknowledgeable")
    elif saved_versions.get("bitsandbytes") != current_versions.get("bitsandbytes"):
        warnings.append(
            "bitsandbytes version changed: "
            f"{saved_versions.get('bitsandbytes')} -> {current_versions.get('bitsandbytes')} "
            "(state-dict layout is compatible across bitsandbytes releases; continuing)")
    saved_torch, current_torch = saved_versions.get("torch"), current_versions.get("torch")
    if saved_torch and current_torch and saved_torch != current_torch:
        if _torch_major_minor(saved_torch) == _torch_major_minor(current_torch):
            warnings.append(
                f"WARNING: torch patch version changed: {saved_torch} -> {current_torch} "
                "(same major.minor — resuming; determinism across patch releases is "
                "not guaranteed)")
        else:
            reject("torch_version", f"{saved_torch} -> {current_torch} (major/minor "
                                    "change: kernel and RNG behaviour can differ)",
                   acknowledgeable=True)
    for name in ("datasets", "tokenizers", "transformers"):
        if saved_versions.get(name) != current_versions.get(name):
            warnings.append(
                f"WARNING: {name} version changed: {saved_versions.get(name)} -> "
                f"{current_versions.get(name)} (recorded, not fatal)")

    # ---- warnings-only settings -----------------------------------------
    for field in ("grad_clip", "weight_decay"):
        if saved.get(field) != current.get(field):
            warnings.append(f"WARNING: {field} changed: {saved.get(field)} -> "
                            f"{current.get(field)} (recorded, not fatal)")

    if rejections:
        raise ResumeConfigError(
            "refusing to resume: the checkpoint's training config does not match "
            "this run. Resuming anyway would silently mix two different runs "
            "(and an optimizer state dict written by one class cannot be loaded by "
            "another). Mismatched fields:\n"
            + "\n".join(rejections)
            + "\nStart a fresh run, or — for the deliberate cases only (a code "
              f"pin bump, a torch upgrade, a larger budget) — re-run with "
              f"{ACKNOWLEDGE_ENV}=<field>[,<field>] naming exactly the fields you "
              f"mean (or {ACKNOWLEDGE_BUDGET_ENV}=1 for the budget)."
        )
    return warnings


def token_budget_extension(
    saved: Optional[Mapping[str, Any]],
    current: Optional[Mapping[str, Any]],
    *,
    acknowledge: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """The acknowledged budget extension to record in the resumed run's metadata.

    ``None`` unless the current budget is **larger** than the saved one (a
    shrink is rejected outright by :func:`validate_training_config`). Recording
    it in the run metadata is what makes the next save carry the new budget
    instead of quietly reverting to the old one.
    """
    saved_budget = int((saved or {}).get("token_budget") or 0)
    current_budget = int((current or {}).get("token_budget") or 0)
    if not saved_budget or current_budget <= saved_budget:
        return None
    acks = frozenset(str(name).strip().lower() for name in (acknowledge or ()))
    return {
        "from": saved_budget,
        "to": current_budget,
        "acknowledged": "token_budget" in acks,
    }


__all__ = [
    "ACKNOWLEDGEABLE_FIELDS",
    "ACKNOWLEDGE_BUDGET_ENV",
    "ACKNOWLEDGE_ENV",
    "DEPENDENCIES",
    "HARD_REJECT_FIELDS",
    "ResumeConfigError",
    "TRAINING_CONFIG_SCHEMA",
    "acknowledged_fields",
    "build_training_config",
    "dependency_versions",
    "token_budget_extension",
    "validate_training_config",
]
