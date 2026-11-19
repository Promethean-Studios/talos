"""PyTorch glue for the Talos 1B notebook — small, audited, CPU-testable.

The notebook's training loop must be *thin*: every decision that can be got
wrong lives in a pure function elsewhere (``talos_stages`` for the budget and
schedule, ``talos_data`` for the stream, ``talos_hardware`` for the plan), and
this module holds only the pieces that genuinely need tensors:

* :func:`masked_cross_entropy` — token-mean CE over a loss mask, computed in
  chunks so a ``(4 × 2048) × 49,152`` logit tensor never materialises a second
  full float32 copy (the 1B logits are ~0.8 GB bf16 and would be ~1.6 GB fp32);
* :func:`autocast_kwargs` / :func:`make_grad_scaler` — bf16 needs no scaler,
  fp16 (T4/V100) does;
* :func:`build_optimizer` — 8-bit AdamW when available, plain AdamW as the
  functional fallback;
* :func:`clip_and_norm` — returns the **pre-clip** norm, which is what the
  checkpoint's ``last_grad_norm`` records and what catches a diverging run;
* :func:`rows_to_tensor` — packed rows (+ optional masks) → device tensors.

Every function here works on CPU with tiny tensors, so the audit items that
touch numerics are covered by ``tests/test_notebook_torch_utils.py`` (skipped
only when torch is absent).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised by the skip path in CI without torch
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # noqa: BLE001 - the module stays importable without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False


def _require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError("talos_torch_utils requires torch")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def masked_cross_entropy(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
    *,
    chunk_tokens: int = 2048,
) -> "torch.Tensor":
    """Token-mean cross-entropy, optionally restricted to masked positions.

    ``logits`` is ``(batch, seq, vocab)``, ``targets`` is ``(batch, seq)`` and
    ``mask`` (optional) is ``(batch, seq)`` with ``1`` on positions that count.
    The mean is over the **supervised** positions only — a Stage-B batch with
    masked user turns must not have its loss diluted by prompt tokens.

    Chunked over positions (``chunk_tokens`` at a time) so peak memory stays
    ``O(chunk × vocab)``: the 1B model's vocab is 49,152 and a 8,192-position
    micro-batch would otherwise need a 1.6 GB float32 logit copy.
    """
    _require_torch()
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if logits.dim() != 3:
        raise ValueError(f"logits must be (batch, seq, vocab), got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"targets {tuple(targets.shape)} must match logits[:2] {tuple(logits.shape[:2])}"
        )
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if mask is None:
        flat_mask = torch.ones_like(flat_targets)
    else:
        if mask.shape != targets.shape:
            raise ValueError(
                f"mask {tuple(mask.shape)} must match targets {tuple(targets.shape)}"
            )
        flat_mask = mask.reshape(-1).to(flat_targets.device)
    total = flat_targets.numel()
    supervised = float(flat_mask.sum().item())
    if supervised <= 0:
        raise ValueError(
            "no supervised tokens in this batch — the loss mask is all zeros, "
            "which would make the step a no-op with a NaN mean"
        )
    loss_sum = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, total, chunk_tokens):
        stop = min(total, start + chunk_tokens)
        chunk_logits = flat_logits[start:stop].float()
        chunk_targets = flat_targets[start:stop]
        chunk_weights = flat_mask[start:stop].to(torch.float32)
        losses = F.cross_entropy(chunk_logits, chunk_targets, reduction="none")
        loss_sum = loss_sum + (losses * chunk_weights).sum()
    return loss_sum / supervised


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def autocast_kwargs(device_type: str, precision: str) -> Dict[str, Any]:
    """``torch.autocast`` kwargs for ``precision`` (``"bf16"``/``"fp16"``/``"fp32"``).

    ``fp32`` disables autocast entirely — that is the honest CPU path and the
    escape hatch if a bf16 build misbehaves.
    """
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}")
    if precision == "fp32":
        return {"device_type": device_type, "enabled": False}
    _require_torch()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return {"device_type": device_type, "dtype": dtype, "enabled": True}


def make_grad_scaler(*, enabled: bool) -> Optional[Any]:
    """A :class:`torch.amp.GradScaler` when fp16 is in use, else ``None``.

    bf16's exponent range covers the loss scale, so a scaler would be dead
    weight; fp16 on a T4/V100 underflows without one.
    """
    if not enabled:
        return None
    _require_torch()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=True)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def build_optimizer(
    parameters: Iterable[Any],
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    use_8bit_adam: bool = True,
) -> Tuple[Any, str]:
    """8-bit AdamW if bitsandbytes is usable, otherwise plain AdamW.

    Returns ``(optimizer, note)`` — the note goes into the run log and the
    checkpoint metadata, so a run always records which optimizer it actually
    used (an 8-bit optimizer state dict is not loadable by a plain AdamW and
    vice versa; the resume path relies on the same class being rebuilt).
    """
    _require_torch()
    params = list(parameters)
    if not params:
        raise ValueError("build_optimizer got no parameters")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=lr, betas=betas,
                                            weight_decay=weight_decay)
            return optimizer, "AdamW8bit (bitsandbytes)"
        except Exception as exc:  # noqa: BLE001 — fall back, never fail the run
            note = f"bitsandbytes unavailable ({type(exc).__name__}: {exc}) — plain AdamW"
    else:
        note = "plain AdamW (8-bit disabled by the hardware plan)"
    return torch.optim.AdamW(params, lr=lr, betas=betas,
                             weight_decay=weight_decay), note


def clip_and_norm(parameters: Iterable[Any], max_norm: float) -> float:
    """Clip gradients to ``max_norm`` and return the **pre-clip** total norm.

    The pre-clip norm is the number worth recording: it rises when a run starts
    to diverge even though clipping hides that from the update itself.
    """
    _require_torch()
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    params = [p for p in parameters if getattr(p, "grad", None) is not None]
    if not params:
        return 0.0
    total = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total.detach().to(torch.float32).item()) if hasattr(total, "detach") else float(total)


# ---------------------------------------------------------------------------
# Non-finite detection (the skipped-update path)
# ---------------------------------------------------------------------------
def loss_is_finite(loss: "torch.Tensor") -> bool:
    """True when every element of ``loss`` is finite (NaN/Inf -> ``False``).

    Called **before** ``backward()``: a non-finite loss produces non-finite
    gradients that poison every parameter they touch, so the loop skips the
    update instead of applying it.
    """
    _require_torch()
    if not torch.is_tensor(loss):
        raise TypeError(f"loss must be a tensor, got {type(loss).__name__}")
    try:
        return bool(torch.isfinite(loss.detach()).all().item())
    except Exception:  # noqa: BLE001 - an unreadable value is not finite
        return False


def grad_norm(parameters: Iterable[Any]) -> float:
    """Total L2 norm of the existing gradients, **without** clipping.

    Separate from :func:`clip_and_norm` because the loop must be able to see a
    non-finite norm *before* clipping: ``clip_grad_norm_`` on an Inf gradient
    multiplies by ``max_norm / Inf == 0``, which silently converts Inf grads to
    NaN instead of reporting them. Returns ``0.0`` when no parameter has a grad.
    """
    _require_torch()
    params = [p for p in parameters if getattr(p, "grad", None) is not None]
    if not params:
        return 0.0
    total = torch.nn.utils.get_total_norm(
        [p.grad.detach() for p in params], foreach=False
    ) if hasattr(torch.nn.utils, "get_total_norm") else None
    if total is None:  # older torch: sum of squares in float32
        total = torch.sqrt(sum((p.grad.detach().to(torch.float32) ** 2).sum()
                               for p in params))
    return float(total.detach().to(torch.float32).item())


def grads_are_finite(parameters: Iterable[Any]) -> bool:
    """True when every existing gradient element is finite."""
    _require_torch()
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all().item()):
            return False
    return True


def zero_gradients(optimizer: Any) -> None:
    """Drop the accumulated gradients of a skipped update (``set_to_none``)."""
    _require_torch()
    optimizer.zero_grad(set_to_none=True)


def clip_gradients(parameters: Iterable[Any], max_norm: float) -> None:
    """Clip the existing gradients to ``max_norm`` in place.

    Called only **after** :func:`grads_are_finite` has passed: clipping an Inf
    gradient would turn it into NaN (``max_norm / Inf == 0``) and hide the
    anomaly instead of reporting it.
    """
    _require_torch()
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    params = [p for p in parameters if getattr(p, "grad", None) is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def rows_to_tensor(
    rows: Sequence[Any],
    *,
    device: Any,
    with_mask: bool = False,
) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    """Packed rows (``.ids`` / ``.mask``) -> ``(ids, mask)`` tensors on ``device``.

    ``ids`` is ``(batch, seq)`` int64; ``mask`` (when requested) is the same
    shape, ``1`` where the position is supervised. A row whose mask is ``None``
    in a masked batch is treated as fully supervised.
    """
    _require_torch()
    if not rows:
        raise ValueError("rows must not be empty")
    ids: List[List[int]] = [list(row.ids) for row in rows]
    width = len(ids[0])
    if any(len(row) != width for row in ids):
        raise ValueError("all rows in a batch must have the same length")
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    if not with_mask:
        return ids_tensor, None
    masks: List[List[int]] = []
    for row in rows:
        mask = getattr(row, "mask", None)
        masks.append([1] * width if mask is None else [int(v) for v in mask])
    return ids_tensor, torch.tensor(masks, dtype=torch.long, device=device)


__all__ = [
    "HAS_TORCH",
    "autocast_kwargs",
    "build_optimizer",
    "clip_and_norm",
    "clip_gradients",
    "grad_norm",
    "grads_are_finite",
    "loss_is_finite",
    "make_grad_scaler",
    "masked_cross_entropy",
    "rows_to_tensor",
    "zero_gradients",
]
