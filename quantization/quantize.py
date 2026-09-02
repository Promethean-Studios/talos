"""Symmetric weight-only quantization: int8 payloads + fp32 scales, and bf16.

Design (kept deliberately minimal — this is a measurement instrument, not a
quantization toolkit):

* **bf16**: ``quantize_model(model, precision="bf16")`` casts all parameters
  and floating buffers to ``torch.bfloat16``. Compute then genuinely runs in
  bf16 (RoPE/RMSNorm internally promote to fp32 and cast back — see
  ``model/rotary.py`` / ``model/rms_norm.py``), so the measurement captures
  real low-precision *compute* error, not just storage error.
* **int8**: every ``nn.Linear`` weight ``W`` (``out, in``) is quantized to
  ``torch.int8`` with symmetric min-max scales: ``q = round(W / s)`` clamped to
  ``[-127, 127]`` with ``s = amax / 127`` per output row (``per_channel``) or
  one scalar for the whole tensor (``per_tensor``). Embeddings quantize their
  ``(vocab, hidden)`` table per row the same way. The quantized modules
  (``QuantizedLinear`` / ``QuantizedEmbedding``) store the int8 payload + fp32
  scales as *buffers* (so ``state_dict`` round-trips include the compressed
  form) and dequantize on every forward (W8A32 weight-only compute).

Round-trip error bound (symmetric, round-to-nearest): ``|W - s*q| <= s``, i.e.
per-channel max abs error is bounded by the largest row scale — ``tests/test_quantization.py``
asserts exactly this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantizationReport",
    "QuantizedLinear",
    "QuantizedEmbedding",
    "quantize_int8_symmetric",
    "dequantize_int8",
    "quantize_model_",
    "quantize_model",
    "payload_bytes",
    "PRECISIONS",
    "GRANULARITIES",
]

PRECISIONS = ("bf16", "int8")
GRANULARITIES = ("per_channel", "per_tensor")
_INT8_MAX = 127.0
# Guard against an all-zero row collapsing the scale (and the row) entirely.
_AMAX_EPS = 1e-12
_BYTES_PER_DTYPE: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
    torch.int32: 4,
    torch.int64: 8,
}


# ---------------------------------------------------------------------------
# Tensor-level symmetric int8 quantization
# ---------------------------------------------------------------------------
def quantize_int8_symmetric(
    weight: torch.Tensor,
    granularity: str = "per_channel",
    num_bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric min-max quantization of ``weight`` to ``torch.int8``.

    Args:
        weight: any float tensor (typically ``(out, in)`` for Linear or
            ``(vocab, hidden)`` for Embedding weights).
        granularity: ``"per_channel"`` — one scale per output row (the last
            dimension is the quantization axis); ``"per_tensor"`` — a single
            scalar scale.
        num_bits: only 8 is implemented (the prototype experiment scope).

    Returns:
        ``(q, scale)`` where ``q`` is ``torch.int8`` in ``[-127, 127]`` and
        ``scale`` is fp32 (shape ``(out, 1)`` for per-channel, ``()`` scalar
        for per-tensor). Dequantization is ``q.float() * scale``.
    """
    if num_bits != 8:
        raise ValueError(f"only num_bits=8 is implemented, got {num_bits}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be a float tensor, got {weight.dtype}")
    w = weight.detach()
    if granularity == "per_channel":
        if w.dim() < 1:
            raise ValueError("per-channel quantization needs at least 1 dim")
        amax = w.abs().amax(dim=-1, keepdim=True)
    else:
        amax = w.abs().amax()
    scale = (amax.float() / _INT8_MAX).clamp_min(_AMAX_EPS)
    q = torch.round(w.float() / scale).clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale


def dequantize_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Invert :func:`quantize_int8_symmetric`: ``q.float() * scale -> dtype``."""
    return (q.float() * scale.float()).to(dtype)


def quantization_error(weight: torch.Tensor, granularity: str = "per_channel") -> Dict[str, float]:
    """Max/mean absolute and relative round-trip error of int8 quantization."""
    q, scale = quantize_int8_symmetric(weight, granularity)
    err = (dequantize_int8(q, scale, weight.dtype) - weight).abs()
    amax = weight.abs().max().clamp_min(_AMAX_EPS).item()
    return {
        "max_abs": float(err.max()),
        "mean_abs": float(err.mean()),
        "max_relative": float(err.max()) / amax,
    }


# ---------------------------------------------------------------------------
# Inference-only quantized modules (int8 payload, fp32 dequantized compute)
# ---------------------------------------------------------------------------
class QuantizedLinear(nn.Module):
    """``nn.Linear`` replacement storing an int8 weight payload + fp32 scales.

    The forward pass dequantizes the payload and runs an fp32 matmul
    (W8A32 weight-only). The fp32 weight is *not* cached — the resident weight
    storage really is int8, which is what makes the peak-RSS measurement
    honest. Gradients do not flow into the payload (inference-only).
    """

    def __init__(
        self,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        granularity: str,
    ) -> None:
        super().__init__()
        self.in_features = weight_int8.shape[1]
        self.out_features = weight_int8.shape[0]
        self.granularity = granularity
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scale", weight_scale)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, granularity: str = "per_channel") -> "QuantizedLinear":
        q, scale = quantize_int8_symmetric(linear.weight, granularity)
        return cls(q, scale, linear.bias, granularity)

    def dequantize_weight(self) -> torch.Tensor:
        return dequantize_int8(self.weight_int8, self.weight_scale, torch.float32)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized fp32 weight, recomputed on access.

        Compatibility shim so code that inspects ``module.weight`` keeps working
        on a quantized module (e.g. ``TalosGPT`` picks the KV-cache dtype from
        ``embed_tokens.weight.dtype``). The *resident* payload stays int8; this
        materializes a temporary fp32 copy each time it is called.
        """
        return self.dequantize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight().to(x.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, granularity={self.granularity}, dtype=int8(weight-only)"
        )


class QuantizedEmbedding(nn.Module):
    """``nn.Embedding`` replacement storing an int8 payload, per-row scales.

    Dequantizes on lookup and returns fp32 vectors (the rest of an int8 model
    runs in fp32). Inference-only.
    """

    def __init__(self, weight_int8: torch.Tensor, weight_scale: torch.Tensor) -> None:
        super().__init__()
        self.num_embeddings = weight_int8.shape[0]
        self.embedding_dim = weight_int8.shape[1]
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scale", weight_scale)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, granularity: str = "per_channel") -> "QuantizedEmbedding":
        q, scale = quantize_int8_symmetric(emb.weight, granularity)
        return cls(q, scale)

    def dequantize_weight(self) -> torch.Tensor:
        return dequantize_int8(self.weight_int8, self.weight_scale, torch.float32)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized fp32 table, recomputed on access (see QuantizedLinear.weight)."""
        return self.dequantize_weight()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.dequantize_weight())


# ---------------------------------------------------------------------------
# Whole-model quantization
# ---------------------------------------------------------------------------
@dataclass
class QuantizationReport:
    """What :func:`quantize_model_` did to a model."""

    precision: str
    granularity: Optional[str]
    replaced: List[Dict[str, Any]] = field(default_factory=list)
    params_quantized: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "granularity": self.granularity,
            "num_modules_replaced": len(self.replaced),
            "params_quantized": self.params_quantized,
            "modules": [
                {"name": r["name"], "kind": r["kind"], "shape": list(r["shape"])}
                for r in self.replaced
            ],
        }


def _replace_children(
    root: torch.nn.Module,
    skip: Sequence[str],
    factory: Any,
) -> List[Dict[str, Any]]:
    """Depth-first replacement of leaf modules selected by ``factory``.

    ``factory(name, module)`` returns either ``None`` (leave in place) or a
    ``(new_module, info_dict)`` pair; the info dict is collected into the
    returned list.
    """
    replaced: List[Dict[str, Any]] = []
    parents = list(root.named_modules())
    for parent_name, parent in parents:
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(full == s or full.startswith(s + ".") for s in skip):
                continue
            made = factory(full, child)
            if made is not None:
                new, info = made
                setattr(parent, child_name, new)
                replaced.append(info)
    return replaced


def quantize_model_(
    model: torch.nn.Module,
    precision: str = "int8",
    granularity: str = "per_channel",
    skip: Sequence[str] = (),
) -> QuantizationReport:
    """Quantize ``model`` **in place** and return a :class:`QuantizationReport`.

    Args:
        model: a Talos model (e.g. :class:`model.gpt.TalosGPT`). Put it in
            ``eval()`` afterwards — quantized modules are inference-only.
        precision: ``"int8"`` (weight-only symmetric payload) or ``"bf16"``
            (true bf16 compute via ``Module.to``).
        granularity: for int8 — ``"per_channel"`` (per output row; default) or
            ``"per_tensor"``.
        skip: module *names* (as in ``named_modules()``) to leave untouched,
            e.g. ``skip=("lm_head",)``.

    The generic module walk (every ``nn.Linear`` / ``nn.Embedding``) is
    deliberate: it works unchanged for dense and MoE presets at any scale.
    """
    if precision == "bf16":
        if granularity != "per_channel":
            raise ValueError("bf16 has no granularity knob; leave granularity at its default")
        model.to(torch.bfloat16)
        report = QuantizationReport(precision="bf16", granularity=None)
        report.params_quantized = sum(p.numel() for p in model.parameters())
        model.quantization_report = report  # type: ignore[attr-defined]
        return report
    if precision != "int8":
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    def factory(name: str, module: torch.nn.Module) -> Optional[Tuple[torch.nn.Module, Dict[str, Any]]]:
        if isinstance(module, nn.Linear):
            new = QuantizedLinear.from_linear(module, granularity)
            info = {"name": name, "kind": "Linear", "shape": tuple(new.weight_int8.shape),
                    "params": new.weight_int8.numel()}
            return (new, info)
        if isinstance(module, nn.Embedding):
            new = QuantizedEmbedding.from_embedding(module, granularity)
            info = {"name": name, "kind": "Embedding", "shape": tuple(new.weight_int8.shape),
                    "params": new.weight_int8.numel()}
            return (new, info)
        return None

    results = _replace_children(model, skip, factory)
    report = QuantizationReport(precision="int8", granularity=granularity)
    # _replace_children returns the collected info dicts (one per replacement).
    for info in results:
        report.replaced.append(info)
        report.params_quantized += info["params"]
    model.quantization_report = report  # type: ignore[attr-defined]
    return report


def quantize_model(
    model: torch.nn.Module,
    precision: str = "int8",
    granularity: str = "per_channel",
    skip: Sequence[str] = (),
) -> Tuple[torch.nn.Module, QuantizationReport]:
    """Non-mutating convenience wrapper: deep-copies then calls :func:`quantize_model_`."""
    import copy

    clone = copy.deepcopy(model)
    report = quantize_model_(clone, precision, granularity, skip)
    return clone, report


def payload_bytes(model: torch.nn.Module) -> Dict[str, Any]:
    """Actual serialized payload size of a model (state_dict tensors only).

    Counts bytes of every persistent tensor (parameters + persistent buffers,
    i.e. exactly what ``torch.save(model.state_dict())`` would write): fp32
    weights are 4 B/param, bf16 2 B, int8 payloads 1 B plus fp32 scales — the
    honest storage footprint of each precision.

    ``params`` is the model's *logical* parameter-element count: every
    state-dict element that represents a weight. Quantization-scale
    bookkeeping (``*.weight_scale`` buffers) adds bytes but represents no
    weights, so it is excluded — on an int8 model the int8 payloads plus the
    surviving fp32 parameters (norm gains, biases) reproduce the original
    parameter count, and ``bytes_per_param`` stays a true per-weight figure
    (~1 B/param) instead of dividing by the handful of ``nn.Parameter``s that
    survive when the weights become buffers.
    """
    per_dtype: Dict[str, int] = {}
    total = 0
    params = 0
    for name, tensor in model.state_dict().items():
        b = tensor.numel() * _BYTES_PER_DTYPE[tensor.dtype]
        key = str(tensor.dtype).replace("torch.", "")
        per_dtype[key] = per_dtype.get(key, 0) + b
        total += b
        if not name.endswith("weight_scale"):
            params += tensor.numel()
    return {
        "total_bytes": total,
        "total_mb": round(total / (1024.0 * 1024.0), 4),
        "per_dtype_bytes": per_dtype,
        "params": params,
        "bytes_per_param": round(total / max(params, 1), 4),
    }
