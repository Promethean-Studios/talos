"""Weight-only quantization utilities for Talos prototype models.

This package is a *small controlled experiment* supporting the prototype-first
validation path (quantize the tiny model, measure, don't redesign anything). It
provides two precision targets and nothing more:

* ``"bf16"``  — cast every parameter/buffer to bfloat16 and run the model with
  bf16 compute (a true low-precision matmul path on CPU via ``model.to()``).
* ``"int8"``  — symmetric min-max **weight-only** quantization of every
  ``nn.Linear`` / ``nn.Embedding`` to ``torch.int8`` with fp32 scales
  (per-output-channel = per-row, or per-tensor), executed W8A32-style: the
  stored payload is int8 (4x smaller than fp32) and the forward dequantizes the
  weight and runs an fp32 matmul. No int8 GEMM kernels are involved — CPU int8
  compute is out of scope for this cycle (deferred with the 400B work).

The quantized modules are inference-only (gradients cannot flow into an int8
payload); training stays in fp32/bf16. Everything here walks the *same* module
graph the tiny model uses, so one code path serves any preset from ``tiny``
upward. See ``tools/quantize_experiment.py`` for the measurement harness that
produces the committed ``benchmarks/quant-tiny-cpu.{json,md}`` numbers.
"""
from quantization.quantize import (
    QuantizationReport,
    QuantizedEmbedding,
    QuantizedLinear,
    dequantize_int8,
    payload_bytes,
    quantize_int8_symmetric,
    quantize_model,
    quantize_model_,
)

__all__ = [
    "QuantizationReport",
    "QuantizedEmbedding",
    "QuantizedLinear",
    "dequantize_int8",
    "payload_bytes",
    "quantize_int8_symmetric",
    "quantize_model",
    "quantize_model_",
]
