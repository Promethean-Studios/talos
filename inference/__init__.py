"""Talos inference: buffered (prefill + KV-cache decode) generation.

The prototype's inference path lives in :mod:`inference.generate`; the
distributed/inference-server scope is described in PLAN.md and is deferred
until after the tiny-prototype validation milestones.
"""
from inference.generate import (
    EquivalenceReport,
    decode_step,
    generate,
    prefill,
    prefill_decode_max_abs_diff,
)

__all__ = [
    "EquivalenceReport",
    "decode_step",
    "generate",
    "prefill",
    "prefill_decode_max_abs_diff",
]
