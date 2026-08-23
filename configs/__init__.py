"""Model presets and compute estimates.

Configs are :class:`~model.config.ModelConfig` dataclasses (see
``configs/presets.py``), the authoritative way to describe a Talos model. The
``compute`` module estimates parameters, weights memory and FLOPs from a config.
"""
from configs.presets import (
    tiny_config,
    small_config,
    medium_config,
    large_config,
    scale_100b_config,
    scale_400b_config,
    ALL_PRESETS,
)

__all__ = [
    "tiny_config",
    "small_config",
    "medium_config",
    "large_config",
    "scale_100b_config",
    "scale_400b_config",
    "ALL_PRESETS",
]
