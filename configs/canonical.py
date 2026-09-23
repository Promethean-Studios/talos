"""Per-preset canonical parameter-count registry.

Each Talos preset that carries an *owner-fixed canonical* parameter count is
registered here. Enforcement sites (the evaluation harness, the generation CLI,
the training guard and the test suite) resolve the expected count from a
rebuilt :class:`~model.config.ModelConfig`'s shape, so *every* canonical preset
is enforced against its exact count by the **same** code path.

Today that is three presets:

* ``tiny``     -> exactly   254,272 params (vocab 1024) — the original prototype;
* ``tiny_1m``  -> exactly 1,000,320 params (vocab 1024) — the ~1M scaling step
  (same architecture, scaled up; see docs/SCALING.md §3.1);
* ``tiny_10m`` -> exactly 9,952,320 params (vocab 1024) — the ~10M scaling step
  (same architecture, scaled up in width; see docs/SCALING.md §3).

The registry deliberately does **not** include the ``small``/``medium``/``large``
and MoE configs: those are design artifacts with *estimated* counts
(``configs.compute``), not owner-fixed canonical prototypes, so they are not
hard-enforced here.
"""
from __future__ import annotations

from typing import Tuple

from configs.presets import ALL_PRESETS
from model.config import ModelConfig


#: preset name -> (canonical parameter count, canonical vocab size)
CANONICAL_PRESETS: dict[str, Tuple[int, int]] = {
    "tiny": (254_272, 1024),
    "tiny_1m": (1_000_320, 1024),
    "tiny_10m": (9_952_320, 1024),
}


def signature(cfg: ModelConfig) -> Tuple[object, ...]:
    """A shape tuple identifying which preset a config corresponds to.

    Built from the fields that determine the parameter count, on the *derived*
    config (so derived ``head_dim``/``intermediate_size`` are folded in). Two
    configs with the same shape signature construct identically-shaped models.
    """
    d = cfg.derive() if cfg.head_dim is None else cfg
    return (
        d.ffn_type,
        d.tie_word_embeddings,
        d.vocab_size,
        d.hidden_size,
        d.num_layers,
        d.num_attention_heads,
        d.num_kv_heads,
        d.head_dim,
        d.intermediate_size,
    )


def resolve_preset(cfg: ModelConfig) -> str:
    """Return the canonical preset name matching ``cfg``'s shape, else raise.

    Raises ``ValueError`` when the config is not one of the registered canonical
    presets, so a drift or an unknown shape fails loudly at the enforcement site
    rather than silently passing.
    """
    sig = signature(cfg)
    for name in CANONICAL_PRESETS:
        if signature(ALL_PRESETS[name]().derive()) == sig:
            return name
    raise ValueError(
        f"config shape {sig} is not a registered canonical preset "
        f"(known canonical presets: {', '.join(sorted(CANONICAL_PRESETS))}) — "
        f"only canonical presets are hard-enforced at their exact count"
    )


def expected_params(cfg: ModelConfig) -> int:
    """Canonical parameter count for ``cfg`` (raises for unknown presets)."""
    return CANONICAL_PRESETS[resolve_preset(cfg)][0]


def expected_vocab(cfg: ModelConfig) -> int:
    """Canonical vocab size for ``cfg`` (raises for unknown presets)."""
    return CANONICAL_PRESETS[resolve_preset(cfg)][1]


def is_canonical(cfg: ModelConfig) -> bool:
    """Whether ``cfg`` is one of the registered canonical presets."""
    try:
        resolve_preset(cfg)
        return True
    except ValueError:
        return False
