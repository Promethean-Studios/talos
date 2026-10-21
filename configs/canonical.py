"""Per-preset canonical parameter-count registry.

Each Talos preset that carries an *owner-fixed canonical* parameter count is
registered here. Enforcement sites (the evaluation harness, the generation CLI,
the training guard and the test suite) resolve the expected count from a
rebuilt :class:`~model.config.ModelConfig`'s shape, so *every* canonical preset
is enforced against its exact count by the **same** code path.

Today that is six presets on two tracks:

The **tiny family** (the original four-scale ladder, vocab 1024):

* ``tiny``       -> exactly    254,272 params (vocab 1024) — the original prototype;
* ``tiny_1m``    -> exactly  1,000,320 params (vocab 1024) — the ~1M scaling step
  (same architecture, scaled up; see docs/SCALING.md §3.1);
* ``tiny_10m``   -> exactly  9,952,320 params (vocab 1024) — the ~10M scaling step
  (same architecture, scaled up in width; see docs/SCALING.md §3);
* ``tiny_100m``  -> exactly 96,482,304 params (vocab 1024) — the ~100M scaling
  step (hidden 1024 × 6 layers; the audit §13 candidate — see
  :func:`configs.presets.tiny_100m_config` for the arithmetic).

The **Talos 1B track** (owner-directed 2026-10, vocab 32,000 — a deliberate
divergence from the 1024 constant, tied embeddings):

* ``talos_1b``       -> exactly    995,973,120 params (vocab 32,000) — the
  owner's 1B spec at 21 layers (depth-adjusted to approach 1B);
* ``talos_1b_spec16``-> exactly    774,440,960 params (vocab 32,000) — the
  literal 16-layer spec variant.

See :func:`configs.presets.talos_1b_config` / :func:`talos_1b_spec16_config`
for the full arithmetic. The registry deliberately does **not** include the
``small``/``medium``/``large`` and MoE configs: those are design artifacts with
*estimated* counts (``configs.compute``), not owner-fixed canonical prototypes,
so they are not hard-enforced here.
"""
from __future__ import annotations

from typing import Tuple

from configs.presets import ALL_PRESETS
from configs.vocab import VOCAB_SIZE
from model.config import ModelConfig


#: preset name -> (canonical parameter count, canonical vocab size).
#: The four tiny-family entries read the single source of truth
#: (configs.vocab.VOCAB_SIZE) so a deliberate vocab change is one edit, not
#: four. The two 1B entries record their own 32,000 vocab inline: their token
#: configs define a separate, deliberately-larger vocabulary track and
#: ``tests/test_vocab_seam.py`` pins that divergence explicitly.
TALOS_1B_VOCAB_SIZE = 32_000
CANONICAL_PRESETS: dict[str, Tuple[int, int]] = {
    "tiny": (254_272, VOCAB_SIZE),
    "tiny_1m": (1_000_320, VOCAB_SIZE),
    "tiny_10m": (9_952_320, VOCAB_SIZE),
    "tiny_100m": (96_482_304, VOCAB_SIZE),
    "talos_1b": (995_973_120, TALOS_1B_VOCAB_SIZE),
    "talos_1b_spec16": (774_440_960, TALOS_1B_VOCAB_SIZE),
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
