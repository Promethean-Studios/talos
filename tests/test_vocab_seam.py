"""Vocab-constant seam test — prove every consumer reads the ONE constant.

The audit's §14 policy and the P1 fix list require a single canonical vocab
source of truth (:data:`configs.vocab.VOCAB_SIZE`) read by every seam that
decides "how many vocab slots does a Talos canonical model/tokenizer have":

* the preset model configs' ``vocab_size`` fields,
* the preset tokenizer configs (``configs.presets.preset_tokenizer_config``),
* the per-preset canonical registry (``configs.canonical.CANONICAL_PRESETS``),
* the tokenizer trainer CLI default (``tokenizer.train`` ``--vocab-size``).

If any seam stops agreeing with the constant (e.g. a future preset is added
with a hand-typed ``vocab_size=2048``), this file fails with the disagreement
named — making a deliberate vocab change a one-line edit to
``configs.vocab.VOCAB_SIZE`` plus this test's update, and anything else an
error.

**The one deliberate divergence:** the owner-directed Talos 1B track
(``talos_1b`` / ``talos_1b_spec16``) uses its own **32,000**-vocab tokenizer —
an explicit owner decision to move off the 1024 prototype contract. Those two
presets are pinned below as *intentionally different*, and every seam that
would silently reuse the 1024 constant for them (the tokenizer-config registry
pull, model↔tokenizer pairing) is asserted to use 32,000 instead.
"""
from __future__ import annotations

from configs.canonical import CANONICAL_PRESETS, TALOS_1B_VOCAB_SIZE
from configs.presets import (
    ALL_PRESETS,
    TOKENIZER_CONFIGS,
    preset_tokenizer_config,
)
from configs.vocab import VOCAB_SIZE
from tokenizer.vocab import (
    BASE_VOCAB_SIZE,
    DEFAULT_SPECIAL_TOKENS,
    TokenizerConfig,
)

CANONICAL_NAMES = ("tiny", "tiny_1m", "tiny_10m", "tiny_100m")
TALOS_1B_NAMES = ("talos_1b", "talos_1b_spec16")
N_SPECIAL_TOKENS = len(DEFAULT_SPECIAL_TOKENS)


def test_vocab_size_constant_is_1024() -> None:
    """Sanity-pin the constant itself so a typo cannot slip through."""
    assert VOCAB_SIZE == 1024


def test_all_preset_model_configs_read_vocab_size() -> None:
    """Every canonical preset model config uses the constant, never a literal."""
    for name in CANONICAL_NAMES:
        cfg = ALL_PRESETS[name]().derive()
        assert cfg.vocab_size == VOCAB_SIZE, (
            f"preset {name} vocab_size={cfg.vocab_size} disagrees with "
            f"configs.vocab.VOCAB_SIZE={VOCAB_SIZE}"
        )


def test_all_preset_tokenizer_configs_read_vocab_size() -> None:
    """Every canonical tokenizer config uses the constant via the registry."""
    for name in CANONICAL_NAMES:
        tok = preset_tokenizer_config(name)
        assert tok.vocab_size == VOCAB_SIZE, (
            f"preset {name} tokenizer vocab_size={tok.vocab_size} disagrees "
            f"with VOCAB_SIZE={VOCAB_SIZE}"
        )
    # The registry builder map itself must list exactly the canonical presets.
    assert set(TOKENIZER_CONFIGS) == set(CANONICAL_NAMES) | set(TALOS_1B_NAMES)


def test_canonical_registry_vocab_reads_vocab_size() -> None:
    """Every CANONICAL_PRESETS (params, vocab) tuple uses the constant."""
    for name in CANONICAL_NAMES:
        params, vocab = CANONICAL_PRESETS[name]
        assert vocab == VOCAB_SIZE, (
            f"registry entry {name} records vocab {vocab} but the canonical "
            f"constant is {VOCAB_SIZE}"
        )
        assert isinstance(params, int) and params > 0


def test_talos_1b_track_deliberately_uses_32000_vocab() -> None:
    """The 1B presets must NOT silently reuse the 1024 contract.

    Every 1B seam — model config, tokenizer config (registry + direct), and
    the canonical-registry entry — must agree on 32,000, and the pairing guard
    (:func:`configs.presets._assert_tokenizer_model_pairing`) must refuse a
    mismatched (e.g. the 1024) config.
    """
    assert TALOS_1B_VOCAB_SIZE == 32_000  # sanity-pin the 1B constant itself
    assert TALOS_1B_VOCAB_SIZE != VOCAB_SIZE  # the divergence is deliberate
    for name in TALOS_1B_NAMES:
        model_cfg = ALL_PRESETS[name]().derive()
        assert model_cfg.vocab_size == TALOS_1B_VOCAB_SIZE, (
            f"1B preset {name} vocab_size={model_cfg.vocab_size} must be "
            f"{TALOS_1B_VOCAB_SIZE}, never the canonical {VOCAB_SIZE}"
        )
        assert model_cfg.tie_word_embeddings is True
        tok = preset_tokenizer_config(name)  # registry path (model-compat seam)
        assert tok.vocab_size == TALOS_1B_VOCAB_SIZE
        assert tok.vocab_size <= model_cfg.vocab_size  # the ≤ contract still holds
        # The canonical registry entry agrees.
        params, vocab = CANONICAL_PRESETS[name]
        assert vocab == TALOS_1B_VOCAB_SIZE
        assert isinstance(params, int) and params > 0
    # The pairing guard itself observes real differences (not a vacuous pass).
    from configs.presets import _assert_tokenizer_model_pairing

    try:
        _assert_tokenizer_model_pairing(
            TokenizerConfig(vocab_size=VOCAB_SIZE), ALL_PRESETS["talos_1b"]()
        )
    except ValueError as exc:
        assert "1024" in str(exc) or "does not match" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("pairing guard accepted a 1024 tokenizer for talos_1b")


def test_tokenizer_trainer_cli_default_reads_vocab_size() -> None:
    """The tokenizer trainer CLI's --vocab-size default is the constant."""
    from tokenizer.train import make_arg_parser

    default = make_arg_parser().get_default("vocab_size")
    assert default == VOCAB_SIZE, (
        f"tokenizer.train --vocab-size default {default} disagrees with "
        f"VOCAB_SIZE={VOCAB_SIZE}"
    )


def test_vocab_size_matches_tokenizer_layout_arithmetic() -> None:
    """VOCAB_SIZE satisfies the byte-level BPE layout exactly (audit §1)."""
    cfg = TokenizerConfig(vocab_size=VOCAB_SIZE)
    assert BASE_VOCAB_SIZE + N_SPECIAL_TOKENS < VOCAB_SIZE  # room for merges
    # 1024 = 256 base bytes + 4 specials + 764 merge slots; specials sit at
    # ids 1020..1023 (top of vocab, LLaMA-style).
    assert cfg.max_merges() == VOCAB_SIZE - BASE_VOCAB_SIZE - N_SPECIAL_TOKENS == 764
    v = {
        "bos": cfg.bos_token,
        "eos": cfg.eos_token,
        "pad": cfg.pad_token,
        "unk": cfg.unk_token,
    }
    assert len(set(v.values())) == 4  # four distinct specials
    # A config that does NOT match the constant must fail the arithmetic too —
    # proving the test observes real differences (not vacuous passes).
    other = TokenizerConfig(vocab_size=VOCAB_SIZE + 1)
    assert other.max_merges() == 765