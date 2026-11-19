"""Fix 2 — the Stage-A budget is a multi-billion-token floor, not "10 hours".

The owner's finding: ``STAGE_A_HOURS = 10`` × the notebook's estimated
tokens/sec gave 62M (T4) to 407M (A100) tokens for a 1.036B-parameter model —
0.06 to 0.39 tokens/parameter, a run that cannot learn language at all. These
tests pin the audit numbers, the corrected default, the floor warning, and the
exact Stage-B quantisation that keeps Stage B comparable with Stage A.
"""

from __future__ import annotations

import os
import sys

import pytest

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

from talos_stages import (STAGE_A_TOKENS_DEFAULT, STAGE_A_TOKENS_FLOOR,  # noqa: E402
                          budget_capability, stage_a_budget, stage_b_budget,
                          tokens_per_parameter)

#: 1,036,060,672 — the primary ``talos_1b`` preset (docs/SCALING.md §7).
PARAMS = 1_036_060_672

#: Estimated tokens/sec per GPU from the notebook's hardware plan
#: (``peak_TFLOPS × 1e12 × MFU / (8 × 1.036B)``), as tabulated in
#: docs/training-1b-notebook.md so the audit is reproducible.
GPU_TOKENS_PER_SEC = {"T4 16GB": 1_725.0, "L4 24GB": 4_379.0, "A100 40GB": 11_293.0}


# ---------------------------------------------------------------------------
# The audit: what the old rule actually produced
# ---------------------------------------------------------------------------
def test_the_old_ten_hour_default_was_below_one_token_per_parameter():
    for gpu, tokens_per_sec in GPU_TOKENS_PER_SEC.items():
        budget = stage_a_budget(tokens_per_sec=tokens_per_sec, hours=10.0)
        assert budget.source == "hours"
        assert budget.tokens == int(10.0 * 3600.0 * tokens_per_sec)
        assert budget.tokens < STAGE_A_TOKENS_FLOOR, gpu
        # 62M-407M tokens for 1.036B parameters: far under 1 token/parameter.
        assert tokens_per_parameter(budget.tokens, PARAMS) < 1.0, gpu
        assert budget.warning is not None, gpu
        assert "BELOW" in budget.warning
        assert "babble" in budget.warning
        assert budget.wall_clock_hours == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# The correction: default to a multi-billion-token budget, warn below the floor
# ---------------------------------------------------------------------------
def test_default_budget_is_multi_billion_tokens_on_every_gpu():
    assert STAGE_A_TOKENS_DEFAULT >= 2_000_000_000
    assert STAGE_A_TOKENS_FLOOR >= 2_000_000_000
    assert STAGE_A_TOKENS_DEFAULT >= STAGE_A_TOKENS_FLOOR
    for gpu, tokens_per_sec in GPU_TOKENS_PER_SEC.items():
        budget = stage_a_budget(tokens_per_sec=tokens_per_sec)   # hours = 0
        assert budget.source == "default", gpu
        assert budget.tokens == STAGE_A_TOKENS_DEFAULT
        assert budget.tokens >= STAGE_A_TOKENS_FLOOR
        assert budget.warning is None, gpu
        # At least 2 tokens/parameter: enough to be grammatical, not babble.
        assert tokens_per_parameter(budget.tokens, PARAMS) >= 2.0
        # The wall clock is still reported — the GPU decides the hours, not the
        # budget.
        assert budget.wall_clock_hours == pytest.approx(
            budget.tokens / tokens_per_sec / 3600.0)


def test_a_short_time_box_still_runs_but_warns_loudly():
    budget = stage_a_budget(tokens_per_sec=1_725.0, hours=2.0)
    assert budget.tokens == int(2.0 * 3600.0 * 1_725.0)
    assert budget.warning is not None
    assert "STAGE_A_HOURS" in budget.warning          # how to fix it
    assert "STAGE_A_TOKENS_OVERRIDE" in budget.warning


def test_override_wins_and_is_still_checked_against_the_floor():
    budget = stage_a_budget(tokens_per_sec=1_725.0, hours=10.0,
                            token_override=STAGE_A_TOKENS_DEFAULT)
    assert (budget.source, budget.tokens, budget.warning) == (
        "override", STAGE_A_TOKENS_DEFAULT, None)

    deliberately_short = stage_a_budget(tokens_per_sec=1_725.0,
                                        token_override=50_000_000)
    assert deliberately_short.source == "override"
    assert deliberately_short.warning is not None   # allowed, but not silent


# ---------------------------------------------------------------------------
# Bad configurations raise instead of quietly training nothing
# ---------------------------------------------------------------------------
def test_hours_without_a_throughput_estimate_is_an_error():
    with pytest.raises(ValueError, match="tokens_per_sec"):
        stage_a_budget(tokens_per_sec=None, hours=10.0)
    with pytest.raises(ValueError, match="tokens_per_sec"):
        stage_a_budget(tokens_per_sec=0.0, hours=10.0)


def test_invalid_knobs_are_errors():
    with pytest.raises(ValueError):
        stage_a_budget(tokens_per_sec=1_725.0, hours=-1.0)
    with pytest.raises(ValueError):
        stage_a_budget(tokens_per_sec=1_725.0, token_override=-5)
    with pytest.raises(ValueError):
        stage_a_budget(tokens_per_sec=1_725.0, floor=0)
    with pytest.raises(ValueError):
        stage_a_budget(tokens_per_sec=1_725.0, default_tokens=0)


def test_a_budget_that_rounds_to_zero_tokens_is_an_error():
    with pytest.raises(ValueError, match="0 tokens"):
        stage_a_budget(tokens_per_sec=1_725.0, hours=1e-9)


# ---------------------------------------------------------------------------
# Honest statements about what a budget buys
# ---------------------------------------------------------------------------
def test_capability_ladder_states_what_each_budget_buys():
    def capability(tokens: int) -> str:
        return budget_capability(tokens, PARAMS)

    assert "babble" in capability(400_000_000)          # ~0.39 tok/param
    assert "1-2 tokens" in capability(1_500_000_000)    # 1.45 tok/param
    assert "2-8 tokens" in capability(STAGE_A_TOKENS_DEFAULT)
    assert "8-15 tokens" in capability(10_000_000_000)
    assert "Chinchilla" in capability(20_000_000_000)   # ~19.3 tok/param
    # Chinchilla is a reference point, not a default: it is not imposed.
    assert STAGE_A_TOKENS_DEFAULT < 20_000_000_000


def test_chinchilla_reference_is_reachable_only_as_an_override():
    chinchilla = 20 * PARAMS
    budget = stage_a_budget(tokens_per_sec=11_293.0, token_override=chinchilla)
    assert budget.tokens == chinchilla
    assert budget.wall_clock_hours is not None and budget.wall_clock_hours > 400


# ---------------------------------------------------------------------------
# Stage B: budget the tokens the packer actually emits
# ---------------------------------------------------------------------------
def test_stage_b_budget_quantises_to_whole_packed_rows():
    # One epoch emits floor(tokens_per_epoch / seq_len) full rows and drops the
    # sub-row remainder; the budget must match that exactly.
    assert stage_b_budget(tokens_per_epoch=1_000, epochs=1, seq_len=128) == 896
    assert stage_b_budget(tokens_per_epoch=1_000, epochs=2, seq_len=128) == 1_792
    assert stage_b_budget(tokens_per_epoch=512, epochs=1, seq_len=512) == 512
    # Below one full row there is nothing to pack: loudly an error, not 0 tokens.
    with pytest.raises(ValueError, match="too small"):
        stage_b_budget(tokens_per_epoch=511, epochs=1, seq_len=512)


def test_stage_b_budget_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        stage_b_budget(tokens_per_epoch=1_000, epochs=0, seq_len=128)
    with pytest.raises(ValueError):
        stage_b_budget(tokens_per_epoch=-1, epochs=1, seq_len=128)
    with pytest.raises(ValueError):
        stage_b_budget(tokens_per_epoch=1_000, epochs=1, seq_len=0)
