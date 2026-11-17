"""Tests for the notebook's honest evaluation (``talos_eval``).

These are the numbers that decide whether the notebook may call its output a
chatbot: repetition/degeneration, EOS behaviour, perplexity and the computed
verdict (which must never overstate the result).
"""
from __future__ import annotations

import os
import sys

import pytest

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import talos_eval as ev  # noqa: E402

EOS = 7
PARAMS = 1_036_060_672


# ---------------------------------------------------------------------------
# Repetition metrics
# ---------------------------------------------------------------------------
def test_ngrams_and_distinct_n():
    assert ev.ngrams([1, 2, 3], 2) == [(1, 2), (2, 3)]
    assert ev.ngrams([1], 2) == []
    assert ev.distinct_n([1, 2, 3, 4], 1) == 1.0
    assert ev.distinct_n([1, 1, 1, 1], 1) == pytest.approx(0.25)
    assert ev.distinct_n([], 1) == 0.0
    assert ev.distinct_n([1], 2) == 1.0
    with pytest.raises(ValueError):
        ev.ngrams([1], 0)


def test_most_repeated_ngram():
    gram, count = ev.most_repeated_ngram([1, 2, 3, 1, 2, 3, 1, 2, 3], 3)
    assert gram == (1, 2, 3) and count == 3
    assert ev.most_repeated_ngram([1], 3) == ((), 0)


# ---------------------------------------------------------------------------
# Generation analysis
# ---------------------------------------------------------------------------
def test_eos_stop_is_detected_and_excluded_from_the_metrics():
    report = ev.analyze_generation([10, 11, 12, EOS], max_new_tokens=8, eos_id=EOS)
    assert report.eos_stopped is True
    assert report.generated_tokens == 4
    assert report.degenerate is False


def test_running_to_the_cap_is_not_an_eos_stop():
    report = ev.analyze_generation([10, 11, 12, 13], max_new_tokens=4, eos_id=EOS)
    assert report.eos_stopped is False
    assert report.generated_tokens == 4


def test_a_looping_generation_is_flagged_degenerate():
    loop = [5, 6, 7] * 10
    report = ev.analyze_generation(loop, max_new_tokens=32, eos_id=EOS)
    assert report.degenerate is True
    assert report.max_ngram_repeat > ev.MAX_ACCEPTABLE_NGRAM_REPEAT
    assert report.distinct_2 < 1.0


def test_varied_text_is_not_flagged():
    varied = list(range(20, 60))
    report = ev.analyze_generation(varied, max_new_tokens=64, eos_id=EOS)
    assert report.degenerate is False
    assert report.distinct_1 == 1.0


def test_analysis_validates_its_inputs():
    with pytest.raises(ValueError):
        ev.analyze_generation([1], max_new_tokens=0, eos_id=EOS)


def test_report_serializes_to_plain_json_types():
    report = ev.analyze_generation([1, 2, 3, EOS], label="factual/greedy",
                                   prompt="User: hi", max_new_tokens=8, eos_id=EOS)
    payload = report.as_dict()
    assert payload["label"] == "factual/greedy"
    assert isinstance(payload["repeated_ngram"], list)
    assert isinstance(payload["distinct_2"], float)


# ---------------------------------------------------------------------------
# Loss, perplexity, verdict
# ---------------------------------------------------------------------------
def test_perplexity_is_exp_of_loss():
    assert ev.perplexity(0.0) == pytest.approx(1.0)
    assert ev.perplexity(2.0) == pytest.approx(7.389, rel=1e-3)
    assert ev.perplexity(None) is None
    assert ev.perplexity(float("nan")) is None
    assert ev.perplexity(1000.0) == float("inf")


def test_tokens_per_parameter_is_the_honesty_metric():
    assert ev.tokens_per_parameter(1_000_000, 1_000_000) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ev.tokens_per_parameter(10, 0)


def test_summarize_aggregates_rates():
    reports = [
        ev.analyze_generation([1, 2, 3, EOS], max_new_tokens=8, eos_id=EOS),
        ev.analyze_generation([4, 5, 6, 8], max_new_tokens=8, eos_id=EOS),
    ]
    summary = ev.summarize(reports)
    assert summary["count"] == 2
    assert summary["eos_rate"] == pytest.approx(0.5)
    assert ev.summarize([])["count"] == 0


def test_verdict_states_the_scale_and_never_claims_a_chatbot():
    reports = [ev.analyze_generation([1, 2, 3, EOS], max_new_tokens=8, eos_id=EOS)]
    verdict = ev.honest_verdict(stage_a_tokens=150_000_000, parameters=PARAMS,
                                val_loss=4.2, reports=reports)
    assert "0.145 tokens/parameter" in verdict
    assert "Chinchilla" in verdict
    assert "perplexity" in verdict
    assert "not" in verdict.lower()
    assert "working chatbot" not in verdict.replace("not evidence of a working chatbot", "")


def test_verdict_at_a_chinchilla_scale_budget_is_qualified_but_measured():
    verdict = ev.honest_verdict(stage_a_tokens=int(20 * PARAMS), parameters=PARAMS,
                                val_loss=2.0, reports=[])
    assert "near-Chinchilla" in verdict
    assert "evidence" in verdict


def test_eval_prompts_are_chat_formatted_and_general_purpose():
    assert len(ev.CHAT_EVAL_PROMPTS) >= 5
    for item in ev.CHAT_EVAL_PROMPTS:
        assert item["prompt"].startswith("User: ")
        assert item["prompt"].endswith("Assistant:")
        assert item["label"]
    labels = {item["label"] for item in ev.CHAT_EVAL_PROMPTS}
    assert {"factual", "reasoning", "creative"} <= labels


def test_eval_table_shows_the_flags():
    records = [{
        "label": "factual/greedy", "prompt": "User: hi", "answer": "hello",
        "generated_tokens": 4, "eos_stopped": True, "distinct_1": 1.0,
        "distinct_2": 0.9, "max_ngram_repeat": 1, "degenerate": True,
    }]
    table = ev.format_eval_table(records)
    assert "factual/greedy" in table
    assert "DEGENERATE" in table
