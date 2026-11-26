"""Regression: the docs tell ONE budget story (fix 4).

`docs/training-1b-notebook.md` carried two contradictory budget stories: the old
"10 h default" table (62.1M / 157.6M / 406.6M tokens, 0.06-0.39 tokens/parameter,
"10 h default") and the corrected 4B-target / 2B-floor one. The stale framing is
gone; the corrected figures are what all three files say now, together with what
the code actually does for fp32 master weights, checkpoint size and the Drive
path (`OUT_DIR` resolves to `/content/drive/MyDrive/Talos/Argo_1B/stage-<a|b>`).
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


NOTEBOOK_DOC = _read("docs", "training-1b-notebook.md")
README = _read("README.md")
STATUS = _read("docs", "STATUS.md")


# ---------------------------------------------------------------------------
# The stale budget estimates are gone
# ---------------------------------------------------------------------------
def test_the_old_token_estimates_are_not_current_guidance():
    for stale in ("62.1M", "406.6M", "0.06\u20130.39", "0.06-0.39"):
        assert stale not in NOTEBOOK_DOC, (
            f"{stale!r} is the old 10-h budget framing and must not be presented "
            "as current guidance")


def test_no_ten_hour_default_is_documented():
    assert "tokens in 10 h" not in NOTEBOOK_DOC
    assert "10 h default" not in NOTEBOOK_DOC
    assert "no 20B default" not in NOTEBOOK_DOC.lower()


# ---------------------------------------------------------------------------
# The corrected story is what all three files say
# ---------------------------------------------------------------------------
def test_the_notebook_doc_states_the_4b_default_and_2b_floor():
    assert "4,000,000,000" in NOTEBOOK_DOC
    assert "STAGE_A_TOKENS_DEFAULT" in NOTEBOOK_DOC
    assert "2,000,000,000" in NOTEBOOK_DOC
    assert "STAGE_A_TOKENS_FLOOR" in NOTEBOOK_DOC
    assert "2B floor" in NOTEBOOK_DOC


def test_readme_and_status_state_the_4b_default_and_2b_floor():
    for text in (README, STATUS):
        assert "4B tokens" in text
        assert "2B floor" in text


def test_all_three_files_document_the_same_operational_facts():
    for text in (NOTEBOOK_DOC, README, STATUS):
        assert "fp32" in text                      # master weights
        assert "6.4 GiB" in text                   # checkpoint size
        assert "MyDrive/Talos/Argo_1B" in text     # the actual Drive path


def test_the_old_wrong_drive_path_and_checkpoint_size_are_gone():
    assert "MyDrive/Talos/Talos_1B" not in NOTEBOOK_DOC
    assert "~4.2 GB (bf16 weights" not in NOTEBOOK_DOC
    # The code's own default: fp32 master weights, 8-bit AdamW required on L4.
    assert "master_weight_dtype" in NOTEBOOK_DOC or "master weights" in NOTEBOOK_DOC
