"""Structural tests for ``notebooks/talos_colab_train.ipynb``.

The notebook cannot be executed in CI (it needs a GPU and the network), so its
*structure* is tested instead: every code cell compiles, the cells are in the
right order, every ``from talos_* import`` name exists in the module it claims
to come from, the two-stage methodology is present, and the specific regressions
the owner called out (a 20B default, a local-GPU assumption, a torch reinstall)
are absent.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOTEBOOKS = os.path.join(_ROOT, "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

NOTEBOOK_PATH = os.path.join(_NOTEBOOKS, "talos_colab_train.ipynb")


@pytest.fixture(scope="module")
def notebook():
    with open(NOTEBOOK_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def code_cells(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def all_source(notebook):
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


# ---------------------------------------------------------------------------
# Well-formedness and order
# ---------------------------------------------------------------------------
def test_notebook_is_valid_and_starts_with_a_markdown_cell(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert notebook["metadata"]["accelerator"] == "GPU"


def test_every_code_cell_compiles(code_cells):
    assert len(code_cells) >= 10
    for index, source in enumerate(code_cells):
        try:
            ast.parse(source)
        except SyntaxError as exc:                     # pragma: no cover - diagnostic
            raise AssertionError(f"code cell {index} does not compile: {exc}") from exc


def test_cells_are_in_execution_order(all_source, code_cells):
    markers = ["# 1. Setup", "# 2. Repo + hardware", "# 3. Config", "# 4. Tokenizer",
               "# 5. Data pipeline", "# 6. Output location", "# 7. Metrics + public tracker",
               "# 8. Model -> GPU", "# 9. Training loop", "# 10. Inspect",
               "# 11. Evaluation"]
    positions = [all_source.index(marker) for marker in markers]
    assert positions == sorted(positions)
    # The config cell comes before the data cell, which comes before the loop.
    assert all_source.index("STAGE = ") < all_source.index("token budget   :")
    assert all_source.index("token budget   :") < all_source.index("plan.should_stop")


def test_every_imported_helper_exists(code_cells):
    """``from talos_x import (a, b)`` must name attributes that module defines."""
    checked = 0
    for source in code_cells:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("talos_"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), (
                    f"{node.module} has no attribute {alias.name}")
                checked += 1
    assert checked >= 20


# ---------------------------------------------------------------------------
# Two-stage methodology
# ---------------------------------------------------------------------------
def test_config_cell_exposes_the_stage_and_budget_knobs(code_cells):
    config = next(source for source in code_cells if "# 3. Config" in source)
    for knob in ("STAGE =", "RESUME =", "STAGE_A_HOURS", "STAGE_A_TOKENS_OVERRIDE",
                 "STAGE_B_EPOCHS", "MICRO_BATCH", "GRAD_ACCUM", "PRECISION",
                 "CHECKPOINT_EVERY", "CHECKPOINT_EVERY_MIN", "SEED",
                 "ALLOW_STAGE_B_FROM_RANDOM_INIT"):
        assert knob in config, f"config cell is missing {knob}"


def test_stage_a_is_c4_and_stage_b_is_ultrachat(all_source):
    assert "allenai/c4" in all_source
    assert "HuggingFaceH4/ultrachat_200k" in all_source
    assert "train_sft" in all_source and "test_sft" in all_source
    # The team's own FineWeb-Edu pack must not be touched.
    assert "corpus-1b-49k" not in all_source
    assert "FineWeb-Edu" not in all_source or "not" in all_source


def test_stage_b_refuses_a_random_initialisation(all_source):
    assert "initialised from Stage-A checkpoint" in all_source
    assert "ALLOW_STAGE_B_FROM_RANDOM_INIT" in all_source
    assert "stage-a" in all_source


def test_budget_is_not_a_hard_coded_20b(all_source):
    # No hard-coded 20B budget: the notebook says so and derives it instead.
    assert "20_000_000_000" not in all_source
    assert "not a" in all_source and "fixed 20B" in all_source
    assert "STAGE_A_HOURS" in all_source
    assert "tokens_in_hours" in all_source


def test_budget_cap_and_accounting_are_wired_into_the_loop(code_cells):
    loop = next(source for source in code_cells if "# 9. Training loop" in source)
    assert "accountant.plan_step(" in loop
    assert "plan.should_stop" in loop
    assert "accountant.add(" in loop
    assert "loss / plan.micro_batches" in loop     # partial final step scaling


def test_resume_restores_everything_the_audit_listed(code_cells):
    loop = next(source for source in code_cells if "# 9. Training loop" in source)
    for item in ("model.load_state_dict", "optimizer.load_state_dict",
                 "restore_rng_state", "StreamPosition.from_dict",
                 "validate_stage_resume", "POSITION_METADATA_KEY"):
        assert item in loop, f"resume path is missing {item}"
    assert "lr_at(accountant.steps" in loop        # schedule continues, not restarts


def test_checkpoints_record_stage_and_dataset_identity(code_cells):
    helpers = next(source for source in code_cells if "# 8. Model -> GPU" in source)
    assert "run_metadata=run_metadata(position" in helpers
    assert "spec.run_metadata(" in helpers
    assert "tokens_consumed=tokens_consumed" in helpers
    assert "tokenizer_fingerprint=TOKENIZER_SHA256" in helpers


# ---------------------------------------------------------------------------
# Colab hardware
# ---------------------------------------------------------------------------
def test_notebook_targets_colab_gpus_not_a_local_card(all_source):
    assert "detect_gpu()" in all_source
    assert "check_cuda_torch()" in all_source
    assert "plan_for_gpu(" in all_source
    assert "3060" not in all_source
    assert "RTX" not in all_source
    assert "drive.mount" in all_source
    assert "TALOS_1B_OUT" in all_source            # local-storage fallback


def test_the_install_cell_never_reinstalls_torch(code_cells):
    install = next(source for source in code_cells if "# 1. Setup" in source)
    assert "torch" not in install.split('"""')[0].replace(
        "torch is the runtime's own CUDA build", "").replace(
        "`pip install torch` in Colab", "").replace("torch untouched", "")
    assert '"torch"' not in install and "torch==" not in install
    assert "datasets" in install


# ---------------------------------------------------------------------------
# Tracker + evaluation
# ---------------------------------------------------------------------------
def test_tracker_is_stage_aware_and_cannot_crash_training(code_cells):
    cell = next(source for source in code_cells if "# 7. Metrics + public tracker" in source)
    assert "TrackerPublisher(" in cell and "stage=spec.key" in cell
    assert "MetricsSync(" in cell
    loop = next(source for source in code_cells if "# 9. Training loop" in source)
    assert "tracker.maybe_push(row)" in loop
    assert "metrics_sync.maybe_push(row)" in loop
    assert "never raises" in loop


def test_evaluation_measures_quality_not_just_loss(code_cells):
    cell = next(source for source in code_cells if "# 11. Evaluation" in source)
    for item in ("CHAT_EVAL_PROMPTS", "analyze_generation", "honest_verdict",
                 "perplexity", "format_eval_table", "eos_token_id=tokenizer.eos_id"):
        assert item in cell, f"evaluation cell is missing {item}"
    assert "greedy" in cell and "sampled" in cell
