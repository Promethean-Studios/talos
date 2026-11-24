"""Integration tests that EXECUTE the notebook's setup/config/logic cells.

Why this file exists: the CPU smoke run (``notebooks/talos_smoke.py``) never
executes cells 1, 2, 7, 8, 9, 11 or 12, which is how ``REPO_PIN_SHA = "__PIN__"``
shipped and made the notebook abort in its second cell. These tests run the real
cell *source* in a sandboxed namespace, with the side-effecting collaborators
(git, the checkpoint loader, the model, the accountant) stubbed — so the logic
is exercised without a GPU, a network or a real checkout.

Nothing here imports torch: the cells are executed as source, not as modules.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOKS = _ROOT / "notebooks"
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import talos_stages as stages  # noqa: E402

NOTEBOOK_PATH = _NOTEBOOKS / "talos_colab_train.ipynb"

#: The pin must be a full 40-hex commit SHA — `__PIN__`, "" or a short SHA is
#: not a commit and `git cat-file -e` fails on it.
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------------------
# Cell loading helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def notebook() -> Dict[str, Any]:
    with open(NOTEBOOK_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _code_cells(notebook: Dict[str, Any]) -> List[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


def _cell(notebook: Dict[str, Any], header: str) -> str:
    """The one code cell whose source contains ``header`` (a cell title line)."""
    matches = [source for source in _code_cells(notebook) if header in source]
    assert len(matches) == 1, f"expected exactly one cell containing {header!r}"
    return matches[0]


def _exec_source(source: str, namespace: Dict[str, Any], name: str) -> None:
    exec(compile(source, f"<notebook:{name}>", "exec"), namespace)


def _exec_function(source: str, function_name: str,
                   namespace: Dict[str, Any]) -> Any:
    """Compile and run one top-level ``def`` from a cell; return the function."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, f"<notebook:{function_name}>", "exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"the cell defines no {function_name}()")


class _Completed:
    """Stand-in for ``subprocess.CompletedProcess`` (no process is spawned)."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Fix 1 — the config cell's repo pin (executed, not regexed)
# ---------------------------------------------------------------------------
def _run_config_cell(notebook: Dict[str, Any], *, pin_override: Optional[str] = None,
                     has_commit: bool = True) -> Dict[str, Any]:
    """Execute cell 2 up to (not including) ``os.chdir(REPO_DIR)``.

    ``subprocess`` is replaced by a recorder (so no clone/fetch ever runs) and
    ``_git``/``_has_commit`` are replaced by stubs — the cell's own definitions
    of those two are stripped from the AST, otherwise they would shadow the
    stubs and shell out to real git.
    """
    source = _cell(notebook, "# 2. Repo (PINNED)")
    if pin_override is not None:
        source = re.sub(r'^REPO_PIN_SHA = "[^"]*"',
                        f'REPO_PIN_SHA = "{pin_override}"', source,
                        count=1, flags=re.M)
    prefix = source[:source.index("os.chdir(REPO_DIR)")]

    tree = ast.parse(prefix)
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, (ast.Import, ast.ImportFrom)))
        and not (isinstance(node, ast.FunctionDef) and node.name in {"_git", "_has_commit"})
    ]

    subprocess_calls: List[Any] = []

    def _fake_run(*args: Any, **kwargs: Any) -> _Completed:
        subprocess_calls.append((args, kwargs))
        return _Completed(0, "", "")

    git_calls: List[Any] = []

    def _git_stub(*args: Any, check: bool = True) -> str:
        git_calls.append(args)
        if args[:1] == ("rev-parse",):
            return namespace["REPO_PIN_SHA"]
        if args[:1] == ("show",):
            return "2026-11-19T00:00:00+00:00"
        return ""

    namespace: Dict[str, Any] = {
        "json": json, "math": __import__("math"), "os": os,
        "random": __import__("random"), "sys": sys,
        "time": __import__("time"), "Path": Path,
        "subprocess": types.SimpleNamespace(run=_fake_run),
        "_git": _git_stub,
        "_has_commit": lambda sha: has_commit,
        "_git_calls": git_calls,
        "_subprocess_calls": subprocess_calls,
    }
    exec(compile(tree, "<notebook:cell2>", "exec"), namespace)
    return namespace


def test_config_cell_pin_is_a_full_commit_sha(notebook):
    """``__PIN__``, an empty string or a short SHA must all fail this test."""
    namespace = _run_config_cell(notebook)
    pin = namespace["REPO_PIN_SHA"]
    assert isinstance(pin, str)
    assert FULL_SHA.match(pin), (
        f"REPO_PIN_SHA is {pin!r}: it must be a full 40-hex commit SHA, not a "
        "placeholder — `git cat-file -e` fails on anything else and cell 2 "
        "aborts with 'cannot obtain the pinned commit'")
    assert pin != "0" * 40


def test_config_cell_takes_the_pinned_checkout_path(notebook):
    """The non-empty pin branch fetches, checks out the pin and verifies HEAD."""
    namespace = _run_config_cell(notebook)
    pin = namespace["REPO_PIN_SHA"]
    calls = namespace["_git_calls"]
    assert ("checkout", "--force", pin) in calls, calls
    assert namespace["HEAD_SHA"] == pin
    assert namespace["COMMIT_DATE"] == "2026-11-19T00:00:00+00:00"
    # The clone is a subprocess call, not a `_git` call, and it ran through the
    # recorder — no real repository was touched.
    assert any("clone" in " ".join(str(part) for part in args[0])
               for args in namespace["_subprocess_calls"])


def test_config_cell_refuses_a_pin_it_cannot_obtain(notebook):
    """An unobtainable pin stops the cell with the loud, actionable error."""
    with pytest.raises(RuntimeError) as excinfo:
        _run_config_cell(notebook, has_commit=False)
    message = str(excinfo.value)
    assert "cannot obtain the pinned commit" in message
    assert "track main deliberately" in message


def test_the_pin_test_bites_on_the_old_placeholder(notebook):
    """Proof the pin tests fail on the pre-fix cell: ``__PIN__`` aborts cell 2."""
    with pytest.raises(RuntimeError, match="cannot obtain the pinned commit __PIN__"):
        _run_config_cell(notebook, pin_override="__PIN__", has_commit=False)
    # ... and the 40-hex assertion rejects it even if git somehow accepted it.
    namespace = _run_config_cell(notebook, pin_override="__PIN__", has_commit=True)
    assert not FULL_SHA.match(namespace["REPO_PIN_SHA"])


# ---------------------------------------------------------------------------
# Fix 6 — the setup cell's install list
# ---------------------------------------------------------------------------
def _install_list(cell_source: str) -> ast.List:
    tree = ast.parse(cell_source)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "INSTALL_PACKAGES"):
            assert isinstance(node.value, ast.List), "INSTALL_PACKAGES must be a list"
            return node.value
    raise AssertionError("the setup cell defines no INSTALL_PACKAGES list")


def test_setup_cell_install_list_is_well_formed_and_has_no_numpy(notebook):
    cell = _cell(notebook, "# 1. Setup")
    packages = _install_list(cell)
    names = []
    for element in packages.elts:
        assert isinstance(element, ast.Constant) and isinstance(element.value, str), (
            "every install entry must be a string literal")
        names.append(element.value)
    assert names, "the install list is empty"
    assert not any(name.split("=")[0].split(">")[0].split("<")[0].strip() == "numpy"
                   for name in names), (
        f"numpy must not be reinstalled (Colab ships a compatible one): {names}")
    for expected in ("datasets", "huggingface_hub", "bitsandbytes"):
        assert any(name.startswith(expected) for name in names), (expected, names)


def test_setup_cell_installs_the_list_it_defines(notebook):
    """The pip call splats INSTALL_PACKAGES — the list is the only source."""
    cell = _cell(notebook, "# 1. Setup")
    tree = ast.parse(cell)
    runs = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
    assert runs, "the setup cell has no subprocess.run call"
    assert any(
        isinstance(part, ast.Starred) and isinstance(part.value, ast.Name)
        and part.value.id == "INSTALL_PACKAGES"
        for call in runs for argument in call.args for part in ast.walk(argument)
    ), "the pip install call must splat INSTALL_PACKAGES"
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert "numpy" not in literals, "numpy must never be passed to pip"


# ---------------------------------------------------------------------------
# Fix 3 — Stage-B init must come from a Stage-A checkpoint
# ---------------------------------------------------------------------------
def _stage_b_initialiser(notebook: Dict[str, Any],
                         checkpoints: Dict[str, Dict[str, Any]],
                         loaded: List[Any]) -> Any:
    """The cell's real ``initialise_from_stage_a`` with the loader stubbed."""
    cell = _cell(notebook, "# 8. Model -> GPU")

    def _load(path: str) -> Dict[str, Any]:
        loaded.append(path)
        return checkpoints[path]

    model = types.SimpleNamespace(load_state_dict=lambda state: loaded.append(state))
    namespace = {
        "load_checkpoint": _load,
        "validate_resume_checkpoint": lambda ckpt, name: None,
        # The REAL guard: stubbing it would make these tests vacuous.
        "validate_stage_b_init": stages.validate_stage_b_init,
        "model": model,
        "TALOS_1B_NAME": "talos_1b",
    }
    return _exec_function(cell, "initialise_from_stage_a", namespace)


def _checkpoint(stage: Optional[str], tokens: int = 4_000_000_000) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"stage_name": "x"}
    if stage is not None:
        metadata["stage"] = stage
    return {"model_state_dict": {"w": 0}, "tokens_consumed": tokens,
            "run_metadata": metadata}


def test_stage_b_init_accepts_a_stage_a_checkpoint(notebook):
    loaded: List[Any] = []
    initialise = _stage_b_initialiser(
        notebook, {"stage-a/step-100.pt": _checkpoint(stages.STAGE_A)}, loaded)
    tokens = initialise("stage-a/step-100.pt")
    assert tokens == 4_000_000_000
    assert {"w": 0} in loaded, "the Stage-A weights must be loaded into the model"


def test_stage_b_init_refuses_a_stage_b_checkpoint(notebook):
    loaded: List[Any] = []
    initialise = _stage_b_initialiser(
        notebook, {"stage-b/step-9.pt": _checkpoint(stages.STAGE_B)}, loaded)
    with pytest.raises(ValueError) as excinfo:
        initialise("stage-b/step-9.pt")
    message = str(excinfo.value)
    assert "Stage B initializes only from a Stage-A checkpoint" in message
    assert "stage='B'" in message
    assert {"w": 0} not in loaded, "a refused checkpoint must not reach the model"


def test_stage_b_init_refuses_an_untagged_checkpoint(notebook):
    loaded: List[Any] = []
    initialise = _stage_b_initialiser(
        notebook, {"old/step-1.pt": _checkpoint(None)}, loaded)
    with pytest.raises(ValueError) as excinfo:
        initialise("old/step-1.pt")
    assert "Stage B initializes only from a Stage-A checkpoint" in str(excinfo.value)
    assert "stage=None" in str(excinfo.value)
    assert {"w": 0} not in loaded


# ---------------------------------------------------------------------------
# Fix 5 — the checkpoint inspection cell survives missing metadata
# ---------------------------------------------------------------------------
_FULL_METADATA: Dict[str, Any] = {
    "stage": "B",
    "stage_name": "conversational tuning",
    "dataset": {"id": "HuggingFaceH4/ultrachat_200k"},
    "token_budget": 4_000_000_000,
    "stage_a_tokens": 4_000_000_000,
    "peak_lr": 5e-5,
    "optimizer": "adamw8bit",
    "dataset_position": {"epoch": 1, "docs_consumed": 1_200, "rows_in_epoch": 3},
}


def _run_inspection_cell(notebook: Dict[str, Any],
                         run_metadata: Any) -> str:
    """Execute cell 11 against one checkpoint; return what it printed."""
    cell = _cell(notebook, "# 10. Inspect the newest checkpoint")
    checkpoint: Dict[str, Any] = {
        "format": stages.CHECKPOINT_FORMAT, "step": 50, "epoch": 1,
        "tokens_consumed": 12_345_678, "train_loss": 2.5, "val_loss": 2.75,
        "last_grad_norm": 0.9, "n_params": 1_036_060_672, "vocab_size": 49_152,
        "tokenizer_fingerprint": "abc123", "optimizer_state_dict": {},
        "rng_state": {},
    }
    if run_metadata is not None:
        checkpoint["run_metadata"] = run_metadata
    namespace = {
        "os": os, "OUT_DIR": "/tmp/out", "TALOS_1B_NAME": "talos_1b",
        "POSITION_METADATA_KEY": stages.POSITION_METADATA_KEY,
        "TOKENIZER_SHA256": "abc123", "spec": object(),
        "list_checkpoint_candidates": lambda directory: ["/tmp/out/step-50.pt"],
        "load_checkpoint": lambda path: checkpoint,
        "validate_resume_checkpoint": lambda ckpt, name: None,
        "validate_stage_resume": lambda *args, **kwargs: None,
    }
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _exec_source(cell, namespace, "cell11")
    return buffer.getvalue()


def test_inspection_cell_prints_a_full_metadata_checkpoint(notebook):
    output = _run_inspection_cell(notebook, dict(_FULL_METADATA))
    assert "tokens consumed   : 12,345,678 of budget 4,000,000,000" in output
    assert "stage-A tokens    : 4,000,000,000" in output
    assert "peak lr           : 5.00e-05" in output
    assert "unknown" not in output


def test_inspection_cell_survives_metadata_without_token_budget(notebook):
    metadata = dict(_FULL_METADATA)
    metadata.pop("token_budget")
    output = _run_inspection_cell(notebook, metadata)
    assert "of budget unknown" in output
    assert "stage-A tokens    : 4,000,000,000" in output


def test_inspection_cell_survives_empty_metadata(notebook):
    output = _run_inspection_cell(notebook, {})
    assert "tokens consumed   : 12,345,678 of budget unknown" in output
    assert "stage             : unknown (unknown)" in output
    assert "stage-A tokens    : unknown" in output
    # A checkpoint with no run_metadata at all is the same case.
    assert "of budget unknown" in _run_inspection_cell(notebook, None)


# ---------------------------------------------------------------------------
# Fix 4 — one authoritative MetricsSync/TrackerPublisher pair
# ---------------------------------------------------------------------------
def test_training_cell_constructs_no_metrics_or_tracker(notebook):
    loop = _cell(notebook, "# 9. Training loop")
    assert "MetricsSync(" not in loop, "cell 7 is the single creation site"
    assert "TrackerPublisher(" not in loop, "cell 7 is the single creation site"
    assert "metrics_sync.maybe_push(row)" in loop
    assert "tracker.maybe_push(row)" in loop
    assert "run cell 7 first" in loop


def test_training_cell_uses_the_cell_7_instances(notebook):
    """Executing the guard without cell 7 raises; with them it does not."""
    loop = _cell(notebook, "# 9. Training loop")
    tree = ast.parse(loop)
    guards = [node for node in tree.body
              if isinstance(node, ast.If) and "run cell 7 first" in ast.get_source_segment(loop, node)]
    assert len(guards) == 1, "expected exactly one 'run cell 7 first' guard"
    lines = loop.splitlines(keepends=True)
    seed = next(index for index, line in enumerate(lines)
                if "tracker.note_skipped(skipped_updates)" in line)
    start = guards[0].lineno - 1
    assert seed > start
    # The guard plus the resumed-run seed that follows it: the real cell source.
    source = "".join(lines[start:seed + 1])

    with pytest.raises(RuntimeError, match="run cell 7 first"):
        exec(compile(source, "<cell9-guard>", "exec"), {})

    tracker = types.SimpleNamespace(skipped=[], note_skipped=lambda n: tracker.skipped.append(n))
    namespace = {"metrics_sync": object(), "tracker": tracker, "skipped_updates": 3}
    exec(compile(source, "<cell9-guard>", "exec"), namespace)     # must not raise
    assert tracker.skipped == [3], "a resumed run's skip count is seeded into cell 7's tracker"


# ---------------------------------------------------------------------------
# Fix 2 — the verdict total keeps the Stage-A tokens on a resumed Stage B
# ---------------------------------------------------------------------------
def test_verdict_total_includes_stage_a_tokens_on_resume(notebook):
    """A resumed Stage-B run whose checkpoint carries stage_a_tokens reports it."""
    cell = _cell(notebook, "# 11. Evaluation")
    recorded: Dict[str, Any] = {}

    def _honest_verdict(**kwargs: Any) -> str:
        recorded.update(kwargs)
        return "VERDICT"

    namespace = {"honest_verdict": _honest_verdict, "STAGE_A": stages.STAGE_A,
                 "STAGE_A_TOKENS_TRAINED": 4_000_000_000,
                 "spec": types.SimpleNamespace(key=stages.STAGE_B),
                 "accountant": types.SimpleNamespace(consumed=50_000_000),
                 "n_params": 1_036_060_672, "eval_loss": 2.5, "reports": []}
    _exec_function(cell, "stage_a_tokens_for_verdict", namespace)
    tail = cell[cell.index("stage_a_tokens = stage_a_tokens_for_verdict("):]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _exec_source(tail, namespace, "cell12-tail")

    assert recorded["stage_a_tokens"] == 4_000_000_000
    assert "unknown" not in buffer.getvalue()


def test_verdict_reports_an_unknown_stage_a_count(notebook):
    """No stage_a_tokens anywhere: the count stays 0 and the note is printed."""
    cell = _cell(notebook, "# 11. Evaluation")
    recorded: Dict[str, Any] = {}

    def _honest_verdict(**kwargs: Any) -> str:
        recorded.update(kwargs)
        return "VERDICT"

    namespace = {"honest_verdict": _honest_verdict, "STAGE_A": stages.STAGE_A,
                 "STAGE_A_TOKENS_TRAINED": 0,
                 "spec": types.SimpleNamespace(key=stages.STAGE_B),
                 "accountant": types.SimpleNamespace(consumed=50_000_000),
                 "n_params": 1_036_060_672, "eval_loss": 2.5, "reports": []}
    _exec_function(cell, "stage_a_tokens_for_verdict", namespace)
    tail = cell[cell.index("stage_a_tokens = stage_a_tokens_for_verdict("):]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _exec_source(tail, namespace, "cell12-tail")

    assert recorded["stage_a_tokens"] == 0
    assert "Stage-A token count unknown" in buffer.getvalue()


# ---------------------------------------------------------------------------
# Checkpoint tiering + the four metrics, executed from the real cells
# ---------------------------------------------------------------------------
def _exec_nodes(source: str, namespace: Dict[str, Any], *, functions: tuple = (),
                assignments: tuple = (), name: str = "cell") -> None:
    """Run only the named top-level defs/assignments of a cell, in order."""
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in assignments
                for target in node.targets):
            body.append(node)
    assert body, f"none of {functions + assignments} found in the cell"
    exec(compile(ast.Module(body=body, type_ignores=[]),
                 f"<notebook:{name}>", "exec"), namespace)


def _retain_checkpoints(notebook, *, raises=False):
    cell = _cell(notebook, "# 8. Model -> GPU")

    class _Report:
        def summary(self):
            return "newest step 12 on Drive; moved 2 to local disk"

    tier = types.SimpleNamespace(
        after_save=(lambda: (_ for _ in ()).throw(RuntimeError("disk full"))
                    if raises else _Report()))
    namespace = {"CHECKPOINT_TIERING": tier}
    _exec_nodes(cell, namespace, functions=("retain_checkpoints",), name="retain")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = namespace["retain_checkpoints"]()
    return result, buffer.getvalue()


def test_cell_8_retention_pass_runs_and_reports(notebook):
    report, output = _retain_checkpoints(notebook)
    assert report is not None
    assert "checkpoints : newest step 12 on Drive" in output


def test_cell_8_retention_pass_never_kills_the_run(notebook):
    """A storage failure is a warning, not the end of a 20B-token run."""
    report, output = _retain_checkpoints(notebook, raises=True)
    assert report is None
    assert "WARNING: checkpoint retention failed" in output
    assert "training continues" in output


def test_cell_8_builds_the_tiering_from_out_dir(notebook):
    cell = _cell(notebook, "# 8. Model -> GPU")
    assert "CheckpointTiering(" in cell and "str(OUT_DIR)" in cell
    assert "LOCAL_CHECKPOINT_LIMIT_BYTES" in cell
    assert "TALOS_LOCAL_CHECKPOINT_DIR" in cell, "the local dir must be overridable"


def test_cell_9_calls_the_retention_pass_after_every_save(notebook):
    cell = _cell(notebook, "# 9. Training loop")
    body = cell[cell.index("def on_checkpoint(info):"):]
    body = body[:body.index("def on_log(info):")]
    assert "save_step(info, stream.position)" in body
    assert body.index("save_step(") < body.index("retain_checkpoints()"), (
        "the retention pass must run AFTER the checkpoint is written")
    # The elapsed time written to run_state.json is already cumulative.
    assert '"elapsed_s": info.elapsed_s' in body
    assert "prior_elapsed + info.elapsed_s" not in body


def test_cell_9_resumes_across_both_stores(notebook):
    cell = _cell(notebook, "# 9. Training loop")
    assert "CHECKPOINT_TIERING.resume_candidates()" in cell
    assert "list_checkpoint_candidates(str(OUT_DIR))" not in cell
    assert 'run_metadata") or {}).get("elapsed_s")' in cell
    assert "elapsed_offset_s=prior_elapsed" in cell
    assert "initial_train_loss=train_loss" in cell


def _on_log(notebook, *, last_finite):
    """Cell 9's real ``on_log`` with the publisher/display stubbed."""
    cell = _cell(notebook, "# 9. Training loop")
    rows: List[Dict[str, Any]] = []
    printed = io.StringIO()

    def _build(**kwargs):
        rows.append(kwargs)
        return types.SimpleNamespace()

    namespace = {
        "math": __import__("math"), "train_loss": float("nan"),
        "build_metrics_row": _build, "render_display_table": lambda row: "<table>",
        "HTML": lambda html: html,
        "display_handle": types.SimpleNamespace(update=lambda html: None),
        "metrics_sync": types.SimpleNamespace(maybe_push=lambda row: True),
        "tracker": types.SimpleNamespace(maybe_push=lambda row: True),
        "spec": types.SimpleNamespace(token_budget=4_000_000_000, key="A"),
        "GPU": types.SimpleNamespace(label="NVIDIA L4"),
        "MODEL_DISPLAY_NAME": "Argo 1B", "n_params": 1_036_060_672,
        "tokenizer": types.SimpleNamespace(vocab_size=49_152), "SEQ_LEN": 2048,
    }
    _exec_nodes(cell, namespace, functions=("on_log",), assignments=("LAST_FINITE_LOSS",),
                name="on_log")
    if last_finite is not None:
        namespace["LAST_FINITE_LOSS"]["value"] = last_finite
    return namespace["on_log"], rows, printed


def _info(**overrides):
    base = dict(step=10, epoch=1, tokens_consumed=1_000, elapsed_s=600.0,
                train_loss=2.5, val_loss=None, checkpoint="step-10.pt",
                skipped_updates=0, tokens_per_sec=2_000.0, eta_s=60.0)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_cell_9_publishes_a_finite_loss_and_the_rolling_rate(notebook):
    on_log, rows, _ = _on_log(notebook, last_finite=None)
    on_log(_info())
    assert len(rows) == 1
    assert rows[0]["train_loss"] == 2.5
    assert rows[0]["tokens_per_sec"] == 2_000.0
    assert rows[0]["eta_s"] == 60.0
    assert rows[0]["elapsed_s"] == 600.0


def test_cell_9_republishes_the_last_finite_loss_on_a_skipped_update(notebook):
    on_log, rows, _ = _on_log(notebook, last_finite=None)
    on_log(_info(train_loss=2.5))
    on_log(_info(step=11, train_loss=float("nan"), skipped_updates=1))
    assert [row["train_loss"] for row in rows] == [2.5, 2.5]
    assert rows[1]["skipped_updates"] == 1


def test_cell_9_publishes_nothing_when_no_finite_loss_exists_yet(notebook):
    on_log, rows, _ = _on_log(notebook, last_finite=None)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        on_log(_info(train_loss=float("nan")))
    assert rows == [], "a NaN point must never reach the graph"
    assert "no metrics row published" in buffer.getvalue()


def test_cell_9_on_checkpoint_records_the_cumulative_elapsed(notebook, tmp_path):
    cell = _cell(notebook, "# 9. Training loop")
    run_state = tmp_path / "run_state.json"
    saved: List[str] = []
    recorded: List[Any] = []
    namespace = {
        "json": json, "RUN_STATE": run_state, "prior_elapsed": 999.0,
        "save_step": lambda info, position: saved.append("step-10.pt") or "step-10.pt",
        "retain_checkpoints": lambda: recorded.append("retain"),
        "stream": types.SimpleNamespace(position="position"),
        "tracker": types.SimpleNamespace(note_checkpoint=lambda *a: recorded.append(a),
                                         note_skipped=lambda n: None),
        "spec": types.SimpleNamespace(key="A"), "last_ckpt_name": "",
    }
    _exec_nodes(cell, namespace, functions=("on_checkpoint",), name="on_checkpoint")
    namespace["on_checkpoint"](_info(elapsed_s=1_234.5))

    assert saved == ["step-10.pt"]
    assert "retain" in recorded, "the retention pass runs after every save"
    state = json.loads(run_state.read_text())
    assert state["elapsed_s"] == 1_234.5, (
        "elapsed_s is cumulative already: adding prior_elapsed would double-count")
    assert state["tokens_consumed"] == 1_000
