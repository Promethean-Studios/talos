"""Regression: the full training-config snapshot and its resume contract (fix 2).

A checkpoint used to say *what* the model was and *how far* the run had got, but
never *how it was being trained*. Resuming across a changed seed, shuffle
buffer, optimizer, precision, effective batch, LR schedule or budget silently
mixed two different runs (and an 8-bit AdamW state dict cannot be loaded by a
plain AdamW at all).

These tests pin the contract:

* the snapshot round-trips through the checkpoint and through JSON;
* every rejected field is named in the error, with both values;
* ``RESUME_ACKNOWLEDGE`` acknowledges **exactly** the named fields (repo pin, a
  torch major/minor change, a larger budget) and nothing else;
* a budget extension needs the acknowledgement, a shrunk budget never passes;
* a torch patch-version change only warns, and a bitsandbytes availability
  change always rejects;
* an old checkpoint without a snapshot warns loudly and does not crash.

The comparison logic is torch-free, so almost all of this runs without torch;
the save/load round-trip test skips where torch is absent.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOTEBOOKS = os.path.join(_ROOT, "notebooks")
for _path in (_ROOT, _NOTEBOOKS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from talos_resume_config import (ACKNOWLEDGE_BUDGET_ENV, ACKNOWLEDGE_ENV,  # noqa: E402
                                 ResumeConfigError, TRAINING_CONFIG_SCHEMA,
                                 acknowledged_fields, build_training_config,
                                 dependency_versions, token_budget_extension,
                                 validate_training_config)

BUDGET = 4_000_000_000


def _config(**overrides):
    kwargs = dict(
        repo_pin="0893171",
        seed=1234,
        shuffle_buffer=10_000,
        optimizer="AdamW8bit (bitsandbytes)",
        peak_lr=3e-4,
        min_lr_ratio=0.1,
        warmup_ratio=0.005,
        grad_clip=1.0,
        weight_decay=0.1,
        precision="bf16",
        master_weight_dtype="fp32",
        micro_batch=4,
        grad_accum=16,
        seq_len=2048,
        token_budget=BUDGET,
        total_steps=30_518,
        versions={"torch": "2.9.0+cu124", "bitsandbytes": "0.45.0",
                  "datasets": "3.1.0", "tokenizers": "0.20.0",
                  "transformers": None},
    )
    kwargs.update(overrides)
    return build_training_config(**kwargs)


# ---------------------------------------------------------------------------
# The snapshot itself
# ---------------------------------------------------------------------------
def test_the_snapshot_records_every_contract_field():
    config = _config()
    assert config["schema"] == TRAINING_CONFIG_SCHEMA
    for field in ("repo_pin", "seed", "shuffle_buffer", "versions", "optimizer",
                  "precision", "master_weight_dtype", "effective_batch",
                  "lr_schedule", "token_budget", "total_steps"):
        assert field in config, field
    assert config["effective_batch"] == {"micro_batch": 4, "grad_accum": 16,
                                         "world_size": 1, "rows": 64,
                                         "tokens": 64 * 2048}
    assert config["lr_schedule"] == {"peak_lr": 3e-4, "min_lr_ratio": 0.1,
                                     "warmup_ratio": 0.005}
    # An empty pin (tracking main) is recorded as None: nothing to compare.
    assert _config(repo_pin="")["repo_pin"] is None


def test_the_snapshot_round_trips_through_json():
    config = _config()
    restored = json.loads(json.dumps(config))
    assert restored == config
    assert validate_training_config(restored, config) == []


def test_build_rejects_a_nonsense_config():
    with pytest.raises(ValueError):
        _config(optimizer="")
    with pytest.raises(ValueError):
        _config(precision="tf32")
    with pytest.raises(ValueError):
        _config(token_budget=0)
    with pytest.raises(ValueError):
        _config(micro_batch=0)


def test_dependency_versions_are_recorded_with_none_for_absent():
    import types
    versions = dependency_versions({"torch": types.SimpleNamespace(__version__="2.9.0")})
    assert versions["torch"] == "2.9.0"
    assert versions["bitsandbytes"] is None
    assert set(versions) == {"torch", "bitsandbytes", "datasets", "tokenizers",
                             "transformers"}


# ---------------------------------------------------------------------------
# Per-field rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field,override,needle", [
    ("seed", {"seed": 7}, "seed"),
    ("shuffle_buffer", {"shuffle_buffer": 1_000}, "shuffle_buffer"),
    ("optimizer", {"optimizer": "AdamW"}, "optimizer"),
    ("precision", {"precision": "fp16"}, "precision"),
    ("master_weight_dtype", {"master_weight_dtype": "bf16"}, "master_weight_dtype"),
    ("effective_batch", {"micro_batch": 2, "grad_accum": 32}, "effective_batch"),
    ("lr_schedule", {"peak_lr": 1e-4}, "lr_schedule"),
])
def test_each_rejected_field_is_named_with_both_values(field, override, needle):
    saved, current = _config(), _config(**override)
    with pytest.raises(ResumeConfigError) as excinfo:
        validate_training_config(saved, current)
    message = str(excinfo.value)
    assert needle in message
    assert "saved" in message and "current" in message
    # An acknowledgement never unlocks these.
    with pytest.raises(ResumeConfigError):
        validate_training_config(saved, current, acknowledge={field})


def test_every_mismatch_is_listed_not_just_the_first():
    saved = _config()
    current = _config(seed=99, precision="fp16", peak_lr=1e-4)
    with pytest.raises(ResumeConfigError) as excinfo:
        validate_training_config(saved, current)
    message = str(excinfo.value)
    assert "seed" in message and "precision" in message and "lr_schedule" in message


# ---------------------------------------------------------------------------
# Repo pin: reject by default, acknowledgeable by name
# ---------------------------------------------------------------------------
def test_a_repo_pin_change_rejects_by_default():
    with pytest.raises(ResumeConfigError, match="repo_pin"):
        validate_training_config(_config(repo_pin="aaa"), _config(repo_pin="bbb"))


def test_a_repo_pin_change_is_accepted_when_acknowledged():
    warnings = validate_training_config(_config(repo_pin="aaa"), _config(repo_pin="bbb"),
                                        acknowledge={"repo_pin"})
    assert any("ACKNOWLEDGED repo_pin" in line for line in warnings)


def test_acknowledging_another_field_does_not_unlock_the_repo_pin():
    with pytest.raises(ResumeConfigError, match="repo_pin"):
        validate_training_config(_config(repo_pin="aaa"), _config(repo_pin="bbb"),
                                 acknowledge={"seed", "token_budget"})


# ---------------------------------------------------------------------------
# Budget: extension needs the ack, shrink never passes
# ---------------------------------------------------------------------------
def test_a_budget_extension_requires_the_acknowledgement():
    saved, current = _config(token_budget=2_000_000_000), _config()
    with pytest.raises(ResumeConfigError, match="token_budget"):
        validate_training_config(saved, current)
    warnings = validate_training_config(saved, current, acknowledge={"token_budget"})
    assert any("ACKNOWLEDGED token_budget" in line for line in warnings)


def test_a_shrunk_budget_rejects_even_with_the_acknowledgement():
    saved = _config()
    current = _config(token_budget=2_000_000_000)
    with pytest.raises(ResumeConfigError) as excinfo:
        validate_training_config(saved, current, acknowledge={"token_budget"})
    assert "SHRUNK" in str(excinfo.value)


def test_the_extension_helper_records_only_growth():
    saved, bigger = _config(token_budget=2_000_000_000), _config()
    assert token_budget_extension(saved, bigger, acknowledge={"token_budget"}) == {
        "from": 2_000_000_000, "to": BUDGET, "acknowledged": True}
    assert token_budget_extension(bigger, saved) is None          # shrink
    assert token_budget_extension(bigger, bigger) is None         # unchanged


# ---------------------------------------------------------------------------
# Versions: patch warns, major/minor rejects, bitsandbytes always rejects
# ---------------------------------------------------------------------------
def test_a_torch_patch_change_only_warns():
    warnings = validate_training_config(
        _config(versions={"torch": "2.9.0+cu124"}),
        _config(versions={"torch": "2.9.1+cu124"}))
    assert any("torch patch version changed" in line for line in warnings)


def test_a_torch_major_minor_change_rejects_unless_acknowledged():
    saved = _config(versions={"torch": "2.9.0+cu124"})
    current = _config(versions={"torch": "2.10.0+cu124"})
    with pytest.raises(ResumeConfigError, match="torch_version"):
        validate_training_config(saved, current)
    warnings = validate_training_config(saved, current, acknowledge={"torch_version"})
    assert any("ACKNOWLEDGED torch_version" in line for line in warnings)


def test_losing_bitsandbytes_always_rejects_even_with_every_acknowledgement():
    saved = _config(optimizer="AdamW8bit (bitsandbytes)",
                    versions={"torch": "2.9.0+cu124", "bitsandbytes": "0.45.0"})
    current = _config(optimizer="AdamW", versions={"torch": "2.9.0+cu124",
                                                   "bitsandbytes": None})
    with pytest.raises(ResumeConfigError) as excinfo:
        validate_training_config(saved, current,
                                 acknowledge={"optimizer", "repo_pin",
                                              "torch_version", "token_budget"})
    message = str(excinfo.value)
    assert "bitsandbytes" in message and "cannot be loaded by plain AdamW" in message


def test_an_old_checkpoint_without_a_snapshot_warns_and_does_not_crash():
    for missing in (None, {}):
        warnings = validate_training_config(missing, _config())
        assert len(warnings) == 1
        assert "no training_config snapshot" in warnings[0]
        assert "WARNING" in warnings[0]


# ---------------------------------------------------------------------------
# The acknowledgement env var
# ---------------------------------------------------------------------------
def test_acknowledged_fields_parses_the_env_var():
    assert acknowledged_fields({}) == frozenset()
    assert acknowledged_fields({ACKNOWLEDGE_ENV: "repo_pin,torch_version"}) == {
        "repo_pin", "torch_version"}
    assert acknowledged_fields({ACKNOWLEDGE_ENV: " repo_pin token_budget "}) == {
        "repo_pin", "token_budget"}
    assert "token_budget" in acknowledged_fields({ACKNOWLEDGE_BUDGET_ENV: "1"})
    assert "token_budget" in acknowledged_fields({ACKNOWLEDGE_BUDGET_ENV: "true"})
    assert acknowledged_fields({ACKNOWLEDGE_BUDGET_ENV: "0"}) == frozenset()


# ---------------------------------------------------------------------------
# The real checkpoint writer/loader (torch)
# ---------------------------------------------------------------------------
def test_the_snapshot_survives_save_and_load_and_guards_the_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from configs.presets import ALL_PRESETS
    from model.gpt import TalosGPT
    from scripts.train_oasst1 import (load_checkpoint, save_checkpoint,
                                      validate_resume_checkpoint)

    model = TalosGPT(ALL_PRESETS["tiny"]().derive())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = _config(optimizer="AdamW", precision="fp32",
                     micro_batch=1, grad_accum=1, seq_len=64,
                     token_budget=4_096, total_steps=64)
    path = str(tmp_path / "step-1.pt")
    save_checkpoint(path, model, 1, 2.5, None, "", optimizer=optimizer,
                    training_config=config)

    ckpt = load_checkpoint(path)
    assert ckpt["training_config"] == config
    # The matching run resumes with no complaint.
    assert validate_resume_checkpoint(ckpt, "tiny", training_config=config) == []
    # A different seed is refused, naming the field.
    other = _config(optimizer="AdamW", precision="fp32", micro_batch=1,
                    grad_accum=1, seq_len=64, token_budget=4_096,
                    total_steps=64, seed=99)
    with pytest.raises(ResumeConfigError, match="seed"):
        validate_resume_checkpoint(ckpt, "tiny", training_config=other)
    # No snapshot passed => the pre-existing checks only, unchanged.
    assert validate_resume_checkpoint(ckpt, "tiny") == []


def test_an_old_checkpoint_without_the_field_still_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    from configs.presets import ALL_PRESETS
    from model.gpt import TalosGPT
    from scripts.train_oasst1 import (load_checkpoint, save_checkpoint,
                                      validate_resume_checkpoint)

    model = TalosGPT(ALL_PRESETS["tiny"]().derive())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "step-1.pt")
    save_checkpoint(path, model, 1, 2.5, None, "", optimizer=optimizer)
    ckpt = load_checkpoint(path)
    assert ckpt["training_config"] is None
    warnings = validate_resume_checkpoint(ckpt, "tiny", training_config=_config())
    assert len(warnings) == 1 and "no training_config snapshot" in warnings[0]
