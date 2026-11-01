"""W4 — ``scripts/benchmark_1b.py``: the owner's RTX 3060 benchmark harness.

The team has no GPU, so these tests pin the parts that must be right *before*
the owner runs it on hardware: the OOM ladder's halving logic, the activation
matrix plan, the guard rails (no overwrite, no silent long run, no 1B fp32 on
a CPU box), the resume-harness verdict logic, the runtime projections, the
checkpoint-retention guarantee, and the results-JSON contract.

Everything here runs on CPU in well under a second: the ladder is exercised
through its injected ``attempt`` callable (no model), the verdict logic through
synthetic checkpoint/metrics dicts (no subprocess), and the one test that
touches a real model uses the 254K-parameter ``tiny`` preset at seq 8.
"""
from __future__ import annotations

import argparse
import json
import os

import pytest
import torch

from scripts.benchmark_1b import (
    DEFAULT_MAX_STEPS,
    LADDER_FLOOR,
    TOKEN_BUDGETS,
    GuardError,
    LadderResult,
    MatrixEntry,
    NvidiaSmiSampler,
    Projection,
    ResumeCheck,
    RetentionPolicy,
    TrainBenchConfig,
    TrainBenchResult,
    _evaluate_resume,
    build_matrix_plan,
    build_report,
    build_resume_cmd,
    checkpoint_storage_report,
    cpu_offload_note,
    format_summary,
    ladder_microbatches,
    make_arg_parser,
    prepare_work_dir,
    project_runtime,
    resolve_results_path,
    retention_decide,
    run_oom_ladder,
    run_train_bench,
    validate_guards,
    write_synthetic_corpus,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _args(**overrides) -> argparse.Namespace:
    """Parse a baseline CLI and apply overrides (the real arg surface)."""
    ns = make_arg_parser().parse_args(["--results-dir", "/tmp/x"])
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _ok(mb: int, tps: float = 100.0) -> TrainBenchResult:
    return TrainBenchResult(
        label=f"mb={mb}", config={}, ok=True, tokens=mb * 8,
        optimizer_steps=1, wall_s=1.0, tokens_per_sec=tps, peak_allocated_mib=100,
    )


def _oom(mb: int) -> TrainBenchResult:
    return TrainBenchResult(
        label=f"mb={mb}", config={}, ok=False, oom=True, reason="CUDA out of memory"
    )


# ---------------------------------------------------------------------------
# 1. OOM ladder
# ---------------------------------------------------------------------------
def test_ladder_halves_from_eight_to_one():
    assert ladder_microbatches(8) == [8, 4, 2, 1]


def test_ladder_non_power_of_two_always_reaches_the_floor():
    assert ladder_microbatches(12) == [12, 6, 3, 1]
    assert ladder_microbatches(3) == [3, 1]
    assert ladder_microbatches(1) == [LADDER_FLOOR]


def test_ladder_start_below_floor_is_clamped():
    assert ladder_microbatches(0) == [LADDER_FLOOR]


def test_ladder_stops_at_first_fitting_microbatch():
    seen: list[int] = []

    def attempt(mb: int) -> TrainBenchResult:
        seen.append(mb)
        return _oom(mb) if mb > 2 else _ok(mb, tps=50.0 * mb)

    result = run_oom_ladder(attempt, start=8, seq=2048)
    assert seen == [8, 4, 2]  # 1 is never attempted: 2 already fits
    assert result.winner_microbatch == 2
    assert result.winner_tokens_per_sec == 100.0
    assert result.max_tokens_per_optimizer_step == 2 * 2048
    assert [r.status for r in result.rungs] == ["oom", "oom", "ok"]
    assert result.seq == 2048


def test_ladder_reports_every_rung_when_nothing_fits():
    result = run_oom_ladder(lambda mb: _oom(mb), start=8, seq=2048)
    assert [r.microbatch for r in result.rungs] == [8, 4, 2, 1]
    assert all(r.status == "oom" for r in result.rungs)
    assert result.winner_microbatch is None
    assert result.max_tokens_per_optimizer_step is None


def test_ladder_distinguishes_errors_and_skips_from_oom():
    def attempt(mb: int) -> TrainBenchResult:
        if mb == 8:
            return TrainBenchResult(label="8", config={}, ok=False, reason="boom")
        if mb == 4:
            return TrainBenchResult(
                label="4", config={}, ok=False, skipped=True, reason="no bf16"
            )
        return _ok(mb)

    result = run_oom_ladder(attempt, start=8, seq=64)
    assert [r.status for r in result.rungs] == ["error", "skipped", "ok"]
    assert result.winner_microbatch == 2


def test_ladder_can_continue_past_first_success_when_asked():
    seen: list[int] = []

    def attempt(mb: int) -> TrainBenchResult:
        seen.append(mb)
        return _ok(mb, tps=float(mb))

    result = run_oom_ladder(attempt, start=4, seq=64, stop_at_first_ok=False)
    assert seen == [4, 2, 1]
    assert result.winner_microbatch == 4  # the FIRST fitting rung wins


# ---------------------------------------------------------------------------
# 2. activation matrix plan
# ---------------------------------------------------------------------------
def test_curated_matrix_covers_every_axis_without_a_full_cross_product():
    plan = build_matrix_plan(microbatches=[8], seq=2048)
    amps = {c.amp for c in plan}
    backends = {c.attention_backend for c in plan}
    ckpts = {c.grad_checkpoint for c in plan}
    accums = {c.grad_accum for c in plan}
    assert {"none", "bf16", "fp16"} <= amps
    assert {"sdpa", "plain"} <= backends
    assert ckpts == {False, True}
    assert max(accums) > 1
    assert len(plan) <= 12


def test_matrix_accum_divides_the_microbatch_to_hold_effective_batch():
    plan = build_matrix_plan(microbatches=[8], seq=2048)
    for cfg in plan:
        assert cfg.microbatch == max(1, 8 // cfg.grad_accum)
        assert cfg.seq == 2048


def test_full_matrix_is_the_cross_product_and_honours_the_cap():
    plan = build_matrix_plan(microbatches=[4], seq=64, full=True, max_combos=100)
    assert len(plan) == 3 * 2 * 2 * 2  # amps x backends x ckpt x accum
    capped = build_matrix_plan(microbatches=[4], seq=64, full=True, max_combos=5)
    assert len(capped) == 5


def test_matrix_respects_disabled_axes():
    plan = build_matrix_plan(
        microbatches=[4], seq=64, amps=("bf16",), backends=("sdpa",),
        grad_checkpoints=(False,), accums=(1,),
    )
    assert plan
    assert all(c.amp == "bf16" and c.attention_backend == "sdpa" for c in plan)
    assert all(not c.grad_checkpoint and c.grad_accum == 1 for c in plan)


def test_matrix_labels_are_unique_and_stable():
    plan = build_matrix_plan(microbatches=[8], seq=2048)
    labels = [c.label() for c in plan]
    assert len(labels) == len(set(labels))
    assert all("amp=" in label and "accum=" in label for label in labels)


def test_cpu_offload_is_reported_as_unsupported_not_faked():
    note = cpu_offload_note()
    assert note["supported"] is False
    assert note["attempted"] is False
    assert "grad-checkpoint" in note["note"]


# ---------------------------------------------------------------------------
# 3. guards
# ---------------------------------------------------------------------------
def test_defaults_are_safe():
    args = _args()
    validate_guards(args)  # does not raise
    assert args.max_steps == DEFAULT_MAX_STEPS
    assert args.preset == "talos_1b"
    assert args.force is False and args.confirm_full_run is False


def test_long_run_needs_confirm_full_run():
    with pytest.raises(GuardError, match="confirm-full-run"):
        validate_guards(_args(max_steps=DEFAULT_MAX_STEPS + 1))
    validate_guards(_args(max_steps=DEFAULT_MAX_STEPS + 1, confirm_full_run=True))


def test_warmup_plus_measure_must_fit_inside_max_steps():
    with pytest.raises(GuardError, match="exceeds --max-steps"):
        validate_guards(_args(warmup_steps=5, measure_steps=6, max_steps=10))


def test_bad_values_are_rejected():
    for kwargs in (
        {"max_steps": 0},
        {"measure_steps": 0},
        {"seq": 1},
        {"microbatch_start": 0},
        {"grad_accum": 0},
        {"preset": "not-a-preset"},
    ):
        with pytest.raises(GuardError):
            validate_guards(_args(**kwargs))


def test_one_billion_fp32_is_refused_on_cpu_without_an_opt_in():
    with pytest.raises(GuardError, match="1B|dry-run"):
        validate_guards(_args(device="cpu", preset="talos_1b", dry_run=False))
    # --dry-run (the CPU wiring check) is allowed: it uses the tiny preset.
    validate_guards(_args(device="cpu", preset="talos_1b", dry_run=True))


def test_one_billion_resume_preset_is_refused_on_cpu():
    with pytest.raises(GuardError, match="resume-preset"):
        validate_guards(
            _args(device="cpu", preset="tiny", resume_preset="talos_1b")
        )


def test_results_file_is_timestamped_and_refuses_overwrite(tmp_path):
    args = _args(results_dir=str(tmp_path))
    path = resolve_results_path(args, "20260101T000000Z")
    assert os.path.basename(path) == "benchmark_1b_talos_1b_20260101T000000Z.json"
    assert path.endswith(".json")

    open(path, "w", encoding="utf-8").write("{}")
    with pytest.raises(GuardError, match="refusing to overwrite"):
        resolve_results_path(args, "20260101T000000Z")
    # --force allows the replacement.
    forced = _args(results_dir=str(tmp_path), force=True)
    assert resolve_results_path(forced, "20260101T000000Z") == path


def test_work_dir_refuses_a_non_empty_directory_without_force(tmp_path):
    args = _args(results_dir=str(tmp_path))
    work = prepare_work_dir(args, "S1")
    open(os.path.join(work, "checkpoint.pt"), "w").close()
    with pytest.raises(GuardError, match="refusing"):
        prepare_work_dir(_args(results_dir=str(tmp_path)), "S1")
    assert prepare_work_dir(_args(results_dir=str(tmp_path), force=True), "S1") == work


# ---------------------------------------------------------------------------
# 4. resume harness verdict logic (pure, no subprocess)
# ---------------------------------------------------------------------------
def _ckpt(step: int = 4, batch: int = 2, seq: int = 64, loss: float = 5.0) -> dict:
    return {
        "step": step,
        "tokens_consumed": step * batch * (seq - 1),
        "train_loss": loss,
        "format": "talos-training-checkpoint-v1",
        "model_state_dict": {},
        "config": {},
    }


def _metrics(steps: int, tokens: int, loss: float) -> dict:
    return {"training": {"steps": steps, "tokens_consumed": tokens,
                         "final_train_loss": loss}}


def _evaluate(**overrides):
    kwargs = dict(
        preset="tiny",
        ckpt_path="/tmp/step-4.pt",
        ckpt=_ckpt(),
        ckpt_step=4,
        batch=2,
        seq=64,
        killed_rc=-9,
        resume_rc=0,
        ref_rc=0,
        resumed_metrics=_metrics(steps=8, tokens=12 * 2 * 63, loss=4.25),
        ref_metrics=_metrics(steps=12, tokens=12 * 2 * 63, loss=4.25),
        loss_rtol=0.05,
    )
    kwargs.update(overrides)
    # validate_resume_checkpoint needs a real shape check; the tiny preset with
    # an empty state dict would fail, so stub it out for the pure-logic tests.
    import scripts.benchmark_1b as mod

    original = mod.validate_resume_checkpoint
    mod.validate_resume_checkpoint = lambda ckpt, preset: None
    try:
        return _evaluate_resume(**kwargs)
    finally:
        mod.validate_resume_checkpoint = original


def test_resume_verdict_passes_on_a_consistent_run():
    ok, reason, checks, steps, tokens, losses = _evaluate()
    assert ok, reason
    assert reason == "all checks passed"
    assert steps["cumulative"] == 12
    assert steps["reference"] == 12
    assert tokens["resumed"] == tokens["reference"]
    assert losses["relative_diff"] == 0.0
    assert {c.name for c in checks} == {
        "checkpoint_valid", "checkpoint_accounting", "process_killed",
        "resume_completed", "reference_completed", "step_accounting",
        "token_accounting", "loss_continuity", "tokens_monotonic",
    }


def test_resume_verdict_fails_when_the_step_count_is_short():
    ok, reason, _, steps, _, _ = _evaluate(
        resumed_metrics=_metrics(steps=4, tokens=8 * 2 * 63, loss=4.25)
    )
    assert not ok
    assert "step_accounting" in reason
    assert steps["cumulative"] == 8


def test_resume_verdict_fails_when_the_process_was_not_killed():
    ok, reason, _, _, _, _ = _evaluate(killed_rc=0)
    assert not ok
    assert "process_killed" in reason


def test_resume_verdict_fails_when_losses_diverge():
    ok, reason, _, _, _, losses = _evaluate(
        resumed_metrics=_metrics(steps=8, tokens=12 * 2 * 63, loss=9.0)
    )
    assert not ok
    assert "loss_continuity" in reason
    assert losses["relative_diff"] > 0.05


def test_resume_verdict_fails_on_a_bad_checkpoint_token_count():
    bad = _ckpt()
    bad["tokens_consumed"] = 999
    ok, reason, _, _, _, _ = _evaluate(ckpt=bad)
    assert not ok
    assert "checkpoint_accounting" in reason


def test_tokens_per_step_is_batch_times_seq_minus_one():
    # The trainer scores x[:, 1:] against x[:, :-1]: 2 x (64 - 1) = 126.
    _, _, _, steps, _, _ = _evaluate()
    assert steps["tokens_per_step"] == 126


def test_resume_cmd_is_explicit_and_reproducible():
    cmd = build_resume_cmd(
        out_dir="/tmp/run", data="/tmp/c.jsonl", preset="tiny", device="cpu",
        seq=64, batch=2, epochs=3, steps_per_epoch=4, seed=0,
    )
    assert "-m" in cmd and "scripts.train_oasst1" in cmd
    assert cmd[cmd.index("--preset") + 1] == "tiny"
    assert "--resume" not in cmd
    resumed = build_resume_cmd(
        out_dir="/tmp/run", data="/tmp/c.jsonl", preset="tiny", device="cpu",
        seq=64, batch=2, epochs=3, steps_per_epoch=4, seed=0,
        resume="/tmp/run/step-4.pt",
    )
    assert resumed[resumed.index("--resume") + 1] == "/tmp/run/step-4.pt"


def test_synthetic_corpus_is_deterministic_and_needs_no_third_party_data(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert write_synthetic_corpus(str(a), n_docs=5, seed=7) == 5
    write_synthetic_corpus(str(b), n_docs=5, seed=7)
    assert a.read_bytes() == b.read_bytes()
    rows = [json.loads(line) for line in a.read_text().splitlines()]
    assert len(rows) == 5 and all(r["text"].strip() for r in rows)


# ---------------------------------------------------------------------------
# 5. runtime projections
# ---------------------------------------------------------------------------
def test_projections_cover_the_owner_budgets_and_are_labelled_measured():
    projs = project_runtime(
        1000.0, rate_is_measured=True, source="OOM ladder winner"
    )
    assert [p.token_budget for p in projs] == list(TOKEN_BUDGETS)
    assert [p.token_budget for p in projs] == [
        200_000_000, 600_000_000, 1_000_000_000, 10_000_000_000
    ]
    for p in projs:
        assert p.label == "measured-rate extrapolation"
        assert p.rate_is_measured is True
        # Reported hours are rounded to 3 decimals.
        assert p.hours == pytest.approx(p.token_budget / 1000.0 / 3600.0, abs=1e-3)


def test_projection_maths_is_monotonic_in_the_budget():
    projs = project_runtime(5.0, rate_is_measured=True, source="x")
    hours = [p.hours for p in projs]
    assert hours == sorted(hours)


def test_projection_without_a_rate_is_labelled_estimated_and_not_a_number():
    projs = project_runtime(None, rate_is_measured=False, source="no measurement")
    assert all(p.label == "estimated (no measured rate)" for p in projs)
    assert all(p.hours != p.hours for p in projs)  # NaN


def test_estimated_rate_is_labelled_differently_from_a_measured_one():
    est = project_runtime(10.0, rate_is_measured=False, source="guess")[0]
    assert est.rate_is_measured is False
    assert "not measured" in est.label


# ---------------------------------------------------------------------------
# 6. checkpoint storage + retention
# ---------------------------------------------------------------------------
def test_checkpoint_storage_counts_optimizer_state():
    report = checkpoint_storage_report(995_973_120, amp="bf16")
    params = 995_973_120
    assert report["weights_bytes"] == params * 4
    assert report["optimizer_bytes"] == params * 8  # AdamW: two fp32 moments
    assert report["total_bytes"] > report["weights_bytes"]
    assert report["weights_only_gib"] < report["total_gib"]
    # bf16 autocast must not be reported as shrinking the stored weights.
    assert "master weights stay fp32" in report["compute_dtype_note"]


def test_retention_keeps_latest_and_known_good_and_never_deletes():
    entries = [("/c/step-4.pt", 4), ("/c/step-8.pt", 8), ("/c/step-12.pt", 12)]
    decision = retention_decide(entries, known_good=["/c/step-4.pt"])
    assert "/c/step-12.pt" in decision["keep"]  # latest
    assert "/c/step-4.pt" in decision["keep"]  # known-good
    assert decision["would_delete"] == []
    assert decision["protected"] == ["/c/step-4.pt"]


def test_retention_never_deletes_a_known_good_even_when_deletion_is_enabled():
    policy = RetentionPolicy(delete_anything_automatically=True)
    entries = [("/c/step-4.pt", 4), ("/c/step-8.pt", 8), ("/c/step-12.pt", 12)]
    decision = retention_decide(
        entries, known_good=["/c/step-4.pt"], policy=policy
    )
    assert "/c/step-4.pt" in decision["keep"]
    assert "/c/step-4.pt" not in decision["would_delete"]
    assert decision["would_delete"] == ["/c/step-8.pt"]


def test_retention_handles_an_empty_directory():
    decision = retention_decide([])
    assert decision["keep"] == [] and decision["would_delete"] == []


# ---------------------------------------------------------------------------
# 7. measured train step on the tiny preset (CPU)
# ---------------------------------------------------------------------------
def _tiny_model():
    from scripts.train_oasst1 import build_preset_model

    return build_preset_model("tiny")


def test_train_bench_measures_tokens_per_second_on_cpu():
    model = _tiny_model()
    cfg = TrainBenchConfig(
        amp="none", attention_backend="sdpa", grad_checkpoint=False,
        grad_accum=1, microbatch=2, seq=8,
    )
    result = run_train_bench(
        model, cfg, torch.device("cpu"), warmup_steps=1, measure_steps=2,
        sample_gpu=False,
    )
    assert result.ok, result.reason
    assert result.tokens == 2 * 8 * 1 * 2
    assert result.optimizer_steps == 2
    assert result.tokens_per_sec and result.tokens_per_sec > 0
    assert result.ms_per_optimizer_step and result.ms_per_optimizer_step > 0
    # No CUDA: VRAM/gpu fields must be None, not a fabricated zero.
    assert result.peak_allocated_mib is None
    assert result.gpu is None


def test_train_bench_survives_grad_accum_and_checkpointing_on_cpu():
    model = _tiny_model()
    cfg = TrainBenchConfig(
        amp="bf16", attention_backend="sdpa", grad_checkpoint=True,
        grad_accum=2, microbatch=1, seq=8,
    )
    result = run_train_bench(
        model, cfg, torch.device("cpu"), warmup_steps=0, measure_steps=1,
        sample_gpu=False,
    )
    assert result.ok, result.reason
    assert result.tokens == 1 * 8 * 2 * 1


def test_run_train_bench_never_propagates_an_out_of_memory_error():
    """An OOM must come back as data, because the ladder depends on it."""
    import scripts.benchmark_1b as mod

    model = _tiny_model()
    cfg = TrainBenchConfig(microbatch=1, seq=8)

    def exploding_batch(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    original = mod.synthetic_batch
    mod.synthetic_batch = exploding_batch  # type: ignore[assignment]
    try:
        result = run_train_bench(
            model, cfg, torch.device("cpu"), warmup_steps=0, measure_steps=1,
            sample_gpu=False,
        )
    finally:
        mod.synthetic_batch = original  # type: ignore[assignment]
    assert result.ok is False
    assert result.oom is True
    assert "out of memory" in (result.reason or "")


# ---------------------------------------------------------------------------
# 8. GPU-absent degradation and the report contract
# ---------------------------------------------------------------------------
def test_nvidia_smi_sampler_degrades_cleanly_without_the_tool():
    sampler = NvidiaSmiSampler(interval_s=0.05)
    samples = sampler.stop()  # never started
    assert isinstance(samples.samples, int)
    assert samples.ok in (True, False)


def _report(**overrides):
    from scripts.benchmark_1b import EnvironmentReport, probe_environment

    env = probe_environment("cpu", torch.device("cpu"))
    kwargs = dict(
        args=_args(),
        env=env,
        model_info={"params": 254_272, "vocab_size": 1024, "layers": 2,
                    "hidden": 64},
        ladder=None,
        matrix=None,
        inference=None,
        resume=None,
        dry_run=True,
    )
    kwargs.update(overrides)
    return build_report(**kwargs)


def test_report_marks_cpu_runs_as_not_gpu_measurements():
    report = _report()
    assert report["validity"]["gpu_measurements"] is False
    assert report["validity"]["dry_run"] is True
    assert report["skipped"], "a CPU run must list what it could not measure"
    assert report["environment"]["cuda_available"] is False
    assert report["benchmark_version"]


def test_report_carries_projections_storage_and_retention():
    ladder = LadderResult(start_microbatch=4, seq=64)
    ladder.winner_microbatch = 4
    ladder.winner_tokens_per_sec = 1000.0
    report = _report(ladder=ladder)
    assert report["projection_basis"]["rate_is_measured"] is True
    assert report["projection_basis"]["tokens_per_sec"] == 1000.0
    assert len(report["projections"]) == len(TOKEN_BUDGETS)
    assert report["checkpoint_storage"]["total_bytes"] > 0
    assert report["retention"]["auto_delete_known_good"] is False
    assert report["measured"]["cpu_offload"]["supported"] is False


def test_report_without_any_measured_rate_says_so():
    report = _report()
    assert report["projection_basis"]["rate_is_measured"] is False
    assert all(p["label"].startswith("estimated") for p in report["projections"])


def test_report_is_json_serialisable():
    report = _report()
    text = json.dumps(report, default=str)
    assert "benchmark_version" in text
    assert json.loads(text)["validity"]["dry_run"] is True


def test_summary_prints_the_resume_verdict_and_skips():
    ladder = LadderResult(start_microbatch=8, seq=2048)
    ladder.winner_microbatch = 4
    ladder.winner_tokens_per_sec = 512.0
    ladder.rungs = []
    report = _report(ladder=ladder)
    text = format_summary(report)
    assert "OOM microbatch ladder" in text
    assert "RESUME HARNESS: not run" in text
    assert "skipped (no GPU)" in text
    assert "measured-rate extrapolation" in text
