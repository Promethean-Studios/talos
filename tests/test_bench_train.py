"""scripts/bench_train.py — CPU dry-run smoke test.

Builds the tiny packed corpus fixture, runs the full three-lane benchmark in
``--dry-run`` mode (CPU, 2 steps/lane), and asserts the pipeline end-to-end:
lane run dirs + metrics, the markdown table, the JSON results, the winner
selection and the recommended launch command.
"""
from __future__ import annotations

import json
import os

from scripts.bench_train import main

from tests.test_trainer_engine import _packed_mini


def test_bench_train_dry_run(tmp_path):
    packed_dir, tok_path, _paths, _manifest = _packed_mini(tmp_path)
    out = str(tmp_path / "bench")
    rc = main([
        "--packed-dir", packed_dir, "--preset", "tiny", "--dry-run",
        "--batch", "2", "--tokenizer-json", tok_path, "--out-dir", out,
        "--lanes", "old,fp32,fp16",
    ])
    assert rc == 0

    # All three lanes ran and each left the trainer's standard artifacts.
    bench = json.load(open(os.path.join(out, "bench.json")))
    rows = {r["lane"]: r for r in bench["rows"]}
    assert set(rows) == {"old", "fp32", "fp16"}
    for lane in ("old", "fp32", "fp16"):
        lane_dir = rows[lane]["out_dir"]
        assert os.path.isfile(os.path.join(lane_dir, "metrics.json"))
        assert os.path.isfile(os.path.join(lane_dir, "train_run_metadata.json"))
        assert rows[lane]["tok_s_run_wide"] is not None, lane
        assert rows[lane]["steps"] == 2, lane
        assert rows[lane]["budget_reached"] is True, lane
        # lane config sanity: old = plain fp32 sync; new lanes = sdpa prefetch
        if lane == "old":
            assert rows[lane]["attention_backend"] == "plain"
            assert rows[lane]["amp"] == "none"
            assert rows[lane]["prefetch"] == 0
            assert rows[lane]["pin_memory"] is False
        else:
            assert rows[lane]["attention_backend"] == "sdpa"
            assert rows[lane]["prefetch"] == 2
            assert rows[lane]["pin_memory"] is True
    assert bench["winner"] in rows

    md = open(os.path.join(out, "bench.md")).read()
    for needle in (
        "## Throughput + memory",
        "## Phase split (ms/step) + checkpoint I/O",
        "## Recommended launch command",
        "--attention-backend",
    ):
        assert needle in md, f"bench.md missing {needle}"
    # The recommended command must be shell-parseable: no comma thousands
    # separators inside numeric flags.
    rec = bench["recommended_command"]
    assert "--token-budget 250000000" in rec
    assert "," not in rec.replace(" ", "").strip("`")

    # --sidecar in standalone mode prints a lane's run-metadata JSON.
    sidecar_path = os.path.join(rows["fp32"]["out_dir"], "train_run_metadata.json")
    assert main(["--sidecar", sidecar_path]) == 0


def test_bench_train_rejects_unknown_lane(tmp_path, capsys):
    packed_dir, tok_path, _paths, _manifest = _packed_mini(tmp_path)
    try:
        main([
            "--packed-dir", packed_dir, "--preset", "tiny", "--dry-run",
            "--batch", "2", "--lanes", "old,bogus",
        ])
        raise AssertionError("expected SystemExit (argparse error)")
    except SystemExit as exc:
        assert exc.code == 2
        assert "unknown lane" in capsys.readouterr().err