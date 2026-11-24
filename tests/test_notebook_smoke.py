"""Fix 8 — the end-to-end CPU smoke run the owner can execute before Colab.

Runs ``notebooks/talos_smoke.py`` as a subprocess: the canonical **tiny** preset
(254,272 parameters, vocab 1,024) through the *real* pipeline — pack a corpus,
train optimizer steps, save a checkpoint, resume from it, evaluate, generate,
publish tracker payloads, then a Stage-B run whose prompt masking is verified and
whose checkpoint a Stage-A resume refuses. No GPU, no network, a couple of
seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "notebooks", "talos_smoke.py")


def test_the_cpu_smoke_run_passes_end_to_end(tmp_path):
    out = tmp_path / "smoke"
    proc = subprocess.run([sys.executable, SMOKE, "--out", str(out)],
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, (proc.stdout[-4000:] + proc.stderr[-4000:])
    stdout = proc.stdout
    assert "PASSED" in stdout
    # It really trained, resumed and evaluated — not a dry run.
    assert "resume" in stdout and "evaluation" in stdout
    assert "stage B" in stdout
    # The four tracker metrics across the save -> resume -> save boundary:
    # cumulative elapsed, rolling tokens/sec, a non-negative ETA and a finite
    # train_loss on every row (checked inside the run, printed here).
    assert "metrics    :" in stdout
    assert "elapsed continues" in stdout

    # Artifacts: at least one checkpoint, and tracker payloads naming Argo.
    checkpoints = [p for p in out.rglob("*.pt")]
    assert checkpoints, f"no checkpoint written under {out}"
    statuses = [p for p in out.rglob("status.json")]
    assert statuses, "the tracker's status.json was not produced"
    payload = json.loads(statuses[0].read_text())
    assert payload.get("model_name") == "Argo 1B"
    assert "skipped_updates" in payload          # additive field, present and 0
    assert payload["skipped_updates"] == 0
