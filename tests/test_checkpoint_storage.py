"""Checkpoint storage tiering: verified moves, bounded local disk, safe eviction.

The contract (owner's directive, 2026-11-24):

* the newest checkpoint stays on Drive (``OUT_DIR``, unchanged);
* older ones MOVE to local disk — copy → verify (sha256 **and** a real load
  through the existing loader/validator) → only then delete the original;
* a failed verification keeps the Drive original, warns, retries once, then
  leaves it alone;
* when local disk passes its byte budget the OLDEST local checkpoint goes first;
* the globally newest checkpoint and the only valid recovery checkpoint are
  never deleted;
* resume tries newest → older across BOTH stores, so a corrupt newest
  checkpoint falls back to an older one;
* a retention pass is idempotent.

No torch here: the loader/validator are injected, so the tests run on plain
files and prove the *policy*, not torch's deserializer.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOKS = _ROOT / "notebooks"
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))

import talos_checkpoints_storage as storage  # noqa: E402


# ---------------------------------------------------------------------------
# A tiny checkpoint stand-in: a JSON file the stub loader can read
# ---------------------------------------------------------------------------
def write_checkpoint(directory: Path, step: int, *, valid: bool = True,
                     filler: int = 1024) -> Path:
    """Write ``step-<N>.pt`` with ``filler`` bytes of padding (distinct sizes)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step}.pt"
    body = {"step": step, "valid": valid, "filler": "x" * int(filler)}
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def write_corrupt_checkpoint(directory: Path, step: int) -> Path:
    """A truncated / interrupted-upload checkpoint: same name, unreadable body."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step}.pt"
    path.write_bytes(b'{"step": %d, "truncated": ' % step)
    return path


def load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate(checkpoint) -> None:
    if not checkpoint.get("valid"):
        raise ValueError("checkpoint is not resumable")
    if "step" not in checkpoint:
        raise ValueError("checkpoint has no step")


def tiering(drive: Path, local: Path, *, limit_bytes: int = 10 ** 9) -> storage.CheckpointTiering:
    return storage.CheckpointTiering(
        str(drive), local_dir=str(local), limit_bytes=limit_bytes,
        load=load, validate=validate, warn=lambda message: warnings.append(message))


warnings: list = []


@pytest.fixture(autouse=True)
def _reset_warnings():
    warnings.clear()
    yield


# ---------------------------------------------------------------------------
# Happy path: newest on Drive, older verified onto local disk
# ---------------------------------------------------------------------------
def test_retention_keeps_the_newest_on_drive_and_moves_the_rest(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    originals = {step: write_checkpoint(drive, step, filler=step * 100).read_bytes()
                 for step in (1, 2, 3, 4)}
    tier = tiering(drive, local)

    report = tier.after_save()

    assert report.ok, report.errors
    assert report.newest_step == 4
    assert [storage.checkpoint_step(p) for p in tier.drive_checkpoints()] == [4]
    assert [storage.checkpoint_step(p) for p in tier.local_checkpoints()] == [3, 2, 1]
    # The local copies are byte-identical to what Drive held (verified move).
    for step, body in originals.items():
        if step == 4:
            continue
        assert (local / f"step-{step}.pt").read_bytes() == body
    assert report.moved and not report.kept_on_drive


def test_move_copies_verifies_and_only_then_deletes(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    source = write_checkpoint(drive, 7)
    body = source.read_bytes()
    tier = tiering(drive, local)

    destination = tier.move_to_local(str(source))

    assert destination == str(local / "step-7.pt")
    assert Path(destination).read_bytes() == body
    assert not source.exists(), "the Drive original must be deleted after verification"


def test_retention_pass_is_idempotent(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    for step in (1, 2, 3):
        write_checkpoint(drive, step)
    tier = tiering(drive, local)

    first = tier.after_save()
    snapshot = {p.name: p.read_bytes() for p in sorted(local.iterdir())}
    second = tier.after_save()

    assert first.moved and not second.moved and not second.evicted
    assert {p.name: p.read_bytes() for p in sorted(local.iterdir())} == snapshot
    assert [storage.checkpoint_step(p) for p in tier.drive_checkpoints()] == [3]


# ---------------------------------------------------------------------------
# Verification failures never destroy the only good copy
# ---------------------------------------------------------------------------
def test_a_copy_that_does_not_load_keeps_the_drive_original(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    source = write_checkpoint(drive, 2, valid=False)      # loads, fails validation
    tier = tiering(drive, local)

    assert tier.move_to_local(str(source)) is None
    assert source.exists(), "the Drive original must survive a failed verification"
    assert not (local / "step-2.pt").exists(), "the unverified copy must be removed"
    # One retry, then it gives up loudly — and says nothing was deleted.
    assert sum("attempt 1/2" in message for message in warnings) == 1
    assert any("ABANDONED" in message for message in warnings)
    assert any("Nothing was deleted" in message for message in warnings)


def test_an_identical_copy_that_does_not_load_is_still_rejected(tmp_path):
    """sha256 equality is not enough: the copy must also load as a checkpoint."""
    drive, local = tmp_path / "drive", tmp_path / "local"
    source = write_checkpoint(drive, 3)
    copy = local / "step-3.pt"
    local.mkdir(parents=True)
    copy.write_bytes(source.read_bytes())               # byte-identical

    def refusing_load(path: str):
        if os.path.abspath(path) == os.path.abspath(str(copy)):
            raise RuntimeError("torch.load: unexpected EOF")
        return load(path)

    tier = storage.CheckpointTiering(
        str(drive), local_dir=str(local), load=refusing_load, validate=validate,
        warn=warnings.append)

    with pytest.raises(storage.CopyVerificationError, match="does not load/validate"):
        tier.verify_copy(str(source), str(copy))
    assert source.exists()


def test_a_sha_mismatch_is_detected(tmp_path):
    tier = tiering(tmp_path / "drive", tmp_path / "local")
    left = write_checkpoint(tmp_path / "drive", 1, filler=64)
    right = tmp_path / "drive" / "step-1-copy.pt"
    right.write_bytes(left.read_bytes().replace(b'"valid": true', b'"valid": fals'))
    with pytest.raises(storage.CopyVerificationError, match="sha256 mismatch"):
        tier.verify_copy(str(left), str(right))


def test_a_short_copy_is_detected_before_any_hashing(tmp_path):
    tier = tiering(tmp_path / "drive", tmp_path / "local")
    left = write_checkpoint(tmp_path / "drive", 1, filler=4096)
    right = tmp_path / "drive" / "short.pt"
    right.write_bytes(b"{}")
    with pytest.raises(storage.CopyVerificationError, match="size mismatch"):
        tier.verify_copy(str(left), str(right))


def test_retention_never_raises_even_when_the_local_store_is_unusable(tmp_path):
    drive = tmp_path / "drive"
    write_checkpoint(drive, 1)
    write_checkpoint(drive, 2)
    # local_dir sits under a *file*, so creating the directory fails every time.
    tier = storage.CheckpointTiering(
        str(drive), local_dir=str(drive / "step-1.pt" / "nested"),
        load=load, validate=validate, warn=warnings.append)

    report = tier.after_save()          # must not raise

    assert report.ok, report.errors
    assert (drive / "step-1.pt").exists() and (drive / "step-2.pt").exists()
    assert len(report.kept_on_drive) == 1
    assert any("ABANDONED" in message for message in warnings)


# ---------------------------------------------------------------------------
# Eviction: oldest local first, at the byte budget
# ---------------------------------------------------------------------------
def test_eviction_deletes_the_oldest_local_checkpoint_first(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    write_checkpoint(drive, 9)                       # the newest, stays on Drive
    for step in (1, 2, 3, 4, 5):
        write_checkpoint(local, step, filler=1000)   # ~equal sizes
    size = os.path.getsize(local / "step-1.pt")
    tier = tiering(drive, local, limit_bytes=size * 3)

    evicted = tier.enforce_local_limit()

    remaining = [storage.checkpoint_step(p) for p in tier.local_checkpoints()]
    assert [storage.checkpoint_step(p) for p in evicted] == [1, 2]
    assert remaining == [5, 4, 3]
    assert storage.total_bytes(tier.local_checkpoints()) <= tier.limit_bytes


def test_the_globally_newest_checkpoint_is_never_evicted(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    drive.mkdir(parents=True)
    for step in (1, 2, 3):
        write_checkpoint(local, step, filler=1000)
    # The newest checkpoint of the whole run is local (e.g. a Drive write failed
    # and the run kept it): it must survive eviction even over budget.
    tier = tiering(drive, local, limit_bytes=1)

    evicted = tier.enforce_local_limit()

    assert [storage.checkpoint_step(p) for p in evicted] == [1, 2]
    assert (local / "step-3.pt").exists()
    assert any("newest checkpoint" in message for message in warnings)


def test_the_only_valid_recovery_checkpoint_is_never_evicted(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    write_corrupt_checkpoint(drive, 10)              # newest, unreadable
    write_checkpoint(local, 5, filler=1000)          # the only valid copy
    tier = tiering(drive, local, limit_bytes=1)

    evicted = tier.enforce_local_limit()

    assert evicted == []
    assert (local / "step-5.pt").exists()
    assert any("only valid recovery checkpoint" in message for message in warnings)


def test_eviction_is_allowed_when_drive_still_holds_a_valid_copy(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    write_checkpoint(drive, 10)
    write_checkpoint(local, 1, filler=1000)
    write_checkpoint(local, 2, filler=1000)
    tier = tiering(drive, local, limit_bytes=1)

    evicted = tier.enforce_local_limit()

    assert [storage.checkpoint_step(p) for p in evicted] == [1, 2]
    assert tier.local_checkpoints() == []


def test_a_store_with_no_local_checkpoints_is_not_an_error(tmp_path):
    tier = tiering(tmp_path / "drive", tmp_path / "local")
    assert tier.enforce_local_limit() == []
    assert tier.after_save().ok


# ---------------------------------------------------------------------------
# Resume: newest → older across both stores
# ---------------------------------------------------------------------------
def test_resume_candidates_are_newest_first_across_both_stores(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    for step in (4, 9):
        write_checkpoint(drive, step)
    for step in (2, 7):
        write_checkpoint(local, step)
    tier = tiering(drive, local)

    assert [(step, store) for step, store, _ in tier.candidates()] == [
        (9, "drive"), (7, "local"), (4, "drive"), (2, "local")]


def test_resume_falls_back_to_an_older_local_checkpoint(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    write_corrupt_checkpoint(drive, 10)              # the newest is broken
    write_checkpoint(drive, 9)                       # older Drive copy
    write_checkpoint(local, 6)                       # older local copy
    tier = tiering(drive, local)
    said = []

    path, checkpoint = tier.select_resume(log=said.append)

    assert path == str(drive / "step-9.pt")
    assert checkpoint["step"] == 9
    assert any("step-10" in message for message in said), said


def test_resume_reaches_the_local_store_when_every_drive_copy_is_broken(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    write_corrupt_checkpoint(drive, 10)
    write_checkpoint(local, 4)
    tier = tiering(drive, local)

    path, checkpoint = tier.select_resume()

    assert path == str(local / "step-4.pt") and checkpoint["step"] == 4


def test_resume_reports_every_failure_when_nothing_valid_exists(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    write_corrupt_checkpoint(drive, 3)
    write_checkpoint(local, 2, valid=False)
    tier = tiering(drive, local)

    with pytest.raises(FileNotFoundError) as excinfo:
        tier.select_resume()

    message = str(excinfo.value)
    assert "either store" in message
    assert "step-3" in message and "step-2" in message


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------
def test_the_policy_constants_are_the_documented_ones():
    assert storage.LOCAL_CHECKPOINT_DIR == "/content/talos_checkpoints"
    assert storage.LOCAL_CHECKPOINT_LIMIT_BYTES == 10 * 1024 ** 3


def test_step_parsing_is_numeric_not_lexicographic(tmp_path):
    for step in (9, 10, 100):
        write_checkpoint(tmp_path, step)
    assert [storage.checkpoint_step(p) for p in storage.list_checkpoints(str(tmp_path))] == [
        100, 10, 9]


def test_listing_a_missing_directory_is_empty():
    assert storage.list_checkpoints("/nonexistent/talos/checkpoints") == []
