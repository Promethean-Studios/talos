"""Checkpoint storage tiering for the Argo 1B notebook (Drive ↔ Colab local disk).

Why this exists: Google Drive gives the run a checkpoint that survives a Colab
recycle, but it is slow to write and the account has a quota — a 1B run that
keeps every ``step-*.pt`` on Drive eventually fills it and the *save* fails,
which is the one failure that loses real work. Colab's own disk is fast and
large but vanishes with the runtime. So the newest checkpoint lives on Drive
(the current ``OUT_DIR``, unchanged) and every older one is demoted to local
disk, where a byte budget bounds how much is kept.

The policy, made precise (owner's directive, 2026-11-24):

1. The **newest** checkpoint stays on Drive. Older Drive checkpoints are moved
   to :data:`LOCAL_CHECKPOINT_DIR`.
2. "Move" = copy → **verify** → only then delete the Drive original. Verify
   means byte-identical ``sha256`` *and* a successful load through the existing
   checkpoint loader/validator (``scripts.train_oasst1.load_checkpoint`` +
   ``validate_resume_checkpoint``). A failed verification keeps the Drive
   original, warns loudly, retries once, and then leaves it alone: the code
   never destroys the only good copy.
3. Local disk has a byte budget (:data:`LOCAL_CHECKPOINT_LIMIT_BYTES`). When it
   is exceeded, the **oldest** local checkpoint is deleted first.
4. Two things are never deleted: the globally newest checkpoint, and the only
   valid recovery checkpoint (when the Drive copy is missing or corrupt and a
   local copy is the sole valid one).
5. Resume tries candidates newest → older **across both stores** through the
   existing validator, so a corrupt newest checkpoint falls back to the next
   older one instead of ending the run.

Nothing here changes the checkpoint format (``talos-training-checkpoint-v1``),
the save/load APIs, or what ``OUT_DIR`` means: the module is called *after* a
successful save and only moves whole files around.

The module imports no ``torch`` at import time (the loader is imported lazily),
so it is unit-testable with plain files and a stub loader.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("talos.notebook.checkpoints_storage")

#: Where demoted checkpoints live on the Colab runtime's local disk. One
#: constant, overridable (the notebook reads an env var, tests pass a tmp dir).
LOCAL_CHECKPOINT_DIR = "/content/talos_checkpoints"

#: Local-disk budget for those demoted checkpoints: 10 GiB, bytes-based, one
#: constant, overridable. Colab gives ~78 GB of ``/content``; the rest of it is
#: the tokenized corpus and the model's activation scratch space.
LOCAL_CHECKPOINT_LIMIT_BYTES = 10 * 1024 ** 3

#: ``step-<N>.pt`` — the only checkpoint filename shape the run writes.
CHECKPOINT_NAME_RE = re.compile(r"^step-(\d+)\.pt$")

#: Read size for hashing/copying (4 MiB: a 4 GB checkpoint is ~1000 reads).
_CHUNK = 4 * 1024 * 1024

#: ``(path) -> checkpoint dict``; the default lazily imports the repo loader.
Loader = Callable[[str], Any]
#: ``(checkpoint dict) -> None``; raises on an invalid checkpoint.
Validator = Callable[[Any], None]
#: ``(message) -> None``; the notebook passes ``print``, tests collect.
Warn = Callable[[str], None]


class CopyVerificationError(RuntimeError):
    """A copied checkpoint did not match its source (size, sha256 or load)."""


# ---------------------------------------------------------------------------
# Default loader / validator (lazy: no torch at import time)
# ---------------------------------------------------------------------------
def default_loader(path: str) -> Any:
    """Load a checkpoint with the repo's own loader (``torch.load``, CPU)."""
    from scripts.train_oasst1 import load_checkpoint  # lazy: imports torch

    return load_checkpoint(path)


def make_validator(preset: str = "talos_1b") -> Validator:
    """A validator that runs the repo's existing resume checks for ``preset``."""

    def _validate(checkpoint: Any) -> None:
        from scripts.train_oasst1 import validate_resume_checkpoint  # lazy

        validate_resume_checkpoint(checkpoint, preset)

    return _validate


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def checkpoint_step(path: str) -> Optional[int]:
    """The step number in ``.../step-<N>.pt``, or ``None`` for any other name."""
    match = CHECKPOINT_NAME_RE.match(os.path.basename(str(path)))
    return int(match.group(1)) if match else None


def list_checkpoints(directory: str) -> List[str]:
    """``step-<N>.pt`` files under ``directory``, NUMERIC step descending.

    Newest first, and numeric (never lexicographic: ``step-9`` sorts "after"
    ``step-100`` in string order). A missing directory is an empty list — the
    local store does not exist until the first demotion creates it.
    """
    try:
        names = os.listdir(directory)
    except (FileNotFoundError, NotADirectoryError):
        return []
    entries: List[Tuple[int, str]] = []
    for name in names:
        step = checkpoint_step(name)
        if step is not None:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                entries.append((step, path))
    entries.sort(key=lambda pair: pair[0], reverse=True)
    return [path for _, path in entries]


def total_bytes(paths: Sequence[str]) -> int:
    """Sum of the files' sizes; a file that vanished counts as 0 bytes."""
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total


def sha256_file(path: str) -> str:
    """Streaming sha256 of a file (4 MiB chunks: no full copy in memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RetentionReport:
    """What one retention pass did — printed by the notebook, asserted by tests."""

    newest_step: Optional[int] = None
    moved: List[str] = field(default_factory=list)
    evicted: List[str] = field(default_factory=list)
    kept_on_drive: List[str] = field(default_factory=list)
    local_bytes: int = 0
    limit_bytes: int = 0
    over_limit: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = [f"newest step {self.newest_step} on Drive"]
        if self.moved:
            parts.append(f"moved {len(self.moved)} to local disk")
        if self.evicted:
            parts.append(f"evicted {len(self.evicted)} oldest local")
        if self.kept_on_drive:
            parts.append(f"{len(self.kept_on_drive)} left on Drive (unverified)")
        parts.append(f"local {self.local_bytes / 2**30:.2f}/"
                     f"{self.limit_bytes / 2**30:.2f} GiB")
        if self.over_limit:
            parts.append("OVER the local limit — nothing left to evict safely")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s): {self.errors[0]}")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------
class CheckpointTiering:
    """Newest checkpoint on Drive, older ones verified onto local disk.

    Args:
        drive_dir: the run's ``OUT_DIR`` (Drive when mounted). Unchanged
            semantics: the newest checkpoint is written here by the save path.
        local_dir: where older checkpoints are demoted to.
        limit_bytes: local-disk budget; the oldest local checkpoint is evicted
            first when it is exceeded.
        preset: preset name for the default validator (``talos_1b``).
        load: injectable loader (default: the repo's ``load_checkpoint``).
        validate: injectable validator (default: the repo's
            ``validate_resume_checkpoint`` for ``preset``).
        warn: injectable warning sink (default: :data:`logger.warning`).
    """

    def __init__(
        self,
        drive_dir: str,
        *,
        local_dir: str = LOCAL_CHECKPOINT_DIR,
        limit_bytes: int = LOCAL_CHECKPOINT_LIMIT_BYTES,
        preset: str = "talos_1b",
        load: Optional[Loader] = None,
        validate: Optional[Validator] = None,
        warn: Optional[Warn] = None,
    ) -> None:
        if not str(drive_dir):
            raise ValueError("drive_dir must be a non-empty path")
        if int(limit_bytes) < 0:
            raise ValueError("limit_bytes must be >= 0")
        self.drive_dir = str(drive_dir)
        self.local_dir = str(local_dir)
        self.limit_bytes = int(limit_bytes)
        self._load: Loader = load or default_loader
        self._validate: Validator = validate or make_validator(preset)
        self._warn: Warn = warn or (lambda message: logger.warning("%s", message))
        #: path -> is a valid resumable checkpoint (one load per file per pass).
        self._valid_cache: Dict[str, bool] = {}

    # -- stores -------------------------------------------------------------
    def drive_checkpoints(self) -> List[str]:
        """``step-<N>.pt`` under ``drive_dir``, newest first."""
        return list_checkpoints(self.drive_dir)

    def local_checkpoints(self) -> List[str]:
        """``step-<N>.pt`` under ``local_dir``, newest first ([] if absent)."""
        return list_checkpoints(self.local_dir)

    def candidates(self) -> List[Tuple[int, str, str]]:
        """Every checkpoint in both stores as ``(step, store, path)``.

        Newest first. On an equal step the Drive copy comes first: it is the
        store the run wrote last, and the local copy is its verified duplicate,
        so trying Drive first then falling back costs nothing when both are good
        and recovers when the Drive copy is the corrupt one.
        """
        entries: List[Tuple[int, int, str, str]] = []
        for path in self.drive_checkpoints():
            entries.append((int(checkpoint_step(path) or 0), 0, "drive", path))
        for path in self.local_checkpoints():
            entries.append((int(checkpoint_step(path) or 0), 1, "local", path))
        entries.sort(key=lambda item: (-item[0], item[1]))
        return [(step, store, path) for step, _, store, path in entries]

    def resume_candidates(self) -> List[str]:
        """Checkpoint paths newest → older, across BOTH stores."""
        return [path for _, _, path in self.candidates()]

    def newest_step(self) -> Optional[int]:
        """The highest step number in either store (``None`` when empty)."""
        steps = [step for step, _, _ in self.candidates()]
        return max(steps) if steps else None

    # -- verification -------------------------------------------------------
    def is_valid(self, path: str) -> bool:
        """Does ``path`` load AND pass the resume validator? (cached per pass)"""
        if path not in self._valid_cache:
            try:
                checkpoint = self._load(path)
                self._validate(checkpoint)
            except Exception as exc:  # noqa: BLE001 — any failure means "not valid"
                logger.info("checkpoint %s is not a valid resume point: %s", path, exc)
                self._valid_cache[path] = False
            else:
                self._valid_cache[path] = True
        return self._valid_cache[path]

    def verify_copy(self, source: str, copy: str) -> None:
        """Raise :class:`CopyVerificationError` unless ``copy`` == ``source``.

        Three independent checks, cheapest first: size, sha256 (byte-identical),
        and a real load through the checkpoint loader + validator (a
        byte-identical file that no longer loads is still not a checkpoint).
        """
        try:
            source_size = os.path.getsize(source)
            copy_size = os.path.getsize(copy)
        except OSError as exc:
            raise CopyVerificationError(f"cannot stat the copy pair: {exc}") from exc
        if source_size != copy_size:
            raise CopyVerificationError(
                f"size mismatch: {source} is {source_size:,} bytes, the copy "
                f"{copy_size:,}")
        if source_size == 0:
            raise CopyVerificationError(f"{copy} is empty (0 bytes)")
        source_sha = sha256_file(source)
        copy_sha = sha256_file(copy)
        if source_sha != copy_sha:
            raise CopyVerificationError(
                f"sha256 mismatch: {source} {source_sha[:16]}… != {copy} "
                f"{copy_sha[:16]}…")
        try:
            checkpoint = self._load(copy)
            self._validate(checkpoint)
        except Exception as exc:  # noqa: BLE001 — reported as a failed verification
            raise CopyVerificationError(
                f"the copy does not load/validate as a checkpoint: "
                f"{type(exc).__name__}: {exc}") from exc

    # -- move ---------------------------------------------------------------
    def move_to_local(self, path: str, *, attempts: int = 2) -> Optional[str]:
        """Copy ``path`` to local disk, verify it, then delete the original.

        Returns the local path on success, ``None`` when the move was abandoned
        (the original is then still in place). ``attempts=2`` is the owner's
        "one retry": a failed verification is retried once, then the Drive
        original is left alone with a loud warning.
        """
        destination = os.path.join(self.local_dir, os.path.basename(path))
        for attempt in range(1, max(1, int(attempts)) + 1):
            try:
                os.makedirs(self.local_dir, exist_ok=True)
                # Remove any previous partial copy first: copy2 would overwrite
                # it, but a stale file that fails verification must not survive.
                _unlink(destination, warn=self._warn)
                shutil.copy2(path, destination)
                self.verify_copy(path, destination)
            except Exception as exc:  # noqa: BLE001 — never raise from a move
                _unlink(destination, warn=self._warn)
                if attempt < attempts:
                    self._warn(
                        f"checkpoint move failed (attempt {attempt}/{attempts}) for "
                        f"{os.path.basename(path)}: {type(exc).__name__}: {exc} — "
                        "retrying")
                    continue
                self._warn(
                    f"checkpoint move ABANDONED for {path} after {attempts} "
                    f"attempts ({type(exc).__name__}: {exc}). The Drive copy is "
                    "still there and untouched; the local copy was removed. Nothing "
                    "was deleted — the next retention pass will try again.")
                return None
            # Verified: the local copy is byte-identical and loads. Now, and only
            # now, the Drive original may go.
            self._valid_cache[destination] = True
            try:
                os.remove(path)
            except OSError as exc:
                self._warn(
                    f"verified copy {destination} but could not delete the Drive "
                    f"original {path} ({exc}); both copies are kept")
            return destination
        return None

    # -- eviction -----------------------------------------------------------
    def _valid_alternative(self, excluded: str) -> Optional[str]:
        """A valid checkpoint other than ``excluded``, or ``None``.

        This is the "never delete the only valid recovery checkpoint" check: it
        asks whether, after the deletion, the run could still resume from
        something. Drive is scanned first because that is where the newest
        checkpoint lives, so the common case costs one validation.
        """
        for path in self.resume_candidates():
            if os.path.abspath(path) == os.path.abspath(excluded):
                continue
            if self.is_valid(path):
                return path
        return None

    def enforce_local_limit(self) -> List[str]:
        """Delete the OLDEST local checkpoints until the budget is met.

        Never deletes (a) the globally newest checkpoint or (b) the only valid
        recovery checkpoint. Returns the paths it deleted; when nothing can be
        deleted safely it warns and leaves the local store over budget rather
        than destroying the run's last good copy.
        """
        paths = self.local_checkpoints()          # newest first
        if not paths:
            return []
        newest = self.newest_step()
        current = total_bytes(paths)
        evicted: List[str] = []
        for path in reversed(paths):              # oldest first
            if current <= self.limit_bytes:
                break
            if checkpoint_step(path) == newest:
                self._warn(
                    f"not evicting {os.path.basename(path)}: it is the newest "
                    "checkpoint in either store")
                continue
            alternative = self._valid_alternative(path)
            if alternative is None:
                self._warn(
                    f"not evicting {os.path.basename(path)}: it is the only valid "
                    "recovery checkpoint (no other copy loads and validates) — "
                    "leaving the local store over its limit instead")
                continue
            size = os.path.getsize(path) if os.path.exists(path) else 0
            try:
                os.remove(path)
            except OSError as exc:
                self._warn(f"could not evict {path}: {exc}")
                continue
            self._valid_cache.pop(path, None)
            current -= size
            evicted.append(path)
            logger.info("evicted local checkpoint %s (%s); resume still possible "
                        "from %s", os.path.basename(path), _human_bytes(size),
                        os.path.basename(alternative))
        return evicted

    # -- the pass the notebook calls ---------------------------------------
    def after_save(self) -> RetentionReport:
        """Run one retention pass after a successful save. Never raises.

        1. find the globally newest checkpoint (it stays on Drive);
        2. demote every older Drive checkpoint to local disk (copy → verify →
           delete), leaving any that fail verification in place;
        3. evict the oldest local checkpoints while over the byte budget.
        """
        report = RetentionReport(limit_bytes=self.limit_bytes)
        try:
            drive = self.drive_checkpoints()
            newest = self.newest_step()
            report.newest_step = newest
            for path in drive:
                if checkpoint_step(path) == newest:
                    continue
                if self.move_to_local(path) is None:
                    report.kept_on_drive.append(path)
                else:
                    report.moved.append(path)
            report.evicted = self.enforce_local_limit()
            report.local_bytes = total_bytes(self.local_checkpoints())
            report.over_limit = report.local_bytes > self.limit_bytes
            if report.over_limit:
                self._warn(
                    f"local checkpoint store is over its limit "
                    f"({_human_bytes(report.local_bytes)} > "
                    f"{_human_bytes(self.limit_bytes)}) and nothing else can be "
                    "evicted safely")
        except Exception as exc:  # noqa: BLE001 — retention must never kill a run
            message = f"{type(exc).__name__}: {exc}"
            report.errors.append(message)
            self._warn(f"checkpoint retention failed ({message}) — training "
                       "continues; no checkpoint was touched by this pass")
        return report

    # -- resume -------------------------------------------------------------
    def select_resume(
        self, *, log: Optional[Callable[[str], None]] = None
    ) -> Tuple[str, Any]:
        """Newest → older across both stores; the first valid one wins.

        This is the "restore an older checkpoint when the newest fails"
        requirement: a corrupt or truncated newest checkpoint (the classic
        interrupted-Drive-upload case) is skipped with a printed reason and the
        next candidate is tried. Raises ``FileNotFoundError`` when no candidate
        in either store loads and validates.
        """
        say = log or (lambda message: logger.info("%s", message))
        tried: List[str] = []
        for step, store, path in self.candidates():
            try:
                checkpoint = self._load(path)
                self._validate(checkpoint)
            except Exception as exc:  # noqa: BLE001 — try the next candidate
                tried.append(f"{store}/step-{step}: {type(exc).__name__}: {exc}")
                say(f"skipping {store} {os.path.basename(path)}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            self._valid_cache[path] = True
            return path, checkpoint
        raise FileNotFoundError(
            "no resumable checkpoint in either store "
            f"({self.drive_dir}, {self.local_dir}); every candidate failed "
            f"validation: {tried or 'no step-*.pt found'}")


def _unlink(path: str, *, warn: Warn) -> None:
    """Remove ``path`` if it exists, warning (never raising) on failure."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        warn(f"could not remove {path}: {exc}")


def _human_bytes(size: int) -> str:
    """``10_737_418_240`` -> ``'10.0 GiB'`` (for the warning messages only)."""
    value = float(max(0, int(size)))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


__all__ = [
    "CHECKPOINT_NAME_RE",
    "CheckpointTiering",
    "CopyVerificationError",
    "LOCAL_CHECKPOINT_DIR",
    "LOCAL_CHECKPOINT_LIMIT_BYTES",
    "RetentionReport",
    "checkpoint_step",
    "default_loader",
    "list_checkpoints",
    "make_validator",
    "sha256_file",
    "total_bytes",
]
