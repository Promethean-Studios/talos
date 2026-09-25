"""Safetensors release-format export/import for Talos (owner requirement #10).

The pickled ``talos-training-checkpoint-v1`` (``step-<N>.pt``) stays the
**training** format: weights + optimizer + RNG state + losses, all in one
file, resume-friendly. Safetensors is the **release** format: raw tensors with
no pickle (safe to untrusted loaders, HF-ecosystem standard) plus a small
JSON sidecar that is sufficient to rebuild and load the model exactly.

Artifact layout in the target directory::

    <out_dir>/
      model.safetensors     # raw fp32 weights in safetensors format
      model.config.json     # sidecar: format + full model_config + n_params +
                            # vocab + canonical preset + tokenizer fingerprint
                            # + provenance (source checkpoint, step, losses)

Export paths:

* :func:`checkpoint_to_safetensors` — from a ``talos-training-checkpoint-v1``
  artifact. The checkpoint is validated through the eval harness
  (:func:`evaluation.harness.load_checkpoint_artifacts`) first, so **every** P0
  guard (format, n_params vs rebuilt config, canonical-preset exact count,
  recorded vocab, tokenizer size + sha256 fingerprint) runs before anything is
  written — export never bypasses the seam checks.
* :func:`export_model` — directly from a live model (used by the tests for
  round-trip validation).

Import: :func:`load_artifact` rebuilds the model from the recorded
``model_config`` sidecar and re-runs the same guards on the sidecar that the
checkpoint path runs on the checkpoint (recorded ``n_params``/``vocab_size``
must match the rebuilt model; a recorded canonical preset must resolve and
match its exact registry count — same code path as the harness). The state
dict is then loaded **strictly** (every key present, every shape matching),
so a bit-exact round-trip is guaranteed by construction and asserted by the
test suite (``tests/test_safetensors_io.py``).

CLI::

    python -m scripts.export_safetensors --checkpoint runs/oasst1-tiny \\
        --out-dir runs/oasst1-tiny/release
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the
# script also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import safetensors.torch  # noqa: E402
import torch  # noqa: E402

from configs.canonical import (  # noqa: E402
    CANONICAL_PRESETS,
    is_canonical,
    resolve_preset,
)
from evaluation.harness import (  # noqa: E402
    find_checkpoint,
    load_checkpoint_artifacts,
)
from model import ModelConfig, TalosGPT  # noqa: E402
from tokenizer.tokenizer import tokenizer_file_sha256  # noqa: E402

#: Release artifact format tag (written into every config.json sidecar).
SAFETENSORS_FORMAT = "talos-release-safetensors-v1"

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "model.config.json"


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write ``payload`` as pretty JSON via a temp file + atomic rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def export_model(
    model: TalosGPT,
    out_dir: str,
    *,
    tokenizer_path: Optional[str] = None,
    step: Optional[int] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    source_checkpoint: Optional[str] = None,
) -> dict:
    """Export ``model`` to ``<out_dir>/model.safetensors`` + sidecar JSON.

    The sidecar records everything needed to rebuild and load the model
    exactly: the full derived ``model_config``, the exact parameter count and
    vocab size, the canonical preset name (when the shape resolves to one — the
    harness's own registry check), and (when given) the tokenizer's file path
    + sha256 fingerprint so a release consumer can also reproduce the
    tokenizer and detect a swapped one.

    Returns the sidecar metadata dict (also persisted as ``model.config.json``).
    """
    cfg = model.config
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, WEIGHTS_FILENAME)
    config_path = os.path.join(out_dir, CONFIG_FILENAME)
    safetensors.torch.save_file(dict(model.state_dict()), weights_path)

    meta: Dict[str, Any] = {
        "format": SAFETENSORS_FORMAT,
        "model_config": asdict(cfg),
        "n_params": model.num_parameters(),
        "vocab_size": cfg.vocab_size,
        "preset": resolve_preset(cfg) if is_canonical(cfg) else None,
        "weights_filename": WEIGHTS_FILENAME,
        "config_filename": CONFIG_FILENAME,
        "step": step,
        "train_loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "source_checkpoint": (
            os.path.abspath(source_checkpoint) if source_checkpoint else None
        ),
        "tokenizer_path": (
            os.path.abspath(tokenizer_path) if tokenizer_path else None
        ),
        "tokenizer_fingerprint": (
            tokenizer_file_sha256(tokenizer_path) if tokenizer_path else None
        ),
        "tokenizer_vocab_size": None,
        "tokenizer_merges": None,
    }
    _write_json_atomic(config_path, meta)
    return meta


def checkpoint_to_safetensors(
    checkpoint_path: str,
    out_dir: str,
) -> dict:
    """Export a ``talos-training-checkpoint-v1`` artifact to safetensors.

    ``checkpoint_path`` may be a ``step-<N>.pt`` file or a directory
    containing one (the newest is used — same resolution the generation CLI
    uses). The checkpoint is fully validated by the eval harness first, then
    its rebuilt model + sidecar tokenizer identity are carried into the
    release artifact. Returns the sidecar metadata dict.
    """
    ckpt_path = find_checkpoint(checkpoint_path)
    ckpt, model, tokenizer = load_checkpoint_artifacts(ckpt_path)
    meta = export_model(
        model,
        out_dir,
        tokenizer_path=ckpt.get("tokenizer_path"),
        step=ckpt.get("step"),
        train_loss=ckpt.get("train_loss"),
        val_loss=ckpt.get("val_loss"),
        source_checkpoint=ckpt_path,
    )
    # Carry the tokenizer's realized vocab/merges (already validated by the
    # harness) into the sidecar so consumers see the release's tokenizer
    # contract without re-parsing the tokenizer file.
    meta["tokenizer_vocab_size"] = tokenizer.vocab_size
    meta["tokenizer_merges"] = tokenizer.merge_count
    _write_json_atomic(os.path.join(out_dir, CONFIG_FILENAME), meta)
    return meta


def _read_sidecar(dir_path: str) -> Tuple[dict, str, str]:
    """Load + sanity-check the sidecar; return ``(meta, weights_path, cfg_path)``."""
    config_path = os.path.join(dir_path, CONFIG_FILENAME)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"not a safetensors release directory: missing {CONFIG_FILENAME} "
            f"in {dir_path}"
        )
    with open(config_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != SAFETENSORS_FORMAT:
        raise ValueError(
            f"unsupported safetensors artifact format {meta.get('format')!r} "
            f"in {config_path}: expected {SAFETENSORS_FORMAT!r}"
        )
    weights_path = os.path.join(dir_path, meta.get("weights_filename", WEIGHTS_FILENAME))
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"safetensors weights file missing: {weights_path}")
    return meta, weights_path, config_path


def load_artifact(dir_path: str) -> Tuple[TalosGPT, dict]:
    """Rebuild + strictly load a model from a safetensors release directory.

    Re-runs the checkpoint seam guards on the sidecar before touching the
    weights: recorded ``n_params`` and ``vocab_size`` must equal the values
    the recorded ``model_config`` actually produces, and a recorded canonical
    preset must resolve to its exact registry count (the same
    ``configs.canonical`` path the harness uses). The state dict loads with
    ``strict=True``, so any missing/unexpected tensor key or shape drift is a
    loud ``RuntimeError``/``ValueError``, never a silent partial load.

    Returns ``(model, meta)`` with the model in ``eval()`` mode.
    """
    meta, weights_path, _ = _read_sidecar(dir_path)
    cfg = ModelConfig(**meta["model_config"]).derive()
    model = TalosGPT(cfg)
    model.eval()
    recorded = int(meta["n_params"])
    actual = model.num_parameters()
    if actual != recorded:
        raise ValueError(
            f"safetensors sidecar n_params mismatch: records {recorded:,} but "
            f"rebuilding the recorded model_config yields {actual:,} params in "
            f"{dir_path} — the artifact is corrupt or the preset drifted"
        )
    recorded_vocab = int(meta["vocab_size"])
    if cfg.vocab_size != recorded_vocab:
        raise ValueError(
            f"safetensors sidecar vocab mismatch: records {recorded_vocab} but "
            f"the recorded model_config yields {cfg.vocab_size}"
        )
    preset = meta.get("preset")
    if preset is not None:
        resolved = resolve_preset(cfg)
        if resolved != preset:
            raise ValueError(
                f"safetensors sidecar preset {preset!r} does not match the "
                f"recorded config's resolved preset {resolved!r} in {dir_path}"
            )
        exp_params, exp_vocab = CANONICAL_PRESETS[preset]
        if actual != exp_params or cfg.vocab_size != exp_vocab:
            raise ValueError(
                f"safetensors artifact is not the canonical {preset} prototype: "
                f"{actual:,} params / vocab {cfg.vocab_size}, expected exactly "
                f"{exp_params:,} / vocab {exp_vocab}"
            )
    tensors = safetensors.torch.load_file(weights_path)
    wanted = set(model.state_dict().keys())
    given = set(tensors.keys())
    if given != wanted:
        missing = sorted(wanted - given)
        unexpected = sorted(given - wanted)
        raise ValueError(
            f"safetensors state_dict mismatch in {weights_path}: "
            f"{'missing ' + str(missing[:5]) if missing else ''}"
            f"{' unexpected ' + str(unexpected[:5]) if unexpected else ''}"
        )
    model.load_state_dict(tensors, strict=True)
    # Free the raw 4-bytes-per-param tensor dict immediately: it is a full copy
    # of the weights (e.g. ~386 MiB for tiny_100m) and callers often hold the
    # source model + this one at the same time for bit-exactness checks.
    del tensors
    return model, meta


def load_tokenizer(dir_path: str, meta: Optional[dict] = None) -> Any:
    """Load the release's sidecar tokenizer when recorded.

    Uses the same content-identity check the harness uses: when the sidecar
    records a ``tokenizer_fingerprint``, the ``tokenizer.json`` at the
    recorded path must hash to it (a swapped/re-trained same-size tokenizer is
    rejected). The tokenizer/model vocab contract (``tokenizer_vocab <= model
    vocab``) is also enforced from the sidecar's recorded values. Returns
    ``None`` when the release has no tokenizer recorded (weights-only export).
    """
    from tokenizer.tokenizer import ByteLevelBPETokenizer

    if meta is None:
        meta, _, _ = _read_sidecar(dir_path)
    tok_path = meta.get("tokenizer_path")
    if not tok_path:
        return None
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(
            f"safetensors sidecar records tokenizer {tok_path!r} but the file "
            f"is missing — release is incomplete"
        )
    recorded_fp = meta.get("tokenizer_fingerprint")
    if recorded_fp:
        actual_fp = tokenizer_file_sha256(tok_path)
        if actual_fp != recorded_fp:
            raise ValueError(
                f"safetensors sidecar records tokenizer sha256 {recorded_fp} "
                f"but {tok_path} hashes to {actual_fp} — the tokenizer was "
                f"swapped/re-trained; refusing to pair it with the release"
            )
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    model_vocab = int(meta["vocab_size"])
    if tokenizer.vocab_size > model_vocab:
        raise ValueError(
            f"safetensors tokenizer vocab {tokenizer.vocab_size} exceeds model "
            f"vocab {model_vocab} — tokenizer/model mismatch"
        )
    return tokenizer


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.export_safetensors",
        description="Export a talos-training-checkpoint-v1 artifact to the "
                    "safetensors release format (model.safetensors + "
                    "model.config.json sidecar).",
    )
    p.add_argument("--checkpoint", required=True,
                   help="step-<N>.pt file, or a directory containing one (the "
                        "newest is used)")
    p.add_argument("--out-dir", required=True,
                   help="where to write model.safetensors + model.config.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    meta = checkpoint_to_safetensors(args.checkpoint, args.out_dir)
    weights_path = os.path.join(args.out_dir, WEIGHTS_FILENAME)
    config_path = os.path.join(args.out_dir, CONFIG_FILENAME)
    print(f"exported safetensors release:")
    print(f"  weights : {weights_path}")
    print(f"  sidecar : {config_path}")
    print(f"  format  : {meta['format']}")
    print(f"  params  : {meta['n_params']:,} (vocab {meta['vocab_size']})")
    print(f"  preset  : {meta['preset'] or 'non-canonical'}")
    print(f"  source  : {meta['source_checkpoint']} "
          f"(step {meta['step']})" if meta.get("source_checkpoint") else "")
    print(f"  weights : {os.path.getsize(weights_path):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())