"""CLI: generate text from a Talos training checkpoint.

Loads a ``talos-training-checkpoint-v1`` artifact (a ``step-<N>.pt`` file, or a
directory containing one — the newest is used) plus its sidecar
``tokenizer.json``, verifies tokenizer/model consistency **before** generating,
then continues a prompt with the canonical KV-cache decode path
(:func:`inference.generate.generate` — the same prefill + incremental-decode
code path whose logits were verified exact in the closed validation cycle).

The generation-time consistency guarantee (any mismatch fails loudly with a
``ValueError`` and **no tokens are generated**):

* the artifact format is ``talos-training-checkpoint-v1``;
* the checkpoint's recorded ``model_config`` rebuilds a model whose parameter
  count equals the recorded ``n_params`` (and, for the canonical presets, the
  per-preset canonical count — 254,272 for ``tiny``, 1,000,320 for ``tiny_1m``,
  9,952,320 for ``tiny_10m`` — enforced by the registry in
  ``configs/canonical.py``);
* the recorded ``vocab_size`` equals the rebuilt config's ``vocab_size``;
* the sidecar ``tokenizer.json`` exists and ``tokenizer_vocab_size <=
  model_vocab_size`` — the ``tokenizer/model_compat.py`` contract, so every
  token id the tokenizer can produce is embeddable by the model;
* the sidecar ``tokenizer.json``'s sha256 matches the fingerprint recorded in
  the checkpoint (``tokenizer_fingerprint``) — a same-size but different-
  content tokenizer (the silent-swap hole the audit found, e.g. a 512-vocab
  sidecar next to a 1024-vocab model) is rejected with both hashes printed.

Decoding is **greedy by default** (deterministic — no RNG anywhere in the
prefill/argmax path). Temperature sampling is available but requires an
explicit ``--seed`` (reproducible sampling needs one; a missing seed is a clear
error, not a silent default).

Sequence length policy (``max_seq_len`` = 512 for the tiny preset): `prompt +
max_new_tokens` must fit within ``max_seq_len``. Prompts longer than
``max_seq_len - max_new_tokens`` are truncated **on the left** — the most
recent tokens are kept, since a causal LM's nearest context is what conditions
the continuation. ``max_new_tokens`` must be strictly smaller than
``max_seq_len`` (there has to be room for a non-empty prompt); an empty /
whitespace-only / zero-token prompt is a clear error, never a crash.

Usage::

    # produces the checkpoint (split -> BPE -> tiny train -> step-*.pt)
    python -m scripts.train_oasst1 --data runs/data.jsonl --out-dir runs/oasst1-tiny \\
        --epochs 3 --seq 64 --batch 4 --seed 0
    # greedy (default, deterministic)
    python -m scripts.generate --checkpoint runs/oasst1-tiny --prompt "The capital of France is"
    # explicit file + longer output
    python -m scripts.generate --checkpoint runs/oasst1-tiny/step-2211.pt \\
        --prompt "Once upon a time" --max-new-tokens 32
    # temperature sampling requires an explicit --seed
    python -m scripts.generate --checkpoint runs/oasst1-tiny --prompt "hi there" \\
        --max-new-tokens 16 --temperature 0.8 --seed 0

The generated continuation (and the full echoed text) is printed; with a fixed
checkpoint + seed the greedy output is bit-identical run to run (asserted by
the test suite).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the
# script also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from evaluation.harness import find_checkpoint, load_checkpoint_artifacts  # noqa: E402
from inference.generate import generate as decode_tokens  # noqa: E402
from model import TalosGPT  # noqa: E402
from model.utils import set_seed  # noqa: E402
from tokenizer.model_compat import map_to_model_vocab  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402


@dataclass
class GenerationResult:
    """Outcome of :func:`generate_from_checkpoint` (one generation call)."""

    checkpoint_path: str
    checkpoint_step: int
    checkpoint_format: str
    params: int
    vocab_size: int
    tokenizer_vocab_size: int
    tokenizer_merges: int
    vocab_padding: int  # model_vocab_size - tokenizer_vocab_size (compat contract)
    prompt: str
    prompt_tokens: int
    prompt_truncated: bool
    max_new_tokens: int
    mode: str  # "greedy" or "sampled(temperature=...)"
    seed: Optional[int]
    token_ids: List[int] = field(default_factory=list)
    text: str = ""  # the generated continuation only
    full_text: str = ""  # echoed prompt bytes + continuation bytes
    wall_s: float = 0.0
    device: str = ""


def generate_from_checkpoint(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> GenerationResult:
    """Generate ``max_new_tokens`` tokens continuing ``prompt`` from a checkpoint.

    Args:
        checkpoint_path: ``step-<N>.pt`` file, or a directory containing one
            (the newest is used).
        prompt: text to continue. Empty / whitespace-only prompts are a clear
            ``ValueError`` (nothing to condition on).
        max_new_tokens: how many tokens to generate (must be ``< max_seq_len``
            so there is always room for a non-empty prompt).
        temperature: when given, sample from the softmax at this temperature
            instead of greedy argmax — and ``seed`` becomes **required**.
        seed: fixed seed for deterministic runs; mandatory for sampling.
        device: compute device (default: auto).

    Returns:
        :class:`GenerationResult` with the consistency report and the generated
        token ids + decoded text.

    Raises:
        ValueError: on any tokenizer/model/config inconsistency (checked before
            generation), an empty prompt, or invalid decode arguments. Nothing
            is generated when this fires.
    """
    # ---- validate decode arguments first (clear errors, never a crash) -----
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is empty — nothing to condition generation on")
    if temperature is not None:
        if temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {temperature} "
                "(temperature <= 0 makes softmax ill-defined)"
            )
        if seed is None:
            raise ValueError(
                "temperature sampling requires an explicit --seed "
                "(sampling without a fixed seed is not reproducible)"
            )
        mode = f"sampled(temperature={temperature:g})"
        greedy = False
    else:
        mode = "greedy"
        greedy = True

    # ---- resolve the artifact (file, or directory -> newest step-*.pt) -----
    ckpt_path = find_checkpoint(checkpoint_path)

    # ---- consistency checks BEFORE any generation --------------------------
    # The loader rebuilds the model from the checkpoint's own model_config,
    # asserts the recorded n_params (and the per-preset canonical count for
    # tiny / tiny_1m), asserts the recorded vocab_size matches the rebuilt
    # config, and enforces tokenizer_vocab_size <= model_vocab_size — see
    # evaluation.harness.load_checkpoint_artifacts. Any mismatch raises here.
    ckpt, model, tokenizer = load_checkpoint_artifacts(ckpt_path)
    # Make the tokenizer/model compat contract explicit on this path (raises
    # the canonical "tokenizer vocab exceeds model vocab" error when violated).
    mapping = map_to_model_vocab(tokenizer, model.config.vocab_size)

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device).eval()

    max_seq_len = model.config.max_seq_len
    max_prompt_tokens = max_seq_len - max_new_tokens
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} leaves no room for a prompt "
            f"within max_seq_len={max_seq_len} (need max_new_tokens < max_seq_len)"
        )
    if seed is not None:
        set_seed(seed)

    # ---- tokenize the prompt, truncating LEFT (keep the most recent text) --
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError(
            "prompt encodes to 0 tokens — cannot condition generation "
            "(is it all special-token text without split_special?)"
        )
    truncated = len(ids) > max_prompt_tokens
    if truncated:
        ids = ids[-max_prompt_tokens:]
    prompt_tensor = torch.tensor([ids], dtype=torch.long, device=resolved_device)

    # ---- generate via the canonical KV-cache prefill + decode path ---------
    t0 = time.monotonic()
    with torch.no_grad():
        new_ids = decode_tokens(
            model,
            prompt_tensor,
            max_new_tokens,
            greedy=greedy,
            temperature=1.0 if temperature is None else temperature,
        )
    wall = time.monotonic() - t0

    return GenerationResult(
        checkpoint_path=os.path.abspath(ckpt_path),
        checkpoint_step=int(ckpt["step"]),
        checkpoint_format=str(ckpt["format"]),
        params=model.num_parameters(),
        vocab_size=model.config.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_merges=tokenizer.merge_count,
        vocab_padding=mapping.padding,
        prompt=prompt,
        prompt_tokens=len(ids),
        prompt_truncated=truncated,
        max_new_tokens=max_new_tokens,
        mode=mode,
        seed=seed,
        token_ids=new_ids,
        text=tokenizer.decode(new_ids),
        full_text=tokenizer.decode(ids + new_ids),
        wall_s=round(wall, 4),
        device=str(resolved_device),
    )


def print_report(r: GenerationResult) -> None:
    print("=" * 72)
    print("Talos generation report")
    print(f"  checkpoint    : {r.checkpoint_path} (step {r.checkpoint_step}, "
          f"{r.checkpoint_format})")
    print(f"  model         : {r.params:,} params, vocab {r.vocab_size} — "
          f"n_params + vocab guards OK")
    print(f"  tokenizer     : vocab {r.tokenizer_vocab_size} "
          f"({r.tokenizer_merges} merges) — compat OK "
          f"(padding {r.vocab_padding} model rows)")
    prompt_note = f"{r.prompt_tokens} tokens" + (
        " (truncated on the left)" if r.prompt_truncated else ""
    )
    print(f"  prompt        : {prompt_note} — {r.prompt!r}")
    print(f"  mode          : {r.mode}"
          + (f" | seed {r.seed}" if r.seed is not None else " | no RNG"))
    print(f"  max_new_tokens: {r.max_new_tokens}")
    print(f"  wall          : {r.wall_s} s "
          f"({r.wall_s / max(r.max_new_tokens, 1) * 1000:.1f} ms/token, "
          f"device {r.device})")
    print(f"  generated     : {r.text!r}")
    print(f"  full text     : {r.full_text!r}")
    print("=" * 72)


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.generate",
        description=__doc__.splitlines()[0],
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="step-<N>.pt file, or a directory containing one (newest is used)",
    )
    p.add_argument("--prompt", required=True, help="text prompt to continue")
    p.add_argument(
        "--max-new-tokens", type=int, default=32,
        help="how many tokens to generate (default 32; must be < max_seq_len)",
    )
    p.add_argument(
        "--temperature", type=float, default=None,
        help="sample at this temperature instead of greedy argmax; requires "
             "--seed (default: greedy, deterministic)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="fixed seed (mandatory when --temperature is given; greedy is "
             "RNG-free either way)",
    )
    p.add_argument("--device", default=None, help="compute device (default: auto)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_arg_parser().parse_args(argv)
    try:
        result = generate_from_checkpoint(
            args.checkpoint,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            device=args.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())