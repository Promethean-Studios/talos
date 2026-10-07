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
  9,952,320 for ``tiny_10m``, 96,482,304 for ``tiny_100m`` — enforced by the
  registry in ``configs/canonical.py``, same path for every preset);
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

Anti-repetition decoding controls (remedy R1 of the Styx generation
diagnosis — the owner's greedy byte-loops): ``--repetition-penalty``,
``--top-k``, ``--top-p``, and EOS early-stopping. They compose per step in a
fixed order — repetition penalty on the raw logits, then temperature, then
top-k, then top-p, then softmax/sample. ``--repetition-penalty`` combined with
greedy (no ``--temperature``) is valid and useful: it demotes already-emitted
tokens so greedy decoding escapes loops while staying deterministic. EOS
early-stopping is ON by default when the checkpoint's tokenizer defines an EOS
token (the canonical tokenizer does); ``--eos-token-id N`` overrides the id.

Sequence length policy (``max_seq_len`` = 512 for the canonical presets):
`prompt + max_new_tokens` must fit within ``max_seq_len``. Prompts longer than
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
    # anti-repetition decoding (R1): penalty + top-p + temperature
    python -m scripts.generate --checkpoint runs/oasst1-tiny --prompt "The capital of France is" \\
        --max-new-tokens 96 --repetition-penalty 1.15 --top-p 0.92 --temperature 0.85 --seed 0
    # deterministic anti-repetition (penalty works with greedy; RNG-free)
    python -m scripts.generate --checkpoint runs/oasst1-tiny --prompt "The capital of France is" \\
        --max-new-tokens 64 --repetition-penalty 1.2

The generated continuation (and the full echoed text) is printed; with a fixed
checkpoint + seed the greedy output is bit-identical run to run (asserted by
the test suite). When generation stops on the EOS token the report says so and
the EOS id is never part of the generated text.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

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
    stop_reason: str  # "length" (all max_new_tokens emitted) or "eos"
    eos_token_id: Optional[int]  # resolved EOS id used for early stopping
    token_ids: List[int] = field(default_factory=list)
    text: str = ""  # the generated continuation only
    full_text: str = ""  # echoed prompt bytes + continuation bytes
    wall_s: float = 0.0
    device: str = ""


def resolve_eos_token_id(
    explicit: Optional[int], tokenizer: Any
) -> Tuple[Optional[int], bool]:
    """Resolve the EOS id for early stopping from a CLI override or the tokenizer.

    An explicit ``--eos-token-id`` always wins. Otherwise, if the (possibly
    exotic) tokenizer object exposes an ``eos_id`` attribute it is used — for
    the canonical :class:`~tokenizer.tokenizer.ByteLevelBPETokenizer` that is
    the ``<|endoftext|>`` special token at the top of the vocab. A tokenizer
    without an EOS token yields ``(None, False)``: EOS-stopping is a no-op and
    the caller should warn.

    Returns:
        ``(eos_id or None, tokenizer_had_eos)``.
    """
    if explicit is not None:
        return explicit, True
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None:
        return None, False
    try:
        return int(eos_id), True
    except (TypeError, ValueError):
        return None, False


def generate_from_checkpoint(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    penalize_prompt: bool = False,
    eos_token_id: Optional[int] = None,
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
        top_k: keep only the ``top_k`` highest-scoring tokens (0 = off).
        top_p: nucleus cutoff in ``(0, 1]`` (1.0 = off).
        repetition_penalty: CTRL-style penalty ``> 0`` on already-emitted ids
            (1.0 = off; ``> 1`` discourages repetition). Valid with greedy.
        penalize_prompt: also apply the repetition penalty to prompt ids.
        eos_token_id: explicit EOS id; ``None`` (default) falls back to the
            tokenizer's EOS token when it defines one (a tokenizer without an
            EOS token makes EOS-stopping a no-op, with a warning).
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
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0 (0 = off), got {top_k}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1] (1.0 = off), got {top_p}")
    if repetition_penalty <= 0:
        raise ValueError(
            f"repetition_penalty must be > 0 (1.0 = off), got {repetition_penalty}"
        )

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
    # EOS early-stopping: explicit --eos-token-id wins; else the tokenizer's
    # EOS token when it defines one; a tokenizer without EOS makes it a no-op.
    resolved_eos, tokenizer_had_eos = resolve_eos_token_id(eos_token_id, tokenizer)
    if resolved_eos is None and not tokenizer_had_eos:
        warnings.warn(
            "tokenizer defines no EOS token — EOS early-stopping is disabled "
            "(pass --eos-token-id N to enable it explicitly)",
            UserWarning,
            stacklevel=2,
        )
    t0 = time.monotonic()
    with torch.no_grad():
        new_ids = decode_tokens(
            model,
            prompt_tensor,
            max_new_tokens,
            greedy=greedy,
            temperature=1.0 if temperature is None else temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            penalize_prompt=penalize_prompt,
            eos_token_id=resolved_eos,
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
        stop_reason="length" if len(new_ids) == max_new_tokens else "eos",
        eos_token_id=resolved_eos,
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
    print(f"  max_new_tokens: {r.max_new_tokens} — stop: {r.stop_reason}"
          + (f" (eos id {r.eos_token_id})" if r.eos_token_id is not None else ""))
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
    p.add_argument(
        "--top-k", type=int, default=0, metavar="K",
        help="keep only the K highest-scoring tokens before softmax/argmax "
             "(default 0 = off)",
    )
    p.add_argument(
        "--top-p", type=float, default=1.0, metavar="P",
        help="nucleus cutoff: keep the smallest set with probability mass >= P "
             "(default 1.0 = off)",
    )
    p.add_argument(
        "--repetition-penalty", type=float, default=1.0, metavar="R",
        help="CTRL-style penalty > 1 against every token already emitted in "
             "this sample (default 1.0 = off; works with greedy — the remedy "
             "for greedy byte-loop repetition)",
    )
    p.add_argument(
        "--penalize-prompt", action="store_true",
        help="also apply --repetition-penalty to the prompt's token ids "
             "(default: only tokens generated in this sample are penalized)",
    )
    p.add_argument(
        "--eos-token-id", type=int, default=None, metavar="ID",
        help="stop generation when this token is sampled (explicit override; "
             "default: the tokenizer's EOS token when it defines one — the "
             "canonical tokenizer does)",
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
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            penalize_prompt=args.penalize_prompt,
            eos_token_id=args.eos_token_id,
            device=args.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())