"""Reproducible performance-benchmark harness for the Talos tiny prototype.

The single command that produces the committed per-item baseline backing the
"benchmark everything" step of the prototype-first plan (before quantization /
DDM experiments)::

    python tools/benchmark.py                      # -> benchmarks/baseline-tiny-cpu.{json,md}
    python tools/benchmark.py --device cuda        # graceful CUDA; CPU otherwise
    python tools/benchmark.py --help               # all knobs

What it measures, for the canonical ``tiny`` preset (254,272 params; hidden 64,
2 layers, GQA, dense FFN — :func:`configs.presets.tiny_config`):

* ``model``      — total / trainable / *active* parameter count (active counted
                   by actually running a forward pass and hooking every module
                   whose parameters participate — generic enough for MoE, where
                   active < total) plus fp32/bf16/AdamW-state size estimates.
* ``train``      — tokens/sec over a fixed number of AdamW steps on the
                   deterministic synthetic recurrent corpus
                   (:func:`training.synthetic.build_recurrent_corpus`), with the
                   properly-aligned causal-LM objective (position ``t`` predicts
                   token ``t+1``), plus the loss trajectory so the baseline is
                   tied to a *learned* model, not a random one.
* ``inference``  — prefill latency for a fixed prompt length, per-token decode
                   latency with the KV cache, and end-to-end
                   :func:`inference.generate.generate` latency. Uses the one
                   canonical prefill/decode path from ``inference/generate.py``.

Determinism: fixed seed (default 0) seeds model init, the corpus and the
sampling/prompt RNGs, so every number is reproducible run-to-run on the same
machine. The committed baseline artifact is a **CPU** baseline — numbers vary
with hardware and thread count; re-run the harness on any machine to
regenerate.

Memory: each measured section runs in its own child process and reports the
process **peak RSS** (``resource.getrusage`` high-water mark, the same approach
as ``tools/measure_memory.py``), so train and inference footprints are isolated.
Peak RSS includes the interpreter + torch import; that is the honest
whole-process footprint. Use ``--in-process`` (tests, quick checks) to run
everything in the current process instead.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

# Make the repo-root packages (`model`, `configs`, `training`, `inference`)
# importable when this file is run directly as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from configs.presets import tiny_config  # noqa: E402
from inference.generate import decode_step, generate, prefill  # noqa: E402
from model import TalosGPT  # noqa: E402
from model.utils import set_seed  # noqa: E402
from training.synthetic import build_recurrent_corpus  # noqa: E402

SCHEMA_VERSION = 1
BASELINE_NAME = "talos-tiny-baseline"
DEFAULT_SEED = 0
DEFAULT_OUT = os.path.join(_REPO_ROOT, "benchmarks", "baseline-tiny-cpu.json")
_MIB = 1024 * 1024

Sections = Sequence[str]
_ALL_SECTIONS: Tuple[str, ...] = ("model", "train", "inference")


# ---------------------------------------------------------------------------
# Environment / machine capture
# ---------------------------------------------------------------------------
def _cpu_model() -> str:
    """Best-effort CPU model string (``/proc/cpuinfo`` first, then platform)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if "model name" in line and ":" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_info() -> Dict[str, Optional[str]]:
    """Commit + branch of the repo this harness runs from (None if not a repo)."""

    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True,
                text=True, timeout=10, check=True,
            )
            return out.stdout.strip() or None
        except Exception:
            return None

    return {"commit": _run("rev-parse", "HEAD"), "branch": _run("rev-parse", "--abbrev-ref", "HEAD")}


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` (CUDA when available, else CPU) or an explicit device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available; use 'auto' or 'cpu'")
    return dev


def environment_info(device: torch.device) -> Dict[str, Any]:
    """Environment provenance recorded alongside every baseline."""
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": cuda,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
    }


def _peak_rss_mb() -> float:
    """Peak RSS of the current process in MiB (Linux: ru_maxrss is KiB)."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0.0
    kb = float(rss) / 1024.0 if sys.platform != "darwin" else float(rss) / 1024.0
    return round(kb, 1)


# ---------------------------------------------------------------------------
# Section: model size
# ---------------------------------------------------------------------------
def active_parameter_count(model: TalosGPT, seq_len: int = 16) -> int:
    """Parameters actually *used* by one forward pass (active params).

    Hooks every leaf module, records the unique parameter tensors that
    participate, and sums their element counts. For a dense model this equals
    the total; for MoE it counts only the experts actually routed (which may
    vary with input, hence we return the count for a fixed deterministic input).
    """
    used: set[int] = set()

    def hook(module: torch.nn.Module, _args: Any) -> None:
        for p in module.parameters(recurse=False):
            used.add(id(p))

    hooks = []
    for m in model.modules():
        if not list(m.children()):  # leaf modules only
            hooks.append(m.register_forward_pre_hook(hook))
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, seq_len), dtype=torch.long, device=next(model.parameters()).device))
        if was_training:
            model.train()
    finally:
        for h in hooks:
            h.remove()
    return sum(p.numel() for p in model.parameters() if id(p) in used)


def run_model_section(device: str, seed: int) -> Dict[str, Any]:
    """Model-size facts for the tiny preset: params (total/trainable/active) + MB."""
    set_seed(seed)
    cfg = tiny_config().derive()
    model = TalosGPT(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = model.num_parameters(trainable_only=True)
    active = active_parameter_count(model)
    return {
        "preset": "tiny",
        "ffn_type": cfg.ffn_type,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "params_total": total,
        "params_trainable": trainable,
        "params_active": active,
        "size_fp32_mb": round(total * 4 / _MIB, 3),
        "size_bf16_mb": round(total * 2 / _MIB, 3),
        # AdamW keeps exp_avg + exp_avg_sq per param, fp32 by default.
        "optimizer_states_fp32_mb_adamw": round(total * 2 * 4 / _MIB, 3),
    }


# ---------------------------------------------------------------------------
# Section: training throughput
# ---------------------------------------------------------------------------
def _build_model(device: str, seed: int) -> TalosGPT:
    set_seed(seed)
    return TalosGPT(tiny_config().derive()).to(device)


def run_train_section(
    device: str,
    seed: int,
    steps: int,
    batch: int,
    seq: int,
    lr: float,
    warmup: int,
    n_seq: int,
) -> Dict[str, Any]:
    """Tokens/sec + loss trajectory over fixed AdamW steps on the synthetic corpus.

    Deterministic: model init, corpus and batch sampling are all seeded. The
    objective is the aligned causal-LM one (input ``x[:, :-1]`` predicts
    ``x[:, 1:]``), so ``tokens_per_step == batch * (seq - 1)``.
    """
    cfg = tiny_config().derive()
    model = _build_model(device, seed)
    corpus = build_recurrent_corpus(
        cfg.vocab_size, n_sequences=n_seq, seq_len=seq, seed=seed, device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    gen = torch.Generator(device=corpus.device).manual_seed(seed)

    # Untrained reference loss over the whole fixed corpus (context: the
    # baseline is tied to a learned model — this is where it starts).
    model.eval()
    with torch.no_grad():
        logits, _ = model(corpus[:, :-1])
        loss_initial = float(loss_fn(logits.reshape(-1, cfg.vocab_size), corpus[:, 1:].reshape(-1)))
    model.train()

    def step() -> float:
        idx = torch.randint(0, corpus.shape[0], (batch,), generator=gen, device=corpus.device)
        x = corpus[idx]
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    for _ in range(warmup):  # untimed warmup (thread pools, allocator, autotune)
        step()
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        losses.append(step())
    wall = time.perf_counter() - t0

    tokens_per_step = batch * (seq - 1)
    return {
        "config": {
            "batch": batch, "seq": seq, "steps": steps, "warmup_steps": warmup,
            "lr": lr, "optimizer": "adamw", "corpus_sequences": n_seq,
            "corpus": "training.synthetic recurrent (seeded)",
        },
        "tokens_per_step": tokens_per_step,
        "steps_measured": steps,
        "wall_time_s": round(wall, 4),
        "tokens_per_sec": round(tokens_per_step * steps / wall, 1),
        "loss_initial_eval": round(loss_initial, 4),
        "loss_start": round(losses[0], 4),
        "loss_end": round(losses[-1], 4),
        "loss_min": round(min(losses), 4),
        "losses": [round(l, 4) for l in losses],
    }


# ---------------------------------------------------------------------------
# Section: inference (prefill + KV-cache decode)
# ---------------------------------------------------------------------------
def run_inference_section(
    device: str,
    seed: int,
    prompt_len: int,
    decode_tokens: int,
    prefill_repeats: int,
    warmup: int,
) -> Dict[str, Any]:
    """Prefill latency, per-token KV-cache decode latency, and generate() e2e.

    Same canonical path as production code: :func:`inference.generate.prefill`
    and :func:`inference.generate.decode_step` (and :func:`inference.generate.generate`
    for the end-to-end number). Deterministic prompt from the seed.
    """
    cfg = tiny_config().derive()
    model = _build_model(device, seed)
    model.eval()
    gen = torch.Generator(device=model.embed_tokens.weight.device).manual_seed(seed)
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), generator=gen,
                           device=model.embed_tokens.weight.device)

    with torch.no_grad():
        for _ in range(warmup):
            prefill(model, prompt)
        prefill_ms: list[float] = []
        for _ in range(prefill_repeats):
            t0 = time.perf_counter()
            prefill(model, prompt)
            prefill_ms.append((time.perf_counter() - t0) * 1000.0)

        logits, cache = prefill(model, prompt)
        nxt = logits[:, -1:, :].argmax(dim=-1)
        for _ in range(warmup):  # untimed decode warmup
            out = decode_step(model, cache, nxt, cache.length)
            nxt = out[:, -1:, :].argmax(dim=-1)
        decode_ms: list[float] = []
        generated: list[int] = []
        for _ in range(decode_tokens):
            t0 = time.perf_counter()
            out = decode_step(model, cache, nxt, cache.length)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            nxt = out[:, -1:, :].argmax(dim=-1)
            generated.append(int(nxt[0, 0]))

        t0 = time.perf_counter()
        generate(model, prompt, max_new_tokens=decode_tokens, greedy=True)
        generate_e2e_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "config": {
            "prompt_len": prompt_len, "decode_tokens": decode_tokens,
            "prefill_repeats": prefill_repeats, "warmup": warmup, "greedy": True,
            "kv_cache": True, "batch": 1,
        },
        "prefill_ms_mean": round(statistics.fmean(prefill_ms), 3),
        "prefill_ms_p50": round(statistics.median(prefill_ms), 3),
        "prefill_ms_min": round(min(prefill_ms), 3),
        "prefill_ms_max": round(max(prefill_ms), 3),
        "prefill_tokens_per_sec": round(prompt_len / (statistics.fmean(prefill_ms) / 1000.0), 1),
        "decode_ms_mean": round(statistics.fmean(decode_ms), 3),
        "decode_ms_p50": round(statistics.median(decode_ms), 3),
        "decode_ms_max": round(max(decode_ms), 3),
        "decode_tokens_per_sec": round(1000.0 / statistics.fmean(decode_ms), 1),
        "generate_e2e_ms": round(generate_e2e_ms, 3),
        "generated": generated,
    }


_SECTION_FUNCS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "model": run_model_section,
    "train": run_train_section,
    "inference": run_inference_section,
}
# Peak RSS is only meaningful for the heavy sections; `model` runs in-process.
_ISOLATED_SECTIONS = ("train", "inference")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _section_worker(payload: Dict[str, Any], queue: Any) -> None:
    """Child-process entry: run one section, report result + own peak RSS."""
    try:
        result = _SECTION_FUNCS[payload["section"]](**payload["kwargs"])
        result["peak_rss_mb"] = _peak_rss_mb()
        queue.put(("ok", payload["section"], result))
    except Exception:  # pragma: no cover - surfaced to the parent verbatim
        import traceback

        queue.put(("error", payload.get("section"), traceback.format_exc()))


def _run_isolated(section: str, kwargs: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    proc = ctx.Process(target=_section_worker, args=({"section": section, "kwargs": kwargs}, queue))
    proc.start()
    status, name, payload = queue.get(timeout=timeout_s)
    proc.join(timeout=60.0)
    if status == "error":  # pragma: no cover
        raise RuntimeError(f"benchmark section '{name}' failed in child process:\n{payload}")
    assert isinstance(payload, dict)
    return payload


def run_all(
    sections: Sections = _ALL_SECTIONS,
    *,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    train_steps: int = 100,
    batch: int = 4,
    seq: int = 64,
    lr: float = 1e-3,
    train_warmup: int = 3,
    n_seq: int = 8,
    prompt_len: int = 64,
    decode_tokens: int = 64,
    prefill_repeats: int = 20,
    infer_warmup: int = 3,
    in_process: bool = False,
) -> Dict[str, Any]:
    """Run the requested sections and return the full machine-readable result dict.

    ``train`` and ``inference`` run in isolated child processes by default so
    their peak-RSS numbers don't contaminate each other; pass
    ``in_process=True`` (tests / debugging) to run everything in-process, in
    which case ``peak_rss_mb`` is the current process's high-water mark.
    """
    dev = resolve_device(device)
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": BASELINE_NAME,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "env": environment_info(dev),
        "git": _git_info(),
        "notes": (
            "Peak RSS includes the Python interpreter + torch import; train and "
            "inference were measured in isolated child processes. Deterministic "
            "given the seed on a fixed machine; numbers are hardware-specific."
        ),
    }
    section_kwargs: Dict[str, Dict[str, Any]] = {
        "model": {"device": str(dev), "seed": seed},
        "train": {
            "device": str(dev), "seed": seed, "steps": train_steps, "batch": batch,
            "seq": seq, "lr": lr, "warmup": train_warmup, "n_seq": n_seq,
        },
        "inference": {
            "device": str(dev), "seed": seed, "prompt_len": prompt_len,
            "decode_tokens": decode_tokens, "prefill_repeats": prefill_repeats,
            "warmup": infer_warmup,
        },
    }
    for section in sections:
        if section not in _SECTION_FUNCS:
            raise SystemExit(f"unknown section '{section}'; choose from {', '.join(_ALL_SECTIONS)}")
        if not in_process and section in _ISOLATED_SECTIONS:
            results[section] = _run_isolated(section, section_kwargs[section], timeout_s=1800.0)
        else:
            results[section] = _SECTION_FUNCS[section](**section_kwargs[section])
            # In-process mode: RSS is a process-wide high-water mark (documented).
            results[section]["peak_rss_mb"] = _peak_rss_mb()
    return results


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------
def render_markdown(results: Dict[str, Any]) -> str:
    """Render the result dict as the committed human-readable summary."""
    env, model = results["env"], results["model"]
    git = results.get("git") or {}
    lines: list[str] = []
    lines.append(f"# Talos tiny prototype — performance baseline ({env['device']})")
    lines.append("")
    lines.append(
        f"Recorded {results['timestamp_utc']} · seed {results['seed']} · "
        f"commit `{git.get('commit') or 'unknown'}` (branch `{git.get('branch') or '?'}`)"
    )
    lines.append("")
    lines.append("| environment | value |")
    lines.append("|---|---|")
    lines.append(f"| torch | {env['torch']} |")
    lines.append(f"| python | {env['python']} |")
    lines.append(f"| cpu | {env['cpu_model']} |")
    lines.append(f"| cores / torch threads | {env['cpu_count']} / {env['torch_num_threads']} |")
    lines.append(f"| device | {env['device']} (cuda_available={env['cuda_available']}) |")
    lines.append("")
    lines.append(f"**This is a CPU baseline** — numbers are specific to this machine and thread count.")
    lines.append("")
    lines.append("## Model — `tiny` preset (dense)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| total parameters | {model['params_total']:,} |")
    lines.append(f"| trainable parameters | {model['params_trainable']:,} |")
    lines.append(f"| active parameters (measured via forward hooks) | {model['params_active']:,} |")
    lines.append(f"| fp32 model size | {model['size_fp32_mb']} MB |")
    lines.append(f"| bf16 model size | {model['size_bf16_mb']} MB |")
    lines.append(f"| AdamW optimizer states (fp32, estimate) | {model['optimizer_states_fp32_mb_adamw']} MB |")
    lines.append("")
    if "train" in results:
        t = results["train"]
        c = t["config"]
        lines.append("## Training throughput (synthetic recurrent corpus)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| config | batch={c['batch']}, seq={c['seq']}, steps={c['steps']} "
                     f"(warmup {c['warmup_steps']}), lr={c['lr']}, AdamW |")
        lines.append(f"| **train throughput** | **{t['tokens_per_sec']:,} tokens/s** "
                     f"({t['tokens_per_step']} tokens/step × {t['steps_measured']} steps "
                     f"in {t['wall_time_s']} s) |")
        lines.append(f"| loss trajectory | {t['loss_initial_eval']} (untrained eval) → "
                     f"{t['loss_start']} → **{t['loss_end']}** (min {t['loss_min']}) |")
        lines.append(f"| peak RSS | {t['peak_rss_mb']} MB |")
        lines.append("")
    if "inference" in results:
        i = results["inference"]
        c = i["config"]
        lines.append("## Inference (prefill + KV-cache decode, batch 1, greedy)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| config | prompt_len={c['prompt_len']}, decode_tokens={c['decode_tokens']}, "
                     f"{c['prefill_repeats']} prefill repeats |")
        lines.append(f"| prefill latency | **{i['prefill_ms_mean']} ms** mean "
                     f"(p50 {i['prefill_ms_p50']}, max {i['prefill_ms_max']}) → "
                     f"{i['prefill_tokens_per_sec']:,} tokens/s |")
        lines.append(f"| decode latency (KV cache) | **{i['decode_ms_mean']} ms/token** mean "
                     f"(p50 {i['decode_ms_p50']}, max {i['decode_ms_max']}) → "
                     f"**{i['decode_tokens_per_sec']:,} tokens/s** |")
        lines.append(f"| generate() end-to-end | {i['generate_e2e_ms']} ms for "
                     f"{c['decode_tokens']} tokens |")
        lines.append(f"| peak RSS | {i['peak_rss_mb']} MB |")
        lines.append("")
    lines.append("> Regenerate on any machine with: `python tools/benchmark.py` "
                 "(writes `benchmarks/baseline-tiny-cpu.json` + this `.md`).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reproducible performance baseline for the Talos tiny prototype "
        "(model size + train throughput + inference latency + peak RSS). "
        "One command: `python tools/benchmark.py` — writes "
        "benchmarks/baseline-tiny-cpu.json (machine-readable) and "
        "baseline-tiny-cpu.md (human-readable) next to the repo's benchmarks/ dir.",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"JSON output path (default: {DEFAULT_OUT})")
    ap.add_argument("--device", default="auto", help="'auto' (default), 'cpu', or 'cuda'")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--threads", type=int, default=None,
                    help="pin torch CPU threads (default: torch's own setting)")
    ap.add_argument("--sections", default=",".join(_ALL_SECTIONS),
                    help=f"comma-separated subset of {_ALL_SECTIONS}")
    ap.add_argument("--train-steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-seq", type=int, default=8, help="sequences in the synthetic corpus")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--prefill-repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmup steps/reps")
    ap.add_argument("--in-process", action="store_true",
                    help="run sections in this process instead of isolated children "
                    "(peak RSS is then a process-wide high-water mark)")
    ap.add_argument("--no-md", action="store_true", help="skip the .markdown summary")
    args = ap.parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    sections = tuple(s.strip() for s in args.sections.split(",") if s.strip())

    results = run_all(
        sections,
        seed=args.seed,
        device=args.device,
        train_steps=args.train_steps,
        batch=args.batch,
        seq=args.seq,
        lr=args.lr,
        train_warmup=args.warmup,
        n_seq=args.n_seq,
        prompt_len=args.prompt_len,
        decode_tokens=args.decode_tokens,
        prefill_repeats=args.prefill_repeats,
        infer_warmup=args.warmup,
        in_process=args.in_process,
    )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, out_path)

    md_path = os.path.splitext(out_path)[0] + ".md"
    markdown = render_markdown(results)
    if not args.no_md:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(markdown)

    print(markdown)
    print(f"WROTE {out_path}" + ("" if args.no_md else f" and {md_path}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
