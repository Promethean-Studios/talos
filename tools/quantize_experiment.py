"""Quantization experiment for the Talos tiny prototype: fp32 vs bf16 vs int8.

The single command that quantizes a *trained* tiny checkpoint and measures what
weight-only quantization does to it::

    python tools/quantize_experiment.py           # -> benchmarks/quant-tiny-cpu.{json,md}
    python tools/quantize_experiment.py --quick   # reduced steps (CI / tests)

Protocol (everything deterministic under ``--seed``):

1. **Train** the tiny preset (:func:`configs.presets.tiny_config`, 254,272
   params) briefly with AdamW on the deterministic synthetic recurrent corpus
   (:func:`training.synthetic.build_recurrent_corpus`) — the #10-benchmark
   training recipe — so quantization acts on *learned* weights, not init.
2. **Reference** the fp32 model: one-shot logits over prompts of several
   lengths drawn from a held-out eval corpus, teacher-forced eval loss over
   that corpus, greedy KV-cache generations (:func:`inference.generate.generate`),
   serialized payload size (:func:`quantization.payload_bytes`), and
   prefill/decode throughput.
3. **Quantize** fresh deep-copies of the same fp32 checkpoint to ``bf16``
   (true bf16 compute) and ``int8`` (symmetric weight-only, per-channel and
   per-tensor scales) and repeat every measurement on each variant.
4. **Compare**: per prompt length — logit mean/max absolute difference, mean
   and min cosine similarity, greedy (argmax) agreement rate; plus
   generation-level agreement and the eval-loss delta.

The committed JSON is a **CPU** artifact (same machine class as
``benchmarks/baseline-tiny-cpu.json``); timing numbers vary with hardware, the
correctness metrics do not (up to BLAS nondeterminism, which the fp32-vs-fp32
control row bounds).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the repo-root packages (`model`, `configs`, `training`, `inference`,
# `quantization`) importable when this file is run directly as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from configs.presets import tiny_config  # noqa: E402
from inference.generate import decode_step, generate, prefill  # noqa: E402
from model import TalosGPT  # noqa: E402
from model.utils import set_seed  # noqa: E402
from quantization import payload_bytes, quantize_model  # noqa: E402
from quantization.evaluate import (  # noqa: E402
    compare_generations,
    compare_logits,
    eval_loss,
    logit_similarity_by_prompt,
)
from tools.benchmark import _cpu_model, _git_info, resolve_device  # noqa: E402

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "talos-tiny-quant"
DEFAULT_OUT = os.path.join(_REPO_ROOT, "benchmarks", "quant-tiny-cpu.json")
_MIB = 1024 * 1024
DEFAULT_PROMPT_LENGTHS: Tuple[int, ...] = (8, 16, 32)
# fp32 loss ceiling sanity: eval-loss deltas below this are "noise-level".
LOSS_DELTA_EPS = 0.05


# ---------------------------------------------------------------------------
# Training (the #10 benchmark recipe, in-process)
# ---------------------------------------------------------------------------
def train_tiny(
    device: str,
    seed: int,
    steps: int,
    batch: int,
    seq: int,
    lr: float,
    n_seq: int,
) -> Tuple[TalosGPT, Dict[str, Any], torch.Tensor]:
    """Train the tiny preset on the synthetic corpus; return (model, stats, eval_corpus).

    Identical recipe to ``tools/benchmark.py``'s train section: aligned causal-LM
    objective, AdamW, seeded batch sampling over the deterministic recurrent
    corpus. The eval corpus is built with ``seed + 1`` so prompts/eval loss are
    held out from training batches' corpus.
    """
    cfg = tiny_config().derive()
    set_seed(seed)
    model = TalosGPT(cfg).to(device)
    corpus = build_corpus(cfg.vocab_size, n_seq, seq, seed, device)
    eval_corpus = build_corpus(cfg.vocab_size, max(n_seq, 8), max(seq, 48), seed + 1, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    gen = torch.Generator(device=corpus.device).manual_seed(seed)

    model.eval()
    with torch.no_grad():
        logits, _ = model(corpus[:, :-1])
        loss_initial = float(loss_fn(logits.reshape(-1, cfg.vocab_size), corpus[:, 1:].reshape(-1)))
    model.train()

    t0 = time.perf_counter()
    final_loss = loss_initial
    for _ in range(steps):
        idx = torch.randint(0, corpus.shape[0], (batch,), generator=gen, device=corpus.device)
        x = corpus[idx]
        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
    train_s = time.perf_counter() - t0
    stats = {
        "steps": steps,
        "batch": batch,
        "seq_len": seq,
        "lr": lr,
        "loss_initial": round(loss_initial, 4),
        "loss_final_batch": round(final_loss, 4),
        "train_tokens": steps * batch * (seq - 1),
        "train_seconds": round(train_s, 3),
        "train_tokens_per_sec": round(steps * batch * (seq - 1) / train_s, 1),
    }
    return model, stats, eval_corpus


def build_corpus(vocab_size: int, n_seq: int, seq_len: int, seed: int, device: str) -> torch.Tensor:
    """Import-lazy wrapper so ``--help`` does not pay the training import."""
    from training.synthetic import build_recurrent_corpus

    return build_recurrent_corpus(vocab_size, n_sequences=n_seq, seq_len=seq_len, seed=seed, device=device)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def _prompts(eval_corpus: torch.Tensor, lengths: Sequence[int]) -> List[torch.Tensor]:
    """Deterministic held-out prompts: leading windows of eval-corpus rows."""
    return [eval_corpus[i % eval_corpus.shape[0]][:L].unsqueeze(0) for i, L in enumerate(lengths)]


def measure_throughput(
    model: TalosGPT,
    prompt: torch.Tensor,
    decode_tokens: int,
    warmup: int,
    prefill_repeats: int,
) -> Dict[str, float]:
    """Prefill tokens/s and KV-decode tokens/s via the canonical inference path."""
    device = model.embed_tokens.weight.device
    prompt_len = prompt.shape[1]
    with torch.no_grad():
        for _ in range(warmup):
            prefill(model, prompt)
        prefill_ms: List[float] = []
        for _ in range(prefill_repeats):
            t0 = time.perf_counter()
            prefill(model, prompt)
            prefill_ms.append((time.perf_counter() - t0) * 1000.0)
        logits, cache = prefill(model, prompt)
        nxt = logits[:, -1:, :].argmax(dim=-1)
        for _ in range(warmup):  # untimed decode warmup
            out = decode_step(model, cache, nxt, cache.length)
            nxt = out[:, -1:, :].argmax(dim=-1)
        decode_ms: List[float] = []
        for _ in range(decode_tokens):
            t0 = time.perf_counter()
            out = decode_step(model, cache, nxt, cache.length)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            nxt = out[:, -1:, :].argmax(dim=-1)
        mean_prefill = sum(prefill_ms) / len(prefill_ms)
        mean_decode = sum(decode_ms) / len(decode_ms)
    return {
        "prefill_ms_mean": round(mean_prefill, 3),
        "prefill_tokens_per_sec": round(prompt_len / (mean_prefill / 1000.0), 1),
        "decode_ms_mean": round(mean_decode, 3),
        "decode_tokens_per_sec": round(1000.0 / mean_decode, 1),
    }


def measure_variant(
    name: str,
    model: TalosGPT,
    ref_logits: Dict[str, torch.Tensor],
    ref_gens: Dict[str, List[int]],
    prompts: List[torch.Tensor],
    eval_corpus: torch.Tensor,
    vocab_size: int,
    gen_prompt: torch.Tensor,
    throughput_cfg: Dict[str, int],
) -> Dict[str, Any]:
    """Full measurement suite for one precision variant vs the fp32 reference."""
    model.eval()
    logits_by_len = {}
    gens: Dict[str, List[int]] = {}
    with torch.no_grad():
        for prompt in prompts:
            lg, _ = model(prompt)
            logits_by_len[str(prompt.shape[1])] = lg
        gens[str(gen_prompt.shape[1])] = generate(
            model, gen_prompt, max_new_tokens=throughput_cfg["decode_tokens"], greedy=True
        )
    gen_key = str(gen_prompt.shape[1])
    return {
        "name": name,
        "payload": payload_bytes(model),
        "eval_loss": round(eval_loss(model, eval_corpus, vocab_size), 4),
        "logit_similarity": {
            length: compare_logits(ref_logits[length], got)
            for length, got in logits_by_len.items()
        },
        "generation": {
            "reference": ref_gens[gen_key],
            "quantized": gens[gen_key],
            **compare_generations(ref_gens[gen_key], gens[gen_key]),
        },
        "throughput": measure_throughput(
            model,
            prompts[-1],
            decode_tokens=throughput_cfg["decode_tokens"],
            warmup=throughput_cfg["warmup"],
            prefill_repeats=throughput_cfg["prefill_repeats"],
        ),
    }


def bottom_line(results: Dict[str, Dict[str, Any]], fp32_loss: float) -> Dict[str, Dict[str, Any]]:
    """Per-variant verdict: did greedy output change, how far did similarity move?"""
    verdict: Dict[str, Dict[str, Any]] = {}
    for name, r in results.items():
        sims = r["logit_similarity"]
        cos = [m["cosine_mean"] for m in sims.values()]
        agree = [m["greedy_agreement"] for m in sims.values()]
        verdict[name] = {
            "greedy_generation_identical": bool(r["generation"]["identical"]),
            "greedy_agreement_min": min(agree),
            "cosine_mean_min": min(cos),
            "eval_loss_delta": round(r["eval_loss"] - fp32_loss, 4),
            "size_mb": r["payload"]["total_mb"],
            "compression_vs_fp32": None,
        }
    fp32_mb = results["fp32"]["payload"]["total_mb"]
    for name, v in verdict.items():
        v["compression_vs_fp32"] = round(fp32_mb / max(v["size_mb"], 1e-9), 2)
    return verdict


def _peak_rss_mb() -> float:
    """Whole-process peak RSS (interpreter + torch included)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_experiment(
    *,
    seed: int = 0,
    device: str = "cpu",
    steps: int = 100,
    batch: int = 4,
    seq: int = 64,
    lr: float = 1e-3,
    n_seq: int = 16,
    prompt_lengths: Sequence[int] = DEFAULT_PROMPT_LENGTHS,
    decode_tokens: int = 16,
    warmup: int = 2,
    prefill_repeats: int = 5,
    skip: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run the full fp32/bf16/int8 experiment and return the result dict."""
    dev = resolve_device(device)
    device_str = str(dev)
    model, train_stats, eval_corpus = train_tiny(
        device_str, seed, steps, batch, seq, lr, n_seq
    )
    cfg = tiny_config().derive()
    model.eval()
    prompts = _prompts(eval_corpus, prompt_lengths)
    gen_prompt = prompts[1]  # the middle length is the generation/throughput prompt
    throughput_cfg = {"decode_tokens": decode_tokens, "warmup": warmup, "prefill_repeats": prefill_repeats}

    # fp32 reference measurements (the control row: fp32 vs fp32 == 0 diff).
    ref_logits: Dict[str, torch.Tensor] = {}
    ref_gens: Dict[str, List[int]] = {}
    with torch.no_grad():
        for prompt in prompts:
            lg, _ = model(prompt)
            ref_logits[str(prompt.shape[1])] = lg
        ref_gens[str(gen_prompt.shape[1])] = generate(
            model, gen_prompt, max_new_tokens=decode_tokens, greedy=True
        )
    results: Dict[str, Dict[str, Any]] = {
        "fp32": measure_variant(
            "fp32", model, ref_logits, ref_gens, prompts, eval_corpus,
            cfg.vocab_size, gen_prompt, throughput_cfg,
        )
    }

    # Variants: each quantizes a FRESH deep-copy of the same fp32 checkpoint.
    variants: Tuple[Tuple[str, str, str], ...] = (
        ("bf16", "bf16", "per_channel"),
        ("int8_per_channel", "int8", "per_channel"),
        ("int8_per_tensor", "int8", "per_tensor"),
    )
    for name, precision, granularity in variants:
        quantized, report = quantize_model(model, precision=precision, granularity=granularity, skip=skip)
        quantized.eval()
        results[name] = measure_variant(
            name, quantized, ref_logits, ref_gens, prompts, eval_corpus,
            cfg.vocab_size, gen_prompt, throughput_cfg,
        )
        results[name]["report"] = report.as_dict()
        del quantized

    return {
        "name": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "preset": "tiny",
        "device": device_str,
        "seed": seed,
        "env": {
            "cpu": _cpu_model(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            **_git_info(),
        },
        "train": train_stats,
        "config": {
            "prompt_lengths": list(prompt_lengths),
            "generation_prompt_len": gen_prompt.shape[1],
            "decode_tokens": decode_tokens,
            "warmup": warmup,
            "prefill_repeats": prefill_repeats,
            "skip": list(skip),
            "eval_corpus_held_out": True,
        },
        "precisions": results,
        "bottom_line": bottom_line(results, results["fp32"]["eval_loss"]),
        "peak_rss_mb": _peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def render_markdown(results: Dict[str, Any]) -> str:
    """Human-readable summary table mirroring the committed JSON."""
    lines: List[str] = [
        f"# Quantization experiment — {results['preset']} preset (CPU)",
        "",
        f"Trained {results['train']['steps']} AdamW steps on the deterministic synthetic "
        f"corpus (loss {results['train']['loss_initial']} → "
        f"{results['train']['loss_final_batch']}); all metrics vs the fp32 model on "
        f"held-out prompts at lengths {results['config']['prompt_lengths']}.",
        "",
        "| variant | size MB | ×vs fp32 | eval loss | Δloss | cosine mean (min) | greedy agree (min) | greedy gen identical | decode tok/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fp32_mb = results["precisions"]["fp32"]["payload"]["total_mb"]
    for name, r in results["precisions"].items():
        sims = list(r["logit_similarity"].values())
        cos_mean = sum(m["cosine_mean"] for m in sims) / len(sims)
        cos_min = min(m["cosine_min"] for m in sims)
        agree_min = min(m["greedy_agreement"] for m in sims)
        bl = results["bottom_line"][name]
        lines.append(
            f"| {name} | {r['payload']['total_mb']} | {bl['compression_vs_fp32']} "
            f"| {r['eval_loss']} | {bl['eval_loss_delta']:+.4f} "
            f"| {cos_mean:.6f} ({cos_min:.6f}) | {agree_min:.4f} "
            f"| {'yes' if bl['greedy_generation_identical'] else 'NO'} "
            f"| {r['throughput']['decode_tokens_per_sec']} |"
        )
    lines += [
        "",
        f"Peak RSS (whole process): {results['peak_rss_mb']} MB. "
        "int8 = symmetric weight-only (int8 payload + fp32 scales, W8A32 dequant-matmul); "
        "bf16 = true bf16 compute. Timing numbers are hardware-specific; correctness "
        "metrics are not. Seed "
        f"{results['seed']}, torch {results['env']['torch']}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Quantization experiment: train tiny, measure fp32 vs bf16 vs int8 "
        "(writes benchmarks/quant-tiny-cpu.json + .md)."
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="'auto', 'cpu', or 'cuda' (default cpu)")
    ap.add_argument("--steps", type=int, default=100, help="AdamW training steps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64, help="training sequence length")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-seq", type=int, default=16, help="corpus sequences")
    ap.add_argument("--prompt-lengths", type=int, nargs="+", default=list(DEFAULT_PROMPT_LENGTHS))
    ap.add_argument("--decode-tokens", type=int, default=16, help="greedy tokens generated/timed")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--prefill-repeats", type=int, default=5)
    ap.add_argument("--skip", nargs="*", default=[], help="module names to leave fp32 (e.g. lm_head)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    ap.add_argument("--no-md", action="store_true", help="skip the .markdown summary")
    ap.add_argument("--quick", action="store_true", help="tiny settings for CI/tests")
    args = ap.parse_args(argv)

    kwargs: Dict[str, Any] = dict(
        seed=args.seed, device=args.device, steps=args.steps, batch=args.batch,
        seq=args.seq, lr=args.lr, n_seq=args.n_seq, prompt_lengths=tuple(args.prompt_lengths),
        decode_tokens=args.decode_tokens, warmup=args.warmup,
        prefill_repeats=args.prefill_repeats, skip=tuple(args.skip),
    )
    if args.quick:
        kwargs.update(steps=10, batch=2, seq=32, n_seq=4, prompt_lengths=(8, 16),
                      decode_tokens=8, warmup=1, prefill_repeats=2)
    results = run_experiment(**kwargs)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
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
    raise SystemExit(main())
