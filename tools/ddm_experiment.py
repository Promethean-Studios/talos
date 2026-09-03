"""DDM KV-tier experiment for the Talos tiny CPU prototype.

Runs the pre-registered memory-tier experiment of
``ddm_experiment_design.md`` on the dense ``tiny`` preset (254,272 params):

* **R — resident control**: the existing :class:`model.cache.KVCache` through
  the canonical :func:`inference.generate.prefill` / :func:`decode_step` path.
* **T — file-tier hypothesis**: :class:`experiments.ddm_kv_cache.DiskTieredKVCache`
  (64-token completed blocks in a run-private temp file, advisory page
  eviction requested), same public path via ``prefill(..., cache=...)``.

Fixed configuration (design §4): 384-token prefill + 128 greedy decode tokens
(batch 1, fp32, CPU, ``torch.set_num_threads(2)``), one deterministic tiny
checkpoint trained for 100 AdamW steps on the recurrent-corpus recipe of
``tools/quantize_experiment.py`` (seed 0; held-out corpus/prompt seed 1),
2 warmup transactions per condition, then R/T pairs executed in alternating
order inside spawned child processes (15 timed pairs by default; ``--smoke``
runs 5 pairs and is labeled a smoke run with **no acceptance claim**).

Measured per design §3: per-condition prefill/decode/transaction timings,
paired ratios with a bootstrap CI on the paired median transaction-throughput
ratio, quality guardrails (logit max abs diff <= 1e-4, greedy agreement at
every compared position, teacher-forced loss delta <= 1e-5), and the exact
4-quantity memory inventory. The automatic verdict lists design §5 rules
verbatim with their pass/fail state. "No benefit at this scale" is a valid
pre-registered outcome (§5.4) — nothing here tunes the mechanism to force a
positive result.

Output: ``benchmarks/ddm-tiny-cpu.json`` + ``benchmarks/ddm-tiny-cpu.md``
(atomically written, ``tools/benchmark.py`` conventions).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import queue as queue_mod
import resource
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the repo-root packages (`model`, `configs`, `experiments`, `inference`,
# `tools`) importable when this file is run directly as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from configs.presets import tiny_config  # noqa: E402
from experiments.ddm_kv_cache import DiskTieredKVCache  # noqa: E402
from inference.generate import decode_step, prefill  # noqa: E402
from model import TalosGPT  # noqa: E402
from model.utils import set_seed  # noqa: E402
from tools.benchmark import _cpu_model, _git_info  # noqa: E402
from tools.quantize_experiment import build_corpus, train_tiny  # noqa: E402

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "talos-tiny-ddm-kv-tier"
DEFAULT_OUT = os.path.join(_REPO_ROOT, "benchmarks", "ddm-tiny-cpu.json")

# -- Fixed configuration (design §4) -----------------------------------------
PROMPT_TOKENS = 384
DECODE_TOKENS = 128  # 128 decode steps; cache ends at exactly max_seq_len=512
TRANSACTION_TOKENS = PROMPT_TOKENS + DECODE_TOKENS  # 512
BLOCK_SIZE = 64
CPU_THREADS = 2
TRAIN_SEED = 0  # checkpoint training / init seed
HELDOUT_SEED = 1  # held-out corpus / prompt seed
TRAIN_STEPS = 100
TRAIN_BATCH = 4
TRAIN_SEQ = 64
TRAIN_LR = 1e-3
TRAIN_N_SEQ = 16
NUM_LOSS_SEQUENCES = 2
WARMUP_PER_CONDITION = 2
FULL_REPEATS = 15
SMOKE_REPEATS = 5
BOOTSTRAP_SAMPLES = 10_000

# -- Guardrails (design §1.3 / §3C) ------------------------------------------
LOGIT_ATOL = 1e-4
LOSS_DELTA_ATOL = 1e-5
RAM_REDUCTION_MIN = 0.75
THROUGHPUT_RATIO_MIN = 0.90

CONDITIONS: Tuple[str, ...] = ("resident", "tiered")

# Design §5 interpretation rules, verbatim (listed in the artifact verdict).
RULE_1 = (
    "Rule 1 (quality): Any violation of the logit, loss, or greedy-agreement "
    "guardrail means the tiered implementation is incorrect."
)
RULE_2 = (
    "Rule 2 (benefit gates): Support the hypothesis only when all three "
    "benefit gates pass: >=75% persistent-KV reduction, paired-median "
    "transaction throughput >=90% of resident with bootstrap CI lower bound "
    ">=0.90, and all quality guards pass."
)
RULE_3 = (
    "Rule 3 (memory-only trade): If persistent RAM falls by >=75% but "
    "throughput is <90% of resident, record the exact slowdown and conclude: "
    "\"mechanism is correct, but no acceptable benefit at the tiny CPU scale.\""
)
RULE_4 = (
    "Rule 4 (valid negative): No benefit at this scale is a successful "
    "experimental outcome."
)
RULE_7 = (
    "Rule 7 (cold-store caveat): If OS page eviction is advisory/unverified, "
    "throughput is labeled file-backed/possibly warm-page; a positive result "
    "under that condition needs later confirmation, a negative result stands."
)


# ---------------------------------------------------------------------------
# Checkpoint + held-out data preparation (deterministic; outside timing scope)
# ---------------------------------------------------------------------------
def prepare_checkpoint(
    device: str,
    workdir: str,
) -> Dict[str, Any]:
    """Train the deterministic tiny checkpoint (quantize-experiment recipe).

    Seed 0 / 100 AdamW steps on the recurrent corpus; held-out prompt and
    loss sequences come from the seed-1 corpus. Saves the fp32 ``state_dict``
    to ``workdir/checkpoint.pt`` for the spawned child processes and returns
    the fixed workload tensors as plain lists (picklable).
    """
    model, train_stats, _ = train_tiny(
        device=device,
        seed=TRAIN_SEED,
        steps=TRAIN_STEPS,
        batch=TRAIN_BATCH,
        seq=TRAIN_SEQ,
        lr=TRAIN_LR,
        n_seq=TRAIN_N_SEQ,
    )
    ckpt_path = os.path.join(workdir, "checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)

    cfg = tiny_config().derive()
    # Held-out (seed 1) recurrent corpus; rows are exactly 512 tokens so one
    # row spans the full transaction and one row is one loss sequence.
    eval_corpus = build_corpus(cfg.vocab_size, 8, TRANSACTION_TOKENS, HELDOUT_SEED, device)
    prompt = eval_corpus[0][:PROMPT_TOKENS].tolist()
    loss_sequences = [eval_corpus[1 + i].tolist() for i in range(NUM_LOSS_SEQUENCES)]
    return {
        "checkpoint_path": ckpt_path,
        "prompt": prompt,
        "loss_sequences": loss_sequences,
        "train_stats": train_stats,
        "train_final_loss": train_stats.get("final_loss"),
    }


def load_model(checkpoint_path: str) -> TalosGPT:
    """Rebuild the tiny model and load the frozen checkpoint deterministically."""
    set_seed(TRAIN_SEED)
    model = TalosGPT(tiny_config().derive())
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# One timed transaction (design §3B)
# ---------------------------------------------------------------------------
def run_transaction(
    model: TalosGPT,
    condition: str,
    prompt: torch.Tensor,
    first_token: torch.Tensor,
    capture_logits: bool = False,
) -> Dict[str, Any]:
    """One fresh-cache transaction: 384-token prefill + 128 greedy decode steps.

    The timing scope starts immediately before the prefill call and ends after
    the 128th decode step (design §3B). Cache/temp-file *construction* happens
    before the scope (both conditions symmetric via ``prefill(cache=...)``);
    the tiered cache's writes/reads/reconstruction are inside it — they are
    the mechanism under test. ``first_token`` is the shared initial next token
    (design §4: same frozen weights, prompt, and initial token both sides).
    """
    if condition == "resident":
        cache: Any = model.new_cache(1, prompt.device, next(model.parameters()).dtype)
        tier = None
    elif condition == "tiered":
        tier = DiskTieredKVCache.for_model(model, block_size=BLOCK_SIZE, sync=True)
        cache = tier
    else:
        raise ValueError(f"unknown condition {condition!r}")
    try:
        prefill_logits_list: List[torch.Tensor] = []
        step_logits_list: List[torch.Tensor] = []
        decode_ms: List[float] = []
        with torch.no_grad():
            t0 = time.perf_counter()
            logits, cache = prefill(model, prompt, cache=cache)
            if capture_logits:
                prefill_logits_list.append(logits[0].clone())
            tok = first_token
            for _ in range(DECODE_TOKENS):
                t1 = time.perf_counter()
                out = decode_step(model, cache, tok, cache.length)
                decode_ms.append((time.perf_counter() - t1) * 1000.0)
                if capture_logits:
                    step_logits_list.append(out[0, 0].clone())
                tok = out[:, -1:, :].argmax(dim=-1)
            total_s = time.perf_counter() - t0
        prefill_ms = total_s * 1000.0 - sum(decode_ms)
        # Greedy continuation = shared first token + argmax of steps 1..127
        # (128 generated tokens at positions 385..512).
        continuation = [int(first_token[0, 0])]
        if capture_logits:
            step_t = torch.stack(step_logits_list)  # (128, vocab)
            continuation.extend(int(i) for i in step_t.argmax(dim=-1)[: DECODE_TOKENS - 1])
        result = {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prefill_plus_decode_ms": total_s * 1000.0,
            "prefill_tokens_per_sec": PROMPT_TOKENS / (prefill_ms / 1000.0),
            "decode_tokens_per_sec": DECODE_TOKENS / (sum(decode_ms) / 1000.0),
            "transaction_tokens_per_sec": TRANSACTION_TOKENS / total_s,
            "cache_final_length": int(cache.length),
        }
        if capture_logits:
            result["prefill_logits"] = prefill_logits_list[0]
            result["step_logits"] = step_t
            result["continuation"] = continuation
        if tier is not None:
            result["ledger"] = tier.ledger.to_dict()
        return result
    finally:
        if tier is not None:
            tier.close()


def _decode_ms_stats(decode_ms: Sequence[float]) -> Dict[str, float]:
    s = sorted(decode_ms)
    n = len(s)

    def p(q: float) -> float:
        return round(s[min(n - 1, int(q * (n - 1) + 0.5))], 4)

    return {
        "decode_ms_mean": round(sum(s) / n, 4),
        "decode_ms_p50": p(0.50),
        "decode_ms_p95": p(0.95),
        "decode_ms_max": round(s[-1], 4),
    }


# ---------------------------------------------------------------------------
# Quality guardrails (design §3C)
# ---------------------------------------------------------------------------
def _compare_logits(ref: torch.Tensor, got: torch.Tensor) -> Dict[str, Any]:
    diff = (ref - got).abs()
    agreement = int((ref.argmax(dim=-1) == got.argmax(dim=-1)).sum())
    ref_f, got_f = ref.reshape(-1).double(), got.reshape(-1).double()
    cos = float(
        torch.dot(ref_f, got_f)
        / (ref_f.norm() * got_f.norm()).clamp_min(1e-30)
    )
    return {
        "positions": int(ref.shape[-2]) if ref.dim() > 1 else 1,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": round(cos, 9),
        "greedy_agreement_count": agreement,
    }


def _teacher_forced_loss(model: TalosGPT, seq: torch.Tensor, cache: Any) -> float:
    """Cross-entropy of a token-by-token cache decode over one (512,) sequence.

    Feeds tokens 0..510 through :func:`decode_step` (exercising the cache on
    every step) and averages CE against the true next tokens 1..511.
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total, count = 0.0, seq.numel() - 1
    with torch.no_grad():
        tok = seq[:1].view(1, 1)
        for pos in range(count):
            out = decode_step(model, cache, tok, cache.length)
            total += float(loss_fn(out[0, 0], seq[pos + 1]))
            tok = seq[pos + 1 : pos + 2].view(1, 1)
    return total / count


def quality_and_inventory_worker(payload: Dict[str, Any], q: Any) -> None:
    """Child-process quality phase: guardrails + 4-quantity memory inventory."""
    try:
        torch.set_num_threads(CPU_THREADS)
        model = load_model(payload["checkpoint_path"])
        device = "cpu"
        prompt = torch.tensor(payload["prompt"], dtype=torch.long, device=device).view(1, -1)
        loss_seqs = [
            torch.tensor(s, dtype=torch.long, device=device) for s in payload["loss_sequences"]
        ]

        with torch.no_grad():
            # Shared initial next token from a resident prefill (design §4).
            seed_logits, _ = prefill(model, prompt)
            first_token = seed_logits[:, -1:, :].argmax(dim=-1)

            # One full transaction per condition with logits captured.
            res = run_transaction(model, "resident", prompt, first_token, capture_logits=True)
            tie = run_transaction(model, "tiered", prompt, first_token, capture_logits=True)

        prefill_cmp = _compare_logits(res["prefill_logits"], tie["prefill_logits"])
        step_cmp = _compare_logits(res["step_logits"], tie["step_logits"])
        continuation_match = res["continuation"] == tie["continuation"]

        losses: Dict[str, Any] = {}
        for i, seq in enumerate(loss_seqs):
            loss_r = _teacher_forced_loss(model, seq, model.new_cache(1, "cpu", torch.float32))
            tier_q = DiskTieredKVCache.for_model(model, block_size=BLOCK_SIZE, sync=True)
            try:
                loss_t = _teacher_forced_loss(model, seq, tier_q)
            finally:
                tier_q.close()
            losses[f"seq{i}"] = {
                "resident": loss_r,
                "tiered": loss_t,
                "loss_delta": loss_t - loss_r,
            }

        # 4-quantity memory inventory (design §3D).
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        resident_cache = model.new_cache(1, "cpu", torch.float32)
        resident_bytes = sum(b.nbytes for b in resident_cache.buffers())
        del resident_cache
        tier_ledger = tie["ledger"]
        inventory = {
            "model_parameter_bytes": param_bytes,
            "resident": {
                "persistent_kv_ram_peak_bytes": resident_bytes,  # preallocated 512-token layout
                "persistent_kv_ram_note": "KVCache preallocates the full 512-token buffer at construction (batch 1)",
                "transient_kv_staging_peak_bytes": 0,
                "transient_note": "update() returns views into the preallocated buffer; no extra staging copies",
                "cold_file_payload_peak_bytes": 0,
                "bytes_read_total": 0,
                "bytes_written_total": 0,
            },
            "tiered": {
                "persistent_kv_ram_peak_bytes": tier_ledger["persistent_kv_ram_peak_bytes"],
                "transient_kv_staging_peak_bytes": tier_ledger["transient_kv_staging_peak_bytes"],
                "cold_file_payload_peak_bytes": tier_ledger["file_payload_peak_bytes"],
                "bytes_read_total": tier_ledger["bytes_read_total"],
                "bytes_written_total": tier_ledger["bytes_written_total"],
                "move_time_total_ms": tier_ledger["move_time_total_ms"],
                "cold_page_eviction": tier_ledger["cold_page_eviction"],
            },
            "ru_maxrss_kb_informational": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        q.put(
            (
                "ok",
                {
                    "prefill_logits": prefill_cmp,
                    "decode_step_logits": step_cmp,
                    "continuation_match": continuation_match,
                    "continuation": tie["continuation"],
                    "losses": losses,
                    "memory_inventory": inventory,
                },
            )
        )
    except Exception:  # pragma: no cover - surfaced to the parent verbatim
        import traceback

        q.put(("error", traceback.format_exc()))


# ---------------------------------------------------------------------------
# Timed pair worker (design §4: spawned child, alternating order)
# ---------------------------------------------------------------------------
def _pair_worker(payload: Dict[str, Any], q: Any) -> None:
    """Child-process entry: warmup both conditions, then one timed R/T pair."""
    try:
        torch.set_num_threads(CPU_THREADS)
        model = load_model(payload["checkpoint_path"])
        prompt = torch.tensor(payload["prompt"], dtype=torch.long).view(1, -1)
        first_token = torch.tensor([[payload["first_token"]]], dtype=torch.long)
        order: Tuple[str, ...] = tuple(payload["order"])
        timed: Dict[str, Dict[str, Any]] = {}
        with torch.no_grad():
            for _ in range(WARMUP_PER_CONDITION):  # untimed warmup, both conditions
                for cond in CONDITIONS:
                    run_transaction(model, cond, prompt, first_token)
            for cond in order:  # alternating pair order, both timed
                timed[cond] = run_transaction(model, cond, prompt, first_token)
        timed["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        q.put(("ok", timed))
    except Exception:  # pragma: no cover - surfaced to the parent verbatim
        import traceback

        q.put(("error", traceback.format_exc()))


def _run_child(target: Any, payload: Dict[str, Any], timeout_s: float = 1800.0) -> Any:
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    proc = ctx.Process(target=target, args=(payload, q))
    proc.start()
    try:
        status, result = q.get(timeout=timeout_s)
    except queue_mod.Empty as exc:  # pragma: no cover
        proc.join(timeout=5.0)
        raise RuntimeError(f"child process for {target.__name__} timed out") from exc
    proc.join(timeout=60.0)
    if status == "error":  # pragma: no cover
        raise RuntimeError(f"{target.__name__} failed in child process:\n{result}")
    return result


# ---------------------------------------------------------------------------
# Statistics: per-condition summaries + paired ratios + bootstrap CI
# ---------------------------------------------------------------------------
def _summary(samples: Sequence[float]) -> Dict[str, float]:
    s = sorted(samples)
    n = len(s)

    def p(q: float) -> float:
        return round(s[min(n - 1, int(q * (n - 1) + 0.5))], 4)

    return {
        "n": n,
        "mean": round(sum(s) / n, 4),
        "p50": p(0.50),
        "p95": p(0.95),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }


def _bootstrap_median_ratio_ci(
    ratios: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> Dict[str, float]:
    """Percentile bootstrap CI for the paired median throughput ratio."""
    g = torch.Generator().manual_seed(seed)
    data = torch.tensor(list(ratios), dtype=torch.float64)
    n = data.numel()
    idx = torch.randint(0, n, (samples, n), generator=g)
    medians = data[idx].median(dim=1).values
    lo, hi = torch.quantile(medians, torch.tensor([0.025, 0.975], dtype=medians.dtype))
    return {
        "bootstrap_samples": samples,
        "ci95_lower": round(float(lo), 4),
        "ci95_upper": round(float(hi), 4),
    }


def _paired_summary(
    resident: List[Dict[str, Any]], tiered: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Paired per-repeat ratios/percent deltas + median ratio with CI."""
    metrics: Dict[str, Dict[str, List[float]]] = {
        "prefill_ms": {"r": [], "t": []},
        "prefill_plus_decode_ms": {"r": [], "t": []},
        "transaction_tokens_per_sec": {"r": [], "t": []},
        "decode_tokens_per_sec": {"r": [], "t": []},
    }
    for r, t in zip(resident, tiered):
        for k in metrics:
            metrics[k]["r"].append(r[k])
            metrics[k]["t"].append(t[k])
    ratios = [
        t / r for r, t in zip(metrics["transaction_tokens_per_sec"]["r"], metrics["transaction_tokens_per_sec"]["t"])
    ]
    paired: Dict[str, Any] = {
        "pair_count": len(ratios),
        "per_repeat_throughput_ratio_t_over_r": [round(x, 4) for x in ratios],
        "median_throughput_ratio_t_over_r": {
            "point": round(sorted(ratios)[len(ratios) // 2], 4),
            **_bootstrap_median_ratio_ci(ratios),
        },
    }
    for k, v in metrics.items():
        mean_r = sum(v["r"]) / len(v["r"])
        mean_t = sum(v["t"]) / len(v["t"])
        paired[k] = {
            "resident": _summary(v["r"]),
            "tiered": _summary(v["t"]),
            "tiered_vs_resident_pct": round((mean_t / mean_r - 1.0) * 100.0, 2),
        }
    # Per-token decode latency distribution (pooled over repeats).
    paired["decode_ms"] = {
        "resident": _decode_ms_stats([x for r in resident for x in r["decode_ms"]]),
        "tiered": _decode_ms_stats([x for t in tiered for x in t["decode_ms"]]),
    }
    return paired


# ---------------------------------------------------------------------------
# Verdict (design §5 — rules verbatim, decided by the data, not tuned)
# ---------------------------------------------------------------------------
def _verdict(quality_ok: bool, ram_reduction: float, median_ratio: Dict[str, float], smoke: bool) -> Dict[str, Any]:
    ratio_ok = median_ratio["point"] >= THROUGHPUT_RATIO_MIN
    ci_ok = median_ratio["ci95_lower"] >= THROUGHPUT_RATIO_MIN
    ram_ok = ram_reduction >= RAM_REDUCTION_MIN
    gates = {
        "persistent_kv_reduction_ge_75pct": ram_ok,
        "paired_median_throughput_ge_90pct": ratio_ok,
        "bootstrap_ci_lower_ge_90pct": ci_ok,
        "quality_guards_pass": quality_ok,
    }
    rules = {
        RULE_1: quality_ok,
        RULE_2: all(gates.values()),
        RULE_3: quality_ok and ram_ok and not ratio_ok,
        RULE_4: not all(gates.values()),
    }
    if not quality_ok:
        conclusion = "QUALITY_FAILURE: the tiered implementation is incorrect; its performance is not interpreted."
    elif all(gates.values()):
        conclusion = "HYPOTHESIS SUPPORTED at tiny scale: all three benefit gates pass."
    elif ram_ok and not ratio_ok:
        conclusion = (
            "MEMORY-ONLY TRADE: mechanism is correct, but no acceptable benefit "
            "at the tiny CPU scale."
        )
    else:
        conclusion = (
            "NO BENEFIT AT THIS SCALE: the DDM form is not justified for the "
            "current prototype."
        )
    if smoke:
        conclusion += " [SMOKE RUN — 5 repeats, no acceptance claim]"
    return {
        "conclusion": conclusion,
        "smoke_run_no_acceptance_claim": smoke,
        "gates": gates,
        "rules_verbatim": {text: ("PASS" if ok else "FAIL") for text, ok in rules.items()},
    }


# ---------------------------------------------------------------------------
# Environment + artifact rendering (tools/benchmark.py conventions)
# ---------------------------------------------------------------------------
def _environment_info() -> Dict[str, Any]:
    tmp_dir = tempfile.gettempdir()
    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": _cpu_model(),
        "git": _git_info(),
        "cold_store_directory": os.path.realpath(tmp_dir),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "scheduler_policy": (
            {0: "SCHED_OTHER", 1: "SCHED_FIFO", 2: "SCHED_RR", 3: "SCHED_BATCH", 5: "SCHED_IDLE", 6: "SCHED_DEADLINE"}.get(
                os.sched_getscheduler(0) if hasattr(os, "sched_getscheduler") else -1, "unknown"
            )
        ),
    }
    try:  # free disk space where the cold tier lives (when obtainable)
        st = os.statvfs(tmp_dir)
        info["cold_store_free_bytes"] = int(st.f_bavail * st.f_bsize)
    except OSError:  # pragma: no cover
        info["cold_store_free_bytes"] = None
    return info


def render_markdown(results: Dict[str, Any]) -> str:
    r = results
    v = r["verdict"]
    inv = r["memory_inventory"]
    ram = inv["ram_reduction_vs_resident_pct"]
    med = r["paired"]["median_throughput_ratio_t_over_r"]
    q = r["quality"]
    lines: List[str] = [
        f"# DDM KV-tier experiment — {r['experiment_name']} (schema v{r['schema_version']})",
        "",
        f"* Branch `{r['environment']['git']['branch']}` @ `{str(r['environment']['git']['commit'])[:12]}` · "
        f"torch {r['environment']['torch']} · {r['environment']['cpu_model']} · "
        f"{r['environment']['torch_num_threads']} threads · {r['config']['repeats']} timed R/T pairs"
        + (" · **SMOKE RUN (no acceptance claim)**" if v["smoke_run_no_acceptance_claim"] else ""),
        "",
        "## Verdict",
        "",
        f"**{v['conclusion']}**",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Persistent-KV RAM reduction (tiered vs resident) | **{ram:.1f}%** "
        f"({inv['tiered']['persistent_kv_ram_peak_bytes']:,} B vs {inv['resident']['persistent_kv_ram_peak_bytes']:,} B peak) |",
        f"| Paired median transaction throughput (T/R) | **{med['point']:.4f}** "
        f"(95% CI [{med['ci95_lower']:.4f}, {med['ci95_upper']:.4f}]) |",
        f"| Logit max abs diff (prefill / 128 decode steps) | {q['prefill_logits']['max_abs_diff']:.3e} / "
        f"{q['decode_step_logits']['max_abs_diff']:.3e} |",
        f"| Greedy agreement (positions) | {q['prefill_logits']['greedy_agreement_count']}/"
        f"{q['prefill_logits']['positions']} prefill, {q['decode_step_logits']['greedy_agreement_count']}/"
        f"{q['decode_step_logits']['positions']} decode; continuation match: {q['continuation_match']} |",
        f"| Teacher-forced loss delta | {q['loss_delta_max_abs']:.3e} (max abs over "
        f"{q['num_loss_sequences']} held-out sequences) |",
        f"| Transient KV staging (tiered peak) | {inv['tiered']['transient_kv_staging_peak_bytes']:,} B "
        f"(not a memory saving — required by full attention) |",
        f"| Cold-tier bytes moved | read {inv['tiered']['bytes_read_total']:,} B / "
        f"written {inv['tiered']['bytes_written_total']:,} B |",
        f"| Cold-page eviction (POSIX_FADV_DONTNEED) | requested="
        f"{inv['tiered']['cold_page_eviction']['requested']}×, failed="
        f"{inv['tiered']['cold_page_eviction']['failed']}×"
        + ("" if inv["tiered"]["cold_page_eviction"]["supported"] else " — **unsupported on this platform; "
            "throughput is a file-backed/possibly warm-page measurement (§5.7)**"),
        "",
        "## Timings (mean over timed pairs, tiered vs resident)",
        "",
        "| Metric | Resident | Tiered | Δ (T vs R) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("prefill_ms", "Prefill (ms)"),
        ("prefill_plus_decode_ms", "Transaction (ms)"),
        ("transaction_tokens_per_sec", "Transaction tokens/s"),
        ("decode_tokens_per_sec", "Decode tokens/s"),
    ):
        rr, tt, dd = r["paired"][key]["resident"]["mean"], r["paired"][key]["tiered"]["mean"], r["paired"][key]["tiered_vs_resident_pct"]
        lines.append(f"| {label} | {rr:,.2f} | {tt:,.2f} | {dd:+.2f}% |")
    lines += [
        f"| Decode p50 / p95 (ms) | {r['paired']['decode_ms']['resident']['decode_ms_p50']:.3f} / "
        f"{r['paired']['decode_ms']['resident']['decode_ms_p95']:.3f} | "
        f"{r['paired']['decode_ms']['tiered']['decode_ms_p50']:.3f} / "
        f"{r['paired']['decode_ms']['tiered']['decode_ms_p95']:.3f} | — |",
        "",
        "## Interpretation rules (design §5, verbatim)",
        "",
    ]
    lines += [f"- {'✅ PASS' if ok == 'PASS' else '❌ FAIL'} — {rule}" for rule, ok in v["rules_verbatim"].items()]
    lines += [
        "",
        "## Memory inventory (design §3D, exact logical bytes)",
        "",
        "| Quantity | Resident | Tiered |",
        "|---|---:|---:|",
        f"| model parameter bytes | {inv['model_parameter_bytes']:,} | {inv['model_parameter_bytes']:,} |",
        f"| persistent KV RAM (peak) | {inv['resident']['persistent_kv_ram_peak_bytes']:,} | "
        f"{inv['tiered']['persistent_kv_ram_peak_bytes']:,} |",
        f"| transient KV staging (peak) | {inv['resident']['transient_kv_staging_peak_bytes']:,} | "
        f"{inv['tiered']['transient_kv_staging_peak_bytes']:,} |",
        f"| cold file payload (peak) | {inv['resident']['cold_file_payload_peak_bytes']:,} | "
        f"{inv['tiered']['cold_file_payload_peak_bytes']:,} |",
        f"| bytes read total | {inv['resident']['bytes_read_total']:,} | {inv['tiered']['bytes_read_total']:,} |",
        f"| bytes written total | {inv['resident']['bytes_written_total']:,} | {inv['tiered']['bytes_written_total']:,} |",
        "",
        f"Informational process peak RSS (quality child): "
        f"{inv['ru_maxrss_kb_informational']:,} kB — dominated by interpreter/Torch/page cache; "
        "the logical persistent-KV measure above is the acceptance metric (§5.6).",
        "",
        "## Fixed configuration",
        "",
        "```json",
        json.dumps(r["config"], indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment(
    repeats: int = FULL_REPEATS,
    smoke: bool = False,
    out_path: str = DEFAULT_OUT,
    write_md: bool = True,
) -> Dict[str, Any]:
    assert smoke or repeats == FULL_REPEATS, "non-smoke runs must use the full 15 repeats"
    device = "cpu"  # design §4 binds CPU
    with tempfile.TemporaryDirectory(prefix="talos-ddm-exp-") as workdir:
        prep = prepare_checkpoint(device, workdir)
        seed_logits_model = load_model(prep["checkpoint_path"])
        with torch.no_grad():  # shared initial next token (design §4)
            prompt_t = torch.tensor(prep["prompt"], dtype=torch.long).view(1, -1)
            first_token = int(seed_logits_model(prompt_t)[0][0, -1].argmax())
        del seed_logits_model

        # One quality + inventory child (guardrails, memory, shared first token).
        quality = _run_child(
            quality_and_inventory_worker,
            {
                "checkpoint_path": prep["checkpoint_path"],
                "prompt": prep["prompt"],
                "loss_sequences": prep["loss_sequences"],
            },
        )
        first_token = quality["continuation"][0]

        # Timed alternating R/T pairs, each in its own spawned child (§4).
        resident, tiered, pairs_meta = [], [], []
        for pair_id in range(repeats):
            order = CONDITIONS if pair_id % 2 == 0 else tuple(reversed(CONDITIONS))
            result = _run_child(
                _pair_worker,
                {
                    "checkpoint_path": prep["checkpoint_path"],
                    "prompt": prep["prompt"],
                    "first_token": first_token,
                    "order": order,
                },
            )
            resident.append(result["resident"])
            tiered.append(result["tiered"])
            pairs_meta.append({"pair_id": pair_id, "order": list(order)})
            print(f"  pair {pair_id + 1}/{repeats} done ({'→'.join(order)})", flush=True)

    inv = quality["memory_inventory"]
    ram_reduction = 1.0 - inv["tiered"]["persistent_kv_ram_peak_bytes"] / inv["resident"]["persistent_kv_ram_peak_bytes"]
    inv["ram_reduction_vs_resident_pct"] = ram_reduction * 100.0
    loss_delta_max = max(abs(v["loss_delta"]) for v in quality["losses"].values())
    quality_ok = (
        quality["prefill_logits"]["max_abs_diff"] <= LOGIT_ATOL
        and quality["decode_step_logits"]["max_abs_diff"] <= LOGIT_ATOL
        and quality["prefill_logits"]["greedy_agreement_count"] == quality["prefill_logits"]["positions"]
        and quality["decode_step_logits"]["greedy_agreement_count"] == quality["decode_step_logits"]["positions"]
        and quality["continuation_match"]
        and loss_delta_max <= LOSS_DELTA_ATOL
    )
    paired = _paired_summary(resident, tiered)
    verdict = _verdict(quality_ok, ram_reduction, paired["median_throughput_ratio_t_over_r"], smoke)
    cold_evict = inv["tiered"]["cold_page_eviction"]
    file_backed_caveat = not (cold_evict["supported"] and cold_evict["requested"] > 0 and cold_evict["failed"] == 0)

    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "run_kind": "smoke" if smoke else "full",
        "acceptance_claim_allowed": not smoke,
        "environment": _environment_info(),
        "config": {
            "preset": "configs.presets.tiny_config().derive()",
            "parameters": 254_272,
            "batch": 1,
            "dtype": "float32",
            "device": device,
            "prompt_tokens": PROMPT_TOKENS,
            "decode_tokens": DECODE_TOKENS,
            "transaction_tokens": TRANSACTION_TOKENS,
            "block_size": BLOCK_SIZE,
            "decode_policy": "greedy",
            "cpu_threads": CPU_THREADS,
            "train_seed": TRAIN_SEED,
            "heldout_seed": HELDOUT_SEED,
            "train_steps": TRAIN_STEPS,
            "warmup_per_condition": WARMUP_PER_CONDITION,
            "repeats": repeats,
            "max_seq_len": 512,
            "loss_delta_atol": LOSS_DELTA_ATOL,
            "logit_atol": LOGIT_ATOL,
            "train_final_loss": prep["train_final_loss"],
            "train_stats": prep["train_stats"],
            "pair_orders": pairs_meta,
        },
        "raw_samples": {
            "resident": [{k: v for k, v in r.items() if k != "ledger"} for r in resident],
            "tiered": [{k: v for k, v in t.items() if k != "ledger"} for t in tiered],
        },
        "per_repeat_tiered_ledgers": [t.get("ledger") for t in tiered],
        "paired": paired,
        "quality": {
            **quality,
            "loss_delta_max_abs": loss_delta_max,
            "num_loss_sequences": len(quality["losses"]),
            "quality_ok": quality_ok,
        },
        "memory_inventory": inv,
        "file_backed_caveat": file_backed_caveat,
        "verdict": verdict,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:  # atomic write
        json.dump(results, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, out_path)
    md_path = os.path.splitext(out_path)[0] + ".md"
    markdown = render_markdown(results)
    if write_md:
        tmp_md = md_path + ".tmp"
        with open(tmp_md, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        os.replace(tmp_md, md_path)
    print(markdown)
    print(f"WROTE {out_path}" + (f" and {md_path}" if write_md else ""))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="DDM KV-tier experiment on the Talos tiny prototype "
        "(design: /home/team/shared/ddm_experiment_design.md). Writes "
        "benchmarks/ddm-tiny-cpu.{json,md}.",
    )
    ap.add_argument("--repeats", type=int, default=FULL_REPEATS,
                    help=f"timed paired repeats (default {FULL_REPEATS})")
    ap.add_argument("--smoke", action="store_true",
                    help=f"quick smoke run ({SMOKE_REPEATS} repeats, labeled, no acceptance claim)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    ap.add_argument("--no-md", action="store_true", help="skip the .markdown summary")
    args = ap.parse_args(argv)
    repeats = SMOKE_REPEATS if args.smoke else args.repeats
    run_experiment(repeats=repeats, smoke=args.smoke, out_path=args.out, write_md=not args.no_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
