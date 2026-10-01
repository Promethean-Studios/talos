"""scripts/bench_train.py — one-command old-vs-new T4 training-engine A/B.

The owner's T4 benchmark gate: run the OLD training engine against the NEW
(SDPA/AMP/fast-data) engine on the SAME packed corpus + token budget + seed,
in one T4 session, and print a markdown table with the recommended launch
command for the real 250M-token milestone run.

Lanes (all on the same ``--packed-dir``, ``--token-budget`` and ``--seed``):

  old      the pre-upgrade engine: plain attention, fp32, synchronous data
           (--attention-backend plain --amp none --prefetch 0) — the engine
           the owner's ~39k-step checkpoint was produced with;
  new-fp32 the upgraded engine in fp32: SDPA attention + mmap-cached pinned
           batches + prefetch (--attention-backend sdpa --amp none
           --prefetch 2 --pin-memory);
  new-fp16 the upgraded engine with AMP fp16 (--attention-backend sdpa
           --amp fp16 --prefetch 2 --pin-memory; fp16 requires SDPA — the
           plain path's -inf mask overflowed Half).

Per lane it reports (reusing the trainer's own phase timers + performance
records): run-wide and steady-state tok/s, steps/s, torch-allocator VRAM
peak + an nvidia-smi sample peak, the data/fwd/bwd/optim/ckpt phase split
in ms/step, checkpoint save+load seconds, wall time. The markdown table is
printed to stdout and the full results + JSON are written to ``--out-dir``.

CPU ``--dry-run`` mode: forces the CPU device and a tiny token budget so the
plumbing (corpus manifest load, three lanes, table, recommended command) can
be exercised without a GPU — the numbers are CPU numbers, not T4 numbers.

Reuses the trainer's structures (train_run + generated metrics dict) so the
benchmark measures the SAME path the real run will take.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the script
# also works when invoked from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.packed import load_packed_manifest  # noqa: E402
from scripts.train_oasst1 import RUN_METADATA_FILENAME, train_run  # noqa: E402

#: Lane name -> engine overrides (the A/B contract).
LANES: Dict[str, Dict[str, Any]] = {
    "old": dict(
        attention_backend="plain", amp="none", prefetch=0, pin_memory=False,
        label="old/plain fp32",
    ),
    "fp32": dict(
        attention_backend="sdpa", amp="none", prefetch=2, pin_memory=True,
        label="SDPA fp32",
    ),
    "fp16": dict(
        attention_backend="sdpa", amp="fp16", prefetch=2, pin_memory=True,
        label="SDPA fp16",
    ),
}
DEFAULT_LANES = ("old", "fp32", "fp16")
DEFAULT_STEPS = 30  # ~4-6 min per lane on a T4 at batch 16 x seq 512
#: The milestone run the recommended command targets (owner's 250M program).
DEFAULT_REC_BUDGET = 250_000_000
DEFAULT_REC_WARMUP = 2_500_000


class _NvSmiSampler:
    """Sample ``nvidia-smi --query-gpu=memory.used`` on a daemon thread.

    Reports the hardware-level VRAM peak (CUDA context + allocator + kernels)
    — complementary to ``torch.cuda.max_memory_allocated``, which only sees
    the torch allocator's own blocks.
    """

    _INTERVAL = 0.2

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._peak = 0.0
        self._thread: Optional[threading.Thread] = None
        self._ok = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._ok:
            return
        self._thread = threading.Thread(
            target=self._sample, name="bench-nvsmi", daemon=True
        )
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    used = max(float(v) for v in out.stdout.split())
                    self._peak = max(self._peak, used)
            except Exception:  # noqa: BLE001 - sampling is best-effort
                pass
            self._stop.wait(self._INTERVAL)

    def peak_mb(self) -> Optional[float]:
        return round(self._peak, 1) if self._ok else None

    def stop(self) -> Optional[float]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        return self.peak_mb()


def _train_args(
    *,
    packed_dir: str,
    out_dir: str,
    preset: str,
    batch: int,
    seq: Optional[int],
    token_budget: int,
    seed: int,
    lr: float,
    device: str,
    engine: Dict[str, Any],
    log_every_steps: int,
    tokenizer_json: Optional[str],
) -> SimpleNamespace:
    """Build the exact ``train_run`` argument namespace for one lane.

    Mirrors the trainer CLI's defaults; every engine knob is taken from the
    lane config so the old lane is numerically the pre-upgrade engine.
    """
    return SimpleNamespace(
        data=None, packed_dir=packed_dir, out_dir=out_dir, seed=seed,
        preset=preset, resume=None, split_ratio=0.9, split_max_docs=None,
        bpe_num_merges=None, bpe_minfreq=2, bpe_max_docs=None,
        bpe_max_chars=None, tokenizer_json=tokenizer_json, epochs=1,
        seq=seq, batch=batch, lr=lr, token_budget=token_budget,
        warmup_tokens=0, lr_decay="none", save_every_tokens=0, grad_clip=0.0,
        max_consecutive_bad_steps=3, max_steps_per_epoch=None,
        val_max_steps=1, device=device,
        attention_backend=engine["attention_backend"], amp=engine["amp"],
        fused_optim=False, pin_memory=engine["pin_memory"],
        prefetch=engine["prefetch"], compile=False,
        log_every_steps=log_every_steps, ckpt_staging_dir=None,
        no_ckpt_fsync=False,
    )


def _run_lane(
    name: str,
    engine: Dict[str, Any],
    base: SimpleNamespace,
    device: torch.device,
) -> dict:
    """Run one lane end-to-end; return the per-lane report row (metrics +
    bench-collected extras)."""
    out_dir = os.path.join(base.out_dir, f"lane-{name}")
    args = _train_args(
        packed_dir=base.packed_dir, out_dir=out_dir, preset=base.preset,
        batch=base.batch, seq=base.seq, token_budget=base.token_budget,
        seed=base.seed, lr=base.lr, device=str(device),
        engine=engine, log_every_steps=base.log_every_steps,
        tokenizer_json=getattr(base, "tokenizer_json", None),
    )
    print(
        f"\n===== lane {name} ({engine['label']}) — "
        f"attention={engine['attention_backend']} amp={engine['amp']} "
        f"prefetch={engine['prefetch']} pin_memory={engine['pin_memory']} "
        f"====="
    )
    t_lane = time.monotonic()
    sampler = _NvSmiSampler() if device.type == "cuda" else None
    if sampler is not None:
        sampler.start()
    try:
        metrics = train_run(args)
    finally:
        if sampler is not None:
            sampler.stop()
    wall_s = round(time.monotonic() - t_lane, 3)
    # Checkpoint SAVE is already timed by the trainer (history.checkpoint_save_s
    # -> metrics.checkpoint_save_total_s). Time the LOAD of the final
    # checkpoint here (the same map_location="cpu" the owner's resume path
    # uses — Drive IO + deserialize).
    ckpt_path = metrics.get("checkpoint")
    ckpt_load_s: Optional[float] = None
    if ckpt_path and os.path.isfile(ckpt_path):
        t_load = time.monotonic()
        torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_load_s = round(time.monotonic() - t_load, 3)
    perf = metrics.get("performance") or {}
    phases = perf.get("phase_mean_ms_per_step") or {}
    row = {
        "lane": name,
        "label": engine["label"],
        "attention_backend": metrics["engine"]["attention_backend"],
        "amp": metrics["engine"]["amp"],
        "prefetch": metrics["engine"]["prefetch"],
        "pin_memory": metrics["engine"]["pin_memory"],
        "tok_s_run_wide": perf.get("run_wide_tok_s"),
        "tok_s_steady": perf.get("steady_state_tok_s"),
        "steps_s": perf.get("run_wide_steps_s"),
        "vram_peak_torch_mb": metrics.get("vram_peak_mb"),
        "vram_peak_nvsmi_mb": sampler.peak_mb() if sampler is not None else None,
        "phase_data_ms": phases.get("data"),
        "phase_fwd_ms": phases.get("fwd"),
        "phase_bwd_ms": phases.get("bwd"),
        "phase_optim_ms": phases.get("optim"),
        "phase_ckpt_ms": phases.get("ckpt"),
        "ckpt_save_total_s": metrics.get("checkpoint_save_total_s"),
        "ckpt_load_s": ckpt_load_s,
        "wall_s": wall_s,
        "tokens": metrics.get("tokens_processed"),
        "steps": metrics.get("steps"),
        "final_train_loss": metrics.get("final_train_loss"),
        "budget_reached": metrics.get("budget_reached"),
        "out_dir": out_dir,
        "checkpoint": ckpt_path,
        "bad_steps": metrics.get("total_bad_steps"),
    }
    return row


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None:
        return "n/a"
    return f"{v:,}{suffix}" if isinstance(v, int) else f"{v:,.3f}{suffix}"


def _render_markdown(
    env: Dict[str, Any],
    rows: List[dict],
    winner: str,
    recommended: str,
) -> str:
    lines: List[str] = []
    add = lines.append
    add(f"# Talos training-engine A/B benchmark — {env['timestamp']}")
    add("")
    add(f"- device: `{env['device']}` | torch {env['torch']} | "
        f"cuda {env['cuda']}")
    add(f"- corpus: `{env['packed_dir']}` — seq {env['seq']}, "
        f"dtype {env['dtype']}, {env['train_rows']:,} train rows "
        f"({env['train_tokens']:,} tokens)")
    add(f"- run: preset `{env['preset']}`, batch {env['batch']}, "
        f"seed {env['seed']}, lr {env['lr']:g}, token budget "
        f"{env['token_budget']:,} (~{env['steps']} steps), "
        f"val_max_steps {env['val_max_steps']}")
    add(f"- out dir: `{env['out_dir']}`")
    add("")
    add("## Throughput + memory")
    add("")
    add("| lane | attention | amp | prefetch | pin | tok/s (run-wide) | "
        "tok/s (steady) | steps/s | VRAM peak (torch, MiB) | VRAM peak "
        "(nvsmi, MiB) | wall s |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['lane']} | {r['attention_backend']} | {r['amp']} | "
            f"{r['prefetch']} | {r['pin_memory']} | "
            f"{_fmt(r['tok_s_run_wide'])} | {_fmt(r['tok_s_steady'])} | "
            f"{_fmt(r['steps_s'])} | {_fmt(r['vram_peak_torch_mb'])} | "
            f"{_fmt(r['vram_peak_nvsmi_mb'])} | {_fmt(r['wall_s'])} |")
    add("")
    add("## Phase split (ms/step) + checkpoint I/O")
    add("")
    add("| lane | data | fwd | bwd | optim | ckpt | ckpt save (s) | "
        "ckpt load (s) |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['lane']} | {_fmt(r['phase_data_ms'])} | "
            f"{_fmt(r['phase_fwd_ms'])} | {_fmt(r['phase_bwd_ms'])} | "
            f"{_fmt(r['phase_optim_ms'])} | {_fmt(r['phase_ckpt_ms'])} | "
            f"{_fmt(r['ckpt_save_total_s'])} | {_fmt(r['ckpt_load_s'])} |")
    add("")
    add("## Loss sanity (same seed; fp16/fp32 paths are NOT bit-identical)")
    add("")
    add("| lane | tokens | steps | final train loss | budget reached |")
    add("|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['lane']} | {_fmt(r['tokens'])} | {_fmt(r['steps'])} | "
            f"{_fmt(r['final_train_loss'])} | {r['budget_reached']} |")
    add("")
    add(f"## Recommended launch command (winner: `{winner}` lane)")
    add("")
    add("```bash")
    add(recommended)
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def _recommended_command(
    env: Dict[str, Any], winner_row: dict, token_budget: int,
    warmup_tokens: int,
) -> str:
    e = winner_row
    return (
        "python -m scripts.train_oasst1 "
        f"--packed-dir {env['packed_dir']} "
        f"--out-dir talos_runs/tiny_100m_milestone "
        f"--preset {env['preset']} --batch {env['batch']} --seed {env['seed']} "
        f"--lr {env['lr']:g} "
        f"--token-budget {token_budget} --warmup-tokens {warmup_tokens} "
        f"--lr-decay cosine "
        f"--attention-backend {e['attention_backend']} --amp {e['amp']} "
        f"--prefetch {e['prefetch']} "
        f"{'--pin-memory' if e['pin_memory'] else '--no-pin-memory'}"
    )


def _suggested_launch_preamble(
    preset: str, batch: int, seq: int, token_budget: int, lanes: List[str]
) -> str:
    """Banner printed before the lanes (mirrors the notebook's gate)."""
    return (
        f"T4 training-engine A/B | {preset} | batch {batch} x "
        f"seq {seq} | {token_budget:,} tokens/lane | "
        f"lanes: {', '.join(lanes)}"
    )


def _print_sidecar(path: str) -> None:
    """Pretty-print a run's checkpoint sidecar ('--sidecar' deliverable).

    Accepts a lane out-dir (reads ``RUN_METADATA_FILENAME``), the sidecar JSON
    file itself, a ``metrics.json`` file, or a ``step-*.pt`` checkpoint (the
    embedded run-metadata block is printed).
    """
    if os.path.isdir(path):
        path = os.path.join(path, RUN_METADATA_FILENAME)
    if not os.path.isfile(path):
        raise SystemExit(f"--sidecar: no such file or dir: {path}")
    if path.endswith(".pt"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        payload = ckpt.get("run_metadata") or ckpt.get("metadata") or ckpt
    else:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/bench_train.py",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--packed-dir", default=None, metavar="DIR",
                   help="packed-token corpus dir (manifest.json + *.npy shards) "
                        "produced by scripts/prepare_corpus.py; required to "
                        "run the benchmark (omit with --sidecar to only print "
                        "a saved sidecar)")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="benchmark output dir (lane run dirs + bench.json + "
                        "bench.md). Default: results/bench-train-<timestamp>")
    p.add_argument("--preset", default="tiny_100m",
                   help="canonical preset to train (default tiny_100m — the "
                        "T4 milestone config; use tiny for CPU smoke)")
    p.add_argument("--batch", type=int, default=16, help="batch size (default 16)")
    p.add_argument("--seq", type=int, default=None,
                   help="sequence length (default: the manifest's seq_len)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"steps per lane (default {DEFAULT_STEPS}); the token "
                        "budget is steps x batch x (seq-1), unless "
                        "--token-budget is given")
    p.add_argument("--token-budget", type=int, default=None, metavar="N",
                   help="exact token budget per lane (overrides --steps)")
    p.add_argument("--seed", type=int, default=0, help="seed for all lanes")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lanes", default=",".join(DEFAULT_LANES),
                   help=f"comma-separated lanes out of {list(LANES)} "
                        "(default: all three)")
    p.add_argument("--prefetch", type=int, default=2,
                   help="prefetch depth for the fp32/fp16 lanes (old lane is "
                        "always 0)")
    p.add_argument("--no-pin-memory", action="store_true",
                   help="disable pinned batches for the fp32/fp16 lanes "
                        "(old lane never pins)")
    p.add_argument("--log-every-steps", type=int, default=10,
                   help="trainer log cadence (default 10)")
    p.add_argument("--tokenizer-json", default=None, metavar="PATH",
                   help="tokenizer.json to verify against the manifest (the "
                        "trainer's own rule applies on the packed path)")
    p.add_argument("--sidecar", default=None, metavar="PATH",
                   help="after the benchmark, pretty-print a run's checkpoint "
                        "sidecar (a lane out-dir, train_run_metadata.json, "
                        "metrics.json, or a step-*.pt checkpoint) and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU plumbing check: force device cpu, 2 steps per "
                        "lane, log every step — validates the corpus + three "
                        "lanes + table without a GPU (numbers are CPU-only)")
    p.add_argument("--device", default=None,
                   help="compute device (default: auto — cuda if available)")
    p.add_argument("--rec-budget", type=int, default=DEFAULT_REC_BUDGET,
                   metavar="N",
                   help=f"token budget in the recommended milestone command "
                        f"(default {DEFAULT_REC_BUDGET:,})")
    p.add_argument("--rec-warmup", type=int, default=DEFAULT_REC_WARMUP,
                   metavar="N",
                   help=f"warmup tokens in the recommended command "
                        f"(default {DEFAULT_REC_WARMUP:,})")
    args = p.parse_args(argv)

    # Sidecar-only mode: no benchmark, just print a saved run's metadata.
    if args.sidecar and not args.packed_dir:
        _print_sidecar(args.sidecar)
        return 0
    if not args.packed_dir:
        p.error("--packed-dir is required to run the benchmark (use "
                "--sidecar without --packed-dir to only print a sidecar)")

    lanes = [l.strip() for l in args.lanes.split(",") if l.strip()]
    unknown = [l for l in lanes if l not in LANES]
    if unknown:
        p.error(f"unknown lane(s) {unknown} — choose from {list(LANES)}")
    if args.batch <= 0 or args.steps <= 0:
        p.error("--batch and --steps must be positive")
    if args.prefetch < 0:
        p.error("--prefetch must be >= 0")
    if args.token_budget is not None and args.token_budget <= 0:
        p.error("--token-budget must be positive")

    # ---- corpus: validate early, resolve defaults from the manifest -------
    manifest = load_packed_manifest(
        args.packed_dir, expected_vocab=None,
    )
    m_seq = int(manifest["seq_len"])
    m_dtype = manifest["dtype"]
    seq = args.seq if args.seq is not None else m_seq
    if seq != m_seq:
        p.error(
            f"--seq {seq} != manifest seq_len {m_seq}; packed rows are used "
            "verbatim"
        )
    counts = manifest["metadata"]["counts"]
    train_rows = int(counts.get("train_rows", 0))
    train_tokens = int(counts.get("train_tokens", 0))
    steps = args.steps
    token_budget = args.token_budget
    if token_budget is None:
        token_budget = steps * args.batch * (seq - 1)
    effective_steps = token_budget // (args.batch * (seq - 1))

    dry_run = args.dry_run
    if dry_run:
        device = torch.device("cpu")
        token_budget = min(token_budget, 2 * args.batch * (seq - 1))
        log_every = 1
        val_max_steps = 1
    else:
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print(
                "note: CUDA not available — lanes run on CPU. Use --dry-run "
                "for a small CPU plumbing check, or run on the T4 for real "
                "numbers.",
                file=sys.stderr,
            )
        log_every = args.log_every_steps
        val_max_steps = 1

    print(f"\n{_suggested_launch_preamble(
        args.preset, args.batch, seq, token_budget, lanes)}\n")

    out_dir = args.out_dir or os.path.join(
        "results", "bench-train-" + datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    # fp32/fp16 lane engine overrides from CLI (old lane stays fixed).
    engine_overrides = dict(prefetch=args.prefetch,
                            pin_memory=not args.no_pin_memory)
    base = SimpleNamespace(
        packed_dir=os.path.abspath(args.packed_dir), out_dir=out_dir,
        preset=args.preset, batch=args.batch, seq=seq,
        token_budget=token_budget, seed=args.seed, lr=args.lr,
        log_every_steps=log_every, tokenizer_json=args.tokenizer_json,
        val_max_steps=val_max_steps,
    )

    rows: List[dict] = []
    for name in lanes:
        engine = dict(LANES[name])
        if name != "old":
            engine.update(engine_overrides)
        row = _run_lane(name, engine, base, device)
        rows.append(row)
        # keep each lane's run metadata + metrics visible for the owner
        print(f"  lane {name}: run-wide {_fmt(row['tok_s_run_wide'])} tok/s | "
              f"steady {_fmt(row['tok_s_steady'])} tok/s | wall "
              f"{_fmt(row['wall_s'])} s | torch vram peak "
              f"{_fmt(row['vram_peak_torch_mb'])} MiB | nvsmi peak "
              f"{_fmt(row['vram_peak_nvsmi_mb'])} MiB | ckpt save "
              f"{_fmt(row['ckpt_save_total_s'])} s | ckpt load "
              f"{_fmt(row['ckpt_load_s'])} s")

    # ---- winner + recommended command -------------------------------
    scored = [r for r in rows if r["tok_s_run_wide"] is not None]
    winner_row = max(scored, key=lambda r: r["tok_s_run_wide"]) if scored \
        else rows[0]
    recommended = _recommended_command(
        dict(packed_dir=base.packed_dir, preset=args.preset,
             batch=args.batch, seq=seq, seed=args.seed, lr=args.lr),
        winner_row, args.rec_budget, args.rec_warmup,
    )

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "packed_dir": base.packed_dir,
        "preset": args.preset,
        "batch": args.batch,
        "seq": seq,
        "dtype": m_dtype,
        "train_rows": train_rows,
        "train_tokens": train_tokens,
        "steps": effective_steps,
        "token_budget": token_budget,
        "seed": args.seed,
        "lr": args.lr,
        "val_max_steps": val_max_steps,
        "lanes": lanes,
        "dry_run": dry_run,
        "out_dir": out_dir,
    }
    md = _render_markdown(env, rows, winner_row["lane"], recommended)
    bench = {
        "env": env,
        "rows": rows,
        "winner": winner_row["lane"],
        "recommended_command": recommended,
    }
    with open(os.path.join(out_dir, "bench.json"), "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(out_dir, "bench.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    print(md)
    print(f"benchmark written to {out_dir}/ (bench.json + bench.md)")
    if args.sidecar:
        print(f"\n===== sidecar: {args.sidecar} =====")
        _print_sidecar(args.sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())