"""Parameter / memory / FLOP estimation for a :class:`ModelConfig`.

These are *analytic estimates* from the config, so they work even for configs
too large to instantiate (e.g. 400B). They are intentionally simple and should
be treated as engineering estimates, not exact measurements.

Conventions
-----------
* ``total_params`` — every weight in the model.
* ``active_params`` — the weights actually executed per token (sparse MoE FFN
  is the only sparsity; attention/norms/embeddings are always active).
* ``weights_mem_bf16`` — ``total_params * 2`` bytes (BF16).
* ``flops_per_token`` — multiply-adds (2 FLOPs per activation) for one forward
  pass of one token.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Optional

from model.config import ModelConfig
from model.utils import human_bytes, human_count
from promethean import loader


@dataclass
class ComputeReport:
    config: ModelConfig
    total_params: int
    active_params: int
    weights_mem_bytes: float
    flops_per_token: float

    @property
    def weights_mem_bf16(self) -> float:
        return self.total_params * 2.0


def _attention_params(cfg: ModelConfig) -> int:
    qd = cfg.num_attention_heads * cfg.head_dim
    kd = cfg.num_kv_heads * cfg.head_dim
    q = cfg.hidden_size * qd
    k = cfg.hidden_size * kd
    v = cfg.hidden_size * kd
    o = qd * cfg.hidden_size
    return q + k + v + o


def _ffn_total_params(cfg: ModelConfig) -> int:
    if cfg.ffn_type == "dense":
        return 3 * cfg.hidden_size * cfg.intermediate_size
    per_expert = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    return per_expert * (cfg.num_experts + cfg.num_shared_experts)


def _ffn_active_params(cfg: ModelConfig) -> int:
    if cfg.ffn_type == "dense":
        return 3 * cfg.hidden_size * cfg.intermediate_size
    per_expert = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    return per_expert * (cfg.num_experts_per_tok + cfg.num_shared_experts)


def estimate(cfg: ModelConfig) -> ComputeReport:
    """Return a full compute/parameter report for ``cfg`` (derived first)."""
    cfg = cfg.derive()
    emb = cfg.vocab_size * cfg.hidden_size
    lm_head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * cfg.hidden_size

    attn = _attention_params(cfg)
    norms_per_layer = 2 * cfg.hidden_size  # 2 RMSNorm gains per layer
    ffn_total = _ffn_total_params(cfg)
    ffn_active = _ffn_active_params(cfg)

    per_layer_total = attn + norms_per_layer + ffn_total
    per_layer_active = attn + norms_per_layer + ffn_active

    total = emb + lm_head + cfg.num_layers * per_layer_total + cfg.hidden_size
    active = emb + lm_head + cfg.num_layers * per_layer_active + cfg.hidden_size
    return ComputeReport(
        config=cfg,
        total_params=total,
        active_params=active,
        weights_mem_bytes=float(total * 2),
        flops_per_token=_flops_per_token(cfg, attn),
    )


def _flops_per_token(cfg: ModelConfig, attn: int) -> float:
    """Rough forward-pass multiply-adds for a single token."""
    qd = cfg.num_attention_heads * cfg.head_dim
    kd = cfg.num_kv_heads * cfg.head_dim
    h = cfg.hidden_size

    # qkv projections + o projection (2 FLOPs per weight used per token).
    proj = (
        2 * (h * qd)          # q
        + 2 * (h * kd)        # k
        + 2 * (h * kd)        # v
        + 2 * (qd * h)        # o
    )
    # Score & weighted-sum over heads (per token, amortised over context).
    attn_compute = 2 * cfg.num_attention_heads * cfg.head_dim * 2

    if cfg.ffn_type == "dense":
        ffn = 2 * 3 * h * cfg.intermediate_size
    else:
        active_experts = cfg.num_experts_per_tok + cfg.num_shared_experts
        ffn = 2 * 3 * h * cfg.moe_intermediate_size * active_experts

    lm = 0 if cfg.tie_word_embeddings else 2 * h * cfg.vocab_size
    per_layer = proj + attn_compute + ffn
    return cfg.num_layers * per_layer + lm


# ------------------------------------------------------------------------------
# Reporting / CLI
# ------------------------------------------------------------------------------

def format_report(report: ComputeReport) -> str:
    cfg = report.config
    ratio = (
        (report.active_params / report.total_params * 100)
        if report.total_params
        else 0.0
    )
    lines = [
        f"{'=' * 66}",
        f"  {cfg.ffn_type.upper():<12} hidden={cfg.hidden_size} layers={cfg.num_layers} "
        f"heads={cfg.num_attention_heads} kv={cfg.num_kv_heads} vocab={cfg.vocab_size}",
        f"{'=' * 66}",
        f"  total params        : {report.total_params:>16,} "
        f"({human_count(report.total_params)})",
        f"  active / token      : {report.active_params:>16,} "
        f"({human_count(report.active_params)})",
        f"  sparsity (active %) : {ratio:>15.2f} %",
        f"  weights mem (BF16)  : {human_bytes(report.weights_mem_bf16):>14}",
        f"  FLOPs / token (fwd) : {human_count(report.flops_per_token):>14}",
        f"  attention scheme    : {cfg.attention_type}",
    ]
    if cfg.attention_type == "hybrid":
        lines.append(
            f"    (sliding window={cfg.sliding_window_size}, "
            f"full every {cfg.periodic_full_every} layers)"
        )
    if cfg.ffn_type == "moe":
        lines.append(
            f"    (experts={cfg.num_experts} routed, top-k={cfg.num_experts_per_tok}, "
            f"shared={cfg.num_shared_experts}, per-expert={cfg.moe_intermediate_size})"
        )
    if cfg.max_seq_len >= 65_536:
        lines.append(f"  context window      : {cfg.max_seq_len:,} tokens")
    lines.append("")
    return "\n".join(lines)


def summarize_all(presets: Optional[Dict[str, callable]] = None) -> str:
    """Build a compact table over every preset config."""
    from configs.presets import ALL_PRESETS

    presets = presets or ALL_PRESETS
    rows = []
    for name, builder in presets.items():
        rep = estimate(builder())
        rows.append(
            f"  {name:<6} {human_count(rep.total_params):>6} total | "
            f"{human_count(rep.active_params):>6} active | "
            f"{human_bytes(rep.weights_mem_bf16):>8} BF16 | "
            f"{human_count(rep.flops_per_token):>6} FLOPs/tok"
        )
    return "\n".join(rows)


def main(argv: Optional[list] = None) -> None:
    """CLI: print compute estimates for one or all preset configs."""
    loader.show("talos")
    parser = argparse.ArgumentParser(
        prog="talos-configs",
        description="Print parameter / memory / FLOP estimates for Talos configs.",
    )
    parser.add_argument(
        "name",
        nargs="*",
        default=None,
        help="Preset names (tiny small medium large 100b 400b). Default: all.",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print a compact one-line table.",
    )
    args = parser.parse_args(argv)

    from configs.presets import ALL_PRESETS

    chosen = args.name or list(ALL_PRESETS.keys())
    if args.summary:
        print(summarize_all({n: ALL_PRESETS[n] for n in chosen}))
        return
    for name in chosen:
        if name not in ALL_PRESETS:
            print(f"Unknown preset {name!r}; choose from {sorted(ALL_PRESETS)}",
                  file=sys.stderr)
            sys.exit(2)
        print(format_report(estimate(ALL_PRESETS[name]())))
    if len(chosen) > 1:
        print("\nSummary:")
        print(summarize_all({n: ALL_PRESETS[n] for n in chosen}))


if __name__ == "__main__":
    main()
