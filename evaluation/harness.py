"""Reproducible checkpoint evaluation harness for Talos.

Given a ``talos-training-checkpoint-v1`` artifact (written by
:func:`scripts.train_oasst1.save_checkpoint`) and its sidecar
``tokenizer.json``, this module recomputes from a JSONL split:

* validation loss (and train loss when a train split is supplied),
* perplexity,
* next-token accuracy (argmax over logits vs. the actual next token),
* parameter count (must equal the checkpoint's recorded ``n_params`` —
  a mismatch fails loudly),
* tokens processed,
* evaluation throughput (tokens/sec),
* peak RSS (``ru_maxrss``, a *process* metric — labelled honestly as such).

Loss convention
    All losses are mean cross-entropy over predicted next tokens computed with
    ``torch.nn.CrossEntropyLoss``, i.e. mean ``-log p`` in **natural log**
    (nats). Perplexity is therefore ``exp(loss)`` in the same natural-log base.

Reproducibility
    Evaluation is a pure function of (checkpoint, tokenizer, JSONL split,
    ``seq_len``/``batch_size``, seed): token batches come from the existing
    :class:`~data.tokenized.StreamingTokenizedDataset` in deterministic file
    order (pack mode, no shuffle, no RNG anywhere in the pipeline) and the
    model runs in ``eval()`` mode under ``torch.no_grad()``. Running the
    evaluator twice on the same inputs yields **bit-identical metrics** (the
    test suite asserts this exact property). Batch shape defaults mirror the
    training loop (``seq_len=64, batch_size=4, drop_last=True``) so the
    recomputed validation loss matches the loss recorded in the checkpoint.

CLI: ``python -m scripts.eval_checkpoint --checkpoint <dir-or-file> ...``
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch

# Repo-root packages are importable when this module is run from the repo root
# (conftest.py does the same for pytest); add the root defensively so the
# module also works when imported from another CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.tokenized import StreamingTokenizedDataset  # noqa: E402
from model import ModelConfig, TalosGPT  # noqa: E402
from model.utils import set_seed  # noqa: E402
from scripts.train_oasst1 import (  # noqa: E402
    CHECKPOINT_FORMAT,
    EXPECTED_TINY_PARAMS,
    EXPECTED_TINY_VOCAB,
    load_checkpoint,
)
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402

EVAL_METRICS_FORMAT = "talos-oasst1-eval-metrics-v1"


@dataclass
class SplitEval:
    """Per-split results, all accumulated over the *same* eval batches."""

    loss: float
    perplexity: float
    accuracy: float
    tokens: int
    batches: int
    wall_s: float
    throughput_tok_per_s: float


@dataclass
class EvalResult:
    """Full evaluation report (also serialized as the metrics JSON file)."""

    format: str
    checkpoint_path: str
    checkpoint_step: int
    checkpoint_format: str
    data: str
    train_data: Optional[str]
    params: int
    vocab_size: int
    tokenizer_vocab_size: int
    tokenizer_merges: int
    val_loss: float
    val_perplexity: float
    val_accuracy: float
    train_loss: Optional[float]
    checkpoint_val_loss: Optional[float]
    tokens_processed: int
    eval_wall_s: float
    throughput_tok_per_s: float
    peak_rss_mb: float
    seed: int
    seq_len: int
    batch_size: int
    drop_last: bool
    device: str
    metrics_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def find_checkpoint(path: str) -> str:
    """Resolve ``--checkpoint``: a file, or a directory's newest ``step-*.pt``.

    Directories are scanned for ``step-<N>.pt`` (the naming convention of
    :func:`scripts.train_oasst1.save_checkpoint`); the highest ``N`` wins.
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        steps = []
        for name in os.listdir(path):
            if name.startswith("step-") and name.endswith(".pt"):
                try:
                    steps.append((int(name[len("step-"): -len(".pt")]), name))
                except ValueError:
                    continue
        if not steps:
            raise FileNotFoundError(
                f"no step-<N>.pt checkpoint found in directory: {path}"
            )
        newest = max(steps, key=lambda pair: pair[0])
        return os.path.join(path, newest[1])
    raise FileNotFoundError(f"checkpoint not found: {path}")


def default_val_split(checkpoint: dict) -> Optional[str]:
    """The checkpoint's own val split, when it sits next to the run's tokenizer.

    ``save_checkpoint`` records an absolute ``tokenizer_path`` pointing at
    ``<out_dir>/tokenizer.json``; the deterministic id split lives at
    ``<out_dir>/data/val.jsonl``. Returns that path when it exists, else
    ``None`` (caller then requires ``--data``).
    """
    tok_path = checkpoint.get("tokenizer_path")
    if not tok_path:
        return None
    candidate = os.path.join(os.path.dirname(tok_path), "data", "val.jsonl")
    return candidate if os.path.isfile(candidate) else None


def load_checkpoint_artifacts(
    checkpoint_path: str,
) -> tuple[dict, TalosGPT, ByteLevelBPETokenizer]:
    """Load and validate a checkpoint + its sidecar tokenizer.

    Fails loudly (``ValueError``) on: unknown checkpoint format, a parameter
    count that disagrees with the checkpoint's recorded ``n_params``, a
    tokenizer whose vocab overflows the model's embedding rows, or a missing
    tokenizer file. The model is rebuilt from the checkpoint's own
    ``model_config`` (no architecture guessing) and returned in ``eval()``
    mode.
    """
    ckpt = load_checkpoint(checkpoint_path)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {ckpt.get('format')!r} in "
            f"{checkpoint_path}: expected {CHECKPOINT_FORMAT!r}"
        )
    cfg = ModelConfig(**ckpt["model_config"]).derive()
    model = TalosGPT(cfg)
    model.eval()
    # Validate the recorded config/params/vocab BEFORE touching the weights:
    # a tampered config must fail with the clean guard error, not a shape
    # mismatch from load_state_dict.
    recorded = int(ckpt["n_params"])
    actual = model.num_parameters()
    if actual != recorded:
        raise ValueError(
            f"checkpoint n_params mismatch: artifact records {recorded:,} "
            f"but rebuilding the checkpoint's model_config yields {actual:,} "
            f"params in {checkpoint_path} — the artifact is corrupt or the "
            f"tiny preset drifted"
        )
    if actual != EXPECTED_TINY_PARAMS:
        raise ValueError(
            f"checkpoint is not the canonical tiny prototype: {actual:,} "
            f"params, expected exactly {EXPECTED_TINY_PARAMS:,} "
            f"(vocab {EXPECTED_TINY_VOCAB})"
        )
    recorded_vocab = ckpt.get("vocab_size")
    if recorded_vocab is not None and int(recorded_vocab) != cfg.vocab_size:
        raise ValueError(
            f"checkpoint vocab_size mismatch: artifact records {recorded_vocab} "
            f"but rebuilding the checkpoint's model_config yields "
            f"{cfg.vocab_size} vocab rows in {checkpoint_path} — the recorded "
            f"config and the recorded vocab_size disagree"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    tok_path = ckpt.get("tokenizer_path")
    if not tok_path or not os.path.isfile(tok_path):
        raise FileNotFoundError(
            f"checkpoint's tokenizer not found ({tok_path!r}) — expected the "
            f"sidecar tokenizer.json next to {checkpoint_path}"
        )
    tokenizer = ByteLevelBPETokenizer.from_file(tok_path)
    if tokenizer.vocab_size > cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} exceeds model vocab "
            f"{cfg.vocab_size} — tokenizer/model mismatch (ids would be "
            f"unembeddable)"
        )
    return ckpt, model, tokenizer


def _evaluate_split_data(
    model: TalosGPT,
    tokenizer: ByteLevelBPETokenizer,
    jsonl_path: str,
    *,
    seq_len: int,
    batch_size: int,
    drop_last: bool,
    device: torch.device,
    max_steps: Optional[int],
) -> SplitEval:
    """One deterministic, batched pass over ``jsonl_path``.

    Loss, perplexity and accuracy are accumulated over exactly the same
    batches/tokens: loss uses ``CrossEntropyLoss(reduction="sum")`` and
    accuracy counts ``argmax(logits) == target`` over ``x[:, :-1] ->
    x[:, 1:]``, matching the training objective. Batch order is the dataset's
    deterministic file order (no shuffle, no RNG).
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    dataset = StreamingTokenizedDataset(
        jsonl_path, tokenizer, seq_len=seq_len, batch_size=batch_size,
        mode="pack", eos=True, drop_last=drop_last,
    )
    vocab = model.config.vocab_size
    total_loss, correct, tokens, batches = 0.0, 0, 0, 0
    t0 = time.monotonic()
    with torch.no_grad():
        for batch in dataset:
            if max_steps is not None and batches >= max_steps:
                break
            x = batch.to(device).long()
            logits, _ = model(x[:, :-1])
            targets = x[:, 1:]
            n = targets.numel()
            total_loss += float(loss_fn(logits.reshape(-1, vocab), targets.reshape(-1)))
            correct += int((logits.argmax(dim=-1) == targets).sum())
            tokens += n
            batches += 1
    wall = time.monotonic() - t0
    if tokens == 0:
        raise ValueError(
            f"no eval tokens in {jsonl_path} at seq_len={seq_len} "
            f"batch_size={batch_size} drop_last={drop_last} — the split is "
            f"empty or too short for one full batch"
        )
    loss = total_loss / tokens
    return SplitEval(
        loss=loss,
        perplexity=float(torch.exp(torch.tensor(loss))),
        accuracy=correct / tokens,
        tokens=tokens,
        batches=batches,
        wall_s=wall,
        throughput_tok_per_s=tokens / wall if wall > 0 else float("nan"),
    )


def run_eval(
    checkpoint_path: str,
    data: Optional[str] = None,
    *,
    train_data: Optional[str] = None,
    seq_len: int = 64,
    batch_size: int = 4,
    drop_last: bool = True,
    max_steps: Optional[int] = None,
    seed: int = 0,
    device: Optional[str] = None,
    out_metrics: Optional[str] = None,
) -> EvalResult:
    """Run the full evaluation report for one checkpoint.

    Args:
        checkpoint_path: ``step-<N>.pt`` file, or a directory containing one
            (the newest is used).
        data: validation JSONL split. Defaults to the checkpoint's own val
            split (``<checkpoint_dir>/data/val.jsonl``) when it exists.
        train_data: optional train JSONL split — when supplied, train loss is
            also reported.
        seq_len / batch_size: eval batch shape. Defaults mirror the training
            loop (64/4) so recomputed loss matches the recorded ``val_loss``.
        drop_last: drop the final partial batch (default True — same as
            training). Pass False to also score the tail tokens.
        max_steps: cap the number of eval batches (CI / smoke runs).
        seed: fixed seed (batch order is deterministic regardless; the seed is
            set for full reproducibility of any downstream randomness).
        device: compute device (default: auto).
        out_metrics: where to write the metrics JSON (default:
            ``<checkpoint dir>/eval-metrics.json``).

    Returns:
        :class:`EvalResult` — also saved as a metrics file so runs are
        comparable.
    """
    ckpt_path = find_checkpoint(checkpoint_path)
    ckpt, model, tokenizer = load_checkpoint_artifacts(ckpt_path)
    if data is None:
        data = default_val_split(ckpt)
        if data is None:
            raise ValueError(
                "no --data given and the checkpoint's own val split "
                "(<tokenizer_dir>/data/val.jsonl) was not found — pass "
                "--data <val.jsonl> explicitly"
            )
    for label, p in (("val", data), ("train", train_data)):
        if p is not None and not os.path.isfile(p):
            raise FileNotFoundError(f"{label} JSONL not found: {p}")

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(seed)
    model = model.to(resolved_device)

    val = _evaluate_split_data(
        model, tokenizer, data, seq_len=seq_len, batch_size=batch_size,
        drop_last=drop_last, device=resolved_device, max_steps=max_steps,
    )
    train = None
    if train_data is not None:
        train = _evaluate_split_data(
            model, tokenizer, train_data, seq_len=seq_len, batch_size=batch_size,
            drop_last=drop_last, device=resolved_device, max_steps=max_steps,
        )

    if out_metrics is None:
        base = os.path.dirname(os.path.abspath(ckpt_path))
        out_metrics = os.path.join(base, "eval-metrics.json")
    out_metrics = os.path.abspath(out_metrics)
    result = EvalResult(
        format=EVAL_METRICS_FORMAT,
        checkpoint_path=os.path.abspath(ckpt_path),
        checkpoint_step=int(ckpt["step"]),
        checkpoint_format=str(ckpt["format"]),
        data=os.path.abspath(data),
        train_data=os.path.abspath(train_data) if train_data else None,
        params=model.num_parameters(),
        vocab_size=model.config.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_merges=tokenizer.merge_count,
        val_loss=val.loss,
        val_perplexity=val.perplexity,
        val_accuracy=val.accuracy,
        train_loss=train.loss if train is not None else None,
        checkpoint_val_loss=(
            None if ckpt.get("val_loss") is None else float(ckpt["val_loss"])
        ),
        tokens_processed=val.tokens + (train.tokens if train is not None else 0),
        eval_wall_s=round(val.wall_s + (train.wall_s if train is not None else 0), 3),
        throughput_tok_per_s=(
            (val.tokens + (train.tokens if train is not None else 0))
            / max(val.wall_s + (train.wall_s if train is not None else 0), 1e-9)
        ),
        peak_rss_mb=round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
        ),
        seed=seed,
        seq_len=seq_len,
        batch_size=batch_size,
        drop_last=drop_last,
        device=str(resolved_device),
        metrics_path=out_metrics,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_metrics)), exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
        fh.write("\n")
    return result