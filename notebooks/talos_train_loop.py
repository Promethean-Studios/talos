"""The Talos/Argo training loop — one code path, tiny dims or 1B, CPU or GPU.

The notebook used to own this loop inline in a code cell, which meant the only
thing that could ever exercise it was a real GPU run. This module extracts the
loop unchanged in behaviour (budget cap, partial-final-step scaling, absolute
LR schedule, resumable dataset position) and adds the two things the owner's
audit asked for:

* **Non-finite detection with skipped updates.** A non-finite loss is caught
  *before* ``backward()`` and a non-finite gradient *before* the optimizer step;
  the update is then skipped (gradients dropped, no optimizer step, no LR
  advance, no step counter advance) and counted in ``skipped_updates``. The
  tokens of the offending micro-batch *are* counted, because the forward pass
  really did consume them — that matches the loop's existing partial-step
  accounting and guarantees forward progress (no silent spin). A burst of
  :data:`~talos_stages.MAX_CONSECUTIVE_SKIPPED` consecutive skips aborts loudly
  with a diagnostic, after writing a checkpoint.
* **Epoch accounting.** ``TrainResult.epochs_completed`` reports how much of the
  requested epochs actually ran, so a Stage-B run that hits its budget cap
  reports "1.98 of 2 epochs" instead of claiming 2.

Everything the loop touches is injected (model, optimizer, scaler, forward-loss
callable, row stream, checkpoint callback, evaluator, log callback), so
``tests/test_notebook_loop.py`` and ``notebooks/talos_smoke.py`` run the exact
same loop on the CPU with the canonical tiny preset — not a parallel toy
implementation.

This module imports no ``torch`` at module scope: tensor work happens inside
:mod:`talos_torch_utils` and inside the callables the caller passes in.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from talos_stages import (MAX_CONSECUTIVE_SKIPPED, NonFiniteAbort, SkipTracker,
                          TokenAccountant, lr_at, plan_step)
from talos_torch_utils import (clip_gradients, grad_norm, grads_are_finite,
                               loss_is_finite, rows_to_tensor)


@dataclass
class TrainLoopConfig:
    """Everything the loop needs to know about the run's shape and cadence."""

    micro_batch: int
    grad_accum: int
    seq_len: int
    total_steps: int
    warmup_steps: int
    peak_lr: float
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 100
    checkpoint_every: int = 50
    checkpoint_every_min: float = 15.0     # wall-clock minutes; <= 0 disables
    max_consecutive_skips: int = MAX_CONSECUTIVE_SKIPPED
    mask_prompt_tokens: bool = False
    #: Tokens in one full pass over the training data, when known: used only to
    #: report ``epochs_completed`` as a fraction (Stage B).
    tokens_per_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("micro_batch", "grad_accum", "seq_len", "total_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.peak_lr <= 0:
            raise ValueError("peak_lr must be positive")
        if not 0.0 <= float(self.min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        for name in ("log_every", "eval_every", "checkpoint_every"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_consecutive_skips < 1:
            raise ValueError("max_consecutive_skips must be >= 1")


@dataclass
class StepInfo:
    """One log interval's numbers — what the notebook renders and publishes."""

    step: int
    epoch: int
    tokens_consumed: int
    supervised_tokens: int
    elapsed_s: float
    train_loss: float
    val_loss: Optional[float]
    last_grad_norm: Optional[float]
    checkpoint: str
    skipped_updates: int
    stage: str
    rows_in_epoch: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "tokens_consumed": self.tokens_consumed,
            "supervised_tokens": self.supervised_tokens,
            "elapsed_s": self.elapsed_s,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "last_grad_norm": self.last_grad_norm,
            "checkpoint": self.checkpoint,
            "skipped_updates": self.skipped_updates,
            "stage": self.stage,
            "rows_in_epoch": self.rows_in_epoch,
        }


@dataclass
class TrainResult:
    """What the run did — including the parts that did not go to plan."""

    steps: int
    tokens_consumed: int
    supervised_tokens: int
    skipped_updates: int
    consecutive_skips: int
    epoch: int
    epochs_completed: float
    reached_budget: bool
    exhausted: bool
    interrupted: bool
    train_loss: float
    val_loss: Optional[float]
    last_grad_norm: Optional[float]
    last_checkpoint: str
    elapsed_s: float
    rows_seen: int
    last_skip_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": int(self.steps),
            "tokens_consumed": int(self.tokens_consumed),
            "supervised_tokens": int(self.supervised_tokens),
            "skipped_updates": int(self.skipped_updates),
            "consecutive_skips": int(self.consecutive_skips),
            "epoch": int(self.epoch),
            "epochs_completed": round(float(self.epochs_completed), 4),
            "reached_budget": bool(self.reached_budget),
            "exhausted": bool(self.exhausted),
            "interrupted": bool(self.interrupted),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "last_grad_norm": self.last_grad_norm,
            "last_checkpoint": self.last_checkpoint,
            "elapsed_s": round(float(self.elapsed_s), 3),
            "rows_seen": int(self.rows_seen),
            "last_skip_reason": self.last_skip_reason,
        }


def _count_supervised(ids: Any, mask: Any) -> int:
    """Supervised positions in one micro-batch (``seq_len - 1`` per row by default).

    Stage A supervises every next-token position; Stage B supervises only the
    assistant turns, so its *loss* sees far fewer tokens than its *budget*
    counts. Reporting both is what makes Stage B's accounting comparable with
    Stage A's instead of silently conflating "tokens trained on" with "tokens
    the loss saw".
    """
    if mask is None:
        return int(ids[:, 1:].numel())
    return int(mask[:, 1:].sum().item())


def train_stage(
    *,
    model: Any,
    optimizer: Any,
    scaler: Optional[Any],
    accountant: TokenAccountant,
    stream: Any,
    forward_loss: Callable[[Any, Any], Any],
    config: TrainLoopConfig,
    stage_key: str = "A",
    device: Any = None,
    on_checkpoint: Optional[Callable[[StepInfo], str]] = None,
    evaluate: Optional[Callable[[], Optional[float]]] = None,
    on_log: Optional[Callable[[StepInfo], None]] = None,
    on_skip: Optional[Callable[[str, StepInfo], None]] = None,
    skip_tracker: Optional[SkipTracker] = None,
    clock: Callable[[], float] = time.time,
    say: Callable[[str], None] = print,
) -> TrainResult:
    """Run one stage until the budget is spent or the data is exhausted.

    Args:
        model: the module being trained (already on ``device``).
        optimizer: built over ``model.parameters()``.
        scaler: a ``GradScaler`` for fp16, or ``None`` (bf16/fp32).
        accountant: the budget counter; pre-load its ``consumed``/``steps`` from
            a checkpoint to resume.
        stream: a ``RowStream``; its ``position`` is read for the epoch count.
        forward_loss: ``(ids, mask) -> loss tensor`` — the stage's objective.
        config: batch shape, schedule and cadence.
        on_checkpoint: ``(StepInfo) -> filename``; writes the checkpoint. Called
            on the cadence **and once at the end** (including on interrupt/abort).
        evaluate: ``() -> Optional[float]`` validation loss, called on the eval
            cadence and once at the end.
        on_log: ``(StepInfo) -> None``; builds the metrics row / display.
        on_skip: ``(reason, StepInfo) -> None``; logs a skipped update.
        skip_tracker: injectable counter (tests inspect it).
        clock: monotonic seconds (injectable for tests).
        say: line printer (``print`` in the notebook, ``list.append`` in tests).

    Returns:
        :class:`TrainResult`.

    Raises:
        NonFiniteAbort: after ``max_consecutive_skips`` skipped updates in a row
            (a checkpoint is written first).
    """
    rows: Iterator[Any] = iter(stream)
    skips = skip_tracker or SkipTracker(max_consecutive=config.max_consecutive_skips)
    start = clock()
    last_ckpt_at = start

    train_loss: float = float("nan")
    val_loss: Optional[float] = None
    last_grad_norm: Optional[float] = None
    last_checkpoint = ""
    supervised_tokens = 0
    rows_seen = 0
    reached_budget = False
    exhausted = False
    interrupted = False

    def _info() -> StepInfo:
        return StepInfo(
            step=int(accountant.steps),
            epoch=int(getattr(stream.position, "epoch", 1)),
            tokens_consumed=int(accountant.consumed),
            supervised_tokens=int(supervised_tokens),
            elapsed_s=float(clock() - start),
            train_loss=float(train_loss),
            val_loss=None if val_loss is None else float(val_loss),
            last_grad_norm=None if last_grad_norm is None else float(last_grad_norm),
            checkpoint=last_checkpoint,
            skipped_updates=int(skips.total),
            stage=str(stage_key),
            rows_in_epoch=int(getattr(stream.position, "rows_in_epoch", 0)),
        )

    model.train()
    try:
        while True:
            # The budget cap: `plan_step` never returns more micro-batches than
            # the remaining budget can pay for, and 0 means stop. A full
            # optimizer step therefore cannot overshoot the stage's budget.
            plan = plan_step(accountant.remaining, micro_batch=config.micro_batch,
                             grad_accum=config.grad_accum, seq_len=config.seq_len)
            if plan.should_stop:
                reached_budget = True
                break

            optimizer.zero_grad(set_to_none=True)
            step_loss_sum, micro_done = 0.0, 0
            step_supervised = 0
            skip_reason = ""

            for _ in range(plan.micro_batches):
                batch_rows: List[Any] = []
                for _ in range(config.micro_batch):
                    try:
                        batch_rows.append(next(rows))
                    except StopIteration:
                        exhausted = True
                        break
                if not batch_rows:
                    break
                ids, mask = rows_to_tensor(batch_rows, device=device,
                                           with_mask=config.mask_prompt_tokens)
                rows_seen += len(batch_rows)
                loss = forward_loss(ids, mask)
                # These tokens were consumed by the forward pass whether or not
                # the update is applied, so they are counted either way (and the
                # loop is guaranteed to make progress even if every update is
                # skipped).
                # steps=0: the optimizer step is counted below, and only when
                # it is actually applied (a skipped update must not advance the
                # step counter or the LR schedule).
                accountant.add(int(ids.numel()), steps=0)
                step_supervised += _count_supervised(ids, mask)

                if not loss_is_finite(loss):
                    skip_reason = (
                        f"non-finite loss ({float(loss.detach())}) in micro-batch "
                        f"{micro_done + 1} of step {accountant.steps}"
                    )
                    break

                # Scale by the number of micro-batches the step was planned for,
                # so a partial final step is not silently down-weighted.
                scaled = loss / plan.micro_batches
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                step_loss_sum += float(loss.detach())
                micro_done += 1

            supervised_tokens += step_supervised

            if skip_reason:
                # Skipped update: drop the partial gradients, no optimizer step,
                # no LR advance, no step counter advance.
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and micro_done > 0:
                    scaler.update()
                last_grad_norm = None
                skips.note_skip(skip_reason)          # may raise NonFiniteAbort
                say(f"SKIPPED update at step {accountant.steps} (reason: {skip_reason}); "
                    f"{skips.total} skipped, {skips.consecutive} in a row")
                if on_skip is not None:
                    on_skip(skip_reason, _info())
                continue

            if micro_done == 0:
                exhausted = True
                break

            if scaler is not None:
                scaler.unscale_(optimizer)
            pre_clip = grad_norm(model.parameters())
            if not math.isfinite(pre_clip) or not grads_are_finite(model.parameters()):
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                last_grad_norm = None
                skips.note_skip(
                    f"non-finite gradient (pre-clip norm {pre_clip}) at step "
                    f"{accountant.steps}"
                )
                say(f"SKIPPED update at step {accountant.steps}: non-finite gradient "
                    f"(pre-clip norm {pre_clip}); {skips.total} skipped, "
                    f"{skips.consecutive} in a row")
                if on_skip is not None:
                    on_skip("non-finite gradient", _info())
                continue

            clip_gradients(model.parameters(), config.grad_clip)
            current_lr = lr_at(accountant.steps, peak_lr=config.peak_lr,
                               total_steps=config.total_steps,
                               warmup=config.warmup_steps,
                               min_lr_ratio=config.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    # GradScaler found Inf/NaN in the gradients and skipped the
                    # step itself: treat it exactly like our own skip.
                    optimizer.zero_grad(set_to_none=True)
                    last_grad_norm = None
                    skips.note_skip("GradScaler skipped the step (non-finite gradients)")
                    say(f"SKIPPED update at step {accountant.steps}: GradScaler found "
                        f"non-finite gradients; {skips.total} skipped")
                    if on_skip is not None:
                        on_skip("GradScaler skip", _info())
                    continue
            else:
                optimizer.step()

            accountant.steps += 1
            skips.note_update()
            train_loss = step_loss_sum / micro_done
            last_grad_norm = pre_clip

            if accountant.steps % config.eval_every == 0 and evaluate is not None:
                val_loss = evaluate()
            if on_checkpoint is not None and (
                accountant.steps % config.checkpoint_every == 0
                or (config.checkpoint_every_min > 0
                    and (clock() - last_ckpt_at) >= config.checkpoint_every_min * 60.0)
            ):
                last_checkpoint = on_checkpoint(_info())
                last_ckpt_at = clock()
            if on_log is not None and accountant.steps % config.log_every == 0:
                on_log(_info())

            if exhausted:
                break
    except KeyboardInterrupt:
        interrupted = True
        say("\ninterrupted — writing a checkpoint before stopping")
    finally:
        if evaluate is not None and val_loss is None and accountant.steps > 0:
            try:
                val_loss = evaluate()
            except Exception as exc:  # noqa: BLE001 - a failed final eval must not hide the run
                say(f"final evaluation failed ({type(exc).__name__}: {exc})")
        if on_checkpoint is not None:
            last_checkpoint = on_checkpoint(_info())

    elapsed = clock() - start
    epochs_completed = _epochs_completed(stream, config)
    result = TrainResult(
        steps=int(accountant.steps),
        tokens_consumed=int(accountant.consumed),
        supervised_tokens=int(supervised_tokens),
        skipped_updates=int(skips.total),
        consecutive_skips=int(skips.consecutive),
        epoch=int(getattr(stream.position, "epoch", 1)),
        epochs_completed=epochs_completed,
        reached_budget=bool(reached_budget),
        exhausted=bool(exhausted),
        interrupted=bool(interrupted),
        train_loss=float(train_loss),
        val_loss=None if val_loss is None else float(val_loss),
        last_grad_norm=None if last_grad_norm is None else float(last_grad_norm),
        last_checkpoint=last_checkpoint,
        elapsed_s=float(elapsed),
        rows_seen=int(rows_seen),
        last_skip_reason=str(skips.last_reason),
    )
    if on_log is not None and math.isfinite(result.train_loss):
        on_log(_info())
    return result


def _epochs_completed(stream: Any, config: TrainLoopConfig) -> float:
    """Epochs finished, plus the fraction of the current epoch already streamed."""
    position = getattr(stream, "position", None)
    if position is None:
        return 0.0
    finished = max(0, int(getattr(position, "epoch", 1)) - 1)
    fraction = 0.0
    if config.tokens_per_epoch:
        streamed = int(getattr(position, "rows_in_epoch", 0)) * config.seq_len
        fraction = min(1.0, streamed / float(config.tokens_per_epoch))
    return float(finished) + fraction


__all__ = ["NonFiniteAbort", "StepInfo", "TrainLoopConfig", "TrainResult", "train_stage"]
