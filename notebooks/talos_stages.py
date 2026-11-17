"""Two-stage training plan for the Talos 1B notebook — pure logic, no torch.

The 1B model is trained **from its own random initialisation** (no pretrained
weights, no external foundation model). That makes the objective two distinct
stages, not one:

* **Stage A — language pretraining.** Next-token prediction on a general-purpose
  English web corpus (``allenai/c4``, config ``en``, the real ``train`` /
  ``validation`` splits). This is where the model learns language at all.
* **Stage B — conversational tuning.** Supervised fine-tuning on
  ``HuggingFaceH4/ultrachat_200k`` (``train_sft`` / ``test_sft``), user turns
  masked out of the loss, 1–3 epochs maximum.

Both stages are resumable and distinct: a checkpoint records the stage it was
written by (:data:`STAGE_METADATA_KEY` inside the checkpoint's ``run_metadata``)
and :func:`validate_stage_resume` refuses to resume a Stage-B checkpoint as
Stage A (or vice versa), because that silently trains the wrong objective on the
wrong data.

Everything in this module is deliberately free of ``torch``, ``datasets`` and
network access so the budget/schedule/accounting contract can be unit-tested on
CPU in CI (``tests/test_notebook_stages.py``). The notebook cell only *glues*
these functions to a PyTorch loop.

Budget rule (owner directive: "the budget cap logic must make it structurally
impossible for a full optimizer step to exceed the stage's token budget"):

* :func:`plan_step` is the **only** way the loop decides how many micro-batches
  a step may consume. It never returns more micro-batches than the remaining
  budget can pay for, and returns ``0`` when nothing is left — the loop then
  stops. :class:`TokenAccountant` re-asserts the invariant after every add, so a
  bookkeeping bug raises instead of silently overrunning the budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

#: The repo's checkpoint format string (``scripts/train_oasst1.py``). Repeated
#: here so this module stays importable without torch.
CHECKPOINT_FORMAT = "talos-training-checkpoint-v1"

#: Stage keys. Exactly two stages — "two stages, not a zoo" (lead directive).
STAGE_A = "A"
STAGE_B = "B"
STAGES = (STAGE_A, STAGE_B)

#: Key inside a checkpoint's ``run_metadata`` that records the stage.
STAGE_METADATA_KEY = "stage"

#: ``run_metadata`` key holding the dataset identity the stage trained on.
DATASET_METADATA_KEY = "dataset"

#: ``run_metadata`` key holding the resumable dataset position.
POSITION_METADATA_KEY = "dataset_position"

#: Human labels for the two stages.
STAGE_NAMES: Dict[str, str] = {
    STAGE_A: "language pretraining",
    STAGE_B: "conversational tuning",
}

#: Stage-scoped output directory names under the run root.
STAGE_DIRS: Dict[str, str] = {STAGE_A: "stage-a", STAGE_B: "stage-b"}


# ---------------------------------------------------------------------------
# Stage specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageSpec:
    """One stage's identity, data, budget and optimisation settings.

    A stage is fully described by this object: the dataset it streams, the
    token budget it must not exceed, its LR schedule shape and whether prompt
    tokens are masked out of the loss. The notebook builds one per run from
    :func:`stage_spec` and puts :meth:`run_metadata` into every checkpoint.
    """

    key: str
    name: str
    objective: str
    dataset_id: str
    dataset_config: Optional[str]
    train_split: str
    val_split: str
    seq_len: int
    token_budget: int
    max_epochs: Optional[int]
    peak_lr: float
    min_lr_ratio: float
    warmup_ratio: float
    grad_clip: float
    weight_decay: float
    mask_prompt_tokens: bool
    seed: int

    # -- identity -----------------------------------------------------------
    def dataset_identity(self) -> Dict[str, Any]:
        """The dataset's identity, recorded in every checkpoint.

        A resume whose dataset identity differs is rejected: resuming Stage A on
        a different corpus silently changes the objective mid-run.
        """
        return {
            "id": self.dataset_id,
            "config": self.dataset_config,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "seq_len": int(self.seq_len),
        }

    # -- schedule -----------------------------------------------------------
    def tokens_per_step(self, micro_batch: int, grad_accum: int) -> int:
        """Tokens consumed by one *full* optimizer step."""
        if micro_batch <= 0 or grad_accum <= 0:
            raise ValueError("micro_batch and grad_accum must be positive")
        return int(micro_batch) * int(grad_accum) * int(self.seq_len)

    def total_steps(self, micro_batch: int, grad_accum: int) -> int:
        """Optimizer steps needed to spend the budget (ceil — the last is partial)."""
        return steps_for_budget(self.token_budget, self.tokens_per_step(micro_batch, grad_accum))

    def warmup_steps(self, micro_batch: int, grad_accum: int, *, minimum: int = 20) -> int:
        """Warmup length: ``warmup_ratio`` of the schedule, never below ``minimum``."""
        total = self.total_steps(micro_batch, grad_accum)
        return warmup_steps(total, self.warmup_ratio, minimum=minimum)

    def run_metadata(
        self,
        *,
        micro_batch: int,
        grad_accum: int,
        precision: str,
        gpu: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The checkpoint ``run_metadata`` sidecar for this stage.

        Carries the stage, the dataset identity and the exact optimisation
        settings, so a checkpoint is self-describing: :func:`validate_stage_resume`
        can tell a Stage-A checkpoint from a Stage-B one, and a reader can
        reconstruct the schedule that produced it.
        """
        metadata: Dict[str, Any] = {
            STAGE_METADATA_KEY: self.key,
            "stage_name": self.name,
            "objective": self.objective,
            DATASET_METADATA_KEY: self.dataset_identity(),
            "token_budget": int(self.token_budget),
            "micro_batch": int(micro_batch),
            "grad_accum": int(grad_accum),
            "tokens_per_step": self.tokens_per_step(micro_batch, grad_accum),
            "precision": str(precision),
            "peak_lr": float(self.peak_lr),
            "min_lr_ratio": float(self.min_lr_ratio),
            "warmup_ratio": float(self.warmup_ratio),
            "grad_clip": float(self.grad_clip),
            "weight_decay": float(self.weight_decay),
            "mask_prompt_tokens": bool(self.mask_prompt_tokens),
            "seed": int(self.seed),
            "gpu": str(gpu),
        }
        if extra:
            metadata.update(dict(extra))
        return metadata


#: Per-stage optimisation defaults. Stage A is a from-scratch pretraining run
#: (higher LR, weight decay, no prompt masking); Stage B is conversational
#: tuning of an existing base model (lower LR, no weight decay on the chat
#: objective, prompt tokens masked so the loss is on the assistant's replies).
STAGE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    STAGE_A: {
        "name": STAGE_NAMES[STAGE_A],
        "objective": "next-token prediction on general English web text",
        "dataset_id": "allenai/c4",
        "dataset_config": "en",
        "train_split": "train",
        "val_split": "validation",
        "peak_lr": 3e-4,
        "min_lr_ratio": 0.1,
        "warmup_ratio": 0.005,
        "grad_clip": 1.0,
        "weight_decay": 0.1,
        "mask_prompt_tokens": False,
        "max_epochs": None,          # c4/en train is far larger than any budget
    },
    STAGE_B: {
        "name": STAGE_NAMES[STAGE_B],
        "objective": "supervised fine-tuning on assistant turns of chat data",
        "dataset_id": "HuggingFaceH4/ultrachat_200k",
        "dataset_config": None,
        "train_split": "train_sft",
        "val_split": "test_sft",
        "peak_lr": 5e-5,
        "min_lr_ratio": 0.1,
        "warmup_ratio": 0.03,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
        "mask_prompt_tokens": True,
        "max_epochs": 3,             # 1-3 epochs max, no unbounded cycling
    },
}


def stage_spec(
    key: str,
    *,
    token_budget: int,
    seq_len: int = 2048,
    seed: int = 1234,
    peak_lr: Optional[float] = None,
) -> StageSpec:
    """Build the :class:`StageSpec` for ``key`` (``"A"`` or ``"B"``).

    ``token_budget`` is required and validated: the budget comes from the
    *hardware plan* (Stage A) or from a measured tokens-per-epoch estimate times
    an explicit epoch count (Stage B), never from a hard-coded round number.
    """
    if key not in STAGE_DEFAULTS:
        raise ValueError(f"unknown stage {key!r}: expected one of {STAGES}")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive — compute it from the hardware plan")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    defaults = dict(STAGE_DEFAULTS[key])
    if peak_lr is not None:
        if peak_lr <= 0:
            raise ValueError("peak_lr must be positive")
        defaults["peak_lr"] = float(peak_lr)
    return StageSpec(
        key=key,
        name=defaults["name"],
        objective=defaults["objective"],
        dataset_id=defaults["dataset_id"],
        dataset_config=defaults["dataset_config"],
        train_split=defaults["train_split"],
        val_split=defaults["val_split"],
        seq_len=int(seq_len),
        token_budget=int(token_budget),
        max_epochs=defaults["max_epochs"],
        peak_lr=float(defaults["peak_lr"]),
        min_lr_ratio=float(defaults["min_lr_ratio"]),
        warmup_ratio=float(defaults["warmup_ratio"]),
        grad_clip=float(defaults["grad_clip"]),
        weight_decay=float(defaults["weight_decay"]),
        mask_prompt_tokens=bool(defaults["mask_prompt_tokens"]),
        seed=int(seed),
    )


def stage_out_dir(root: str, stage: str) -> str:
    """``root`` -> ``root/stage-a`` (or ``stage-b``); creates nothing."""
    if stage not in STAGE_DIRS:
        raise ValueError(f"unknown stage {stage!r}: expected one of {STAGES}")
    return f"{str(root).rstrip('/')}/{STAGE_DIRS[stage]}"


# ---------------------------------------------------------------------------
# Budget arithmetic — the overshoot guard
# ---------------------------------------------------------------------------
def steps_for_budget(token_budget: int, tokens_per_step: int) -> int:
    """Optimizer steps to spend exactly ``token_budget`` (ceil: last step partial)."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    return int(math.ceil(token_budget / tokens_per_step))


@dataclass(frozen=True)
class StepPlan:
    """How much of one optimizer step the remaining budget can pay for.

    ``micro_batches == 0`` means "the budget is spent — stop". A value below
    ``grad_accum`` is the final partial step: the loop must scale the loss by
    ``1 / micro_batches`` (not ``1 / grad_accum``) so the last gradient is not
    silently down-weighted.
    """

    micro_batches: int
    tokens_this_step: int
    partial: bool

    @property
    def is_final(self) -> bool:
        """True when the budget is spent after this step."""
        return self.partial

    @property
    def should_stop(self) -> bool:
        """True when there is nothing left to spend."""
        return self.micro_batches == 0


def plan_step(
    remaining_tokens: int,
    *,
    micro_batch: int,
    grad_accum: int,
    seq_len: int,
) -> StepPlan:
    """Cap one optimizer step at the remaining budget — structurally.

    This is the single decision point that makes budget overshoot impossible:

    * ``remaining_tokens >= micro_batch * grad_accum * seq_len`` -> a full step;
    * a smaller remainder -> as many whole micro-batches as it can pay for
      (``partial=True``), because a micro-batch is the finest granularity a
      forward/backward pass has;
    * ``remaining_tokens < micro_batch * seq_len`` -> ``0`` micro-batches, i.e.
      the loop stops instead of stepping past the budget.
    """
    if micro_batch <= 0 or grad_accum <= 0 or seq_len <= 0:
        raise ValueError("micro_batch, grad_accum and seq_len must be positive")
    if remaining_tokens < 0:
        raise ValueError("remaining_tokens must be >= 0")
    tokens_per_micro = micro_batch * seq_len
    full = micro_batch * grad_accum * seq_len
    if remaining_tokens >= full:
        return StepPlan(micro_batches=int(grad_accum), tokens_this_step=int(full),
                        partial=False)
    affordable = int(remaining_tokens // tokens_per_micro)
    affordable = max(0, min(int(grad_accum), affordable))
    if affordable == 0:
        return StepPlan(micro_batches=0, tokens_this_step=0, partial=True)
    return StepPlan(micro_batches=affordable,
                    tokens_this_step=affordable * tokens_per_micro,
                    partial=True)


@dataclass
class TokenAccountant:
    """The run's token counter, with the budget invariant enforced on every add.

    ``consumed`` counts tokens *this stage* has trained on (a Stage-B run starts
    from zero even though the model already saw Stage A's tokens: the budget is
    a per-stage contract). :meth:`add` raises if an add would cross the budget —
    a bug in the loop surfaces immediately instead of producing a run that
    quietly trained past its stated budget.
    """

    token_budget: int
    consumed: int = 0
    steps: int = 0

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.consumed < 0:
            raise ValueError("consumed must be >= 0")
        if self.consumed > self.token_budget:
            raise ValueError(
                f"consumed {self.consumed:,} already exceeds the budget "
                f"{self.token_budget:,}"
            )

    @property
    def remaining(self) -> int:
        return max(0, int(self.token_budget) - int(self.consumed))

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def fraction(self) -> float:
        return float(self.consumed) / float(self.token_budget)

    def add(self, tokens: int, *, steps: int = 1) -> None:
        """Add ``tokens`` consumed by ``steps`` optimizer step(s)."""
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        if steps < 0:
            raise ValueError("steps must be >= 0")
        new_total = int(self.consumed) + int(tokens)
        if new_total > int(self.token_budget):
            raise ValueError(
                f"budget overshoot: {self.consumed:,} + {tokens:,} = {new_total:,} "
                f"> budget {self.token_budget:,}"
            )
        self.consumed = new_total
        self.steps += int(steps)

    def plan_step(self, *, micro_batch: int, grad_accum: int, seq_len: int) -> StepPlan:
        """Shorthand for :func:`plan_step` against this accountant's remainder."""
        return plan_step(self.remaining, micro_batch=micro_batch,
                         grad_accum=grad_accum, seq_len=seq_len)

    def as_dict(self) -> Dict[str, int]:
        return {
            "tokens_consumed": int(self.consumed),
            "token_budget": int(self.token_budget),
            "steps": int(self.steps),
        }


def simulate_run(
    token_budget: int,
    *,
    micro_batch: int,
    grad_accum: int,
    seq_len: int,
) -> Dict[str, int]:
    """Replay the loop's accounting with no model — the budget contract as a test.

    Returns ``{"steps", "tokens_consumed", "final_micro_batches", "overshoot"}``.
    ``overshoot`` must always be ``0``: it is the number of tokens the simulated
    run consumed beyond its budget (impossible by construction, asserted by the
    test suite for awkward budgets such as 1 token, budget-1, and non-multiples).
    """
    accountant = TokenAccountant(token_budget)
    steps = 0
    final_micro = 0
    while True:
        plan = accountant.plan_step(micro_batch=micro_batch, grad_accum=grad_accum,
                                    seq_len=seq_len)
        if plan.should_stop:
            break
        accountant.add(plan.tokens_this_step, steps=1)
        steps += 1
        final_micro = plan.micro_batches
    return {
        "steps": steps,
        "tokens_consumed": accountant.consumed,
        "final_micro_batches": final_micro,
        "overshoot": max(0, accountant.consumed - token_budget),
    }


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
def warmup_steps(total_steps: int, warmup_ratio: float, *, minimum: int = 20) -> int:
    """Warmup length in optimizer steps (``ratio × total``, clamped to the run)."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= float(warmup_ratio) < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if int(total_steps) <= 1:
        return 0
    raw = int(round(float(warmup_ratio) * int(total_steps)))
    # Clamp into the run AFTER applying the minimum, so a very short schedule
    # (a smoke run) never gets a warmup longer than the schedule itself.
    return max(0, min(int(total_steps) - 1, max(int(minimum), raw)))


def lr_at(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    warmup: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup from ``peak_lr/warmup`` to ``peak_lr``, then cosine to the floor.

    ``step`` is the 0-based index of the step about to run, so step 0 uses
    ``peak_lr / warmup`` (never 0) and step ``warmup-1`` uses ``peak_lr``. After
    ``total_steps`` the LR sits at ``peak_lr * min_lr_ratio`` and stays there,
    which is what makes a *resumed* run's LR identical to the uninterrupted one.
    """
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= float(min_lr_ratio) <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    step = max(0, int(step))
    warmup = max(0, int(warmup))
    if warmup > 0 and step < warmup:
        return float(peak_lr) * float(step + 1) / float(warmup)
    span = max(1, int(total_steps) - warmup)
    progress = min(1.0, float(step - warmup) / float(span))
    floor = float(min_lr_ratio)
    return float(peak_lr) * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Checkpoint / resume contract
# ---------------------------------------------------------------------------
def validate_stage_resume(
    ckpt: Mapping[str, Any],
    spec: StageSpec,
    *,
    tokenizer_fingerprint: Optional[str] = None,
) -> None:
    """Validate a checkpoint against the stage about to resume it.

    Raises ``ValueError`` naming the broken contract. Checks, in order:

    1. format is :data:`CHECKPOINT_FORMAT`;
    2. the checkpoint records the **same stage** (a Stage-B checkpoint resumed as
       Stage A would train the pretraining objective on chat data and wreck the
       run — this is the "must not regress to the other stage" guard);
    3. the dataset identity matches (id/config/splits/seq_len);
    4. the tokenizer fingerprint matches when both sides record one;
    5. optimizer state and RNG state are present (a checkpoint without them can
       be *evaluated*, but resuming from it silently restarts the schedule and
       the RNG);
    6. the recorded token count fits inside the stage budget.
    """
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint format {ckpt.get('format')!r} is not {CHECKPOINT_FORMAT!r}"
        )
    metadata = ckpt.get("run_metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint run_metadata is not a mapping")
    recorded_stage = metadata.get(STAGE_METADATA_KEY)
    if recorded_stage is None:
        raise ValueError(
            "checkpoint does not record a training stage — it was written by the "
            "old single-stage notebook and cannot be resumed as Stage "
            f"{spec.key}; start this stage fresh"
        )
    if str(recorded_stage) != spec.key:
        raise ValueError(
            f"checkpoint belongs to stage {recorded_stage!r} "
            f"({STAGE_NAMES.get(str(recorded_stage), 'unknown')}) but this run is "
            f"stage {spec.key!r} ({spec.name}). Point OUT_ROOT at the stage-"
            f"{str(recorded_stage).lower()} directory, or set STAGE="
            f"{recorded_stage!r} — resuming one stage as the other trains the "
            "wrong objective on the wrong data"
        )
    recorded_dataset = metadata.get(DATASET_METADATA_KEY) or {}
    expected_dataset = spec.dataset_identity()
    if dict(recorded_dataset) != expected_dataset:
        raise ValueError(
            f"checkpoint was trained on {recorded_dataset} but this stage is "
            f"{expected_dataset} — the dataset identity must match to resume"
        )
    recorded_fingerprint = ckpt.get("tokenizer_fingerprint")
    if tokenizer_fingerprint and recorded_fingerprint and (
        str(recorded_fingerprint) != str(tokenizer_fingerprint)
    ):
        raise ValueError(
            "checkpoint tokenizer fingerprint "
            f"{recorded_fingerprint!r} != the loaded tokenizer's "
            f"{tokenizer_fingerprint!r} — a different tokenizer means different ids"
        )
    if ckpt.get("optimizer_state_dict") is None:
        raise ValueError(
            "checkpoint has no optimizer state: it can be evaluated, but resuming "
            "from it would restart Adam's moments from zero"
        )
    if not ckpt.get("rng_state"):
        raise ValueError(
            "checkpoint has no RNG state: resuming would not reproduce the "
            "uninterrupted run's data order or dropout stream"
        )
    tokens = ckpt.get("tokens_consumed")
    if tokens is not None and int(tokens) > int(spec.token_budget):
        raise ValueError(
            f"checkpoint already consumed {int(tokens):,} tokens, more than this "
            f"stage's budget {spec.token_budget:,} — raise the budget or start "
            "a new stage"
        )


def resume_position(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """The resumable state a checkpoint carries, in one dict.

    ``epoch``/``dataset_position`` come from ``run_metadata``; a checkpoint
    written before position tracking reports epoch 1 / position 0, which the
    caller treats as "re-stream this epoch from its start" (documented cost: the
    epoch's documents are re-read, tokens are *not* recounted).
    """
    metadata = ckpt.get("run_metadata") or {}
    position = metadata.get(POSITION_METADATA_KEY) or {}
    return {
        "stage": metadata.get(STAGE_METADATA_KEY),
        "step": int(ckpt.get("step") or 0),
        "tokens_consumed": int(ckpt.get("tokens_consumed") or 0),
        "epoch": int(ckpt.get("epoch") or position.get("epoch") or 1),
        "docs_consumed": int(position.get("docs_consumed") or 0),
        "rows_in_epoch": int(position.get("rows_in_epoch") or 0),
        "train_loss": ckpt.get("train_loss"),
        "val_loss": ckpt.get("val_loss"),
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "DATASET_METADATA_KEY",
    "POSITION_METADATA_KEY",
    "STAGES",
    "STAGE_A",
    "STAGE_B",
    "STAGE_DEFAULTS",
    "STAGE_DIRS",
    "STAGE_METADATA_KEY",
    "STAGE_NAMES",
    "StageSpec",
    "StepPlan",
    "TokenAccountant",
    "lr_at",
    "plan_step",
    "resume_position",
    "simulate_run",
    "stage_out_dir",
    "stage_spec",
    "steps_for_budget",
    "validate_stage_resume",
    "warmup_steps",
]
