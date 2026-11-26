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

#: Pinned Hub revisions (commit SHAs) for the two datasets. A dataset that moves
#: under a resumed run silently changes the objective mid-training, so the
#: revision is part of the dataset identity and is validated on resume.
#:
#: Resolved **2026-11-19** from ``https://huggingface.co/api/datasets/<id>``
#: (``"sha"`` field); bump deliberately, never implicitly.
#: ``allenai/c4`` last modified 2024-01-09; ``HuggingFaceH4/ultrachat_200k``
#: last modified 2024-10-16.
C4_EN_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"

# ---------------------------------------------------------------------------
# Stage A budget policy (owner audit: "severely insufficient")
# ---------------------------------------------------------------------------
#: The smallest Stage-A budget that is worth running at all. A 1B model trained
#: from scratch on fewer than ~2 tokens/parameter learns token statistics and
#: word shapes, not language; below this the notebook warns loudly instead of
#: starting a run that cannot produce a usable model.
STAGE_A_TOKENS_FLOOR = 2_000_000_000

#: The default Stage-A target: 4B tokens ≈ 3.9 tokens/parameter ≈ 19 % of
#: Chinchilla-optimal (~20.7B for 1.036B params). Justified in
#: ``docs/training-1b-notebook.md`` with the per-GPU hours → tokens → capability
#: table: this is the smallest budget at which a from-scratch 1B produces
#: grammatical, locally coherent English instead of babble.
STAGE_A_TOKENS_DEFAULT = 4_000_000_000

#: Chinchilla-optimal tokens for the 1B preset (≈20 tokens/parameter). Documented
#: as the reference point, **not** imposed as a default: on Colab hardware it is
#: weeks of wall-clock, and pretending otherwise is what made the old budget lie.
CHINCHILLA_TOKENS = 20_000_000_000

#: Consecutive skipped optimizer updates (non-finite loss or gradient) after
#: which the run aborts loudly instead of spinning on a broken model.
MAX_CONSECUTIVE_SKIPPED = 20

#: Stage-scoped output directory names under the run root.
STAGE_DIRS: Dict[str, str] = {STAGE_A: "stage-a", STAGE_B: "stage-b"}

#: Sentinel for :func:`stage_spec` overrides: "argument not given" (so ``None``
#: can be passed *as* a value, e.g. ``dataset_config=None`` for a config-less
#: dataset).
_UNSET: Any = object()


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
    #: Pinned Hub revision (commit SHA) of ``dataset_id``. Part of the dataset
    #: identity: a dataset whose ``main`` moved between two sessions would
    #: otherwise change the corpus under a resumed run without any error.
    dataset_revision: Optional[str] = None

    # -- identity -----------------------------------------------------------
    def dataset_identity(self) -> Dict[str, Any]:
        """The dataset's identity, recorded in every checkpoint.

        A resume whose dataset identity differs is rejected: resuming Stage A on
        a different corpus (or on a different *revision* of the same corpus)
        silently changes the objective mid-run.
        """
        return {
            "id": self.dataset_id,
            "config": self.dataset_config,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "seq_len": int(self.seq_len),
            "revision": self.dataset_revision,
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
        "dataset_revision": C4_EN_REVISION,
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
        "dataset_revision": ULTRACHAT_REVISION,
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
    dataset_revision: Optional[str] = None,
    dataset_id: Any = _UNSET,
    dataset_config: Any = _UNSET,
    train_split: Any = _UNSET,
    val_split: Any = _UNSET,
) -> StageSpec:
    """Build the :class:`StageSpec` for ``key`` (``"A"`` or ``"B"``).

    ``token_budget`` is required and validated: the budget comes from the
    *budget policy* (:func:`stage_a_budget` for Stage A) or from a measured
    tokens-per-epoch estimate times an explicit epoch count (Stage B), never
    from a hard-coded round number.

    ``dataset_revision`` defaults to the stage's pinned Hub revision
    (:data:`C4_EN_REVISION` / :data:`ULTRACHAT_REVISION`); pass an explicit SHA
    only to reproduce an older corpus, and note that it becomes part of the
    checkpoint's dataset identity, so a resume must use the same one.

    The ``dataset_id`` / ``dataset_config`` / split overrides exist for the CPU
    smoke run (``notebooks/talos_smoke.py``), which trains on a synthetic corpus
    and must record *that* as its dataset identity rather than claiming c4. They
    are not a licence to swap corpora in a real run: the identity goes into every
    checkpoint and a resume with a different one is refused.
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
    if dataset_id is not _UNSET:
        defaults["dataset_id"] = dataset_id
    if dataset_config is not _UNSET:
        defaults["dataset_config"] = dataset_config
    if train_split is not _UNSET:
        defaults["train_split"] = train_split
    if val_split is not _UNSET:
        defaults["val_split"] = val_split
    revision = defaults.get("dataset_revision") if dataset_revision is None else dataset_revision
    if dataset_id is not _UNSET and dataset_revision is None:
        revision = None
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
        dataset_revision=revision,
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
# Stage A budget policy — hours → tokens → what the model can honestly become
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageABudget:
    """The Stage-A token budget and *why* it is what it is.

    ``source`` is one of ``"default"`` (the corrected default target),
    ``"hours"`` (the user time-boxed the run: ``STAGE_A_HOURS × tokens/sec``) or
    ``"override"`` (an explicit token count). ``warning`` is non-``None`` when the
    resulting budget is below :data:`STAGE_A_TOKENS_FLOOR` — the notebook prints
    it as a loud WARNING rather than silently starting a run that cannot learn
    language.
    """

    tokens: int
    source: str
    floor: int
    warning: Optional[str]
    wall_clock_hours: Optional[float]

    @property
    def below_floor(self) -> bool:
        return self.tokens < self.floor

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tokens": int(self.tokens),
            "source": self.source,
            "floor": int(self.floor),
            "below_floor": self.below_floor,
            "warning": self.warning,
            "wall_clock_hours": (None if self.wall_clock_hours is None
                                 else round(float(self.wall_clock_hours), 2)),
        }


def tokens_per_parameter(tokens: int, parameters: int) -> float:
    """``tokens / parameters`` — the ratio that decides what the run can become."""
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    return float(tokens) / float(parameters)


def budget_capability(tokens: int, parameters: int) -> str:
    """What a from-scratch model of ``parameters`` params can honestly become.

    The bands are deliberately blunt. They describe *language competence*, not
    chat quality: no Stage-A budget that fits Colab produces a usable assistant
    on its own, and Stage B cannot invent knowledge the base model never saw.
    """
    ratio = tokens_per_parameter(tokens, parameters)
    if ratio < 1.0:
        return ("below 1 token/parameter: token statistics and word shapes only — "
                "the model will emit babble, not language")
    if ratio < 2.0:
        return ("1-2 tokens/parameter: grammatical fragments and short memorised "
                "spans; frequent drift, no reliable syntax across a sentence")
    if ratio < 8.0:
        return ("2-8 tokens/parameter: grammatical, locally coherent English for a "
                "few sentences; unreliable facts, weak long-range coherence")
    if ratio < 15.0:
        return ("8-15 tokens/parameter: readable paragraphs and basic instruction "
                "following after Stage B; still well below Chinchilla, factual "
                "errors remain common")
    return ("near-Chinchilla (>=15 tokens/parameter): a genuine small base model; "
            "quality still bounded by data and tuning")


def stage_a_budget(
    *,
    tokens_per_sec: Optional[float],
    hours: float = 0.0,
    token_override: int = 0,
    floor: int = STAGE_A_TOKENS_FLOOR,
    default_tokens: int = STAGE_A_TOKENS_DEFAULT,
) -> StageABudget:
    """Derive the Stage-A budget, warning (not silently accepting) a useless run.

    Precedence:

    1. ``token_override > 0`` — an explicit token budget wins outright;
    2. ``hours > 0`` — a time-boxed run: ``hours × 3600 × tokens_per_sec``;
    3. otherwise — the corrected default target :data:`STAGE_A_TOKENS_DEFAULT`
       (multi-billion tokens), independent of the GPU, because the *budget* is
       what decides whether the model learns language and the *wall clock* is
       what the GPU decides.

    Any result below ``floor`` carries a ``warning``; the notebook prints it as a
    WARNING block before training. This is the fix for the owner's finding that
    ``STAGE_A_HOURS × tokens/sec`` at the old default (10 h) yielded 62M-407M
    tokens — 0.06-0.39 tokens/parameter, i.e. a run that could not work.
    """
    if floor <= 0 or default_tokens <= 0:
        raise ValueError("floor and default_tokens must be positive")
    if token_override < 0:
        raise ValueError("token_override must be >= 0")
    if hours < 0:
        raise ValueError("hours must be >= 0")

    wall_clock_hours: Optional[float] = None
    if token_override:
        tokens, source = int(token_override), "override"
    elif hours > 0:
        if tokens_per_sec is None or tokens_per_sec <= 0:
            raise ValueError(
                "hours > 0 needs a positive tokens_per_sec estimate from the "
                "hardware plan — the runtime reported none"
            )
        tokens = int(float(hours) * 3600.0 * float(tokens_per_sec))
        source = "hours"
    else:
        tokens, source = int(default_tokens), "default"
    if tokens <= 0:
        raise ValueError(
            "the derived Stage-A budget is 0 tokens — raise STAGE_A_HOURS or set "
            "STAGE_A_TOKENS_OVERRIDE"
        )
    if tokens_per_sec is not None and float(tokens_per_sec) > 0:
        wall_clock_hours = float(tokens) / float(tokens_per_sec) / 3600.0

    warning: Optional[str] = None
    if tokens < floor:
        warning = (
            f"Stage-A budget {tokens:,} tokens is BELOW the {floor:,}-token floor "
            f"(~{tokens_per_parameter(tokens, 1_036_060_672):.2f} tokens/parameter). "
            f"A from-scratch 1B model at this budget cannot learn language: expect "
            f"babble. This is what {source} gives you. Raise STAGE_A_HOURS, clear "
            f"STAGE_A_HOURS to use the {default_tokens:,}-token default, or set "
            f"STAGE_A_TOKENS_OVERRIDE — the run is allowed, but it is not a "
            f"language model at the end of it."
        )
    return StageABudget(tokens=int(tokens), source=source, floor=int(floor),
                        warning=warning, wall_clock_hours=wall_clock_hours)


def stage_b_budget(
    *,
    tokens_per_epoch: int,
    epochs: int,
    seq_len: int,
) -> int:
    """Stage-B token budget that matches **exactly** what the stream emits.

    The stream packs documents and drops the sub-``seq_len`` remainder at each
    epoch boundary, so one epoch emits ``floor(tokens_per_epoch / seq_len) ×
    seq_len`` tokens — not ``tokens_per_epoch``. Budgeting the raw estimate (the
    previous behaviour) let the cap truncate the final epoch by however much the
    measured mean was off, which made "2 epochs" quietly mean "1.97 epochs" and
    made Stage-B token counts not comparable with Stage A's.

    The budget is therefore quantised to whole packed rows, and the loop reports
    the epochs actually completed (``train_result.epochs_completed``).
    """
    if tokens_per_epoch <= 0:
        raise ValueError("tokens_per_epoch must be positive — measure it")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    rows_per_epoch = int(tokens_per_epoch) // int(seq_len)
    if rows_per_epoch < 1:
        raise ValueError(
            f"tokens_per_epoch {tokens_per_epoch:,} is smaller than one {seq_len}-token "
            "row — the dataset is too small to pack even one row"
        )
    return rows_per_epoch * int(seq_len) * int(epochs)


# ---------------------------------------------------------------------------
# Non-finite detection: skipped updates
# ---------------------------------------------------------------------------
class NonFiniteAbort(RuntimeError):
    """Raised when too many consecutive optimizer updates were skipped.

    A burst of skipped updates means the run is broken (exploded LR, a corrupt
    batch, a bad checkpoint); continuing would burn GPU hours producing nothing,
    so the loop stops with a diagnostic instead of spinning silently.
    """


@dataclass
class SkipTracker:
    """Counts skipped optimizer updates and aborts on a consecutive burst.

    A skipped update is one the loop *decided not to apply* because the loss or
    the gradients were not finite. Skipping is the correct response (one bad
    batch must not poison the weights), but a long run of them is not — so the
    tracker raises :class:`NonFiniteAbort` past
    :data:`MAX_CONSECUTIVE_SKIPPED` in a row.

    Nothing here imports torch: the loop calls :meth:`note_skip` /
    :meth:`note_update`, and the counters go into the checkpoint metadata, the
    metrics row and the tracker's ``status.json`` (``skipped_updates``).
    """

    max_consecutive: int = MAX_CONSECUTIVE_SKIPPED
    total: int = 0
    consecutive: int = 0
    last_reason: str = ""

    def __post_init__(self) -> None:
        if self.max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")

    def note_skip(self, reason: str) -> None:
        """Record a skipped update; raise once the consecutive limit is passed."""
        self.total += 1
        self.consecutive += 1
        self.last_reason = str(reason)
        if self.consecutive >= self.max_consecutive:
            raise NonFiniteAbort(
                f"{self.consecutive} consecutive optimizer updates were skipped "
                f"(last reason: {reason}; {self.total} total). The run is not "
                "learning anything — the loss or the gradients are not finite. "
                "Stop and check: (1) the LR schedule and peak LR, (2) the newest "
                "checkpoint (resume from an older step), (3) the data (a corrupt "
                "shard produces NaN logits), (4) fp16 overflow on a T4 — set "
                "PRECISION='bf16' on an L4/A100 or lower the LR. A checkpoint was "
                "written before this abort."
            )

    def note_update(self) -> None:
        """Record an applied update (resets the consecutive counter)."""
        self.consecutive = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "skipped_updates": int(self.total),
            "consecutive_skips": int(self.consecutive),
            "last_skip_reason": self.last_reason,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, max_consecutive: int) -> "SkipTracker":
        """Restore a tracker from a checkpoint's persisted counters.

        ``state`` is what :func:`skip_state_from_metadata` returns; the limit
        always comes from the current config, never from the checkpoint (the
        policy is this session's, the history is the run's).
        """
        if not isinstance(state, Mapping):
            raise ValueError("skip_state must be a mapping")
        return cls(
            max_consecutive=int(max_consecutive),
            total=max(0, int(state.get("total") or 0)),
            consecutive=max(0, int(state.get("consecutive") or 0)),
            last_reason=str(state.get("last_reason") or ""),
        )


def skip_state_from_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """A checkpoint's persisted skip counters, or ``None`` when it has none.

    All three of :class:`SkipTracker`'s fields live in the checkpoint's
    ``run_metadata`` (``skipped_updates`` / ``consecutive_skips`` /
    ``last_skip_reason``). A checkpoint written before they existed returns
    ``None``: the caller warns once and the counters start at zero, rather than
    crashing on a missing field (the audit round's ``_fmt_int`` lesson).
    """
    if not isinstance(metadata, Mapping):
        return None
    keys = ("skipped_updates", "consecutive_skips", "last_skip_reason")
    if not any(key in metadata for key in keys):
        return None
    return {
        "total": max(0, int(metadata.get("skipped_updates") or 0)),
        "consecutive": max(0, int(metadata.get("consecutive_skips") or 0)),
        "last_reason": str(metadata.get("last_skip_reason") or ""),
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


def validate_stage_b_init(ckpt: Mapping[str, Any]) -> str:
    """Refuse a Stage-B initialisation that did not come from Stage A.

    Stage B is conversational tuning *of the pretrained base model*, so its init
    checkpoint must be the Stage-A final checkpoint. :func:`validate_stage_resume`
    guards the resume path (a stage may only resume itself); this guards the other
    direction — Stage B starting **fresh** — which otherwise accepted any
    checkpoint of the right format and preset. Loading a Stage-B (or an untagged)
    checkpoint there silently skips the pretraining and reports a run that never
    happened.

    Returns the stage tag it accepted (``"A"``); raises ``ValueError`` naming the
    fix otherwise.
    """
    metadata = ckpt.get("run_metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint run_metadata is not a mapping")
    recorded = metadata.get(STAGE_METADATA_KEY)
    if recorded is None:
        raise ValueError(
            "Stage B initializes only from a Stage-A checkpoint; got stage=None "
            f"(the checkpoint records no {STAGE_METADATA_KEY!r} tag). Set "
            "STAGE_A_INIT_CKPT to the Stage-A final checkpoint (its run_metadata "
            "carries stage='A'), or run Stage A first"
        )
    if str(recorded) != STAGE_A:
        raise ValueError(
            "Stage B initializes only from a Stage-A checkpoint; got "
            f"stage={recorded!r} ({STAGE_NAMES.get(str(recorded), 'unknown')}). "
            "Point STAGE_A_INIT_CKPT at the Stage-A final checkpoint under "
            "<out>/stage-a — a Stage-B checkpoint is already conversationally "
            "tuned and cannot pretrain itself"
        )
    return str(recorded)


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
    "CHINCHILLA_TOKENS",
    "C4_EN_REVISION",
    "DATASET_METADATA_KEY",
    "MAX_CONSECUTIVE_SKIPPED",
    "NonFiniteAbort",
    "POSITION_METADATA_KEY",
    "STAGES",
    "STAGE_A",
    "STAGE_A_TOKENS_DEFAULT",
    "STAGE_A_TOKENS_FLOOR",
    "STAGE_B",
    "STAGE_DEFAULTS",
    "STAGE_DIRS",
    "STAGE_METADATA_KEY",
    "STAGE_NAMES",
    "SkipTracker",
    "StageABudget",
    "StageSpec",
    "StepPlan",
    "TokenAccountant",
    "ULTRACHAT_REVISION",
    "budget_capability",
    "lr_at",
    "plan_step",
    "resume_position",
    "simulate_run",
    "skip_state_from_metadata",
    "stage_a_budget",
    "stage_b_budget",
    "stage_out_dir",
    "stage_spec",
    "steps_for_budget",
    "tokens_per_parameter",
    "validate_stage_b_init",
    "validate_stage_resume",
    "warmup_steps",
]
