"""End-to-end CPU smoke run for the Argo 1B notebook — no GPU, no network.

The notebook's training loop is only ever exercised for real on a Colab GPU,
which is exactly the wrong place to discover that a checkpoint cannot be
resumed or that Stage B's masking broke. This module runs the **same** code
paths the 1B run uses — :func:`talos_data.RowStream` for packing,
:func:`talos_train_loop.train_stage` for the loop, the repo's
``save_checkpoint`` / ``validate_resume_checkpoint`` / ``restore_rng_state``,
``inference.generate.generate`` for decoding, and
:class:`talos_training_lib.TrackerPublisher` for the tracker payloads — with the
canonical **tiny** preset (exactly 254,272 parameters, vocab 1,024) and a
synthetic text corpus, so it finishes in seconds on a CPU.

What it proves, in order:

1. data → packed rows → a few optimizer steps with the real budget cap;
2. a checkpoint is written in the repo's ``talos-training-checkpoint-v1`` schema
   and passes the repo's own resume validation;
3. resuming from it lands on the *same* token count and step count as an
   uninterrupted run (and reports the loss delta);
4. evaluation (validation loss → perplexity) and generation (greedy decode +
   degeneracy analysis) run on the checkpointed weights;
5. the tracker and metrics payloads are produced (``dry_run``: written locally,
   nothing uploaded);
6. Stage B runs with assistant-only masking and its checkpoint is *refused* by a
   Stage-A resume — the two-stage discipline.

Run it before a real Colab run::

    python notebooks/talos_smoke.py --out /tmp/talos-smoke

or from pytest: ``tests/test_notebook_smoke.py``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "notebooks"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402 - after the sys.path fix-up

from configs.presets import ALL_PRESETS  # noqa: E402
from inference.generate import generate  # noqa: E402
from model.gpt import TalosGPT  # noqa: E402
from scripts.train_oasst1 import (capture_rng_state, list_checkpoint_candidates,  # noqa: E402
                                  load_checkpoint, restore_rng_state,
                                  save_checkpoint, validate_resume_checkpoint)
from talos_data import (RowStream, StreamPosition, encode_chat_document,  # noqa: E402
                        encode_text_document, take_rows)
from talos_eval import analyze_generation, perplexity  # noqa: E402
from talos_stages import (POSITION_METADATA_KEY, STAGE_A, STAGE_B,  # noqa: E402
                          TokenAccountant, stage_b_budget, stage_spec,
                          validate_stage_resume)
from talos_torch_utils import (autocast_kwargs, build_optimizer,  # noqa: E402
                               make_grad_scaler, masked_cross_entropy,
                               rows_to_tensor)
from talos_tokenizer import (TALOS_1B_49K_MERGES, TALOS_1B_49K_SHA256,  # noqa: E402
                             TALOS_1B_49K_VOCAB)
from talos_training_lib import (MODEL_DISPLAY_NAME, MetricsSync,  # noqa: E402
                                TrackerPublisher, build_metrics_row,
                                render_display_table)
from talos_train_loop import TrainLoopConfig, TrainResult, train_stage  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402

#: The canonical tiny preset — the same one every other repo test uses.
SMOKE_PRESET = "tiny"
SMOKE_PARAMS = 254_272
SMOKE_MODEL_VOCAB = 1024

#: Synthetic corpus size (documents) — enough to pack a few hundred rows.
SMOKE_DOCS = 400
SMOKE_VAL_DOCS = 8
SMOKE_SEQ_LEN = 64
SMOKE_MICRO_BATCH = 2
SMOKE_GRAD_ACCUM = 2
SMOKE_STEPS = 4
SMOKE_RESUME_AFTER = 2
SMOKE_STAGE_B_STEPS = 2
SMOKE_MAX_NEW_TOKENS = 8
SMOKE_SEED = 1234

_WORDS = (
    "the of and to in is was for on that with as by at from his her their its "
    "it be been this not are but had they which will one all were when we there "
    "can an your what out so said have do has more about up who new time also"
).split()


@dataclass
class SmokeResult:
    """Everything the smoke run measured — asserted by the pytest wrapper."""

    out_dir: str
    params: int
    vocab: int
    tokenizer_fingerprint: str
    #: Uninterrupted reference run.
    full_steps: int
    full_tokens: int
    full_train_loss: float
    #: Interrupted-then-resumed run.
    resumed_steps: int
    resumed_tokens: int
    resumed_train_loss: float
    resume_loss_delta: float
    checkpoint: str
    checkpoint_format: str
    val_loss: Optional[float]
    perplexity: Optional[float]
    generated_tokens: int
    distinct_2: float
    stage_b_tokens: int
    stage_b_supervised_tokens: int
    stage_b_checkpoint: str
    stage_b_rejected_by_stage_a: bool
    tracker_status_path: str
    tracker_history_path: str
    metrics_path: str
    tracker_status: Dict[str, Any] = field(default_factory=dict)
    display_rows: int = 0
    elapsed_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Synthetic data (no network)
# ---------------------------------------------------------------------------
def synthetic_text_documents(count: int, *, seed: int = SMOKE_SEED) -> List[str]:
    """Deterministic pseudo-English documents for the smoke run."""
    rng = random.Random(seed)
    return [" ".join(rng.choice(_WORDS) for _ in range(rng.randint(40, 90))) + "."
            for _ in range(int(count))]


def synthetic_chat_documents(count: int, *, seed: int = SMOKE_SEED) -> List[List[Dict[str, str]]]:
    """Deterministic ``messages`` rows in ultrachat's shape (user + assistant)."""
    rng = random.Random(seed + 1)
    rows: List[List[Dict[str, str]]] = []
    for _ in range(int(count)):
        question = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 14)))
        answer = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(10, 30)))
        rows.append([{"role": "user", "content": question + "?"},
                     {"role": "assistant", "content": answer + "."}])
    return rows


def make_tokenizer() -> ByteLevelBPETokenizer:
    """The canonical tiny preset's byte-level tokenizer (vocab <= 1,024, no merges).

    Byte-level encoding is lossless with zero merges, which is what lets the
    smoke run use the tiny preset's *own* tokenizer config instead of training
    one — the packing/masking code paths are identical either way.
    """
    from configs.presets import tiny_tokenizer_config

    tokenizer = ByteLevelBPETokenizer(tiny_tokenizer_config())
    if tokenizer.vocab_size > SMOKE_MODEL_VOCAB:
        raise AssertionError(
            f"smoke tokenizer vocab {tokenizer.vocab_size} exceeds the tiny "
            f"model's {SMOKE_MODEL_VOCAB} embedding rows"
        )
    return tokenizer


# ---------------------------------------------------------------------------
# The smoke run
# ---------------------------------------------------------------------------
def run_smoke(out_dir: Optional[str] = None, *,
              say: Callable[[str], None] = print) -> SmokeResult:
    """Run the whole pipeline end to end on CPU; return the measured summary."""
    started = time.time()
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="talos-smoke-")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SMOKE_SEED)
    device = torch.device("cpu")
    tokenizer = make_tokenizer()
    fingerprint = f"smoke-byte-tokenizer-vocab{tokenizer.vocab_size}"
    model_config = ALL_PRESETS[SMOKE_PRESET]().derive()
    model = TalosGPT(model_config)
    if model.num_parameters() != SMOKE_PARAMS:
        raise AssertionError(
            f"tiny preset is {model.num_parameters():,} params, expected {SMOKE_PARAMS:,}"
        )
    say(f"model      : {SMOKE_PRESET} ({model.num_parameters():,} params, "
        f"vocab {model_config.vocab_size:,}) — display name {MODEL_DISPLAY_NAME}")

    def encode(text: str, bos: bool = True, eos: bool = True) -> List[int]:
        return tokenizer.encode(text, bos=bos, eos=eos)

    train_docs = synthetic_text_documents(SMOKE_DOCS)
    val_docs = synthetic_text_documents(SMOKE_VAL_DOCS, seed=SMOKE_SEED + 7)

    def doc_source(docs: Sequence[str]) -> Callable[[int, int], Any]:
        def source(epoch: int, skip: int):
            for text in list(docs)[int(skip):]:
                ids, mask = encode_text_document(text, encode)
                if ids:
                    yield ids, mask
        return source

    # --- budget: a few real optimizer steps, capped exactly like the 1B run ---
    tokens_per_step = SMOKE_MICRO_BATCH * SMOKE_GRAD_ACCUM * SMOKE_SEQ_LEN
    budget = tokens_per_step * SMOKE_STEPS
    spec = stage_spec(STAGE_A, token_budget=budget, seq_len=SMOKE_SEQ_LEN,
                      seed=SMOKE_SEED, dataset_id="smoke/synthetic-text",
                      dataset_config=None, dataset_revision="local",
                      train_split="train", val_split="validation")
    val_rows = take_rows(
        RowStream(doc_source(val_docs), seq_len=SMOKE_SEQ_LEN,
                  vocab_size=SMOKE_MODEL_VOCAB, epochs=1),
        SMOKE_MICRO_BATCH * 2,
    )
    if not val_rows:
        raise AssertionError("the synthetic validation set packed no rows")
    say(f"budget     : {budget:,} tokens = {SMOKE_STEPS} steps x {tokens_per_step:,}")

    AUTOCAST = autocast_kwargs("cpu", "fp32")

    def make_forward_loss(m: TalosGPT) -> Callable[[Any, Any], Any]:
        def forward_loss(ids: Any, mask: Any = None):
            with torch.autocast(**AUTOCAST):
                logits, _ = m(ids[:, :-1])
                return masked_cross_entropy(logits, ids[:, 1:],
                                            None if mask is None else mask[:, 1:])
        return forward_loss

    @torch.no_grad()
    def evaluate(m: TalosGPT) -> Optional[float]:
        m.eval()
        total, count = 0.0, 0
        for start in range(0, len(val_rows), SMOKE_MICRO_BATCH):
            chunk = val_rows[start:start + SMOKE_MICRO_BATCH]
            ids, mask = rows_to_tensor(chunk, device=device, with_mask=False)
            total += float(make_forward_loss(m)(ids, mask))
            count += 1
        m.train()
        return total / count if count else None

    def make_optimizer(m: TalosGPT):
        return build_optimizer(m.parameters(), lr=spec.peak_lr,
                               weight_decay=spec.weight_decay, use_8bit_adam=False)

    def make_config(total_steps: int) -> TrainLoopConfig:
        return TrainLoopConfig(
            micro_batch=SMOKE_MICRO_BATCH, grad_accum=SMOKE_GRAD_ACCUM,
            seq_len=SMOKE_SEQ_LEN, total_steps=total_steps,
            warmup_steps=0, peak_lr=spec.peak_lr,
            min_lr_ratio=spec.min_lr_ratio, grad_clip=spec.grad_clip,
            log_every=1, eval_every=1, checkpoint_every=1,
            checkpoint_every_min=0.0, mask_prompt_tokens=False,
        )

    def make_saver(m: TalosGPT, optimizer: Any, stage_dir: Path,
                   stage_spec_obj: Any, position_of: Callable[[], StreamPosition],
                   ) -> Callable[[Any], str]:
        stage_dir.mkdir(parents=True, exist_ok=True)

        def on_checkpoint(info) -> str:
            name = f"step-{info.step}.pt"
            position = position_of()
            if len(position.pending_ids) >= SMOKE_SEQ_LEN:
                # A position captured between two packer yields can hold a
                # full row of pending ids; the packer would re-emit that row on
                # resume, so drop the buffer and re-pack from the document
                # boundary (docs_consumed stays exact, so tokens do not drift).
                position = StreamPosition(
                    epoch=position.epoch, docs_consumed=position.docs_consumed,
                    rows_in_epoch=position.rows_in_epoch)
            save_checkpoint(
                str(stage_dir / name), m, info.step, info.train_loss, info.val_loss,
                tokenizer_path=f"<smoke>/{fingerprint}",
                optimizer=optimizer, rng_state=capture_rng_state(),
                epoch=info.epoch, tokens_consumed=info.tokens_consumed,
                tokenizer_fingerprint=fingerprint,
                run_metadata=stage_spec_obj.run_metadata(
                    micro_batch=SMOKE_MICRO_BATCH, grad_accum=SMOKE_GRAD_ACCUM,
                    precision="fp32", gpu="CPU",
                    extra={
                        POSITION_METADATA_KEY: position.as_dict(),
                        "skipped_updates": info.skipped_updates,
                        "supervised_tokens": info.supervised_tokens,
                        "smoke": True,
                    },
                ),
                last_grad_norm=info.last_grad_norm,
            )
            return name
        return on_checkpoint

    # --- run A: uninterrupted reference ------------------------------------
    say("run A      : 4 uninterrupted optimizer steps")
    stream_a = RowStream(doc_source(train_docs), seq_len=SMOKE_SEQ_LEN,
                         vocab_size=SMOKE_MODEL_VOCAB, epochs=1)
    optimizer_a, note_a = make_optimizer(model)
    accountant_a = TokenAccountant(budget)
    result_a = train_stage(
        model=model, optimizer=optimizer_a, scaler=None, accountant=accountant_a,
        stream=stream_a, forward_loss=make_forward_loss(model),
        config=make_config(SMOKE_STEPS), stage_key=spec.key, device=device,
        on_checkpoint=make_saver(model, optimizer_a, root / "stage-a", spec,
                                 lambda: stream_a.position),
        evaluate=lambda: evaluate(model), say=say,
    )
    if result_a.skipped_updates:
        raise AssertionError(f"run A skipped {result_a.skipped_updates} update(s)")
    if result_a.tokens_consumed != budget:
        raise AssertionError(
            f"run A consumed {result_a.tokens_consumed:,} of {budget:,} budgeted tokens"
        )
    say(f"run A      : {result_a.steps} steps, {result_a.tokens_consumed:,} tokens, "
        f"train loss {result_a.train_loss:.4f}, ckpt {result_a.last_checkpoint}")

    # --- run B: stop after 2 steps, resume from the checkpoint --------------
    say(f"run B      : {SMOKE_RESUME_AFTER} steps, then resume to the same budget")
    torch.manual_seed(SMOKE_SEED)
    model_b = TalosGPT(model_config)
    optimizer_b, _ = make_optimizer(model_b)
    # The first session is a genuinely short one (its own 2-step budget), which
    # is how a Colab session that gets cut off looks; the resumed session then
    # uses the full stage budget.
    accountant_b = TokenAccountant(tokens_per_step * SMOKE_RESUME_AFTER)
    stream_b = RowStream(doc_source(train_docs), seq_len=SMOKE_SEQ_LEN,
                         vocab_size=SMOKE_MODEL_VOCAB, epochs=1)
    result_b1 = train_stage(
        model=model_b, optimizer=optimizer_b, scaler=None, accountant=accountant_b,
        stream=stream_b, forward_loss=make_forward_loss(model_b),
        config=make_config(SMOKE_STEPS), stage_key=spec.key, device=device,
        on_checkpoint=make_saver(model_b, optimizer_b, root / "stage-b-resume", spec,
                                 lambda: stream_b.position),
        evaluate=None, say=say,
    )
    if result_b1.steps != SMOKE_RESUME_AFTER:
        raise AssertionError(
            f"run B ran {result_b1.steps} steps before the checkpoint, expected "
            f"{SMOKE_RESUME_AFTER}"
        )
    if result_b1.tokens_consumed <= 0:
        raise AssertionError("run B consumed no tokens")
    candidates = list_checkpoint_candidates(str(root / "stage-b-resume"))
    if not candidates:
        raise AssertionError("run B wrote no checkpoint")
    ckpt = load_checkpoint(candidates[0])
    validate_resume_checkpoint(ckpt, SMOKE_PRESET)
    validate_stage_resume(ckpt, spec, tokenizer_fingerprint=fingerprint)

    model_b.load_state_dict(ckpt["model_state_dict"])
    optimizer_b.load_state_dict(ckpt["optimizer_state_dict"])
    restore_rng_state(ckpt["rng_state"])
    accountant_b = TokenAccountant(budget)          # the full stage budget
    accountant_b.consumed = int(ckpt["tokens_consumed"])
    accountant_b.steps = int(ckpt["step"])
    position = StreamPosition.from_dict(
        (ckpt.get("run_metadata") or {}).get(POSITION_METADATA_KEY) or {})
    stream_b2 = RowStream(doc_source(train_docs), seq_len=SMOKE_SEQ_LEN,
                          vocab_size=SMOKE_MODEL_VOCAB, epochs=1, start=position)
    result_b2 = train_stage(
        model=model_b, optimizer=optimizer_b, scaler=None, accountant=accountant_b,
        stream=stream_b2, forward_loss=make_forward_loss(model_b),
        config=make_config(SMOKE_STEPS), stage_key=spec.key, device=device,
        on_checkpoint=make_saver(model_b, optimizer_b, root / "stage-b-resume", spec,
                                 lambda: stream_b2.position),
        evaluate=lambda: evaluate(model_b), say=say,
    )
    if result_b2.tokens_consumed != result_a.tokens_consumed:
        raise AssertionError(
            f"resumed run consumed {result_b2.tokens_consumed:,} tokens, the "
            f"uninterrupted run {result_a.tokens_consumed:,} — token accounting drifted"
        )
    if result_b2.steps != result_a.steps:
        raise AssertionError(
            f"resumed run did {result_b2.steps} steps, the uninterrupted run "
            f"{result_a.steps}"
        )
    loss_delta = abs(float(result_b2.train_loss) - float(result_a.train_loss))
    say(f"resume     : {result_b2.steps} steps, {result_b2.tokens_consumed:,} tokens "
        f"(identical to run A), |Δ train loss| {loss_delta:.2e}")

    # --- evaluation + generation on the checkpointed weights ----------------
    val_loss = evaluate(model_b)
    ppl = perplexity(val_loss)
    prompt_ids = torch.tensor([encode("the of and", bos=True, eos=False)],
                              dtype=torch.long, device=device)
    model_b.eval()
    with torch.no_grad():
        generated = generate(model_b, prompt_ids, max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                             greedy=True, eos_token_id=tokenizer.eos_id)
    report = analyze_generation(generated, label="smoke/greedy", prompt="the of and",
                               max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                               eos_id=tokenizer.eos_id)
    model_b.train()
    say(f"evaluation : val loss {val_loss:.4f}, perplexity {ppl:.2f}, "
        f"{report.generated_tokens} generated tokens, distinct-2 {report.distinct_2:.2f}")

    # --- tracker + metrics payloads (dry run: local files, no network) ------
    tracker = TrackerPublisher(
        "PrometheanStudio/talos-training-tracker", dry_run=True, min_interval_s=0.0,
        local_dir=str(root / "tracker"), gpu="CPU", stage=spec.key,
        token_target=spec.token_budget, tokens_per_step=tokens_per_step,
        model_name=MODEL_DISPLAY_NAME, parameters=model.num_parameters(),
        vocab_size=model_config.vocab_size, sequence_length=SMOKE_SEQ_LEN,
        skipped_updates=result_b2.skipped_updates,
    )
    metrics = MetricsSync("smoke/talos-metrics", dry_run=True, min_interval_s=0.0,
                          local_path=str(root / "tracker" / "metrics.jsonl"))
    row = build_metrics_row(
        step=result_b2.steps, epoch=result_b2.epoch,
        tokens_so_far=result_b2.tokens_consumed, total_tokens=spec.token_budget,
        elapsed_s=result_b2.elapsed_s, train_loss=result_b2.train_loss,
        val_loss=val_loss, checkpoint=result_b2.last_checkpoint, gpu="CPU",
        model=MODEL_DISPLAY_NAME, params=model.num_parameters(),
        vocab=model_config.vocab_size, seq_len=SMOKE_SEQ_LEN, stage=spec.key,
        skipped_updates=result_b2.skipped_updates,
    )
    tracker.note_checkpoint(result_b2.steps, result_b2.tokens_consumed)
    tracker.push(row)
    metrics.maybe_push(row)
    metrics.flush()
    display_html = render_display_table(row)
    status_payload = json.loads(Path(tracker.status_path).read_text())
    say(f"tracker    : status.json model_name={status_payload['model_name']!r}, "
        f"skipped_updates={status_payload['skipped_updates']}, "
        f"display rows {display_html.count('<tr>')}")

    # --- Stage B: masked loss + stage discipline ---------------------------
    chat_docs = synthetic_chat_documents(SMOKE_DOCS)

    def chat_source(epoch: int, skip: int):
        for messages in list(chat_docs)[int(skip):]:
            ids, mask = encode_chat_document(messages, encode,
                                             eos_id=tokenizer.eos_id, mask_prompt=True)
            if ids:
                yield ids, mask

    b_tokens_per_step = tokens_per_step
    b_tokens_per_epoch = b_tokens_per_step * 2
    b_budget = stage_b_budget(tokens_per_epoch=b_tokens_per_epoch,
                              epochs=1, seq_len=SMOKE_SEQ_LEN)
    spec_b = stage_spec(STAGE_B, token_budget=b_budget, seq_len=SMOKE_SEQ_LEN,
                        seed=SMOKE_SEED, dataset_id="smoke/synthetic-chat",
                        dataset_config=None, dataset_revision="local",
                        train_split="train_sft", val_split="test_sft")
    torch.manual_seed(SMOKE_SEED)
    model_b_stage = TalosGPT(model_config)
    model_b_stage.load_state_dict(ckpt["model_state_dict"])   # Stage A -> Stage B init
    optimizer_sb, _ = make_optimizer(model_b_stage)
    accountant_sb = TokenAccountant(b_budget)
    config_sb = TrainLoopConfig(
        micro_batch=SMOKE_MICRO_BATCH, grad_accum=SMOKE_GRAD_ACCUM,
        seq_len=SMOKE_SEQ_LEN, total_steps=SMOKE_STAGE_B_STEPS,
        warmup_steps=0, peak_lr=spec_b.peak_lr, min_lr_ratio=spec_b.min_lr_ratio,
        grad_clip=spec_b.grad_clip, log_every=1, eval_every=10**6,
        checkpoint_every=1, checkpoint_every_min=0.0, mask_prompt_tokens=True,
        tokens_per_epoch=b_tokens_per_epoch,
    )
    stream_sb = RowStream(chat_source, seq_len=SMOKE_SEQ_LEN,
                          vocab_size=SMOKE_MODEL_VOCAB, epochs=1)
    result_sb = train_stage(
        model=model_b_stage, optimizer=optimizer_sb, scaler=None,
        accountant=accountant_sb,
        stream=stream_sb, forward_loss=make_forward_loss(model_b_stage),
        config=config_sb, stage_key=spec_b.key, device=device,
        on_checkpoint=make_saver(model_b_stage, optimizer_sb, root / "stage-b", spec_b,
                                 lambda: stream_sb.position),
        evaluate=None, say=say,
    )
    if result_sb.supervised_tokens >= result_sb.tokens_consumed:
        raise AssertionError(
            f"Stage B supervised {result_sb.supervised_tokens:,} of "
            f"{result_sb.tokens_consumed:,} tokens — prompt masking is not applied"
        )
    sb_ckpt = load_checkpoint(str(root / "stage-b" / result_sb.last_checkpoint))
    rejected = False
    try:
        validate_stage_resume(sb_ckpt, spec, tokenizer_fingerprint=fingerprint)
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("a Stage-B checkpoint was accepted by a Stage-A resume")
    say(f"stage B    : {result_sb.steps} steps, {result_sb.tokens_consumed:,} tokens "
        f"({result_sb.supervised_tokens:,} supervised), epochs "
        f"{result_sb.epochs_completed:.2f}; Stage-A resume refused it correctly")

    result = SmokeResult(
        out_dir=str(root),
        params=model.num_parameters(),
        vocab=model_config.vocab_size,
        tokenizer_fingerprint=fingerprint,
        full_steps=result_a.steps,
        full_tokens=result_a.tokens_consumed,
        full_train_loss=float(result_a.train_loss),
        resumed_steps=result_b2.steps,
        resumed_tokens=result_b2.tokens_consumed,
        resumed_train_loss=float(result_b2.train_loss),
        resume_loss_delta=loss_delta,
        checkpoint=result_b2.last_checkpoint,
        checkpoint_format=str(ckpt["format"]),
        val_loss=None if val_loss is None else float(val_loss),
        perplexity=ppl,
        generated_tokens=int(report.generated_tokens),
        distinct_2=float(report.distinct_2),
        stage_b_tokens=result_sb.tokens_consumed,
        stage_b_supervised_tokens=result_sb.supervised_tokens,
        stage_b_checkpoint=result_sb.last_checkpoint,
        stage_b_rejected_by_stage_a=rejected,
        tracker_status_path=tracker.status_path,
        tracker_history_path=tracker.history_path,
        metrics_path=metrics.local_path,
        tracker_status=status_payload,
        display_rows=display_html.count("<tr>"),
        elapsed_s=time.time() - started,
        notes=[note_a, f"tokenizer vocab {tokenizer.vocab_size}",
               f"1B pins: {TALOS_1B_49K_SHA256[:12]}… vocab {TALOS_1B_49K_VOCAB:,} "
               f"merges {TALOS_1B_49K_MERGES:,}"],
    )
    say(f"smoke run  : PASSED in {result.elapsed_s:.1f}s — artifacts under {root}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None,
                        help="directory for the smoke artifacts (default: a temp dir)")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as JSON instead of text")
    args = parser.parse_args(argv)
    result = run_smoke(args.out, say=(lambda _line: None) if args.json else print)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
