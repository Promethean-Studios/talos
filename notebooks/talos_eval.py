"""Honest evaluation for the Talos 1B notebook — pure logic, no torch.

"Does the loss go down" is not evidence that a model can hold a conversation.
This module turns generations into the numbers that make a from-scratch 1B
model's real quality legible, so the notebook cannot imply more than it has:

* validation loss **and** perplexity (perplexity = ``exp(loss)``);
* whether each generation stopped on EOS (the model's own turn-ending token) or
  ran to the length cap;
* repetition/degeneration: distinct-1, distinct-2, and the most-repeated 3-gram
  (a model that loops "the the the" scores badly here even with a low loss);
* a computed verdict that states the tokens-per-parameter it was trained on
  against Chinchilla's ~20, and refuses to call the result a working chatbot.

Everything is a pure function of token ids (``tests/test_notebook_eval.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Conversational prompts for the generation smoke test. General-purpose, no
#: domain specialisation (owner: "a simple, general-purpose conversational
#: chatbot"): factual questions, an instruction, a small reasoning step, a
#: creative ask, and a follow-up that depends on the previous turn's context.
#: Each is formatted exactly the way Stage B trains (``User: …\\n\\nAssistant:``).
CHAT_EVAL_PROMPTS: Tuple[Dict[str, str], ...] = (
    {"label": "factual", "prompt": "User: What is the capital of France?\n\nAssistant:"},
    {"label": "explain", "prompt": "User: Explain what a computer is to a ten year old.\n\nAssistant:"},
    {"label": "instruction", "prompt": "User: Give me three tips for learning to cook.\n\nAssistant:"},
    {"label": "reasoning", "prompt": "User: If I have 3 apples and buy 2 more, how many do I have?\n\nAssistant:"},
    {"label": "creative", "prompt": "User: Write the first two sentences of a story about a lighthouse.\n\nAssistant:"},
    {"label": "smalltalk", "prompt": "User: Hello! How are you today?\n\nAssistant:"},
)

#: A from-scratch 1B needs roughly 20 tokens/parameter to be a competent base
#: model (Chinchilla). Colab-scale budgets land far below that, and the verdict
#: says so instead of hiding it.
CHINCHILLA_TOKENS_PER_PARAM = 20.0

#: Thresholds that flag a *degenerate* generation rather than merely a weak one.
MAX_ACCEPTABLE_NGRAM_REPEAT = 4      # the same 3-gram appearing more than 4x
MIN_ACCEPTABLE_DISTINCT_2 = 0.20     # fewer distinct bigrams than this = looping


# ---------------------------------------------------------------------------
# Repetition / degeneration metrics
# ---------------------------------------------------------------------------
def ngrams(tokens: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    """All ``n``-grams of ``tokens`` as tuples (empty when too short)."""
    if n <= 0:
        raise ValueError("n must be positive")
    seq = [int(token) for token in tokens]
    if len(seq) < n:
        return []
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(tokens: Sequence[int], n: int) -> float:
    """Distinct-``n``: unique ``n``-grams / all ``n``-grams (1.0 = no repeats).

    Returns ``1.0`` for a sequence shorter than ``n`` (nothing has repeated yet)
    and ``0.0`` for an empty sequence, so a length-capped stub is not credited.
    """
    if not tokens:
        return 0.0
    grams = ngrams(tokens, n)
    if not grams:
        return 1.0
    return len(set(grams)) / float(len(grams))


def most_repeated_ngram(tokens: Sequence[int], n: int = 3) -> Tuple[Tuple[int, ...], int]:
    """The most frequent ``n``-gram and its count (``((), 0)`` when none)."""
    grams = ngrams(tokens, n)
    if not grams:
        return (), 0
    counts: Dict[Tuple[int, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    gram, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return gram, count


@dataclass(frozen=True)
class GenerationReport:
    """One generation, measured."""

    label: str
    prompt: str
    generated_tokens: int
    max_new_tokens: int
    eos_stopped: bool
    distinct_1: float
    distinct_2: float
    max_ngram_repeat: int
    repeated_ngram: Tuple[int, ...]
    degenerate: bool

    @property
    def stopped_early(self) -> bool:
        return self.eos_stopped and self.generated_tokens < self.max_new_tokens

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "prompt": self.prompt,
            "generated_tokens": int(self.generated_tokens),
            "max_new_tokens": int(self.max_new_tokens),
            "eos_stopped": bool(self.eos_stopped),
            "distinct_1": round(float(self.distinct_1), 4),
            "distinct_2": round(float(self.distinct_2), 4),
            "max_ngram_repeat": int(self.max_ngram_repeat),
            "repeated_ngram": list(self.repeated_ngram),
            "degenerate": bool(self.degenerate),
        }


def analyze_generation(
    generated_ids: Sequence[int],
    *,
    label: str = "",
    prompt: str = "",
    max_new_tokens: int,
    eos_id: int,
) -> GenerationReport:
    """Measure one generation: length, EOS behaviour, repetition, degeneration.

    ``generated_ids`` are the **new** tokens only (the prompt is not counted):
    EOS stops count only if the final token is ``eos_id``; the model repeating a
    prompt token is not degeneration, the model looping its own output is.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    ids = [int(token) for token in generated_ids]
    eos_stopped = bool(ids) and ids[-1] == int(eos_id)
    body = ids[:-1] if eos_stopped else ids
    gram, count = most_repeated_ngram(body, 3)
    distinct_2 = distinct_n(body, 2)
    degenerate = bool(body) and (
        count > MAX_ACCEPTABLE_NGRAM_REPEAT or distinct_2 < MIN_ACCEPTABLE_DISTINCT_2
    )
    return GenerationReport(
        label=str(label),
        prompt=str(prompt),
        generated_tokens=len(ids),
        max_new_tokens=int(max_new_tokens),
        eos_stopped=eos_stopped,
        distinct_1=distinct_n(body, 1),
        distinct_2=distinct_2,
        max_ngram_repeat=int(count),
        repeated_ngram=tuple(int(token) for token in gram),
        degenerate=degenerate,
    )


# ---------------------------------------------------------------------------
# Loss / perplexity
# ---------------------------------------------------------------------------
def perplexity(loss: Optional[float]) -> Optional[float]:
    """``exp(loss)``; ``None`` for a missing loss, ``inf`` for a huge one."""
    if loss is None:
        return None
    value = float(loss)
    if value != value:
        return None
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def tokens_per_parameter(tokens_trained: int, parameters: int) -> float:
    """The ratio that decides what a model can honestly be."""
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if tokens_trained < 0:
        raise ValueError("tokens_trained must be >= 0")
    return float(tokens_trained) / float(parameters)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def summarize(reports: Sequence[GenerationReport]) -> Dict[str, Any]:
    """Aggregate generation reports into rates the notebook prints."""
    count = len(reports)
    if count == 0:
        return {"count": 0, "eos_rate": None, "degenerate_rate": None,
                "mean_distinct_2": None, "mean_generated_tokens": None}
    return {
        "count": count,
        "eos_rate": sum(1 for report in reports if report.eos_stopped) / count,
        "degenerate_rate": sum(1 for report in reports if report.degenerate) / count,
        "mean_distinct_2": sum(report.distinct_2 for report in reports) / count,
        "mean_generated_tokens": sum(report.generated_tokens for report in reports) / count,
    }


def honest_verdict(
    *,
    stage_a_tokens: int,
    parameters: int,
    val_loss: Optional[float],
    reports: Sequence[GenerationReport],
) -> str:
    """A plain-language verdict computed from the numbers — no marketing.

    States the tokens-per-parameter actually trained, what that buys, what it
    does not, and what the generations showed. It never says "working chatbot".
    """
    ratio = tokens_per_parameter(stage_a_tokens, parameters)
    ppl = perplexity(val_loss)
    summary = summarize(reports)
    ppl_text = "not measured" if ppl is None else (
        "inf" if ppl == float("inf") else f"{ppl:,.1f}")
    lines = [
        f"Trained on {stage_a_tokens:,} tokens of pretraining = {ratio:.3f} "
        f"tokens/parameter (Chinchilla-scale is ~{CHINCHILLA_TOKENS_PER_PARAM:.0f}).",
        f"Validation loss {val_loss if val_loss is not None else 'n/a'} "
        f"(perplexity {ppl_text}).",
    ]
    if summary["count"]:
        lines.append(
            f"{summary['count']} prompts: EOS stop on {summary['eos_rate']:.0%}, "
            f"degenerate repetition on {summary['degenerate_rate']:.0%}, "
            f"mean distinct-2 {summary['mean_distinct_2']:.2f}."
        )
    if ratio < 1.0:
        lines.append(
            "At under ~1 token/parameter this is an early-training artifact: "
            "expect locally plausible word sequences, frequent factual errors, "
            "topic drift and mimicry of the 'User:/Assistant:' format. It will "
            "not hold a coherent conversation."
        )
    elif ratio < CHINCHILLA_TOKENS_PER_PARAM / 2:
        lines.append(
            "This is a small-scale base model: it can produce fluent spans and "
            "follow the chat format after Stage B, but expect factual errors, "
            "weak long-range coherence and unreliable instruction following. "
            "It is not a production chatbot."
        )
    else:
        lines.append(
            "This is a near-Chinchilla-scale base model. Quality still depends on "
            "data and tuning; the generation numbers above are the evidence, not "
            "the loss curve."
        )
    lines.append(
        "Judge it by the numbers above: a completed notebook is not evidence of "
        "a working chatbot."
    )
    return "\n".join(lines)


def format_eval_table(records: Iterable[Mapping[str, Any]]) -> str:
    """Plain-text table of prompt → response → flags (notebook output)."""
    lines: List[str] = []
    for record in records:
        lines.append("-" * 78)
        lines.append(f"[{record.get('label', '')}] prompt : {record.get('prompt', '')!r}")
        lines.append(f"            answer : {record.get('answer', '')!r}")
        lines.append(
            "            tokens={tokens} eos_stop={eos} distinct-1={d1} "
            "distinct-2={d2} max-3gram-repeat={rep}{flag}".format(
                tokens=record.get("generated_tokens"),
                eos=record.get("eos_stopped"),
                d1=record.get("distinct_1"),
                d2=record.get("distinct_2"),
                rep=record.get("max_ngram_repeat"),
                flag="  <-- DEGENERATE" if record.get("degenerate") else "",
            )
        )
    lines.append("-" * 78)
    return "\n".join(lines)


__all__ = [
    "CHAT_EVAL_PROMPTS",
    "CHINCHILLA_TOKENS_PER_PARAM",
    "GenerationReport",
    "MAX_ACCEPTABLE_NGRAM_REPEAT",
    "MIN_ACCEPTABLE_DISTINCT_2",
    "analyze_generation",
    "distinct_n",
    "format_eval_table",
    "honest_verdict",
    "most_repeated_ngram",
    "ngrams",
    "perplexity",
    "summarize",
    "tokens_per_parameter",
]
