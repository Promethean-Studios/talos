"""Dataset pipeline for the Talos 1B notebook — pure logic, no torch/datasets.

Two stages need two different shapes of data, and both must be *resumable at an
exact position* (the owner's audit item: "a resumed run must not silently
restart the dataset from its first epoch"):

* **Stage A** streams a general-purpose corpus (``allenai/c4``, config ``en``).
  Each document is encoded as ``<bos> … <eos>``; every token contributes to the
  loss; documents are concatenated and sliced into ``seq_len`` rows, so a row can
  span two documents (standard packing, no padding waste).
* **Stage B** streams ``HuggingFaceH4/ultrachat_200k``. Each dialog becomes
  ``User: …`` / ``Assistant: …`` segments; the loss mask is ``0`` on user (and
  system) tokens and ``1`` on assistant tokens **and** on the EOS that ends each
  assistant turn, so the model learns *when to stop* instead of learning to
  produce user turns.

Position tracking: a checkpoint stores ``epoch``, ``docs_consumed``,
``rows_in_epoch`` and the packer's leftover ids (``pending_ids`` /
``pending_mask``, at most ``seq_len - 1`` each). That quadruple pins the exact
place in the stream, so a resume continues mid-epoch at the same token boundary
as the uninterrupted run instead of re-reading the epoch or misaligning the
packing by a partial row.

Nothing here imports ``torch`` or ``datasets``: the caller injects a
``doc_source(epoch, skip_docs)`` callable and an ``encode(text, bos, eos)``
callable, which is what makes the whole pipeline testable on CPU
(``tests/test_notebook_data.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

#: ``(ids, loss_mask_or_None)`` — one encoded document.
EncodedDoc = Tuple[List[int], Optional[List[int]]]

#: ``doc_source(epoch, skip_docs) -> Iterable[EncodedDoc]``: the caller's
#: dataset stream for one epoch, with the first ``skip_docs`` raw documents
#: skipped (that is what makes a mid-epoch resume cheap: documents before the
#: checkpoint are never tokenized again).
DocSource = Callable[[int, int], Iterable[EncodedDoc]]

#: ``encode(text, bos, eos) -> List[int]`` — the tokenizer's encoder.
Encoder = Callable[..., Sequence[int]]

#: Roles the chat formatter understands. Unknown roles are kept, not dropped.
ROLE_ALIASES: Dict[str, str] = {
    "user": "user",
    "human": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}

#: Separator between turns inside one document.
TURN_SEP = "\n\n"


def normalize_role(role: Any) -> str:
    """Map a dataset role onto ``user``/``assistant``/``system``/``tool``.

    Unknown or missing roles become ``"unknown"`` rather than being dropped: a
    malformed row should degrade into a visible odd document, never silently
    vanish from the token stream (silent drops make token accounting lie).
    """
    key = str(role or "").strip().lower()
    return ROLE_ALIASES.get(key, key or "unknown")


def chat_segments(messages: Iterable[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """``messages`` -> ``[(role, text), …]`` with empty turns removed.

    ultrachat_200k rows are ``{"messages": [{"role": …, "content": …}, …]}``;
    the ``prompt`` column is the same first turn, so it is not used twice.
    """
    segments: List[Tuple[str, str]] = []
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        role = normalize_role(message.get("role"))
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            continue
        segments.append((role, text))
    return segments


def format_chat_document(messages: Iterable[Mapping[str, Any]]) -> str:
    """Linearize a dialog into ``User: …\\n\\nAssistant: …`` (for inspection)."""
    return TURN_SEP.join(f"{role.capitalize()}: {text}"
                         for role, text in chat_segments(messages))


def encode_chat_document(
    messages: Iterable[Mapping[str, Any]],
    encode: Encoder,
    *,
    eos_id: int,
    mask_prompt: bool = True,
) -> EncodedDoc:
    """Encode one dialog to ``(ids, loss_mask)`` with assistant-only supervision.

    Layout per turn: ``"<Role>: <text>"`` encoded without special tokens, then
    ``eos_id`` (assistant turns only). A ``<bos>`` is prepended by encoding the
    first segment with ``bos=True``. ``loss_mask`` is ``1`` on every assistant
    token and on the EOS that closes an assistant turn, ``0`` on user/system
    tokens and on the EOS that closes a user turn (the model must not be taught
    to emit user text).

    With ``mask_prompt=False`` the mask is ``None`` (every token supervised),
    which is what Stage A uses and what the plain-text/continued-pretraining
    fallback for chat data would use.
    """
    segments = chat_segments(messages)
    ids: List[int] = []
    mask: List[int] = []
    if not segments:
        return ids, (None if not mask_prompt else mask)
    for index, (role, text) in enumerate(segments):
        piece = list(encode(f"{role.capitalize()}: {text}",
                            bos=(index == 0), eos=False))
        supervised = 1 if (role == "assistant" or not mask_prompt) else 0
        ids.extend(piece)
        mask.extend([supervised] * len(piece))
        if role == "assistant":
            ids.append(int(eos_id))
            mask.append(1)
    if not mask_prompt:
        return ids, None
    return ids, mask


def encode_text_document(
    text: str,
    encode: Encoder,
    *,
    bos: bool = True,
    eos: bool = True,
) -> EncodedDoc:
    """Encode one plain document (Stage A) — every token is supervised."""
    ids = [int(token) for token in encode(text, bos=bos, eos=eos)]
    return ids, None


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackedRow:
    """One training row: ``seq_len`` token ids and an optional 0/1 loss mask."""

    ids: List[int]
    mask: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.ids)


def validate_ids(ids: Sequence[int], vocab_size: int, *, where: str = "document") -> None:
    """Bounds-check ids against the model vocab, with context on failure.

    Repo convention (``data.tokenized.validate_id_array``): an out-of-range id
    must fail at the data source, naming the document, never as a device-side
    assert in the middle of a step.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    for token in ids:
        value = int(token)
        if value < 0 or value >= vocab_size:
            raise ValueError(
                f"token id {value} outside [0, {vocab_size}) in {where} — the "
                "tokenizer does not match the model vocab"
            )


def pack_pairs(
    docs: Iterable[EncodedDoc],
    *,
    seq_len: int,
    vocab_size: int,
    pad_id: int = 0,
    drop_last: bool = True,
    buffer: Optional[EncodedDoc] = None,
) -> Iterator[Tuple[EncodedDoc, EncodedDoc]]:
    """Concatenate encoded docs and slice them into ``seq_len`` rows.

    Yields ``(row, leftover)``: ``row`` is the completed ``(ids, mask)`` pair and
    ``leftover`` is the packer's remaining buffer **after** this row. The caller
    that needs resumability stores the final ``leftover`` in the checkpoint, so
    it can seed :class:`RowStream` and continue at the same token boundary.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not (0 <= pad_id < max(vocab_size, 1)):
        raise ValueError(f"pad_id {pad_id} outside [0, {vocab_size})")
    ids: List[int] = list(buffer[0]) if buffer else []
    mask: Optional[List[int]] = list(buffer[1]) if buffer and buffer[1] is not None else None
    for doc_ids, doc_mask in docs:
        validate_ids(doc_ids, vocab_size)
        ids.extend(int(token) for token in doc_ids)
        if mask is not None or doc_mask is not None:
            if mask is None:
                mask = [1] * (len(ids) - len(doc_ids))
            mask.extend(list(doc_mask) if doc_mask is not None else [1] * len(doc_ids))
        while len(ids) >= seq_len:
            row_ids = ids[:seq_len]
            row_mask = None if mask is None else mask[:seq_len]
            ids = ids[seq_len:]
            if mask is not None:
                mask = mask[seq_len:]
            yield (row_ids, row_mask), (list(ids), None if mask is None else list(mask))
    if ids and not drop_last:
        pad = seq_len - len(ids)
        row_ids = ids + [pad_id] * pad
        row_mask = None if mask is None else mask + [0] * pad
        yield (row_ids, row_mask), ([], None)


# ---------------------------------------------------------------------------
# Resumable stream
# ---------------------------------------------------------------------------
@dataclass
class StreamPosition:
    """Where the stream is — the exact state a checkpoint records.

    ``epoch`` is 1-based. ``docs_consumed``/``rows_in_epoch`` count inside the
    current epoch. ``pending_ids``/``pending_mask`` are the packer's leftover
    (``< seq_len`` tokens) that have been read but not yet emitted as a row.
    """

    epoch: int = 1
    docs_consumed: int = 0
    rows_in_epoch: int = 0
    pending_ids: List[int] = field(default_factory=list)
    pending_mask: Optional[List[int]] = None

    def tokens_in_epoch(self, seq_len: int) -> int:
        """Tokens of this epoch already consumed (emitted rows + pending)."""
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        return int(self.rows_in_epoch) * int(seq_len) + len(self.pending_ids)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready position (what lands in ``run_metadata.dataset_position``)."""
        return {
            "epoch": int(self.epoch),
            "docs_consumed": int(self.docs_consumed),
            "rows_in_epoch": int(self.rows_in_epoch),
            "pending_ids": [int(token) for token in self.pending_ids],
            "pending_mask": (None if self.pending_mask is None
                             else [int(value) for value in self.pending_mask]),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StreamPosition":
        """Rebuild a position from :meth:`as_dict` output (defensive on types)."""
        mask = payload.get("pending_mask")
        return cls(
            epoch=int(payload.get("epoch") or 1),
            docs_consumed=int(payload.get("docs_consumed") or 0),
            rows_in_epoch=int(payload.get("rows_in_epoch") or 0),
            pending_ids=[int(token) for token in (payload.get("pending_ids") or [])],
            pending_mask=None if mask is None else [int(value) for value in mask],
        )


class RowStream:
    """Deterministic, resumable stream of packed rows over ``epochs`` epochs.

    ``doc_source(epoch, skip_docs)`` must be deterministic for a given
    ``(epoch, skip_docs)``: the notebook seeds each epoch's shuffle with
    ``seed + epoch`` and skips ``skip_docs`` raw rows with ``itertools.islice``
    before tokenizing, so resuming re-reads only the documents it must.

    Iterating updates :attr:`position` in place — the training loop saves
    ``position.as_dict()`` in every checkpoint.
    """

    def __init__(
        self,
        doc_source: DocSource,
        *,
        seq_len: int,
        vocab_size: int,
        epochs: int = 1,
        start: Optional[StreamPosition] = None,
        pad_id: int = 0,
        drop_last: bool = True,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.doc_source = doc_source
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.epochs = int(epochs)
        self.pad_id = int(pad_id)
        self.drop_last = bool(drop_last)
        self.position = start or StreamPosition()
        if self.position.epoch < 1:
            raise ValueError("start position epoch must be >= 1")
        if len(self.position.pending_ids) >= self.seq_len:
            raise ValueError("start position pending_ids must be shorter than seq_len")

    # -- iteration ----------------------------------------------------------
    def _counted_source(self, position: StreamPosition) -> DocSource:
        """``doc_source`` that advances ``position.docs_consumed`` per document.

        A document is counted as soon as the packer pulls it (the packer then
        extends its buffer with *all* of that document's ids before yielding any
        row), so ``docs_consumed`` + ``pending_ids`` is an exact position even
        when a checkpoint lands mid-document.
        """

        def source(epoch: int, skip_docs: int) -> Iterator[EncodedDoc]:
            for doc in self.doc_source(int(epoch), int(skip_docs)):
                position.docs_consumed += 1
                yield doc

        return source

    def __iter__(self) -> Iterator[PackedRow]:
        position = self.position
        while position.epoch <= self.epochs:
            start_epoch = position.epoch
            skip = position.docs_consumed
            buffer: EncodedDoc = (
                list(position.pending_ids),
                None if position.pending_mask is None else list(position.pending_mask),
            )
            for row, leftover in pack_pairs(
                self._counted_source(position)(start_epoch, skip),
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                pad_id=self.pad_id,
                drop_last=self.drop_last,
                buffer=buffer,
            ):
                position.pending_ids, position.pending_mask = leftover
                position.rows_in_epoch += 1
                yield PackedRow(ids=list(row[0]),
                                mask=None if row[1] is None else list(row[1]))
            # epoch finished: the leftover buffer is dropped (a new epoch starts
            # on a fresh token stream, exactly like an uninterrupted run).
            position.epoch += 1
            position.docs_consumed = 0
            position.rows_in_epoch = 0
            position.pending_ids = []
            position.pending_mask = None


def take_rows(rows: Iterable[PackedRow], count: int) -> List[PackedRow]:
    """First ``count`` rows of ``rows`` (finite helper for validation sets)."""
    if count < 0:
        raise ValueError("count must be >= 0")
    taken: List[PackedRow] = []
    for row in rows:
        taken.append(row)
        if len(taken) >= count:
            break
    return taken


# ---------------------------------------------------------------------------
# Repetition / size arithmetic (the "don't cycle a small set silently" guard)
# ---------------------------------------------------------------------------
def estimate_tokens_per_epoch(rows: int, avg_tokens_per_row: float) -> int:
    """Tokens in one pass over a dataset: ``rows × measured tokens/row``."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    if avg_tokens_per_row <= 0:
        raise ValueError("avg_tokens_per_row must be positive — measure it, do not guess")
    return int(round(float(rows) * float(avg_tokens_per_row)))


def estimate_passes(token_budget: int, tokens_per_epoch: int) -> float:
    """How many passes over the dataset a budget implies."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if tokens_per_epoch <= 0:
        raise ValueError("tokens_per_epoch must be positive")
    return float(token_budget) / float(tokens_per_epoch)


def epochs_for_budget(token_budget: int, tokens_per_epoch: int) -> int:
    """Smallest epoch count that covers ``token_budget`` (at least 1)."""
    return max(1, int(math.ceil(estimate_passes(token_budget, tokens_per_epoch))))


def assert_repeat_exposure(
    passes: float,
    *,
    max_passes: float,
    what: str = "dataset",
) -> None:
    """Refuse to cycle a small dataset more than ``max_passes`` times.

    The old notebook re-cycled ultrachat_200k ~100 times to reach 20B tokens.
    That is not pretraining, it is memorisation with extra steps, so the budget
    must be justified by the data and the compute — and this function makes an
    unjustified cycle count fail loudly instead of running.
    """
    if passes <= 0:
        raise ValueError("passes must be positive")
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    if passes > max_passes:
        raise ValueError(
            f"this budget would make {passes:.2f} passes over the {what}, more "
            f"than the {max_passes:g} allowed — lower the token budget or use a "
            "larger corpus; do not re-cycle a small dataset without justification"
        )


__all__ = [
    "DocSource",
    "EncodedDoc",
    "Encoder",
    "PackedRow",
    "ROLE_ALIASES",
    "RowStream",
    "StreamPosition",
    "TURN_SEP",
    "assert_repeat_exposure",
    "chat_segments",
    "encode_chat_document",
    "encode_text_document",
    "epochs_for_budget",
    "estimate_passes",
    "estimate_tokens_per_epoch",
    "format_chat_document",
    "normalize_role",
    "pack_pairs",
    "take_rows",
    "validate_ids",
]
