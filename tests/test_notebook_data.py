"""Tests for the notebook's dataset pipeline (``talos_data``).

Covers the owner's audit items: split/format correctness, role handling, loss
masking, EOS placement, packing, deterministic shuffling, repeat exposure and —
the important one — that a resumed stream continues at the same token boundary
as an uninterrupted run.
"""
from __future__ import annotations

import os
import sys

import pytest

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import talos_data as data  # noqa: E402

VOCAB = 64
EOS = 2


def encode(text, bos=True, eos=True):
    """Deterministic toy encoder: one id per word (bounded by VOCAB)."""
    ids = [3 + (abs(hash(word)) % (VOCAB - 4)) for word in text.split()]
    return ([1] if bos else []) + ids + ([EOS] if eos else [])


# ---------------------------------------------------------------------------
# Roles and chat formatting
# ---------------------------------------------------------------------------
def test_roles_are_normalized_and_unknown_roles_survive():
    assert data.normalize_role("Human") == "user"
    assert data.normalize_role("GPT") == "assistant"
    assert data.normalize_role("SYSTEM") == "system"
    assert data.normalize_role(None) == "unknown"
    assert data.normalize_role("moderator") == "moderator"


def test_chat_segments_drops_empty_turns_but_keeps_order():
    messages = [
        {"role": "user", "content": "  hello  "},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "hi there"},
        {"content": "no role"},
    ]
    assert data.chat_segments(messages) == [
        ("user", "hello"), ("assistant", "hi there"), ("unknown", "no role"),
    ]


def test_format_chat_document_is_the_training_format():
    text = data.format_chat_document([
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
    ])
    assert text == "User: hi\n\nAssistant: hello"


def test_encode_chat_document_masks_user_turns_and_ends_assistant_turns_with_eos():
    ids, mask = data.encode_chat_document(
        [{"role": "user", "content": "hello there"},
         {"role": "assistant", "content": "hi"},
         {"role": "user", "content": "bye"},
         {"role": "assistant", "content": "goodbye"}],
        encode, eos_id=EOS)
    assert len(ids) == len(mask)
    assert ids[0] == 1                      # BOS at the document start
    assert mask[0] == 0                     # ... and it is not supervised
    assert set(mask) == {0, 1}
    # Each assistant turn's closing EOS is supervised (the model learns to stop).
    assistant_eos = [i for i, token in enumerate(ids) if token == EOS and mask[i] == 1]
    assert len(assistant_eos) == 2
    # User turns contribute no supervised tokens at all.
    assert mask.count(0) > 0 and mask.count(1) > 0


def test_encode_chat_document_without_masking_returns_no_mask():
    ids, mask = data.encode_chat_document(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        encode, eos_id=EOS, mask_prompt=False)
    assert mask is None and ids


def test_encode_text_document_is_fully_supervised():
    ids, mask = data.encode_text_document("a b c", encode)
    assert ids[0] == 1 and ids[-1] == EOS
    assert mask is None


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def _rows(docs, seq_len, vocab=VOCAB, drop_last=True):
    return [row for row, _ in data.pack_pairs(docs, seq_len=seq_len, vocab_size=vocab,
                                              drop_last=drop_last)]


def test_packing_joins_documents_and_slices_fixed_rows():
    docs = [([5] * 5, None), ([6] * 5, None), ([7] * 5, None)]
    rows = _rows(docs, 10)
    # 15 tokens -> one full row; the 5-token tail is dropped (drop_last=True).
    assert [len(ids) for ids, _ in rows] == [10]
    assert rows[0][0] == [5] * 5 + [6] * 5
    # ... and kept (padded) when the caller asks for it.
    padded = _rows(docs, 10, drop_last=False)
    assert [len(ids) for ids, _ in padded] == [10, 10]
    assert padded[1][0] == [7] * 5 + [0] * 5


def test_packing_pads_only_when_asked():
    docs = [([5] * 4, None)]
    assert len(_rows(docs, 10)) == 0
    rows = _rows(docs, 10, drop_last=False)
    assert rows[0][0] == [5] * 4 + [0] * 6


def test_packing_carries_the_loss_mask_alongside_the_ids():
    docs = [([5] * 4, [0] * 4), ([6] * 8, [1] * 8)]
    rows = _rows(docs, 6)
    assert rows[0] == ([5] * 4 + [6] * 2, [0] * 4 + [1] * 2)
    assert rows[1] == ([6] * 6, [1] * 6)


def test_packing_rejects_out_of_vocab_ids_with_context():
    with pytest.raises(ValueError, match="outside"):
        _rows([([VOCAB], None)], 4)


def test_validate_ids_reports_the_where():
    with pytest.raises(ValueError, match="document 7"):
        data.validate_ids([1, 999], VOCAB, where="document 7")


# ---------------------------------------------------------------------------
# Resumable stream
# ---------------------------------------------------------------------------
def _source(total_docs=40, doc_len=7):
    """Deterministic document source: doc i is doc_len ids of value i+3."""
    def source(epoch, skip):
        for index in range(skip, total_docs):
            yield ([3 + (index % 50)] * doc_len, [1] * doc_len)
    return source


def test_stream_is_deterministic_for_a_seed_and_epoch():
    a = [row.ids for row in data.RowStream(_source(), seq_len=10, vocab_size=VOCAB, epochs=1)]
    b = [row.ids for row in data.RowStream(_source(), seq_len=10, vocab_size=VOCAB, epochs=1)]
    assert a == b and a


def test_stream_position_round_trips_through_json():
    position = data.StreamPosition(epoch=2, docs_consumed=5, rows_in_epoch=3,
                                   pending_ids=[1, 2], pending_mask=[1, 0])
    restored = data.StreamPosition.from_dict(position.as_dict())
    assert restored == position
    assert restored.tokens_in_epoch(10) == 3 * 10 + 2


def test_resume_continues_at_the_same_token_boundary():
    """The audit item: a resumed stream must not restart or misalign packing."""
    seq_len = 10
    uninterrupted = [row.ids for row in
                     data.RowStream(_source(), seq_len=seq_len, vocab_size=VOCAB, epochs=1)]

    # Consume 7 rows, save the position, then resume from it.
    stream = data.RowStream(_source(), seq_len=seq_len, vocab_size=VOCAB, epochs=1)
    head = [row.ids for row in data.take_rows(stream, 7)]
    saved = data.StreamPosition.from_dict(stream.position.as_dict())
    assert len(head) == 7

    resumed_stream = data.RowStream(_source(), seq_len=seq_len, vocab_size=VOCAB,
                                    epochs=1, start=saved)
    tail = [row.ids for row in resumed_stream]
    assert head + tail == uninterrupted


def test_resume_in_a_later_epoch_restarts_that_epoch_cleanly():
    stream = data.RowStream(_source(total_docs=6), seq_len=10, vocab_size=VOCAB, epochs=2)
    rows = [row.ids for row in stream]
    # Two epochs over 6 docs of 7 tokens: 4 rows per epoch (42 tokens -> 4 rows).
    per_epoch = 4
    assert len(rows) == per_epoch * 2
    assert stream.position.epoch == 3          # ran past the last epoch
    assert rows[:per_epoch] == rows[per_epoch:]


def test_stream_rejects_a_position_that_cannot_be_resumed():
    with pytest.raises(ValueError):
        data.RowStream(_source(), seq_len=4, vocab_size=VOCAB, epochs=1,
                       start=data.StreamPosition(pending_ids=[1] * 4))
    with pytest.raises(ValueError):
        data.RowStream(_source(), seq_len=4, vocab_size=VOCAB, epochs=0)


def test_take_rows_is_bounded():
    rows = data.take_rows(
        data.RowStream(_source(total_docs=100), seq_len=10, vocab_size=VOCAB, epochs=1), 3)
    assert len(rows) == 3
    with pytest.raises(ValueError):
        data.take_rows([], -1)


# ---------------------------------------------------------------------------
# Sizes and repeat exposure
# ---------------------------------------------------------------------------
def test_token_estimates_are_measured_not_guessed():
    per_epoch = data.estimate_tokens_per_epoch(207_865, 1226.1)
    assert per_epoch == pytest.approx(254_862_000, rel=0.01)
    assert data.estimate_passes(per_epoch * 2, per_epoch) == pytest.approx(2.0)
    assert data.epochs_for_budget(per_epoch + 1, per_epoch) == 2
    with pytest.raises(ValueError):
        data.estimate_tokens_per_epoch(0, 1.0)
    with pytest.raises(ValueError):
        data.estimate_tokens_per_epoch(10, 0.0)


def test_repeat_exposure_refuses_cycling_a_small_dataset():
    with pytest.raises(ValueError, match="passes"):
        data.assert_repeat_exposure(100.0, max_passes=3, what="ultrachat train_sft")
    data.assert_repeat_exposure(2.0, max_passes=3)
    with pytest.raises(ValueError):
        data.assert_repeat_exposure(0.0, max_passes=3)
