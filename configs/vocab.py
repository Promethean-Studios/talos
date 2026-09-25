"""Canonical vocab size — the single named source of truth for the prototype.

Every seam that decides "how many vocab slots does a Talos canonical model /
tokenizer have" reads :data:`VOCAB_SIZE` from here instead of hard-coding
``1024``:

* the preset configs (``configs.presets`` — model ``vocab_size`` fields and the
  preset tokenizer configs);
* the per-preset canonical registry (``configs.canonical.CANONICAL_PRESETS`` —
  the second element of every ``(params, vocab)`` tuple);
* the tokenizer trainer CLI default (``tokenizer/train.py`` ``--vocab-size``);
* the training guard (``scripts/train_oasst1.py`` tokenizer-vs-model check);
* the checkpoint/harness and generation seams (they read the *model's*
  ``cfg.vocab_size``, which is now built from this constant).

This module deliberately imports nothing: it is a leaf so ``configs.presets``
and ``configs.canonical`` (which import each other's symbols) and the
lower-level ``tokenizer`` package can all read it without any import cycle.

Any future change to the prototype family's vocab (more merge slots, new
special tokens, a second canonical budget) must be a deliberate edit to
:data:`VOCAB_SIZE` here — the "vocab-constant seam" test
(``tests/test_vocab_seam.py``) then mechanically proves every consumer still
agrees.
"""
from __future__ import annotations

#: Canonical vocab size for every Talos prototype preset (bytes + specials +
#: merges). 1024 = 256 base byte tokens + 4 specials (ids 1020..1023, LLaMA-style
#: at the top) + up to 764 learned BPE merges — see ``tokenizer.vocab`` for the
#: layout arithmetic. Kept at 1024 per the audit's §14 recommendation: the
#: canonical tokenizer already saturates all 1024 slots on the ~15M-token
#: corpus, and a larger budget would only grow the O(B·S·V) logits/CE cost
#: without any data-side gain.
VOCAB_SIZE = 1024