# Phase B — greedy generation samples (254K vs 1M vs published baseline)

All generations: **greedy argmax, deterministic (no RNG)**, `max_new_tokens=32`, CPU
(Intel Xeon @ 2.90 GHz, 2 cores, torch 2.13.0+cpu). Outputs are unstructured
babble — the honest expectation at this scale; the point is guard behavior and
determinism. ms/token is the **clean CLI measurement** (`scripts.generate`,
fresh process per checkpoint: local tiny 1.3 ms/token · tiny_1m 2.0 ms/token ·
published 1.1 ms/token; a batch driver that reloaded checkpoints in one process
showed inflated 32–51 ms/token, so per-model CLI latency is used here).

| Model | Prompt | Continuation | ms/token |
|---|---|---|---:|
| local-tiny-254272 | 'The capital of France is' | ` de sin de sin de sin de sin de ` | 1.3 |
| local-tiny-254272 | 'OpenAssistant is an open-source project that' | ` de sin and and and and and de s` | 1.3 |
| local-tiny-254272 | 'What is the meaning of life' | `ren and and and and and and and ` | 1.3 |
| local-tiny-254272 | 'Once upon a time' | `n de de de de de sin de sin de s` | 1.3 |
| local-tiny-254272 | 'The quick brown fox' | `t de sin de sin de sin de sin de` | 1.3 |
| local-tiny-1m-1000320 | 'The capital of France is' | ` eine der eine der eine der eine` | 2.0 |
| local-tiny-1m-1000320 | 'OpenAssistant is an open-source project that' | ` and einen and eine de seinen ei` | 2.0 |
| local-tiny-1m-1000320 | 'What is the meaning of life' | `ction de seine de seine de seine` | 2.0 |
| local-tiny-1m-1000320 | 'Once upon a time' | `n eine Machine de seine de seine` | 2.0 |
| local-tiny-1m-1000320 | 'The quick brown fox' | `t in Авание Д Д Дван` | 2.0 |
| published-tiny-254272-hf-oasst1 | 'The capital of France is' | ` connected such and seen such an` | 1.1 |
| published-tiny-254272-hf-oasst1 | 'OpenAssistant is an open-source project that' | ` area such and seen seen such se` | 1.1 |
| published-tiny-254272-hf-oasst1 | 'What is the meaning of life' | `ly seen such and seen such and s` | 1.1 |
| published-tiny-254272-hf-oasst1 | 'Once upon a time' | ` seen and seen sustainable seen ` | 1.1 |
| published-tiny-254272-hf-oasst1 | 'The quick brown fox' | `el en el ser el ser el ser einer` | 1.1 |

The `published-tiny-254272-hf-oasst1` row is the **owner's published baseline**
(HF `PrometheanStudio/talos-mini-254k-oasst1`, step-7680, trained on a T4) —
generated locally CPU with the same tokenizer (byte-identical). Reference only:
different hardware/training regimen. All three checkpoints passed the
generation-time consistency guards (format, canonical params, vocab, tokenizer
compat) before producing tokens.

## Full texts (echos)

```json
[
  {"model": "local-tiny-254272", "prompt": "The capital of France is", "continuation": " de sin de sin de sin de sin de ", "full_text": "The capital of France is de sin de sin de sin de sin de ", "params": 254272, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-254272", "prompt": "OpenAssistant is an open-source project that", "continuation": " de sin and and and and and de s", "full_text": "OpenAssistant is an open-source project that de sin and and and and and de s", "params": 254272, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-254272", "prompt": "What is the meaning of life", "continuation": "ren and and and and and and and ", "full_text": "What is the meaning of liferen and and and and and and and ", "params": 254272, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-254272", "prompt": "Once upon a time", "continuation": "n de de de de de sin de sin de s", "full_text": "Once upon a timen de de de de de sin de sin de s", "params": 254272, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-254272", "prompt": "The quick brown fox", "continuation": "t de sin de sin de sin de sin de", "full_text": "The quick brown foxt de sin de sin de sin de sin de", "params": 254272, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-1m-1000320", "prompt": "The capital of France is", "continuation": " eine der eine der eine der eine", "full_text": "The capital of France is eine der eine der eine der eine", "params": 1000320, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-1m-1000320", "prompt": "OpenAssistant is an open-source project that", "continuation": " and einen and eine de seinen ei", "full_text": "OpenAssistant is an open-source project that and einen and eine de seinen ei", "params": 1000320, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-1m-1000320", "prompt": "What is the meaning of life", "continuation": "ction de seine de seine de seine", "full_text": "What is the meaning of lifection de seine de seine de seine", "params": 1000320, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-1m-1000320", "prompt": "Once upon a time", "continuation": "n eine Machine de seine de seine", "full_text": "Once upon a timen eine Machine de seine de seine", "params": 1000320, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "local-tiny-1m-1000320", "prompt": "The quick brown fox", "continuation": "t in Авание Д Д Дван", "full_text": "The quick brown foxt in Авание Д Д Дван", "params": 1000320, "checkpoint_step": 11541, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "published-tiny-254272-hf-oasst1", "prompt": "The capital of France is", "continuation": " connected such and seen such an", "full_text": "The capital of France is connected such and seen such an", "params": 254272, "checkpoint_step": 7680, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "published-tiny-254272-hf-oasst1", "prompt": "OpenAssistant is an open-source project that", "continuation": " area such and seen seen such se", "full_text": "OpenAssistant is an open-source project that area such and seen seen such se", "params": 254272, "checkpoint_step": 7680, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "published-tiny-254272-hf-oasst1", "prompt": "What is the meaning of life", "continuation": "ly seen such and seen such and s", "full_text": "What is the meaning of lifely seen such and seen such and s", "params": 254272, "checkpoint_step": 7680, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "published-tiny-254272-hf-oasst1", "prompt": "Once upon a time", "continuation": " seen and seen sustainable seen ", "full_text": "Once upon a time seen and seen sustainable seen ", "params": 254272, "checkpoint_step": 7680, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"},
  {"model": "published-tiny-254272-hf-oasst1", "prompt": "The quick brown fox", "continuation": "el en el ser el ser el ser einer", "full_text": "The quick brown foxel en el ser el ser el ser einer", "params": 254272, "checkpoint_step": 7680, "mode": "greedy", "max_new_tokens": 32, "device": "cpu"}
]
```