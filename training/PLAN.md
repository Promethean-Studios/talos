# training (Phase 2)
Scope: distributed training (DP/TP/PP/EP), BF16/FP8, mixed-precision optimizer
states, gradient checkpointing, checkpoints/resume, fault tolerance, LR schedule.

**Status (2026-11-06):** the single-device pieces are IMPLEMENTED — the trainer
lives in `scripts/train_oasst1.py` with BF16/FP16 AMP, gradient checkpointing,
gradient accumulation, checkpoint/resume guards and a token-based LR schedule
(`docs/trainer.md`). **Distributed training (DP/TP/PP/EP) and FP8 are still not
implemented** — see `distributed/PLAN.md`.
