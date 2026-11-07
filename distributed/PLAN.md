# distributed (Phase 2)
Scope: DP/TP/PP/EP parallelisms, BF16/FP8 matmul, all-reduce, expert-parallel
routing, checkpointing, fault tolerance.

**Status (2026-11-06): still NOT implemented.** BF16 autocast exists in the
single-device trainer (`--amp bf16`), but there is no distributed execution of
any kind and no FP8 path.
