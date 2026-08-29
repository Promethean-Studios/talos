# Talos Colab Notebooks

This directory holds ready-to-run Google Colab notebooks for Talos's
development workflow.

## `talos_colab_train.ipynb` — First GPU training run (tiny prototype)

This is the basic end-to-end acceptance test for the Talos **`tiny` prototype**
(~254,272 params). It trains the canonical `tiny` preset on the Colab GPU and
confirms the training pipeline genuinely learns.

### 3 steps to run it

1. **Open the notebook in Colab** —
   `File → Upload notebook` (drag `talos_colab_train.ipynb` in) or `GitHub` →
   `Promethean-Studios/talos` → `notebooks/talos_colab_train.ipynb`.
2. **Choose a GPU runtime** — Runtime → Change runtime type → Hardware
   accelerator → **GPU** → Save. (A CPU runtime also works; the scripts detect
   the device automatically and fall back, just slower.)
3. **Run all** — Runtime → Run all. Each cell explains itself; no `pip install`
   is needed because Colab already ships `torch` + `numpy`, which are Talos's
   only runtime dependencies.

### What a successful run looks like

The training cell ends with a line like:

```
OK: dense tiny (254272 params, device=cuda) loss 6.9170 -> 0.12XX over 300 steps (peak drop 6.8XXX)
```

- **Initial loss ≈ 6.9** ≈ `log(vocab)` for vocab=1024 (model starts guessing
  uniformly).
- **Final loss well below 1** — the model has learned and overfit the fixed
  corpus, proving forward + backward + AdamW actually work.
- **`device=cuda`** confirms the GPU was used.

The last cell sanity-checks inference: it prefills a prompt in one pass and
decodes token-by-token via the KV cache, ending with `OK: generated 8 tokens:
[...]`.

### Pointing it at real data later

The notebook trains on a small **fixed synthetic corpus**
(`training.synthetic.build_recurrent_corpus`) — a deliberate, fast,
deterministic way to prove the loop learns; it is *not* real language data.

To train on real data, replace the synthetic corpus with tokens produced by the
Phase 3 data pipeline:

```bash
python -m data.pipeline \
    --config data/configs/example.json --out /tmp/mycorp \
    --max-docs 10000 --seed 0
```

That writes sharded JSONL under `output.dir`. From there, the next step is a
training entry point that reads those shards (a real-dataset trainer) rather
than `training.synthetic`. That trainer is the natural follow-up to this basic
test; it also should save checkpoints so inference can run on the *trained*
model instead of a fresh one.

### Notes / gotchas

- The example scripts (`examples/tiny_train.py`, `examples/run_inference.py`)
  auto-detect CUDA and are runnable directly as scripts (they put the repo root
  on `sys.path` themselves), so the notebook's `!python examples/...` calls
  work with no extra setup.
- The training script is deterministic for a fixed `--seed` (default `0`), so
  repeated runs reproduce the same loss curve.
- This notebook is the **validated vehicle** for the prototype (Phase 4).
  Distributed training, checkpoints, and a real-dataset trainer are later
  steps, not part of this basic test.
