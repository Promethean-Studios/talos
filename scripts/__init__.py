"""Developer CLI scripts for Talos.
Entry points::
    python -m scripts.cli               # tiny model forward/backward demo
    python -m scripts.train_oasst1      # OASST1 JSONL -> BPE -> tiny training (split/eval/ckpt)
    python -m scripts.eval_checkpoint   # reproducible checkpoint eval (loss/ppl/acc/throughput)
    python -m configs.compute           # print config parameter/FLOP estimates
    bash scripts/run_tests.sh           # run the full pytest suite
"""