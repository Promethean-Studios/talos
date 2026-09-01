# Quantization experiment — tiny preset (CPU)

Trained 100 AdamW steps on the deterministic synthetic corpus (loss 6.9411 → 3.2361); all metrics vs the fp32 model on held-out prompts at lengths [8, 16, 32].

| variant | size MB | ×vs fp32 | eval loss | Δloss | cosine mean (min) | greedy agree (min) | greedy gen identical | decode tok/s |
|---|---|---|---|---|---|---|---|---|
| fp32 | 0.97 | 1.0 | 7.2545 | +0.0000 | 1.000000 (1.000000) | 1.0000 | yes | 1020.8 |
| bf16 | 0.485 | 2.0 | 7.2541 | -0.0004 | 0.999993 (0.999973) | 0.9688 | yes | 942.3 |
| int8_per_channel | 0.2571 | 3.77 | 7.2543 | -0.0002 | 0.999988 (0.999944) | 0.9375 | yes | 806.2 |
| int8_per_tensor | 0.2435 | 3.98 | 7.2544 | -0.0001 | 0.999968 (0.999734) | 1.0000 | yes | 813.6 |

Peak RSS (whole process): 321.6 MB. int8 = symmetric weight-only (int8 payload + fp32 scales, W8A32 dequant-matmul); bf16 = true bf16 compute. Timing numbers are hardware-specific; correctness metrics are not. Seed 0, torch 2.13.0+cpu.
