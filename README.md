<img width="432" height="124" alt="Screenshot 2026-09-17 4 30 09 PM" src="https://github.com/user-attachments/assets/b4cbb3f5-875f-419d-a74f-1eb073459a78" />

# 

**Talos** is an experimental language-model project by **Promethean Studios**, focused on building and testing small, efficient transformer architectures before scaling toward larger models.

> **Status:** Experimental / active development
> **Current model:** Talos Mini
> **Trained checkpoint:** Not yet available

## Talos Mini

The current canonical Talos Mini configuration contains **254,272 trainable parameters**.

| Specification           |                     Value |
| ----------------------- | ------------------------: |
| Parameters              |               **254,272** |
| Hidden size             |                    **64** |
| Transformer layers      |                     **2** |
| Attention heads         |                     **4** |
| KV heads                |                     **2** |
| Vocabulary size         |                 **1,024** |
| Maximum sequence length |            **512 tokens** |
| Precision               |                  **FP32** |
| Attention backend       | **PlainAttentionBackend** |

The parameter count describes the configured architecture. It does **not** represent a released trained model checkpoint.

## Current Development

Talos currently focuses on validating the complete model and training stack:

* Transformer architecture
* Attention and KV-cache implementation
* Rotary positional embeddings
* RMS normalization
* Feed-forward networks
* Training pipeline
* Inference pipeline
* Evaluation
* Benchmarking
* Quantization experiments
* Future scaling experiments

The project has successfully completed architecture and CUDA smoke testing, but a **usable trained Talos checkpoint has not yet been produced**.

## Training

Early development tests have demonstrated that the training pipeline can reduce loss on extremely small test data.

These experiments are **not a benchmark of model intelligence or general language performance**. They are primarily used to verify that forward passes, backpropagation, optimization, and loss calculation behave as expected.

A production-quality training run and released checkpoint are still pending.

## Experimental MoE Configuration

The repository also contains an experimental Mixture-of-Experts code path.

| Specification   |     Value |
| --------------- | --------: |
| Hidden size     |   **128** |
| Layers          |     **3** |
| Attention heads |     **8** |
| KV heads        |     **4** |
| Head dimension  |    **16** |
| Experts         |    **16** |
| Active experts  |     **3** |
| Shared experts  |     **1** |
| Vocabulary      | **2,048** |

This configuration is currently a **code-path experiment**, not the primary Talos Mini model.

## Project Direction

Talos is being developed incrementally:

**254K → 1M+ parameters → larger architectures**

Future research directions include larger context windows, sparse/MoE architectures, quantization, efficient inference, and heterogeneous memory/compute systems.

Long-term experimental concepts include architectures with substantially larger total parameter counts while keeping active parameters significantly lower through sparsity and expert routing.

## Repository Structure

```text
talos/
├── model/          # Model architecture
├── configs/        # Model configurations
├── data/           # Data utilities
├── training/       # Training components
├── inference/      # Inference components
├── evaluation/     # Evaluation tools
├── benchmarks/     # Benchmarking
├── distributed/    # Distributed-training experiments
├── experiments/    # Experimental implementations
├── examples/       # Usage examples
├── docs/           # Documentation
└── tests/          # Test suite
```

## Current State

| Component                    | Status                  |
| ---------------------------- | ----------------------- |
| Model architecture           | 🟢 Implemented          |
| 254K parameter configuration | 🟢 Implemented          |
| CUDA execution               | 🟢 Tested               |
| Training pipeline            | 🟡 In development       |
| Full training run            | 🟡 Pending              |
| Trained checkpoint           | 🔴 Not released         |
| HF model weights             | 🔴 Not released         |
| 1M+ parameter model          | 🔴 Not started          |
| Production model             | 🔴 Not the current goal |

## License

See the repository license for the current licensing terms.
