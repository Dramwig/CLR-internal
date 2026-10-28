---
license: mit
tags:
  - latent-reasoning
  - chain-of-thought
  - continuous-thought
  - coconut
  - prosqa
  - graph-traversal
  - curriculum-learning
  - pause-tokens
  - gpt2
base_model: openai-community/gpt2
datasets:
  - prosqa
language:
  - en
---

# COCONUT Curriculum Checkpoints

Pretrained model checkpoints for **"The Curriculum Is the Mechanism: Dissecting COCONUT's Latent Thought Gains on ProsQA."**

We show that Meta's COCONUT architecture derives its ProsQA accuracy from the 7-stage training curriculum, not the continuous thought (hidden-state recycling) mechanism. A simple pause-token control trained with the same curriculum matches COCONUT at 96.6% vs. 97.0% (McNemar p = 0.845). A multi-pass pause control further decomposes out-of-distribution generalization into two factors: recycled content (which hurts extrapolation) and sequential processing (which drives topological generalization).

## Models

All models are fine-tuned from `openai-community/gpt2` (124M parameters) on ProsQA with a 7-stage curriculum (50 epochs, seed 0).

| Directory | Paper ID | Description | Feedback Mode | Best Epoch | ProsQA Test |
|-----------|----------|-------------|---------------|:----------:|:-----------:|
| `cot-baseline/` | M1 | Standard chain-of-thought fine-tuning | `cot` | 44 | 83.0% |
| `coconut/` | M2 | Meta's COCONUT with hidden-state recycling | `continuous` | 49 | 97.0% |
| `pause-curriculum/` | M3 | Learned pause embedding, single forward pass, same curriculum | `pause_curriculum` | 43 | 96.6% |
| `pause-multipass/` | M4 | Learned pause embedding, 6 sequential passes | `pause_multipass` | 30 | 94.8% |

**Key finding:** M3 matches M2 on in-distribution accuracy (McNemar p = 0.845), demonstrating that the curriculum drives the gains, not the mechanism.

### Out-of-Distribution Generalization

| Test Set | M1 (CoT) | M2 (COCONUT) | M3 (Pause) | M4 (Multipass) |
|----------|----------:|-------------:|-----------:|---------------:|
| 7-hop | 10.7% | 66.0% | 75.4% | **76.9%** |
| 8-hop | 8.2% | 67.5% | 75.1% | **75.2%** |
| DAG | 28.2% | 59.2% | 51.9% | **59.8%** |
| Dense | 14.1% | 61.2% | **68.4%** | 64.8% |

Factorial decomposition via M4:
- **Recycled content hurts extrapolation:** M4 outperforms M2 on 7-hop by 10.9 pp (p < 0.001)
- **Sequential processing drives DAG generalization:** M4 outperforms M3 on DAG by 7.9 pp (p < 0.001)

## Repository Structure

```
bmarti44/coconut-curriculum-checkpoints/
├── cot-baseline/                # M1 — Chain-of-thought baseline
│   ├── checkpoint_1/
│   ├── checkpoint_2/
│   ├── ...
│   ├── checkpoint_50/
│   └── checkpoint_best/         # → checkpoint_44
├── coconut/                     # M2 — COCONUT (hidden-state recycling)
│   ├── checkpoint_1/
│   ├── ...
│   ├── checkpoint_50/
│   └── checkpoint_best/         # → checkpoint_49
├── pause-curriculum/            # M3 — Pause-token, single pass
│   ├── checkpoint_1/
│   ├── ...
│   ├── checkpoint_50/
│   └── checkpoint_best/         # → checkpoint_43
├── pause-multipass/             # M4 — Pause-token, 6 sequential passes
│   ├── checkpoint_1/
│   ├── ...
│   ├── checkpoint_50/
│   └── checkpoint_best/         # → checkpoint_30
└── experiments/                 # Experiment results (JSON, NPZ)
    ├── causal_sanity/           # Causal intervention sanity checks
    ├── corruption/              # Thought corruption analysis (M2/M3/M4)
    ├── m6/                      # M4 full experiment suite (accuracy, corruption, McNemar, probing)
    ├── m6_epoch39/              # M4 at epoch 39 (comparison checkpoint)
    ├── ood/                     # Out-of-distribution evaluation results
    ├── probing/                 # Linear probing experiments
    └── probing_corrected/       # Corrected probing with permutation tests
```

Each model directory contains 51 checkpoints: `checkpoint_1` through `checkpoint_50` plus `checkpoint_best` (symlink to the peak-validation epoch).

## Experiments

The `experiments/` directory contains all raw experiment results referenced in the paper:

| Subdirectory | Description | Key Files |
|-------------|-------------|-----------|
| `causal_sanity/` | Causal intervention sanity gate (Experiment 0) | `exp0_sanity_result.json`, `m1_causal.json` |
| `corruption/` | Thought token corruption ablation (Experiment 1) — permutation, transplant, position-by-position degradation | `results.json`, `m3_corruption.json`, `m5_corruption.json` |
| `m6/` | M4 (Pause-Multipass) full experiment suite — accuracy, corruption, McNemar tests, probing, transplant | `accuracy.json`, `corruption.json`, `mcnemar.json`, `summary.json` |
| `m6_epoch39/` | M4 evaluated at epoch 39 (second validation peak) for plateau analysis | `accuracy.json`, `summary.json` |
| `ood/` | Out-of-distribution generalization (7-hop, 8-hop, DAG, Dense) | `results.json`, `detailed_outputs.json` |
| `probing/` | Linear probing of intermediate representations | `results.json` |
| `probing_corrected/` | Corrected probing with permutation-based significance testing | `results_linear_perm.json`, `m3_linear_perm.json`, `m5_linear_perm.json` |

## Quick Start

### Download

```bash
# Via huggingface_hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('bmarti44/coconut-curriculum-checkpoints', local_dir='results/')
"
```

Or use the automated reproduction script from the [GitHub repo](https://github.com/bmarti44/research-pipeline):

```bash
python reproduce.py --from-checkpoints
```

### Loading a Checkpoint

```python
from code.exp_utils import load_model_by_name

# Load any model by directory name
model, tokenizer, info = load_model_by_name("coconut", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("cot-baseline", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("pause-curriculum", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("pause-multipass", "results/", device="cuda")
```

`load_model_by_name()` searches for `checkpoint_best` first, then falls back to `checkpoint_50` (final epoch), then the highest-numbered checkpoint.

### Manual Loading

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({"additional_special_tokens": ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]})

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
model.resize_token_embeddings(len(tokenizer))  # 50260

state_dict = torch.load("results/coconut/checkpoint_best/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
```

## Checkpoint Format

Each checkpoint is a PyTorch `state_dict` saved with `torch.save()`. The state dict contains:

- `base_causallm.*` -- GPT-2 weights (124M parameters)
- `pause_embedding` -- Learned pause embedding (M3 and M4 only)

All models extend the GPT-2 vocabulary with 3 special tokens (`<|start-latent|>`, `<|end-latent|>`, `<|latent|>`), giving a vocab size of 50,260.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `openai-community/gpt2` (124M) |
| Dataset | ProsQA (17,886 training samples) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 128 |
| Epochs | 50 |
| Precision | fp32 |
| Seed | 0 |
| Curriculum | 7 stages, ~5 epochs per stage, max 6 latent stages |

## Citation

```bibtex
@article{coconut-curriculum-2026,
  title={The Curriculum Is the Mechanism: Dissecting {COCONUT}'s Latent Thought Gains on {ProsQA}},
  author={Anonymous},
  year={2026},
  note={Preprint}
}
```

## Paper and Code

- **Paper:** See `manuscript/` in the [GitHub repository](https://github.com/bmarti44/research-pipeline)
- **Code:** [github.com/bmarti44/research-pipeline](https://github.com/bmarti44/research-pipeline)

## References

- Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769). arXiv:2412.06769.
- Zhang et al. (2025). [On the Causal Role of Continuous Thought Tokens](https://arxiv.org/abs/2512.21711). arXiv:2512.21711.
- Goyal et al. (2024). [Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226). ICLR 2024.

## License

MIT
