---
base_model:
- ModalityDance/latent-tts-coconut
license: mit
pipeline_tag: text-generation
library_name: transformers
tags:
- latent-reasoning
- continuous-thought
- coconut
- grpo
- reinforcement-learning
datasets:
- zen-E/GSM8k-Aug
---

# SVP-V-GRPO · COCONUT GPT-2

A [COCONUT](https://huggingface.co/ModalityDance/latent-tts-coconut) GPT-2 (124M)
latent-reasoning model post-trained with **SVP-V-GRPO** — reinforcement learning whose
rollout exploration comes from *weight-space* perturbation of the attention value
projections. For each layer and rollout, SVP independently samples Gaussian noise and
rescales coefficients in the fixed SVD basis of `W_V` as `sᵢ → sᵢ(1+αgᵢ)`. 

Only the value-projection weights (the V columns of each `c_attn`) differ from the base
checkpoint. The perturbation is a training-time exploration mechanism and is **not** used
at deployment: inference is ordinary greedy decoding.
This repository contains the paper's GPT-2 SVP-V-GRPO checkpoint from epoch 9 `B=32`, `lr=6e-5` run. 

## Results

Clean greedy decoding, `max_new_tokens=64`, exact-match on the final answer.

| Model (GPT-2 124M) | GSM8K | GSM-Hard | MultiArith | SVAMP | ASDiv-A | GSM-Plus |
|---|---|---|---|---|---|---|
| COCONUT (base) | 34.1 | 7.7 | 80.9 | 35.6 | 60.2 | 17.4 |
| SLPO | 34.9 | 7.6 | 82.8 | 34.3 | 58.7 | 18.2 |
| SIM-CoT | 44.7 | 9.3 | 90.5 | 40.6 | 67.2 | 21.5 |
| CoDi | 42.5 | 9.3 | 91.9 | 40.0 | 65.4 | 23.1 |
| **This model** (SVP on `W_V`, α=0.6) | **50.3** | **11.3** | **93.4** | 43.6 | **71.2** | **27.6** |

## Usage

This is a fixed-length continuous-thought model: the prompt ends with
`<|start-latent|>`, six latent steps feed each step's last hidden state back as the next
input embedding, `<|end-latent|>` closes the latent phase, and the answer is then decoded
as ordinary tokens. 

```python
from transformers import AutoTokenizer, GPT2LMHeadModel

REPO = "."  # path to this repository
tok = AutoTokenizer.from_pretrained(REPO)
model = GPT2LMHeadModel.from_pretrained(REPO).eval().cuda()
# then run the two-phase latent loop: 6 latent steps, then decode the answer
```

A reference implementation of the loop ships with the SVP code release; the original
COCONUT inference code works as well.

## Training

| | |
|---|---|
| Base | `ModalityDance/latent-tts-coconut` (COCONUT GPT-2, 124M) |
| Data | 383,891 eligible prompts from [GSM8K-Aug](https://huggingface.co/datasets/zen-E/GSM8k-Aug) |
| Algorithm | GRPO: `G=32` rollouts per prompt; `2B=64` candidate prompts per iteration with up to `B=32` mixed-outcome groups retained; `μ=2` optimization passes; centered advantages; symmetric clipping `ε=0.2`; k3 KL (`β=0.02`) to the frozen base |
| Exploration | SVP with `α=0.6`: independent Gaussian coefficient perturbations for each Value-projection layer and rollout, held fixed through prefill and all six latent steps; clean answer decoding; SVD factors refreshed every 50 iterations |
| Trained parameters | V columns of every `c_attn` (≈ 7M of 124M) |
| Optimizer | AdamW, constant `lr=6e-5`, seed 0, no warmup or weight decay, gradient clipping at 1.0; epoch-9 checkpoint from 15 epochs |
| Reward | Final-answer correctness only |

## License

MIT, inherited from the base checkpoint (which derives from `openai-community/gpt2`).
