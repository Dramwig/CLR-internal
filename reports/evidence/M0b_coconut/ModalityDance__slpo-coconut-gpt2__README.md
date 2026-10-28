---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - latent-reasoning
  - coconut
  - slpo
  - gpt2
  - reinforcement-learning
base_model: ModalityDance/latent-tts-coconut
---

# COCONUT + SLPO (GPT-2)

Surrogate Latent Policy Optimization (**SLPO**) checkpoint on top of [COCONUT](https://huggingface.co/ModalityDance/latent-tts-coconut) (GPT-2 124M).
This is the **COCONUT+SLPO** model reported in the paper *SLPO: Scaling Latent Reasoning via a Surrogate Policy*.

## Model Details

- **Backbone**: COCONUT / GPT-2 (`ModalityDance/latent-tts-coconut`)
- **Method**: stopping-gate cold start → SLPO (RLOO) with adaptive latent stopping
- **Special tokens**: `<|latent|>`, `<|start-latent|>`, `<|end-latent|>`
- **Recommended gate threshold**: `0.6`
- **Max latent length**: `12`

## Results (paper main table, Acc)

Deterministic accuracy with dropout disabled and learned stop gate:

| Benchmark | Acc | Mean latent length |
|-----------|-----|--------------------|
| GSM8K | 35.63 | 5.73 |
| GSM-Hard | 7.66 | 6.08 |
| MultiArith | 83.10 | 6.28 |

## Related

- Paper (arXiv): [2607.19691](https://arxiv.org/abs/2607.19691)
- Hugging Face Paper: [2607.19691](https://huggingface.co/papers/2607.19691)
- Code: [ModalityDance/SLPO](https://github.com/ModalityDance/SLPO)
- Project page: [modalitydance.github.io/SLPO](https://modalitydance.github.io/SLPO/)
- Base model: [ModalityDance/latent-tts-coconut](https://huggingface.co/ModalityDance/latent-tts-coconut)
- Sibling: [ModalityDance/slpo-codi-gpt2](https://huggingface.co/ModalityDance/slpo-codi-gpt2)
- Collection: [ModalityDance/SLPO](https://huggingface.co/collections/ModalityDance/slpo)

## Installation

```bash
git clone https://github.com/ModalityDance/SLPO.git
cd SLPO
pip install -r requirements.txt   # plus a CUDA PyTorch build
hf download ModalityDance/slpo-coconut-gpt2 --local-dir checkpoints/slpo-coconut-gpt2
```

## Quick Start

Batched eval (paper Acc settings):

```bash
CKPT=checkpoints/slpo-coconut-gpt2 \
MODEL_TYPE=coconut STOP_POLICY=gate \
STOP_GATE_THRESHOLD=0.6 MAX_LATENT_LENGTH=12 \
DATA=data/gsm_test.json \
bash scripts/eval.sh
```

Minimal Python (from the repo root; needs the SLPO latent generation stack):

```python
import torch
from transformers import AutoTokenizer

from src.models.generation import LatentGenerationMixin, LatentGenerationConfig
from src.paths import get_model_class

model_id = "ModalityDance/slpo-coconut-gpt2"
backbone_cls = get_model_class("coconut")

class LatentModel(backbone_cls, LatentGenerationMixin):
    pass

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LatentModel.from_pretrained(model_id)
model.eval()

question = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells the remainder "
    "at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?"
)
prompt = question + "\n<|start-latent|>"
inputs = tokenizer(prompt, return_tensors="pt")

gen_cfg = LatentGenerationConfig(
    stop_policy="gate",
    max_latent_length=12,
    stop_gate_threshold=0.6,
    max_new_tokens=128,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
)

with torch.no_grad():
    output = model.generate(**inputs, generation_config=gen_cfg)
sequences = output.sequences if hasattr(output, "sequences") else output
print(tokenizer.decode(sequences[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{you2026slpo,
  title   = {SLPO: Scaling Latent Reasoning via a Surrogate Policy},
  author  = {You, Runyang and Liu, Zhiyuan and Li, Yongqi and Li, Wenjie},
  year    = {2026},
  eprint  = {2607.19691},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url     = {https://arxiv.org/abs/2607.19691}
}
```
