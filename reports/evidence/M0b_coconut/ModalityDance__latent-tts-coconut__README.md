---
base_model:
- openai-community/gpt2
license: mit
pipeline_tag: text-generation
library_name: transformers
---

# COCONUT Model

<div align="center">

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/ModalityDance/latent-tts-coconut)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2510.07745)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?style=for-the-badge&logo=github)](https://github.com/ModalityDance/LatentTTS)

</div>

## Overview

**COCONUT** (Chain of Continuous Thought) is a latent reasoning model based on GPT-2 that enables continuous thought generation in latent space. This model is part of the research presented in the paper [Parallel Test-Time Scaling for Latent Reasoning Models](https://huggingface.co/papers/2510.07745).

Official Code: [https://github.com/ModalityDance/LatentTTS](https://github.com/ModalityDance/LatentTTS)

## Model Details

- **Base Architecture**: GPT-2 Language Model
- **Model Class**: `COCONUTGPT2` (extends `GPT2LMHeadModel`)
- **Latent Tokens**: Uses special tokens `<|latent|>`, `<|start-latent|>`, `<|end-latent|>` for latent reasoning
- **Input Format**: Requires newline after input question before `<|start-latent|>` token

## Related Models

This repository includes other latent reasoning models that you might find useful:

[ModalityDance/latent-tts](https://huggingface.co/collections/ModalityDance/latent-tts)

## Installation

Download the model from HuggingFace:

```bash
huggingface-cli download ModalityDance/latent-tts-coconut --local-dir checkpoints/coconut
```

## Quick Start

### Basic Usage

Note: Inference requires the `src` directory and custom implementation files from the [official GitHub repository](https://github.com/ModalityDance/LatentTTS).

```python
from transformers import AutoTokenizer
from src.generation_mixin import LatentGenerationMixin, LatentGenerationConfig
from src.paths import MODELS

# Load tokenizer
model_id = "checkpoints/coconut"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get latent token IDs
latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

# Create model class with generation mixin
class LatentCOCONUT(MODELS["coconut"]["class"], LatentGenerationMixin):
    def __init__(self, config):
        super().__init__(config)

# Load model
model = LatentCOCONUT.from_pretrained(
    model_id,
    latent_id=latent_id,
    latent_start_id=start_id,
    latent_end_id=end_id,
    device_map="auto",
)

# Prepare input (note: newline before <|start-latent|>)
question = "What is 2 + 2?
<|start-latent|>"
inputs = tokenizer(question, return_tensors="pt").to(model.device)

# Configure generation
generation_config = LatentGenerationConfig(
    max_new_tokens=512,
    latent_length=6,
    latent_do_sample=True,
    latent_do_sample_by="dropout",  # or "noise"
    dropout_p=0.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Generate
output = model.generate(
    **inputs,
    generation_config=generation_config,
    num_return_sequences=1,
)

# Decode result
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
```

### Batch Processing

The model fully supports batch processing:

```python
# Prepare batch inputs
questions = [
    "What is 2 + 2?
<|start-latent|>",
    "What is 5 * 3?
<|start-latent|>",
    "What is 10 - 4?
<|start-latent|>",
]
inputs = tokenizer(questions, return_tensors="pt", padding=True).to(model.device)

# Generate for batch
outputs = model.generate(
    **inputs,
    generation_config=generation_config,
    num_return_sequences=1,
)

# Decode batch results
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for result in results:
    print(result)
```

## Generation Parameters

### LatentGenerationConfig

- `max_new_tokens` (int): Maximum number of tokens to generate
- `latent_length` (int): Number of latent tokens (default: 6)
- `latent_do_sample` (bool): Whether to use stochastic sampling
- `latent_do_sample_by` (str): Sampling method - `"dropout"` or `"noise"`
- `dropout_p` (float): Dropout probability for Monte Carlo Dropout (e.g., 0.1)
- `noise_std` (float): Standard deviation for Additive Gaussian Noise

### Sampling Methods

1. **Monte Carlo Dropout**: Randomly drops activations during forward passes

   ```python
   generation_config = LatentGenerationConfig(
       latent_do_sample_by="dropout",
       dropout_p=0.1,
       # ...
   )
   ```
2. **Additive Gaussian Noise**: Injects noise into latent embeddings

   ```python
   generation_config = LatentGenerationConfig(
       latent_do_sample_by="noise",
       noise_std=0.1,
       # ...
   )
   ```

## Answer Extraction

COCONUT uses a special answer format with `#` separator:

```python
from src.paths import coconut_extract_answer_number

# Extract answer from generated text
answer = coconut_extract_answer_number(result)
print(f"Answer: {answer}")
```

## Evaluation

Run evaluation using the provided scripts in the official repository:

```bash
# For COCONUT (GPT-2 based models)
./run_tests.sh
```

## Model Card

- **Paper**: [Parallel Test-Time Scaling for Latent Reasoning Models](https://arxiv.org/abs/2510.07745)
- **HuggingFace**: [ModalityDance/latent-tts-coconut](https://huggingface.co/ModalityDance/latent-tts-coconut)
- **Benchmarks**: GSM8K Test, GSM8K Hard, MultiArith

## Citation

If you use this model, please cite:

```bibtex
@misc{you2025paralleltesttimescalinglatent,
      title={Parallel Test-Time Scaling for Latent Reasoning Models}, 
      author={Runyang You and Yongqi Li and Meng Liu and Wenjie Wang and Liqiang Nie and Wenjie Li},
      year={2025},
      eprint={2510.07745},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07745}, 
}

@misc{hao2025traininglargelanguagemodels,
      title={Training Large Language Models to Reason in a Continuous Latent Space}, 
      author={Shibo Hao and Sainbayar Sukhbaatar and DiJia Su and Xian Li and Zhiting Hu and Jason Weston and Yuandong Tian},
      year={2025},
      eprint={2412.06769},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.06769}, 
}
```