---
license: mit
base_model: openai-community/gpt2
language:
  - en
tags:
  - latent-reasoning
  - interpretability
  - reasoning
  - coconut
  - prosqa
---

# Coconut · gpt2 · ProsQA

This is the **Coconut** checkpoint trained on **ProsQA** with base model
[`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2), from the paper
[*Are Latent Reasoning Models Easily Interpretable?*](https://arxiv.org/abs/2604.04902) (Dilgren & Wiegreffe, 2026).

- 📄 **Paper:** https://arxiv.org/abs/2604.04902
- 💻 **Code:** https://github.com/connordilgren/are-lrms-easily-interpretable
- 📚 **Collection (all checkpoints):** https://huggingface.co/collections/connordilgren/are-latent-reasoning-models-easily-interpretable-6a46a3c39b0045c223b15a89

## Files

This repository contains a single raw PyTorch checkpoint, **`checkpoint_40`** — the state dict as
saved by the training framework. It is not a `from_pretrained`-style model; it is loaded by
the paper's evaluation code, which builds the base model and applies this checkpoint.

## Usage

The evaluation code in the [repository](https://github.com/connordilgren/are-lrms-easily-interpretable) loads this checkpoint from the local path
configured in `model_paths.yaml`. Download it to the expected location with:

```bash
hf download connordilgren/gpt2-prosqa-coconut checkpoint_40 --local-dir checkpoints/prosqa-coconut
```

This places the file at `checkpoints/prosqa-coconut/checkpoint_40`, which is the path referenced for this model
(`gpt2` → `prosqa` → `coconut`) in `model_paths.yaml`. See the
repository README for full setup and evaluation instructions.

## Citation

```bibtex
@misc{dilgren2026latentreasoningmodelseasily,
      title={Are Latent Reasoning Models Easily Interpretable?},
      author={Connor Dilgren and Sarah Wiegreffe},
      year={2026},
      eprint={2604.04902},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.04902},
}
```
