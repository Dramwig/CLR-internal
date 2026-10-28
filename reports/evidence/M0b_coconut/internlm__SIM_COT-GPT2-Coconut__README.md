---
language:
- en
license: mit
tags:
- chain-of-thought
- implicit-reasoning
- multimodal
- gpt2
- instruction-tuned
datasets:
- gsm8k
- svamp
- multi_arith
model-index:
- name: SIM_COT-GPT2-Coconut
  results:
  - task:
      type: math-word-problems
      name: Arithmetic Reasoning
    dataset:
      name: GSM8K
      type: gsm8k
    metrics:
    - type: accuracy
      value: xx.x
  - task:
      type: math-word-problems
      name: MultiArith
    dataset:
      name: MultiArith
      type: multi_arith
    metrics:
    - type: accuracy
      value: xx.x
  - task:
      type: math-word-problems
      name: SVAMP
    dataset:
      name: SVAMP
      type: svamp
    metrics:
    - type: accuracy
      value: xx.x
---

# 🚀 SIM_COT-GPT2-Coconut

[![🤗 Model Repo](https://img.shields.io/badge/HuggingFace-Model-blue)]()
[![📂 GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/InternLM/SIM-CoT)  
[![📄 Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/pdf/2509.20317)  

<p align="center">
  <img src="./assets/coconut_teaser.png" alt="Teaser Figure" width="600"/>
</p>

## 📖 Introduction

Chain-of-Thought (CoT) prompting has become a widely adopted strategy for enhancing the reasoning capabilities of Large Language Models (LLMs). By decomposing problems into intermediate steps, explicit CoT improves accuracy across a variety of reasoning tasks. However, the token cost of explicit reasoning severely limits its scalability, especially when applied to long-horizon tasks or deployed under strict computational budgets.

Implicit CoT methods attempt to address this issue by replacing explicit intermediate steps with continuous latent representations. These approaches achieve higher token efficiency while retaining some of the benefits of step-wise reasoning. Despite this promise, a persistent performance gap remains: implicit CoT methods often underperform compared to explicit reasoning, especially as the number of latent tokens is scaled. Our analysis identifies a fundamental **latent instability problem**: as more implicit reasoning tokens are introduced, training frequently becomes unstable, with latent representations collapsing into homogeneous states that lack semantic diversity. This failure is largely due to the absence of fine-grained, step-level supervision in existing approaches.

To overcome this limitation, we introduce **SIM-CoT**, a plug-and-play training module designed to stabilize and enrich the latent reasoning space. SIM-CoT leverages an auxiliary decoder during training that aligns each implicit token with its corresponding explicit reasoning step. This step-level supervision ensures that latent states encode distinct and meaningful information. Importantly, the auxiliary decoder is removed during inference, meaning that SIM-CoT preserves the computational efficiency of implicit CoT without adding runtime overhead.

Empirical results demonstrate that SIM-CoT substantially improves both **in-domain accuracy** and **out-of-domain stability**. On smaller models such as GPT-2, SIM-CoT not only boosts implicit baselines like Coconut by +8.2% but also **surpasses explicit CoT by +2.1% while being 2.3× more token-efficient**. On larger models, including LLaMA-3.1 8B, SIM-CoT delivers consistent gains, improving CODI by +3.0% and significantly narrowing the performance gap with explicit reasoning. These findings highlight SIM-CoT as an effective and scalable solution for advancing implicit reasoning in LLMs.

---

**SIM_COT-GPT2-Coconut** is a large implicit language model based on **GPT2**, fine-tuned with **SIM-CoT (Supervised Implicit Chain-of-Thought)** on top of the **Coconut latent reasoning framework**.  
It is designed to improve ✨ *implicit reasoning* and 🧮 *arithmetic multi-step problem solving* across benchmarks such as **GSM8K, GSM-Hard, MultiArith, and SVAMP**.

---

## 📊 Experimental Results

We evaluate **SIM-CoT** across both **in-domain** (GSM8K-Aug) and **out-of-domain** (GSM-Hard, MultiArith, SVAMP) benchmarks, using **GPT-2**, **LLaMA-3.2 1B**, **LLaMA-3.2 3B**, and **LLaMA-3.1 8B** as backbones, applied to both **Coconut** and **CODI** frameworks.


<p align="center">
  <img src="./assets/gpt2.png" alt="Main Results on GPT2" width="750"/>
</p>

*Main results on GPT-2. We report accuracy % on in-domain (GSM8k-Aug) and out-of-domain (GSM-Hard, MultiArith, SVAMP) benchmarks. Our SIM-CoT is shown to provide accuracy gains on top of existing methods such as Coconut and CODI.*

<p align="center">
  <img src="./assets/llama1b.png" alt="Main Results on LLaMA3 1B" width="750"/>
</p>

*Main results on LLaMA 3.2 1B. We report accuracy % on in-domain (GSM8k-Aug) and out-of-domain (GSM-Hard, MultiArith, SVAMP) benchmarks. Our SIM-CoT builds on CODI to achieve a new SOTA in implicit reasoning while setting performance comparable to explicit CoT.*

<p align="center">
  <img src="./assets/llama3b_8b.png" alt="Main Results on LLaMA3 3B and 8B" width="750"/>
</p>

*Main results on LLaMA 3.2 3B and 8B. We report accuracy % on in-domain (GSM8k-Aug) and out-of-domain (GSM-Hard, MultiArith, SVAMP) benchmarks.*

---


The model integrates **implicit reasoning tokens** during training and inference.  
Unlike standard explicit CoT models, SIM-CoT encourages the model to generate **latent structured thoughts** that are decoded only during training, while remaining implicit during inference.

---

## 🎯 Intended Uses

- 🔬 **AI-related research** (reasoning, representation learning, interpretability)  
- 📊 **Benchmarking** on arithmetic reasoning datasets (e.g., GSM8K, SVAMP, MultiArith, GSM-Hard)  
- 🧩 Studying **latent representation learning** and **reasoning generalization**  

⚠️ *Not intended for deployment in production without careful alignment and safety evaluation.*

---

## 💻 Usage

To reproduce our results, follow the steps below:

### 1. Clone the repository
```bash
git clone https://github.com/InternLM/SIM-CoT.git
cd SIM-CoT/Coconut
```

### 2. Run the evaluation script
We provide shell scripts for different backbones and datasets.
```
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot_eval.yaml
```



