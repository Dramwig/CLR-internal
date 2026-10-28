# 运行登记

尚未运行实验。后续逐次登记运行目的、配置、代码版本、数据划分、种子、`runs/<run_id>/` 路径与 `metrics.json` 来源。

## 2026-10-28 — M0 v1（部分完成，非整体验收）

完成环境盘点、Idea A只读复用审计、官方代码/checkpoint核实、三种源接口、问题级冻结划分、15项测试。CODI-GPT2完成smoke、全量GSM8K test复现和64题val噪声曲线。其余源保持N/A，详见BLOCKERS。

### 实际结果（seeds=11/22/33；均值 ± 95% Student-t CI）

| 项目 | 结果 | 解释 |
|---|---:|---|
| 官方CODI-GPT2 test.py / GSM8K test | 43.59 ± 0.00% | 论文43.7%，差-0.106pp |
| CLR CODI-GPT2等长接口 / 同一test | 41.70 ± 0.00% | 与论文差-2.002pp；官方协议对照净增1.895pp，保留差异 |
| bf16重复潜状态最大逐元素差 | 0 | 固定输入/seed；不等同于batch1/16完全一致 |
| ProsQA训练预算，显式batch8成本探针 | 117.75 ± 27.87 GPU小时/seed | 吞吐外推，不是实际训练时间；>40小时，暂缓 |
| 单元测试 | 15 passed | 包括真实小型官方架构fixture，不是研究精度 |

CI=0来自同一固定checkpoint的确定性推理重复，不能解释成总体准确率零不确定性；不是三次独立训练。

| sigma | val准确率 % ±95%CI | K=16不同有效答案数 ±95%CI |
|---:|---:|---:|
| 0 | 37.50 ± 0.00 | 1.000 ± 0.000 |
| 0.01 | 36.39 ± 0.50 | 1.156 ± 0.039 |
| 0.03 | 36.00 ± 0.61 | 1.193 ± 0.022 |
| 0.1 | 36.23 ± 0.87 | 1.339 ± 0.098 |
| 0.3 | 36.20 ± 1.58 | 1.771 ± 0.195 |

负面结果：噪声提高多样性但未提高该面板平均准确率；单次val baseline39.0625%和K16 sigma0 baseline37.5%不一致，证明batch数值/输出协议需要固定。所有sigma的曲线内部比较使用同一K16协议，无效答案率均为0。

### 资源与可追溯性

设备由驱动报告为RTX4090/49140MiB，共8张；实际环境和共享负载见reports/M0_env.md。forward是实际模型调用次数，batch调用计一次；不是样本数。GPU小时为单卡wall时间，不是独占GPU利用率积分。

| 运行 | run_id | GPU小时 | source forward |
|---|---|---:|---:|
| CLR smoke | `20261028T092620797688Z_8f7b65449647_bfd675` | 0.0153 | 783 |
| CLR test | `20261028T092858690199Z_231d82430916_1c95a0` | 0.1013 | 8265 |
| CLR noise | `20261028T092859683784Z_4c9ba94e51a9_789243` | 0.2036 | 16368 |
| official test.py | `20261028T094129023827Z_fbf746deb744_12ecd6` | 0.0120 | 510 |

主要配置与命令（cwd=/home/me/PaperField/CalReader）：

```bash
PYTHONPATH=. .venv/bin/python scripts/m0_inventory.py configs/M0_inventory.yaml
PYTHONPATH=. .venv/bin/python scripts/m0_data.py configs/M0_data.yaml
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python scripts/m0_tests.py configs/M0_tests.yaml
PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_eval.py configs/M0_codi_gpt2_smoke_b16.yaml
PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_eval.py configs/M0_codi_gpt2_repro_b16.yaml smoke_run=runs/20261028T092620797688Z_8f7b65449647_bfd675
PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_eval.py configs/M0_codi_gpt2_noise_b16.yaml smoke_run=runs/20261028T092620797688Z_8f7b65449647_bfd675
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_codi_reference.py configs/M0_codi_reference_smoke.yaml
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_codi_reference.py configs/M0_codi_reference_full.yaml smoke_run=runs/20261028T093843988383Z_c5632b1754b1_567572
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python scripts/m0_report.py configs/M0_report.yaml
```

下载、成本探针和失败记录的具体命令/config均保存于对应run。初次未进入Run的动态库失败及OOM成本计数缺口已在BLOCKERS披露。没有正在运行的训练/评估或下载任务。

报告：[M0_env](../reports/M0_env.md)、[M0_repro](../reports/M0_repro.md)。已完成的CODI结果以SHA-256清单固定，后续协议更改应建v2；M0其余部分仍未完成。报告版本存档为 `reports/M0_repro_v1.md`。
