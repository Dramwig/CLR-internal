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

## M0.1 v2：协议修补、源训练去重与噪声smoke（2026-10-28，未完成/未验收）

完整可追溯报告：`reports/M0.1_protocol.md`、`reports/M0.1_noise.md`。v1代码与结果保持冻结，所有候选运行另存。

| 实验 | 结果 | 95% CI / 说明 |
|---|---|---|
| v1对官方逐题一致率 | 65.580%（454题不同） | 按题bootstrap10k：[63.002%,68.158%] |
| v2 bf16官方batch128复放 | 99.924%（1题不同） | [99.773%,100%]；不等于fp32数据生成通过 |
| fp32 batch1 test准确率 | 43.745% | [41.092%,46.399%] |
| fp32 batch1对官方bf16一致率 | 95.299% | [94.162%,96.437%]，未达到99% |
| 去重 | 精确0；13-gram旧val/cal/test=3/4/7 | 保守检查全385620条训练发布语料 |
| v2划分 | val300 / cal193 / test1312 | 原500验证池经用户允许重划分；v1不变 |
| 噪声smoke | n=1，K16，seed11，18设置，183.706秒 | 无全量结果、无推荐sigma |

确定性指标不再重复三个推理seed制造零CI；使用题目bootstrap。三个随机seed仅用于完整噪声设计（11/22/33），当前完整试验未运行，不能填造“均值±CI”。具体命令与覆盖项见报告及各run/config.yaml。基本命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/m01_dedup.py configs/M01_dedup.yaml
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/m01_protocol_eval.py configs/M01_protocol_smoke.yaml
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/m01_noise.py configs/M01_noise_smoke.yaml
PYTHONPATH=. .venv/bin/python scripts/m01_report.py configs/M01_report.yaml
PYTHONPATH=. .venv/bin/python -m pytest tests/test_m01.py -q
```

负面结果：padding修正并不足以对齐；bf16投影张量布局和物理batch会改变逐题答案；fp32固定batch并未自然满足原官方bf16≥99%的条件。报告保留所有失败候选，不根据准确率挑版本。保守去重可能排除仅共享模板的问题，完整语料检查不能证明实际训练参与情况。协议问题解决前不进入M1。

M0b第三方Coconut仅元数据检索：`runs/20261028T112341606990Z_4b258d612bbb_cd4f89`，GPU=0、source forwards=0，精度为第三方声称，未复现。Huginn官方恢复下载/分片校验进度见 `reports/M0b_huginn.md`，未将下载进度记作复现完成。

## 2026-10-28 — M0.1 用户审查决定与完整实验续行
用户明确接受bf16复放验收，授权固定batch1/fp32继续完整噪声试验。完整计划保持300题×18设置×K16×3seed（259200次rollout），预计约46 GPU wall小时（粗估，非实耗）。协议hash因审查元数据更新而变化，重跑smoke后方可全量。原fp32与官方95.299%一致率不追改，原失败run保留。G-σ与M1尚未验收；结果产生前不填写均值或选择sigma。

Huginn支线：4/4官方权重SHA验证通过，真实加载成功，但smoke OOM；已记录0.0728161 GPUh、340次源forward。完整复现/噪声未启动，无准确率结论；测试8 passed。参见独立M0b报告，不将元数据下载成功算作模型复现成功。

### 完整噪声已实际启动（2026-10-28 21:15，Asia/Shanghai）
- 授权后smoke：`20261028T131027495042Z_d4e87644eae3_1eef38`，217.574秒、4640次forward，通过。
- 完整运行：`20261028T131534016226Z_83bfd1aab8dd_d4e55b`；配置入口`configs/M01_noise_job.yaml`，tmux会话`clr-m01-noise-v2`。
- 固定300题×18设置×16rollout×3seed，fp32 batch1。当前结果未完成，不填造表格；完成后自动追加完整统计与推荐/负面结果。
- 新smoke线性成本估计54.4 GPU wall小时（此前46小时也是粗估），共享GPU吞吐与生成长度会变化。没有更改规模或模型。

## 2026-10-29 — ModelScope LLaMA底模 v2 获取与校验完成
下载run `20261029T010824516751Z_89dc7e0b7668_b742ff`：9文件/2,480,842,630字节，218.771秒，GPU小时0、source forward0。全部文件通过ModelScope SHA-256与上游Git对象同一性验证。
离线格式验证run `20261029T011218552649Z_a2b3255c27bd_7e0397` 通过。命令入口为 `scripts/m0b_modelscope_download.py configs/M0b_llama_modelscope_download_v2.yaml` 与 `scripts/m0b_llama_base_validate.py configs/M0b_llama_base_validate_v2.yaml`。来源、版本、文件哈希、假设和剩余checkpoint阻塞见 `reports/M0b_llama_modelscope_v2.md`。本轮无准确率实验，故无3seed准确率/CI；未把资源校验当作CODI复现。

## 2026-10-29 — CODI-LLaMA作者checkpoint获取与复现续行
用户明确授权不限平台获取。使用配置 `configs/M0b_codi_llama_download_v2.yaml`、入口 `scripts/m0b_codi_checkpoint.py` 下载公开作者权重并对官方LFS SHA校验。后续v2 protocol使用ModelScope底模与作者训练checkpoint；源接口和官方test.py分别做smoke，之后才允许全量复现。单元检查 `tests/test_codi_llama_v2.py tests/test_m01.py`：7 passed，随机微型LLaMA只是反馈/扰动机制测试，不是模型复现结果。
