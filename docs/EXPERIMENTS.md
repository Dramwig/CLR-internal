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

### CODI-LLaMA v2权重已校验，验证/复现流程已启动
下载run `20261029T011806667943Z_8b873a7eb8a6_7735de`：578.473秒、3,408,265,790字节，GPU小时0，源前向0，官方SHA通过。验证/复现由 `scripts/m0b_codi_llama_job.py configs/M0b_codi_llama_job_v2.yaml` 在tmux `clr-m0b-codi-llama-v2`执行，固定GPU5、bf16、6潜在步、官方LoRA/projector设置。完整阶段严格依赖各自smoke通过，未填写未完成的准确率。原GPT2完整噪声任务继续独立运行。

### CODI-LLaMA GPU4验证通过，官方全量已启动
CLR smoke `20261029T013734197681Z_7bcbbe6fbd3d_77ffd6` 与官方smoke `20261029T013822186629Z_e9f09c818e4c_64a9d7` 均完成；8题确定性run `20261029T013901994681Z_66cf21a10c33_09328c` 最大潜状态差0、答案一致。真实smoke不是全量准确率结论。新pipeline配置 `configs/M0b_codi_llama_job_v2_gpu4.yaml`，GPU4资源调度显式记录，完整1319题官方协议复现随后CLR逐题对照正在推进。早先未知辅助键及GPU5预检失败均保留，无静默降级。

## CODI-LLaMA v2复现流水线结束
详见 `reports/M0b_codi_llama_repro_v2.md`；噪声与最终审查待完成。

### CODI-LLaMA v2复现与同batch控制完成（2026-10-29）

| 运行 | test1319准确率（按题bootstrap10k 95% CI） | 与官方逐题一致率 |
|---|---|---|
| 官方test.py batch128 | 55.497% [52.843,58.150] | 100%自身 |
| CLR batch16 | 55.269% [52.616,57.923] | 97.953%，27题不同 |
| CLR batch128控制 | 55.421% [52.767,58.074] | 98.939%，14题不同 |

clean test1312官方55.335% [52.591,58.003]。同输入8题重复潜状态差0；不是跨batch不变性保证。本轮确定性结果不使用重复seed CI，三个随机seed要求留给噪声实验。详细来源、成本、失败记录、>2pp偏差分析与命令见 `reports/M0b_codi_llama_analysis_v2.md`、`reports/M0b_codi_llama_repro_v2.md`。统计汇总run `20261029T015330021682Z_a261881100a5_29e463`。当前不声称全部复现条件或噪声任务完成。


## M0.1 完整噪声 v2 完成（待审查）

运行 `/home/me/PaperField/CalReader/runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b`；完整配置、成本、图与负面结果见 `reports/M0.1_noise_full_v2.md`。

| sigma | from_step | temp | accuracy | distinct valid | nondegenerate | invalid |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.4233 [0.3667, 0.4800] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 0.1 | 0 | 0 | 0.4244 [0.4230, 0.4257] | 1.1400 [1.1317, 1.1483] | 0.0144 [0.0049, 0.0240] | 0.0000 [0.0000, 0.0000] |
| 0.3 | 0 | 0 | 0.4245 [0.4232, 0.4258] | 1.5444 [1.5110, 1.5779] | 0.0667 [0.0667, 0.0667] | 0.0000 [0.0000, 0.0000] |
| 0.5 | 0 | 0 | 0.4188 [0.4159, 0.4218] | 1.9611 [1.9155, 2.0067] | 0.1122 [0.0914, 0.1331] | 0.0000 [0.0000, 0.0000] |
| 0.75 | 0 | 0 | 0.4117 [0.4098, 0.4137] | 2.5378 [2.3969, 2.6786] | 0.1922 [0.1714, 0.2131] | 0.0000 [0.0000, 0.0000] |
| 1.0 | 0 | 0 | 0.3924 [0.3880, 0.3969] | 3.3178 [3.1610, 3.4745] | 0.2844 [0.2636, 0.3053] | 0.0000 [0.0000, 0.0000] |
| 1.5 | 0 | 0 | 0.3004 [0.2892, 0.3116] | 5.6678 [5.5747, 5.7608] | 0.4489 [0.4281, 0.4697] | 0.0000 [0.0000, 0.0000] |
| 2.0 | 0 | 0 | 0.1813 [0.1668, 0.1959] | 7.8311 [7.7082, 7.9540] | 0.4244 [0.3681, 0.4808] | 0.0000 [0.0000, 0.0000] |
| 0 | 3 | 0 | 0.4233 [0.3667, 0.4800] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 0.1 | 3 | 0 | 0.4239 [0.4231, 0.4247] | 1.1078 [1.0905, 1.1250] | 0.0122 [0.0074, 0.0170] | 0.0000 [0.0000, 0.0000] |
| 0.3 | 3 | 0 | 0.4247 [0.4222, 0.4271] | 1.4089 [1.3836, 1.4342] | 0.0533 [0.0533, 0.0533] | 0.0000 [0.0000, 0.0000] |
| 0.5 | 3 | 0 | 0.4233 [0.4199, 0.4267] | 1.6933 [1.6685, 1.7182] | 0.0867 [0.0580, 0.1154] | 0.0000 [0.0000, 0.0000] |
| 0.75 | 3 | 0 | 0.4200 [0.4177, 0.4223] | 2.0456 [1.9791, 2.1120] | 0.1433 [0.1054, 0.1813] | 0.0000 [0.0000, 0.0000] |
| 1.0 | 3 | 0 | 0.4117 [0.4102, 0.4132] | 2.5278 [2.4223, 2.6333] | 0.2000 [0.1701, 0.2299] | 0.0000 [0.0000, 0.0000] |
| 1.5 | 3 | 0 | 0.3739 [0.3712, 0.3765] | 3.8056 [3.5579, 4.0532] | 0.3289 [0.3081, 0.3497] | 0.0000 [0.0000, 0.0000] |
| 2.0 | 3 | 0 | 0.3210 [0.3175, 0.3246] | 5.1356 [4.9853, 5.2858] | 0.3867 [0.3220, 0.4513] | 0.0000 [0.0000, 0.0000] |
| 0 | 0 | 0.5 | 0.4223 [0.4187, 0.4259] | 3.6556 [3.4947, 3.8164] | 0.1711 [0.1294, 0.2128] | 0.0000 [0.0000, 0.0000] |
| 0 | 0 | 1.0 | 0.4010 [0.3959, 0.4060] | 5.8322 [5.7334, 5.9310] | 0.3178 [0.2804, 0.3551] | 0.0001 [-0.0005, 0.0007] |

推荐输出：`{"selected": {"sigma": 1.5, "from_step": 3, "temperature": 0, "accuracy_drop": 0.04944444444444446, "nondegenerate": 0.3288888888888889}, "tie_break": "nondegenerate descending, drop ascending, sigma ascending, from_step ascending"}`

## 2026-10-30 — M0.1 收尾核验与 M1 状态


- 完整噪声 run：`20261028T131534016226Z_83bfd1aab8dd_d4e55b`；54/54 设置-seed，300题、K16、seed11/22/33，259200次rollout。冻结清单71个文件全部SHA-256匹配。
- 划分：train385533、val300、cal193、test1312个problem ID；各划分内无重复、两两无交集。这里的train是来源语料ID登记，不代表已完成M1训练数据或潜状态dump。
- 回归命令：`PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/m0_tests.py configs/M0_tests.yaml milestone=M0.1 stage=closure_tests`。
- 回归 run：`20261030T010511419639Z_a559efd03407_0bfab2`；23 passed、1 skipped；skip为独立Huginn环境依赖相关检查，不计为通过。
- 噪声成本35.3689 GPU wall小时、3428312次源forward；回归测试未调用源checkpoint推理。

## 主要结果与限制

| 设置 | 准确率（均值，95% CI） | 非退化比例（均值，95% CI） |
|---|---|---|
| 无噪声贪心 |42.333% [36.667%,48.000%]|0%|
| sigma1.5，from_step3 |37.389% [37.12%,37.65%]|32.889% [30.81%,34.97%]|
| sigma0，temperature1.0 |40.097% [39.59%,40.60%]|31.778% [28.04%,35.51%]|

确定性区间按题bootstrap10k；随机行区间来自三个seed均值的Student-t，表示固定面板的seed变动，不是新题泛化区间。逐seed按题bootstrap另存原run/summary.json。推荐噪声候选准确率下降4.944pp，仅距5pp门槛0.056pp；不得宣称稳健满足门槛。温度对照不替换预先规定的sigma选择规则。

官方bf16复放99.924%一致；fp32生产协议95.299%，用户已接受区别。bf16残差1题定位到首次投影数值路径，底层dispatch原因未完全确定。固定batch1/fp32协议不宣称跨batch不变。去重是保守文本筛查，不能证明发布checkpoint的实际训练参与记录。

完整协议：[M0.1_protocol.md](M0.1_protocol.md)。噪声结果及图：[M0.1_noise_full_v2.md](M0.1_noise_full_v2.md)。上述报告中的历史“等待完整噪声”状态由本收尾记录补充，不修改冻结结果。


收尾报告：`reports/M0.1_closure_v2.md`。M1未启动：用户选择另行提供M1任务书，不将数据管线建议视为已确认范围。

## 2026-10-30 — M1 新任务书与实现（未完成）

M0.1审查通过；用户接受CODI-LLaMA现状、噪声延到M6前，Huginn暂停。M1只生成CODI-GPT2数据，不训练reader/probe。新任务书替代原M1，源模型调用标签提前完成。详见 `reports/M1_data.md`。

候选准备：`PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_prepare.py configs/M1_data.yaml`，成功run `20261030T011913127306Z_cb2983bfb3ec_109195`，20000/2000/300/500/1005题，尚待smoke预算后冻结v3。5项新增单元测试通过。

200题smoke命令：`PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_smoke.py configs/M1_data.yaml stage=smoke device=cuda:2`，run `20261030T012126501342Z_c7c7a651e47e_6eef7a`。900秒截止，代表性工作量定义和不足见M1报告；未完成时不填准确率或成本总数。完整val300批处理验收在smoke成功后自动运行；未通过不会全量生成。

## M1 smoke 超时与显式修正（2026-10-30）

首轮GPU2 smoke run `20261030T012126501342Z_c7c7a651e47e_6eef7a` 在900秒完成161/200题，failed，23655次forward、0.2500001GPU小时；未计作通过。按正式split分工修正：同200题clean+7个早退，100train做K8轨迹，100val做两种K16单步MC；完整阶段仍做所有t，配置未降精度/模型。修正版run `20261030T013639323603Z_13ad784ef5d6_518bf2`正在运行。预算按阶段实际题数估计，包含失败尝试成本。详细状态见 `reports/M1_data.md` 与 `reports/evidence_M1/full_job.json`。

## 2026-10-30 — M1 200题smoke通过，进入val300关卡

修正版run `20261030T013639323603Z_13ad784ef5d6_518bf2` 完成200/200，898.671秒，0.249631GPU小时，22388次源forward；全部200题早退T与正常推理一致，前三题×3seed的M0.1 from_step3反馈向量逐元素一致。前一版超时不改写为成功。

|存储子集|状态行数|字节|fp16最大绝对误差|最大逐向量相对L2|RMSE|
|---|---:|---:|---:|---:|---:|
|100 reader_train（含K8轨迹、前10全层）|5147|15867184|0.0604553|0.000439416|0.000596085|
|100 val（clean）|700|2189950|0.0307541|0.000369053|0.000361824|

真实smoke产物已检查800条轨迹、800个terminal状态、problem无跨split、fp16读取正确。数据完整性检查不是3seed重复性能实验，不制造重复seed的零方差CI；三seed批处理比较在下一个关卡进行。

独立上下文核验run `20261030T014942634492Z_d617fee4e7ef_fbf580`：val300 padding全部与M0.1噪声面板一致，原test1312恢复历史1319队列的cohort，其中284题候选重组宽度需要修正。尚未对cal/test做任何M1推理，v3最终冻结时应用该恢复映射，原候选不覆盖。

smoke后成本估算run `20261030T015310339508Z_f61e4a4d38d9_378f0f`：保留20000 reader_train的完整生成预计38.408 GPU小时（已加25%吞吐余量，不含最终关卡成本），预计6.816GiB（2倍存储余量）。最终预算还要加入所有失败smoke/关卡的实耗；仅预算最终通过后冻结v3并启动全量。

val300关卡正在run `20261030T015147548907Z_f9aeb43dc9b2_daa0ae` 执行，尚无最终通过结论。

## M1 pipeline — generation_failed

Runs: {"smoke_run": "runs/20261030T013639323603Z_13ad784ef5d6_518bf2", "gate_run": "runs/20261030T015147548907Z_f9aeb43dc9b2_daa0ae", "budget_run": "runs/20261030T022507631329Z_ae2889e005b8_1fe957", "generation_run": "runs/20261030T022518234836Z_39c11741d8fb_f9026e"}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job.json`. No failed gate was bypassed.

## 2026-11-01 — M1 状态核查与恢复

val300验收已通过，run `20261030T015147548907Z_f9aeb43dc9b2_daa0ae`：早退T一致率100%，原fp32与fp16往返状态的零噪声批处理续算均无答案不一致；三seed噪声比较通过。v3已冻结：reader_train20000 / reader_val2000 / val300 / cal500 / test1005，预算39.462GPU小时（含预检）、6.816GiB，没有缩减规模。

此前全量run `20261030T022518234836Z_39c11741d8fb_f9026e` 在磁盘预检退出（0次源forward、0GPU小时），不是持续运行或数据生成完成。2026-11-01复查磁盘配重自动填充已恢复开启（谁恢复未核实），当前free18.628GiB。按用户已有授权release10GiB并持久暂停自动填充，恢复空间约28.6GiB；GPU2实测空闲14.35GiB。

原输出目录只有空目录，使用rmdir移除（非空会拒绝），未删除任何数据。旧失败状态和log复制为 `reports/evidence_M1/full_job_disk_failure_20261030.*`。使用相同已冻结配置显式重启，tmux `clr-m1-full-restart`，日志 `reports/evidence_M1/full_job_restart_20261101.log`。M1仍未完成，成功与否以新run/metrics.json为准。

确认重启已进入真实源模型计算：新run `20261031T170800331382Z_39c11741d8fb_ec36b0`（北京时间2026-11-01），config SHA与旧失败尝试完全一致。检查时已完成4题、417次forward；当前磁盘约29GiB，物理GPU2。此前失败run与所有验收结果均保留。

## M1 pipeline — generation_failed

Runs: {"smoke_run": "runs/20261030T013639323603Z_13ad784ef5d6_518bf2", "gate_run": "runs/20261030T015147548907Z_f9aeb43dc9b2_daa0ae", "budget_run": "runs/20261030T022507631329Z_ae2889e005b8_1fe957", "generation_run": "runs/20261031T170800331382Z_39c11741d8fb_ec36b0"}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job.json`. No failed gate was bypassed.


## 2026-11-01 — 全量中断后状态核验

全量 run `20261031T170800331382Z_39c11741d8fb_ec36b0` 于北京时间11月1日10:25因磁盘余量低于1GiB保护退出；当前无生成进程。计算进度 reader_train 10147/20000，耗时9.291732 GPU小时，1105526次源forward。持久化manifest含10112题，早退日志含10140题；内存中的末尾不能视为已落盘。现有文件保留，其他split尚未完成。当前磁盘恢复约27.9GiB、配重自动填充仍暂停；运行期间空间骤降原因尚未核实。原生成器禁止覆盖已有输出且不支持恢复，需要验证分片与行为日志边界后实现显式续跑，不能直接原命令重跑或宣称M1完成。


## 2026-11-01 — M1 v2 显式断点恢复

用户要求继续且不因磁盘余量停止。新增 `scripts/m1_generate_v2.py`、`clr/data/resume.py` 和 `configs/M1_data_v2.yaml`：在新目录 `data/dumps/codi_gpt2_v2` 复用完整分片，原结果不覆盖。核验连续分片、张量索引、冻结ID前缀、生产cohort上下文、7步clean、每题8条完整轨迹及早退日志后才续跑。未落盘尾部重新按原种子计算；重复计算成本保留。

取消低余量主动退出，改为低于20GiB时调用配重release补至约25GiB；已提前释放30GiB并保持自动填充暂停。实际I/O失败仍如实报错；60GPU小时（包含前次9.291732小时和预检）及8GiB产物预算保持。恢复逻辑及原policy/data测试6项通过，首次测试发现可变列表共享已修复。

已知统计限制：原中断运行没有持久保存fp32→fp16误差聚合，不能从fp16文件反推；原分片写入时已执行误差阈值检查，恢复后误差汇总仅涵盖新生成状态，报告必须标明，不冒充全量误差统计。尚未完成M1，不训练reader/probe。


2026-11-01 北京时间21:54启动v2续跑，run `20261101T135435455123Z_ed305481698c_e92e9a`，物理GPU2；命令 `PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_finish_job_v2.py configs/M1_data_v2.yaml device=cuda:2`。恢复核验316个文件/469806行/10112题通过，审计见 `reports/evidence_M1/recovery_audit_v2.json`。自动完成链将继续生成剩余split，计算标签统计和图，更新报告及冻结清单。当前仍在运行，不是完成结果。


## 2026-11-02 — M1续跑进度与空间管理修正

当前v2运行仍在执行。reader_train20000、reader_val2000、val300、cal500已完成；检查时test492/1005，总23292/23805题。续跑耗时11.702GPU小时，含此前中断生成累计20.994GPU小时（另计预检），尚未进入最终报告生成。

发现reserve自动释放参数为小数，CLI只接受整数，历次调用返回2；生成未因该错误停止。已修正为向上取整，并新增回归测试，两项恢复测试通过。当前在途进程仍使用内存中的旧函数，未声称热更新；已显式release30GiB成功，余量约52GiB，自动填充保持暂停，以保障本次剩余生成。研究协议、结果和种子未改变。


## 2026-11-02 — 服务器重启后 v3 恢复

用户确认服务器重启并要求续跑。原v2进程已消失，metrics残留running是重启前快照，不代表仍在执行。重启前test798/1005、v2成本12.800449GPU小时/1606828次forward；原失败生成9.291732GPU小时也计入预算。四个已完成split复用，不重跑。test已持久化768题/5376状态行、9216条MC记录和44条K64记录，通过冻结前缀、cohort、完整步骤和行为日志核验；第769题起重算，共剩237题。

新代码与配置v3、新输出 `data/dumps/codi_gpt2_v3`，原v1/v2不覆盖。已完成文件用硬链接复用，生成器不修改复用文件；本次记录SHA和祖先run成本。重启恢复与policy/data测试8项通过。物理GPU2当前空闲，显式继续使用；release20GiB成功且自动填充保持暂停。自动配重调用已使用整数参数。

命令：`PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_finish_job_v3.py configs/M1_data_v3.yaml device=cuda:2`。tmux `clr-m1-reboot-v3`，状态 `reports/evidence_M1/full_job_v3.json`。保持原协议、划分、seed及全部实验设置；中断test段的量化误差聚合未持久化，恢复后的test误差统计仅覆盖新生成部分，限制与reader_train一致，不能称为全量统计。

## M1 pipeline — generation_completed

Runs: {"smoke_run": "runs/20261030T013639323603Z_13ad784ef5d6_518bf2", "gate_run": "runs/20261030T015147548907Z_f9aeb43dc9b2_daa0ae", "budget_run": "runs/20261030T022507631329Z_ae2889e005b8_1fe957", "generation_run": "runs/20261102T031137055963Z_24b03138a255_a2c9f2"}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job_v3.json`. No failed gate was bypassed.


## 2026-11-03 — 完成状态与文档同步

M1 v3数据及报告均已完成，待用户审查。共23,805题，产物约3.03GiB，生成累计22.615373GPU小时（另计预检），770个数据文件及6个冻结文件SHA复核通过。当前接续索引为 `docs/STATUS.md`，`AGENTS.md` 已同步，后续状态变化必须同时持久化。保留reader_train/test中断部分量化误差聚合缺失、cohort依赖等限制；无新训练任务启动，原冻结结果未改动。


## 2026-11-03 — M1审查通过，M2–M3启动

用户授权合并里程碑，当前不训练CLR。输入隔离源题目文本，状态只用模型实际输出。Q1规则经用户明确：clean仅轨迹起点独立Bernoulli标签及clean终点，traj起点之后用自身观测及终点；不监督更早clean前缀。reader_val报告clean/traj分别指标和各t样本覆盖。cal_A/cal_B250/250已固定。

QA首轮smoke因少量题不足4种答案失败，未覆盖；固定训练集数值池补足后run `20261103T042122100178Z_172c4808908a_81b8b5`通过，输出 `data/M23_v1a/smoke`。全量QA任务tmux `clr-m23-questions`。Q4成对交换正值作为对方负值，保证保留问题上数值边际精确一致；未匹配/解析失败单独统计。全量不得使用失败smoke。

源调用范围冲突按具体要求解释为B5a/C/D/E例外，其他仅读M1；不新增轨迹。GPU当前均忙，显式使用cuda3训练轻量probe、cuda1量化复算，负载与实际OOM如实记录；没有替换模型或降精度。D候选许可和谱系正在核实；尚无完整基线结论或M2–M3完成声明。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "running", "asdiv_run": "running", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "running", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "running", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23结构审查修正：中止基线run20261103T044101470059Z_19e9238e5441_14ac1a（先前已保存尝试保留不纳入最终）。B0/B1/B3改为对z+模板特征+选项数值特征联合两层MLP，Q3共享逐选项打分；避免旧加法结构无法表达Q4状态×数值交互。新增交互测试，7项测试通过。此为实现修正，未按test结果选择模型或seed；超参网格/样本量保持。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

2026-11-03 M23收尾验证：全套单元测试38 passed、1 skipped（reports/M23_tests.log）。官方500题79.8%，同题M1为79.4%；SVAMP993题43.001%，ASDiv1000题54.9%，两池完整评估合计0.410463 GPU小时。量化抽查200题的重算fp32转fp16与保存值完全一致。完整三seed基线与B5a仍在运行；不宣称里程碑已完成。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

2026-11-03 B5a batch门槛通过：run 20261103T053137666964Z_b807ffac4af3_0d5c8e；四类各128条固定reader_val实例，答案一致100%，max概率差2.27094e-5<1e-4；max原始logit差0.00120163。batch64 smoke运行中，旧batch8完整运行继续，未替换最终结果。导入顺序失败已解决且记录，未模型/精度降级。

2026-11-03 B5a Q3 reader_val已完成（source_forward_calls累计69253）；batch64 smoke完成但耗时78.75秒，原batch8为55.72秒，未见吞吐收益（不同共享GPU，不能归因为batch本身），因此不采用v2替换完整run。保留gate及smoke结果、额外成本；最终协议仍fp32/batch8。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。

M23自动状态更新：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "completed", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`。运行索引 `data/M23_v1a/pipeline_status.json`。


## M2–M3最终完整运行（2026-11-03）

完成类型化问题库、B0–B5a、Mondrian、域/层诊断、未见池和M1量化补查。未训练CLR。完整命令以各run配置快照为准；报告入口 `.venv/bin/python scripts/m23_report.py configs/M23_report.yaml`。

三seed主表（均值±95% CI，B5a另按题bootstrap）：


|baseline|qtype|split|accuracy|AUROC|NLL|Brier|MAE|ECE|
|---|---|---|---|---|---|---|---|---|
|B0|Q1|reader_val|0.7523 ± 0.0000|NA|0.6886 ± 0.0000|0.2478 ± 0.0000|0.4977 ± 0.0000|0.2478 ± 0.0000|
|B0|Q1|test|0.3598 ± 0.0000|NA|0.6957 ± 0.0000|0.2513 ± 0.0000|0.4481 ± 0.0000|0.1447 ± 0.0000|
|B0|Q2|reader_val|0.6829 ± 0.0000|0.4963 ± 0.0198|0.6906 ± 0.0001|0.2487 ± 0.0001|0.4987 ± 0.0001|0.1882 ± 0.0001|
|B0|Q2|test|0.3248 ± 0.0000|0.5082 ± 0.0227|0.6956 ± 0.0001|0.2512 ± 0.0001|0.5012 ± 0.0001|0.1787 ± 0.0002|
|B0|Q3|reader_val|0.3470 ± 0.0009|NA|1.3340 ± 0.0066|0.7251 ± 0.0019|NA|0.0395 ± 0.0107|
|B0|Q3|test|0.3594 ± 0.0079|NA|1.3225 ± 0.0029|0.7202 ± 0.0015|NA|0.0372 ± 0.0141|
|B0|Q4|reader_val|0.5014 ± 0.0036|0.5017 ± 0.0018|0.6933 ± 0.0003|0.2501 ± 0.0001|0.5000 ± 0.0000|0.0075 ± 0.0100|
|B0|Q4|test|0.4975 ± 0.0074|0.4987 ± 0.0050|0.6934 ± 0.0004|0.2501 ± 0.0002|0.5000 ± 0.0001|0.0111 ± 0.0130|
|B1|Q1|reader_val|0.7523 ± 0.0000|NA|0.6888 ± 0.0001|0.2478 ± 0.0001|0.4978 ± 0.0001|0.2480 ± 0.0002|
|B1|Q1|test|0.3598 ± 0.0000|NA|0.6955 ± 0.0001|0.2512 ± 0.0001|0.4480 ± 0.0001|0.1445 ± 0.0002|
|B1|Q2|reader_val|0.6858 ± 0.0021|0.4956 ± 0.0183|0.6893 ± 0.0001|0.2481 ± 0.0000|0.4981 ± 0.0000|0.1836 ± 0.0026|
|B1|Q2|test|0.3633 ± 0.0268|0.4944 ± 0.0338|0.6953 ± 0.0002|0.2511 ± 0.0001|0.5010 ± 0.0001|0.1441 ± 0.0173|
|B1|Q3|reader_val|0.3294 ± 0.0033|NA|1.3541 ± 0.0013|0.7349 ± 0.0009|NA|0.0283 ± 0.0075|
|B1|Q3|test|0.3518 ± 0.0030|NA|1.3421 ± 0.0016|0.7283 ± 0.0007|NA|0.0335 ± 0.0079|
|B1|Q4|reader_val|0.4997 ± 0.0090|0.4986 ± 0.0075|0.6932 ± 0.0000|0.2500 ± 0.0000|0.5000 ± 0.0000|0.0063 ± 0.0049|
|B1|Q4|test|0.4996 ± 0.0038|0.4992 ± 0.0087|0.6932 ± 0.0001|0.2500 ± 0.0000|0.5000 ± 0.0000|0.0083 ± 0.0089|
|B2|Q1|reader_val|0.8304 ± 0.0018|NA|0.3958 ± 0.0026|0.1240 ± 0.0008|0.2831 ± 0.0037|0.0654 ± 0.0053|
|B2|Q1|test|0.7169 ± 0.0031|NA|0.5532 ± 0.0056|0.1868 ± 0.0022|0.3273 ± 0.0053|0.0430 ± 0.0077|
|B2|Q2|reader_val|0.7955 ± 0.0016|0.8515 ± 0.0058|0.4298 ± 0.0065|0.1391 ± 0.0023|0.2953 ± 0.0028|0.0412 ± 0.0013|
|B2|Q2|test|0.7464 ± 0.0059|0.8362 ± 0.0056|0.5228 ± 0.0015|0.1734 ± 0.0004|0.3504 ± 0.0015|0.0418 ± 0.0045|
|B2|Q3|reader_val|0.5840 ± 0.0033|NA|0.9518 ± 0.0014|0.5306 ± 0.0011|NA|0.0648 ± 0.0051|
|B2|Q3|test|0.5335 ± 0.0036|NA|1.0155 ± 0.0051|0.5684 ± 0.0021|NA|0.0524 ± 0.0086|
|B2|Q4|reader_val|0.5005 ± 0.0055|0.4994 ± 0.0046|0.6931 ± 0.0000|0.2500 ± 0.0000|0.5000 ± 0.0000|0.0036 ± 0.0019|
|B2|Q4|test|0.5010 ± 0.0085|0.5002 ± 0.0064|0.6931 ± 0.0000|0.2500 ± 0.0000|0.5000 ± 0.0000|0.0035 ± 0.0028|
|B3|Q1|reader_val|0.8375 ± 0.0041|NA|0.3752 ± 0.0009|0.1182 ± 0.0004|0.2626 ± 0.0015|0.0494 ± 0.0037|
|B3|Q1|test|0.7520 ± 0.0046|NA|0.5228 ± 0.0073|0.1733 ± 0.0029|0.3026 ± 0.0065|0.0345 ± 0.0049|
|B3|Q2|reader_val|0.8093 ± 0.0067|0.8674 ± 0.0014|0.3976 ± 0.0036|0.1291 ± 0.0011|0.2723 ± 0.0081|0.0308 ± 0.0057|
|B3|Q2|test|0.7603 ± 0.0133|0.8460 ± 0.0056|0.5183 ± 0.0167|0.1698 ± 0.0074|0.3453 ± 0.0157|0.0484 ± 0.0053|
|B3|Q3|reader_val|0.6565 ± 0.0054|NA|0.7718 ± 0.0081|0.4374 ± 0.0036|NA|0.0553 ± 0.0126|
|B3|Q3|test|0.5859 ± 0.0019|NA|0.8860 ± 0.0035|0.5048 ± 0.0004|NA|0.0412 ± 0.0039|
|B3|Q4|reader_val|0.7142 ± 0.0050|0.7797 ± 0.0052|0.5568 ± 0.0021|0.1891 ± 0.0013|0.3854 ± 0.0027|0.0227 ± 0.0081|
|B3|Q4|test|0.6826 ± 0.0109|0.7451 ± 0.0053|0.5889 ± 0.0044|0.2027 ± 0.0018|0.4055 ± 0.0005|0.0187 ± 0.0058|
|B4|Q1|reader_val|0.8299 ± 0.0017|NA|0.3955 ± 0.0016|0.1237 ± 0.0008|0.2768 ± 0.0022|0.0531 ± 0.0035|
|B4|Q1|test|0.7526 ± 0.0032|NA|0.5067 ± 0.0025|0.1676 ± 0.0009|0.2910 ± 0.0010|0.0323 ± 0.0035|
|B4|Q2|reader_val|0.7793 ± 0.0015|0.8479 ± 0.0003|0.4550 ± 0.0052|0.1486 ± 0.0015|0.3095 ± 0.0036|0.0710 ± 0.0044|
|B4|Q2|test|0.7272 ± 0.0028|0.8450 ± 0.0028|0.5340 ± 0.0010|0.1794 ± 0.0003|0.3595 ± 0.0004|0.0469 ± 0.0028|
|B5a|Q1|reader_val|0.2770 (deterministic)|NA|0.8096 (deterministic)|0.3065 (deterministic)|0.5438 (deterministic)|0.3224 (deterministic)|
|B5a|Q1|test|0.6282 (deterministic)|NA|0.6580 (deterministic)|0.2327 (deterministic)|0.4176 (deterministic)|0.0487 (deterministic)|
|B5a|Q2|reader_val|0.3716 (deterministic)|0.5185 (deterministic)|0.7906 (deterministic)|0.2965 (deterministic)|0.5314 (deterministic)|0.2336 (deterministic)|
|B5a|Q2|test|0.6360 (deterministic)|0.4495 (deterministic)|0.6612 (deterministic)|0.2340 (deterministic)|0.4686 (deterministic)|0.0826 (deterministic)|
|B5a|Q3|reader_val|0.2534 (deterministic)|NA|1.3861 (deterministic)|0.7499 (deterministic)|NA|0.0098 (deterministic)|
|B5a|Q3|test|0.2455 (deterministic)|NA|1.3867 (deterministic)|0.7502 (deterministic)|NA|0.0180 (deterministic)|
|B5a|Q4|reader_val|0.5035 (deterministic)|0.5014 (deterministic)|0.6933 (deterministic)|0.2501 (deterministic)|0.5000 (deterministic)|0.0057 (deterministic)|
|B5a|Q4|test|0.5037 (deterministic)|0.5010 (deterministic)|0.6933 (deterministic)|0.2501 (deterministic)|0.5000 (deterministic)|0.0057 (deterministic)|


成本与偏离：


|stage|run|GPU hours|source forwards|
|---|---|---|---|
|question_run|runs/20261103T042241646739Z_d327b6a2b4fa_428508|0|0|
|baseline_run|runs/20261103T045628621237Z_2964c31f2153_11a348|0.5343011922869442|0|
|patchscope_run|runs/20261103T043008727615Z_3d1392832851_85503d|1.364403531752222|88010|
|layers_run|runs/20261103T045629556778Z_c1273b972f55_418497|0.04864521111444143|0|
|quant_run|runs/20261103T042503115213Z_b650da2692e8_d82a89|0.009226457894441814|1400|
|official_run|runs/20261103T043032691526Z_7213a3d8ae2c_852e9d|0.005168258419726044|61|
|svamp_run|runs/20261103T043414920503Z_0ffa8eb65906_5d11c6|0.19991807649610968|54668|
|asdiv_run|runs/20261103T043815088245Z_a0358d220af9_70e23b|0.21054489036000126|56360|

所有M2–M3已记录运行（含smoke、重复运行）的成本下界：2.6075 GPU小时、202641次源前向；逐run记录及缺失项：`/home/me/PaperField/CalReader/runs/20261103T055223163924Z_faee26776ad3_cea0d0/cost_ledger.json`。缺失计数的失败/中断run意味着这不是精确总成本。

B5a额外检查batch64：固定reader_val50题、四类各128实例，答案一致100%，max概率差2.27094e-5，低于预设1e-4；但smoke为78.7秒，原batch8为55.7秒。不同共享GPU上的时间受资源竞争影响，未观察到加速，因此完整结果保留原batch8，不采用v2。额外gate/smoke及启动前两次入口错误均留存，未据test指标选择batch。

GPU为NVIDIA GeForce RTX 4090，设备及环境快照见各run；GPU小时按设备占用墙钟计，包含CPU预处理等待。上表为选用的完整run，不能当作包含所有smoke/失败/被替代运行的总成本。部分早期中断未写出最终计数，成本缺口保留，不按零处理。

QA在提交代码后完整重建，五个split parquet逐字节SHA一致，见data/M23_v1a/qa_reproduction.json。初期未提交脚本的运行另有明确标注为事后保存的m23_source_audit；不冒充启动时快照。基线首版的Q4加性结构不能表达状态与数值交互，已在完整评估完成前中止，旧运行已写出部分测试预测但未据此选择结构；按结构检查改用联合两层MLP v2并重新smoke；没有据test结果选模型。

命令：`PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python scripts/m23_baselines.py configs/M23_baseline_v2.yaml mode=full smoke_run=runs/20261103T045443038876Z_eb5c2140141e_437383`；其余入口及配置快照见本表各run。测试记录：reports/M23_tests.log。


负面结果与三项审查问题见 reports/M3_baselines.md 末尾；不能将这些基线结果作为CLR有效性结论。


M2–M3问题库、B0–B5a、层/域诊断、官方500题、两候选未见池与量化抽查均完成，待用户审查。B5b推迟M6，未训练CLR。报告：reports/M2_questions.md、M3_baselines.md、M3_shift.md、M3_unseen_pools.md；冻结索引 data/M23_v1a/freeze.json。


## M2–M3最终核验（2026-11-03）

实现、全部必需实验、四份报告及图已完成，待用户审查；未训练CLR，B5b按任务许可推迟M6。Q1按用户确认的起点/后续观测状态Bernoulli规则执行。
冻结清单 `data/M23_v1a/freeze.json` 的635个文件SHA全部一致；核验记录 `data/M23_v1a/verification.json`。51个基线checkpoint、216份基线预测、16份B5a预测齐全。测试40 passed；1个已暂停Huginn模块因可选lm_eval缺失跳过。最终报告run `runs/20261103T055223163924Z_faee26776ad3_cea0d0`。
已记录成本下界2.6075 GPU小时、202641次源前向，历史中断缺失成本未记成零。D两池含smoke合计0.417348 GPU小时，低于5小时。
必须随结果保留的审查限制：早期smoke触及10题test，旧中止run保存过部分test预测，未用于调参但不符合严格test-only-final约定；不得声称该偏离已获接受。Q4边际匹配在原split保证，cal_A/B半集没有独立重配。conformal依赖假设、来源未核实候选池及成本缺口见BLOCKERS；不更换划分/seed掩盖问题。
B3 test Q3准确率58.59% ± 0.19pp，Q4 68.26% ± 1.09pp（三seed95% t区间、clean/seen/t等权）；B5a Q3约24.55%、Q4约50.37%，如实保留负面结果。M3_baselines.md末尾已逐项回答三项审查问题。
