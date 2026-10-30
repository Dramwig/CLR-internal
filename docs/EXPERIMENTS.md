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
