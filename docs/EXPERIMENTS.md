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

## M4 v0 — smoke与正式训练启动（2026-11-03，未完成）
模型采用官方固定revision的ModernBERT-base；val-only源/reader/校准评估smoke通过，单元测试9通过；尚无最终test结论。
命令：`PYTHONPATH=. .venv/bin/python scripts/m4_train_suite.py configs/M4_train_v0.yaml`；D与QA由 `scripts/m4_data_followup.py configs/M4_data_followup.yaml` 接续。
吞吐26.6训练样本/s、93.8评估样本/s、峰值2.42GiB，smoke置换差0；这些是工程验收数，不是三seed科研结果。
完整估算66.6 GPUh（含20%裕量），上限80；D估算0.451，上限6。详见 [预算](../reports/M4_budget.md) 与 [任务](M4_TASK.md)。
负面/限制：172条未见eval与新train存在13-gram重叠；259题公式解析失败；均保留记录，不改划分。控制器状态与阈值JSON边界错误已修复，失败run留存。
最终三seed指标、按题bootstrap与G1结论尚待完成，不能以本记录代替里程碑验收。

## M4 v0 — 完整实验完成，待G1审查
执行：`scripts/m4_train_suite.py configs/M4_train_v2.yaml`，`scripts/m4_pipeline.py configs/M4_pipeline.yaml`（均使用项目Python与PYTHONPATH=.）。
三seed11/22/33全部纳入；均值±95%CI、按题10000次配对bootstrap、核心结果表见 [唯一审查文件](../reports/M4_REVIEW.md)。
完整指标与负面结果见 [完整报告](../reports/M4_clr_v0.md)，所有t/模板/clean-traj结果在 `reports/M4_metrics.csv`。
预算与实际成本见报告；可选消融未执行、未见池重叠/解析失败及修复记录如实保留。完成不等于G1通过。
冻结：`data/M4_v0/freeze.json`；M1/M23输入冻结哈希复核一致。未经审查不进入M5。

## M4.1 — A诊断与B学习能力关卡完成，主比较未完成
命令：`scripts/m41_diagnose.py configs/M41_diagnose_v2.yaml`；`scripts/m41_gate_controller.py configs/M41_gate_controller_retry_v2.yaml`。
原v0存在非零状态→选项通路，合成Q3/Q4为98.6%/99.6%；1k Q4原配置81.2%，降低LR并追加训练97.0%；曝光不足已确认，未定位单个断路算子。
候选v1/lite的1k Q4拟合99.2%/99.3%，合成Q3均100%、Q4为98.9%/99.2%，六项≥98%工程门槛通过。上述为固定seed诊断，不冒充三seed泛化结果或CI。
根因≤3KB及详细证据见[当前审查摘要](../reports/M4.1_REVIEW.md)与data/M41_v1/diagnosis_complete.json、gate_lock.json；M4冻结12946文件SHA一致。
C真实入口val smoke运行中，最终test仍未访问；最终三seedCI与四项G1待完整实验，不提前宣布修复有效。

## 2026-11-05 — M5a预检完成，全量待预算/范围决定
实现S0–S3、B4/B3(Q1)/oracle K16及冻结val-only入口；9测试通过，7个冻结probe/normalization哈希一致，未触碰test。
完整网格5题smoke261秒、50.2秒/题/seed；val300三seed外推12.5GPUh，含20%裕量/已有成本约15.5>10，未启动全量。
备选同5题smoke116秒：仅S3 oracle限N16/b4全步，其他原设置不变；总估算6.88GPUh，尚未获准用于全量。
命令/配置/run/成本CSV及负面工程记录见[唯一审查文件](../reports/M5a_REVIEW.md)和[详细报告](../reports/M5a_preflight.md)。
研究均值±95%CI和accuracy–compute未执行；不能用smoke回答oracle上限或probe比例。预检保守计费0.354GPUh，含中断成本上界预留，非精确总数。

## M4.1 bf16通路复核与恢复验证（2026-11-05）
`configs/M41_diagnose_bf16.yaml`对应run `20261105T032226938789Z_933a64cec066_27aa41`：80/80批z梯度及置换中心化logit变化非零，0.03255GPUh，test未打开。
seed11显式checkpointing恢复后训练正常；7项M4.1测试通过。主比较仍运行，未产生最终三seed指标。证据见data/M41_v1/diagnosis_complete.json及docs/M41_LIVE.json。

## 2026-11-05T06:03:47.954732+00:00 — M5a v13继续授权
用户“允许 继续完成”后执行10GPUh备选，S3 oracle限N16/b4全步；旧预检冻结不改。
固定5题val smoke102秒、250条通过；9测试通过，各N独立池/价值成本纳入。
配置configs/M5a_full_v13.yaml；预算data/M5a_v13/budget_plan.json，估算5.89GPUh。
尚无完整研究结论；协议与预先固定分析见docs/M5a_TASK.md。

## 2026-11-05T11:34:44.657867+00:00 — M5a v13完成待审查
M5a val300×seed11/22/33完成；oracle同预算S1增益32.8pp，95%CI[27.8,38]（题内平均seed后按题bootstrap10,000次）。
命令：scripts/m5a_search.py configs/M5a_full_v13.yaml；scripts/m5a_report.py configs/M5a_report_v13.yaml。
全部B4/B3/S1/S2网格，S3 oracle按继续授权限N16/b4全步；负面结果与比例不稳定情况均保留。
累计保守5.80961GPUh含smoke/失败/中断预留；无test访问。
唯一审查reports/M5a_REVIEW.md；完整表/CI/曲线/成本reports/M5a_search.md及data/M5a_v13/analysis/；冻结data/M5a_v13/freeze.json。

## M4.1 用户授权提高预算（2026-11-05）
总上限35→50 GPUh，新训练单run时限9→14h；设置和seed不变，旧成本全部计入。
旧进程不被打断或修改快照；必要时从epoch checkpoint新建恢复run。接续configs/M41_budget50_resume.yaml，详见M4.1_TASK.md。

## M4.1 完整实验完成，待审查
诊断→学习能力门槛→全曝光三seed主比较→val选型迁移→最终校准与Mondrian完成。
命令：`scripts/m41_pipeline.py configs/M41_pipeline.yaml`；三seed均值±95%CI及10k按题bootstrap见[审查报告](../reports/M4.1_REVIEW.md)。
全部结果、成本、负面结论与协议偏离见[详细报告](../reports/M4.1_details.md)；冻结data/M41_v1/freeze.json，M4不改。

## 2026-11-06T05:13:28.693528+00:00 — M5b完成待审查
A重训前缀/单状态消融、B同父MC诊断、C含clean搜索v2与D排序微调，三seed及10k题bootstrap完成。
命令：scripts/m5b_finish.py configs/M5b_finish.yaml；scripts/m5b_report.py configs/M5b_report.yaml。
三seed均值/95%CI、全部负面结果与偏离见[唯一审查](../reports/M5b_REVIEW.md)及[详细报告](../reports/M5b_details.md)。
累计16.8066 GPUh≤25，含失败/smoke/无效run；B/C/D无test，A早期无效test暴露保留并排除。
冻结data/M5b_v1/freeze.json；run/CSV/曲线/成本索引data/M5b_v1/completion.json与analysis/index.json。

## 2026-11-06T05:16:33.248867+00:00 — M5b完成待审查
A重训前缀/单状态消融、B同父MC诊断、C含clean搜索v2与D排序微调，三seed及10k题bootstrap完成。
命令：scripts/m5b_finish.py configs/M5b_finish.yaml；scripts/m5b_report.py configs/M5b_report.yaml。
三seed均值/95%CI、全部负面结果与偏离见[唯一审查](../reports/M5b_REVIEW.md)及[详细报告](../reports/M5b_details.md)。
累计16.8066 GPUh≤25，含失败/smoke/无效run；B/C/D无test，A早期无效test暴露保留并排除。
冻结data/M5b_v1/freeze.json；run/CSV/曲线/成本索引data/M5b_v1/completion.json与analysis/index.json。

## M6 A收尾与B/C/D smoke（2026-11-06）
A：val300、seed11/22/33，sigma=.3/.75、S3 N8/b4/6步；CLR相对greedy +.111/−.667pp，同时95%CI均跨0，RQ1保留负面分析；详见reports/M6_rq1_closure.md。
B：40train+2val source smoke 409秒，batch1024训练smoke、val覆盖率代理通过；11项测试通过，最终test尚未推断。
命令/全成本：configs/M6_*.yaml及data/M6_v1/runs/；控制器tmux clr-m6-pipeline等待完整noise关卡。M6未完成，接续docs/M6_LIVE.json。

## 2026-11-06 — M7 环境、独立KV验收与smoke
命令 `.venv-comm/bin/python scripts/m7_acceptance.py configs/M7/{smoke,acceptance_full,smoke_cot}.yaml`（逐次独立运行）。
历史8题receiver logits与重建消息误差0；新raw KV两路径逐元素一致率100%。direct固定val80：message/none/mismatch=45.0/26.3/23.8%。
cot512 val8在11.9分钟完成：62.5/50.0/12.5%；非同规模比较，不据此声称全量cot已验收。
源码/标签/选项置换/边界等6项测试通过；source smoke_v2通过2个val基础episode（14秒），不读test。
完整val长度估算与HiddenBench格式核验见data/M7_v1/plan.json；仅synthetic进入主比较，M7尚未完成。

## M6 B1扩展扫描完成（2026-11-06）
val300/K16/相对t/seed11，预登记sigma=3/4/6/8均下降≥5pp；完整八点无同时满足两门槛者。
按用户事前规则选sigma=2：下降1.99pp、非退化14.8%，门槛未达，降级使用；LLaMA Q1单列。
扩展含smoke/失败/重启1.64GPUh≤2；单seed噪声选择获任务授权，按题10000次CI见reports/M6_noise_selection_v2.md。
命令：.venv/bin/python scripts/m6_run_extension.py configs/M6_extension_pipeline.yaml；正式数据/训练尚未完成。

## 2026-11-06 — M7完成待审查
M7非自适应实验完成待审查；唯一审查reports/M7_REVIEW.md。冻结data/M7_v1/freeze.json；缩量/历史raw缺证据/1%无穷门限等限制见审查文件；未启动M8。
CF/test/k16/名义α=.05：CLR 0.575% [0,1.44]；B3 1.44% [0.287,2.87]；实测FPR与其他基线见唯一审查文件。
A1/A4 test CLR−B3：-0.018pp [-0.126,0.0898] / 0.251pp [-3.52,3.95]；累计7.73GPUh。
三seed主比较、按group bootstrap10000次95%CI、natural_commitment附加分析齐全；详见reports/M7_REVIEW.md、analysis/CSV。
命令/源码快照见configs/M7/、runs/；上游冻结哈希全部一致，test仅最终锁后评估。

## M6 2026-11-07
M6实验与统计完成，待审查；唯一审查reports/M6_REVIEW.md，冻结data/M6_v1/freeze.json。
接续：.venv/bin/python -m scripts.m6_pipeline configs/M6_pipeline_noise_v2.yaml；三seed配对CI及负面结果见审查/详细报告。
新增GPUh=47.6（含smoke/失败/重启）；成本data/M6_v1/analysis/costs.csv。

## 2026-11-07 — M7.1 source smoke
命令：.venv-comm/bin/python scripts/m71_source.py configs/M71/source.yaml；详情reports/M71_evidence/source_smoke_v2.log。
val12组21.5秒通过；实际KV到f的K/V梯度有限非零；CPU最小测试5 passed。
每组clean/CF特征约614kB；全规模含cal/test扰动约10.2GiB，sender保守6.23GPUh，尚非实际总成本。
首轮确定性cumsum失败已保留0.00650GPUh；无test访问、无模型或规模回退。
层扫描与主比较尚未完成；接续docs/M71_LIVE.json、任务docs/M7.1_TASK.md。

## 2026-11-07 — M7.1无损存储与附加协议smoke
fp16+gzip val12组23.4秒、raw controls12组25.4秒通过；28项最小测试通过。
旧/新GPU与存储3.56e6元素逐项一致、max误差0；完整数据估计9.05GiB，sender保守6.76GPUh。
M7 2912/M6 35688冻结文件与三生成器源码SHA一致；证据data/M71_v1/upstream_verification.json、storage_comparison.json。
用户授权receiver cal_A/B各300、natural val100/test300；都按组SHA固定，natural不进阈值或调参。
scan v1优化振荡保留，v2降低lr继续；C1/主比较未执行，test未打开。日志reports/M71_evidence/；任务docs/M7.1_TASK.md。

## 2026-11-07 — M7.1完成待审查
C1/C2通过，足量train8000/val1000/cal1500/test2000和三seed主比较/组bootstrap完成；累计15.9GPUh。
名义α5%的CF：CLR 93.1%[92.5,93.8]、B3 24.0%[22.4,25.5]、norm6.50%[5.45,7.60]；text300为0%；CLR实测FPR5.07%。
A1/A4较B3 +31.3/+53.8pp；A2/A3近随机，natural正确/错误分布重叠；完整三seed/CI见报告。
最终冻结22830文件一致，9.65GiB；复核M7 2912/M6 35688文件一致。
唯一审查reports/M7.1_REVIEW.md；完整证据reports/M7.1_details.md、data/M71_v1/analysis_v3/。

## 2026-11-08 M8a 固定测试完成
P0公开期望通过，无需重训/重定阈值；固定500组三seed/E0–E4完成，44协议+4统计测试通过，累计9.57/25GPUh（含旧P0、smoke/失败）。
ε=.03：E1/E2鲁棒失效均69.3%[65.4,73.0]，识别5.07%=clean FPR，实际fp16特征逐位不变；目标达成100%，其中26.8%原本已是目标，已从改变指标排除。
E3有限搜索失效随ε=.01/.03/.1/.3为69.1/68.3/66.5/66.0%；组bootstrap10000次CI及单λ/seed见报告。早停使50/200预算主曲线相同，不冒称实际执行全部步数。
命令configs/M8a/full_{0,1,2,3}.yaml→configs/M8a/report_v1.yaml；唯一审查reports/M8a_REVIEW.md，详细reports/M8a_details.md，冻结data/M8a_v1/freeze.json（933文件）。M7/M6/M7.1冻结只读。

## 2026-11-08 — M8b 开始
任务：R4 receiver状态、R2随机f、R1随机问题、R3全层，预算25GPUh。
命令：`.venv-comm/bin/python -m scripts.m8b_prepare configs/M8b/base.yaml`；后续入口configs/M8b/。
提取smoke train100/val20约104秒，4项协议单测通过；A4小样本接近随机，尚非有效性结论。
详细协议docs/M8b_TASK.md；接续docs/M8b_LIVE.json，尚未最终test。

## 2026-11-09 M8b 设计比较完成
R1随机问集、R2随机读取、R3全层与R4接收方状态；三seed，test500/PGD val20，clean QA val200/test500。
命令configs/M8b/final_*.yaml→configs/M8b/statistics_final.yaml；问答、FPR、E1/E2/E3失效及10000次组bootstrap 95%CI见reports/M8b_REVIEW.md。
全部失败/额外训练/加载及孤儿成本上界计入25GPUh；完整负面结果、k/ε/步数曲线、时延和复现证据见reports/M8b_details.md。
冻结data/M8b_v1/freeze.json；上游62363项最终SHA复核一致，不更改M8a/M71/M7/M6。

## 2026-11-09 M8c test前预算关卡
short direct-lookup val200准入93.0/27.0/26.0%；R4三seed曝光达标，val A4均≥99.5%，D0冻结权重重定阈值。
正式val矩阵90分钟超时，19组完整/末组部分保留；新三ε主矩阵预算估计最大124组（点估计181）<用户下限200。
CPU预算/报告命令：.venv-comm/bin/python -m scripts.m8c_budget configs/M8c/budget_feasibility.yaml；-m scripts.m8c_budget_report configs/M8c/budget_pause_report.yaml。
累计6.43/15GPUh；主200组保守累计20.2GPUh，按用户规则停在test前。10项协议测试通过，33897个上游SHA一致。
均值/95%组bootstrap CI、部分val与预算限制：reports/M8c_REVIEW.md；M8c未完成，未创建final_lock或test预测。

## 2026-11-09 M8c 21GPUh最终协议启动
用户提高累计硬限至21GPUh，固定主200组/三seed/三ε/λ1/PGD200，已用6.43计入；保守主累计预测20.2。
先写final_lock.json（n=200、前100次要ID、预算、顺序及5634项SHA），后启动scripts.m8c_controller configs/M8c/controller.yaml。
GPU1/3/4每seed全部200组；次要只在主完成后按最小成本→E1/E2→λ敏感性顺序。按耗时预测超21则停、不裁主矩阵；尚未完成。

## 2026-11-09 M8c 主矩阵接续完成
seed22于189组完整后触发8h单run上限；按锁定200ID/三ε/λ1/原部署key只补缺失行，原run和成本保留。
原final_lock只读；执行补充data/M8c_v1/execution_supplement.json，CPU合并run逐trial校验无重复且全200×6条，不冒充GPU源run。
共享GPU显存门槛由15改8GiB并明确记录，模型/精度/矩阵不变；主三seed200组完成后才启动次要最小扰动。
R4 val/test A1/A4=94.2/99.8%与91.7/99.5%；主统计data/M8c_v1/analysis_final21，完整审查待次要项预算停止/收尾。

## 2026-11-09 M8c 依赖设置鲁棒性
short direct-lookup准入val200：有/无/错配93.0/27.0/26.0%；R4三seed各≥4.224M曝光，D0冻结权重重定阈值。
最终矩阵由configs/M8c/final_*.yaml与minimum*.yaml驱动；三seed均值/10000组bootstrap CI与次要项进度见reports/M8c_REVIEW.md。
独立21GPUh硬限，含加载/失败/中断上界；主E3使用冻结test500哈希序前缀200组，不按test选模型或λ。
冻结data/M8c_v1/freeze.json；上游M71/M8a/M8b SHA不变。

## 2026-11-10 M9启动
仅补M6 C2：共享CLR-lite、两源adapter/source token、3seed，独立10GPUh账本；原M6数据/训练协议只读复用。
命令：scripts.m9_train configs/M9/smoke.yaml；scripts.m9_pipeline configs/M9/pipeline.yaml；smoke70.7秒、两源各5训练/val题通过，不触碰test。
上游105515唯一文件SHA一致；论文B–D仅CPU派生/冻结CSV制图，不新增模型运行。接续docs/M9_LIVE.json。

## 2026-11-10 M9完成
M6 C2三seed共享CLR-lite补齐；单源/联合Q1–Q4与conformal配对差及95%题bootstrap CI见reports/M9_REVIEW.md。
A累计7.38/10GPUh（含smoke/加载/训练/校准/评估），源前向0；训练锁后才test。
命令：scripts.m9_resume_pipeline configs/M9/resume_pipeline.yaml；scripts.m9_finish configs/M9/finish.yaml。
19份独立表图脚本、输入CSV冻结SHA、隔离制图环境逐字节重建；四贡献/两局限逐条核验，不能主张另列。
负面/缺失控制与历史偏离保留；唯一审查reports/M9_REVIEW.md，详细reports/M9_details.md，冻结data/M9_v1/freeze.json。

## 2026-11-10 M10 P1 smoke与正式训练启动
7个GSM设置及M71 B6均通过≤5%/15min的val-only smoke；最小测试3通过，test未打开。
命令入口：scripts/m10_smoke_queue.py configs/M10/smoke_queue.yaml；吞吐面板configs/M10/throughput_queue.yaml。
三seed正式训练由configs/M10/full_gsm_queue.yaml和full_audit_queue.yaml驱动；未产生最终指标。
审计B6 hidden896参数4866177≥单类型B3的4719876；6epoch≥B3三seed总曝光对应5/5/6epoch。
证据reports/M10_evidence；初步P1估算11–12GPUh，运行账本守卫12/40GPUh硬上限。

## 2026-11-10 M10 HiddenBench/迁移执行进展（未完成里程碑）
HiddenBench source/reader/eval val-only smoke通过，三种读取器三seed训练完成；63独立组全使用，原始65case。
跨用途N100/300/1000训练完成，full仍运行；全部GSM评估分支smoke及11项测试通过，未开test。
上游110122文件SHA全部一致；paper原66文件已逐字节归档；CPU收尾控制器等最终exports后再制图/审查。
正式吞吐持续复核，成本见docs/M10_LIVE.json；释放25GiB配重并保持自动补足暂停，记录reports/M10_evidence/disk_reserve_release.json。
这些是运行进展，不能代替最终主表或M10审查/冻结；接续docs/M10_EXECUTION_CONTINUATION.md。

## 2026-11-10 M10最终冻结
LLaMA缺失格、B6/每类型CLR、HiddenBench和M9主体初始化三seed完成；B5c预算准入失败NR。
累计13.521/40GPUh，含smoke/失败/校准/推断；全训练hash锁后才test。
命令configs/M10/final_controller.yaml；论文独立重建、最终证据及CI见reports/M10_REVIEW.md和M10_details.md。
M9原paper66文件归档，旧freeze不改；更新paper与FINAL_SUMMARY并最终冻结data/M10_v1/freeze.json。

## M10b 启动与准入
仅CODI-GPT2 B5c Q3/Q4，完整冻结标签、一遍训练、seed11/22/33；预算22GPUh含失败加载校准推断。
训练smoke20/20组76.7秒、校准/推断reader_val代理15组10.6秒；15项协议单测通过，未访问test。
实测训练15.2GPUh、训练期val0.461、预留2，准入17.6/22；configs/M10b/controller.yaml，tmux clr-m10b。
旧M10只读，paper原76文件归档data/M10b_v1/m10_paper_snapshot；最终审查reports/M10b_REVIEW.md（尚未完成）。

## M10b 最终冻结
仅GPT2 B5c Q3/Q4完整一遍×三seed；CLR−B5c test Q3 -0.659 [-1.51,0.19]pp；Q4 -0.0864 [-0.984,0.83]pp。
命令configs/M10b/controller.yaml；成本17.2/22 GPUh，含失败加载/校准/推断。
固定seed整problem bootstrap10000 CI；审计框架修订、附录定位、复现和证据见reports/M10b_REVIEW.md。
实验永久冻结data/M10b_v1/freeze.json，旧M10 paper字节归档。

## M10c 只读精度–开销
固定reader_val64，同卡RTX4090、batch1/64三轮中位；无训练/新test推断/重校准。
CLR对B5c单查询FLOPs低648/437倍，精度CI沿用三seed；conformal整problem bootstrap10000。
新增GPU成本0.0941h含失败，源前向0；命令configs/M10c/measure.yaml、summarize.yaml、finish.yaml。
表/配对差/原始三轮/局限见reports/M10c_REVIEW.md；冻结data/M10c_v1/freeze.json。

## 2026-11-12 — M10d 四格补齐
GPT2 B6复用M10；LLaMA B4按M3训练6817题、三seed/两LR/8epoch，统一锁hash后cal_A独立温度及test最终评估。
入口：`.venv/bin/python -m scripts.m10d configs/M10d/run.yaml [stage=train|stage=final]`；CPU汇总`scripts.m10d_summarize`。
Q1 NLL / Q2 AUROC：B6 0.574 / 0.853；B4 0.551 / 0.806；95%按题bootstrap CI见data/M10d_v1/four_cells.json。
新增0.0289412GPUh≤2，源前向0；20训练/20val smoke4.82秒，3项测试通过。
论文四TODO/必要加粗及number_audit追加完成，T1/Times编译8页通过；唯一审查reports/M10d_REVIEW.md。

## 2026-11-12T15:34:59.142298+08:00 — M10e
两源HSRM/CLUE及LLaMA M3 B2完成，三seed或确定性按题bootstrap10000 CI见data/M10e_v1/summary.csv。
命令：python -m scripts.m10e configs/M10e/run.yaml stage={smoke,train,lock,final}；后接m10e_summarize_v2/audit/finish。
新增0.942GPUh、源前向0；HSRM后期val NLL上升及负面比较如实保留，不改test/旧冻结。
用户改为Markdown论文交接，未改稿编译；审查reports/M10e_REVIEW.md，详情reports/M10e_details.md。

## 2026-11-12 — M11-qual
只读冻结权重/预测导出；命令`python -m scripts.m11_qual configs/M11_qual/export.yaml stage=full`及同配置finish/package。
val-only smoke通过；12单测；两源各3题概率逐位一致；所需规模齐全。保守0.0849GPUh，源前向0。
确定性哈希选样，无新统计/CI；三seed已有预测保留；M8c仅已有分数。
交付/规模/限制见[审查](../reports/M11_qual_REVIEW.md)及`data/M11_qual_v1/qual_export.zip`。

## M12 — 启动与协议smoke
用户授权A1–A5/三seed，M41完整曝光及早停/校准相同，预算≤30GPUh、11-15 12:00截止，不改论文。
新增组件隔离/原模型等价/序列化缓存；12项协议测试通过，训练与val代理smoke复核中，尚未访问test。
入口`.venv/bin/python -m scripts.m12_start configs/M12/base.yaml`；状态docs/M12_LIVE.json，任务docs/M12_TASK.md。

## M12 — CLR-lite组件消融
M12组件消融完成待审查；5变体三seed，18.1/30GPUh；唯一审查reports/M12_REVIEW.md，冻结data/M12_ablation/freeze.json；未改论文。
入口：`.venv/bin/python -m scripts.m12_pipeline configs/M12/base.yaml`。
Q1/Q2/Q3/Q4、按题CI和配对差：data/M12_ablation/summary.csv；置换permutation.csv。
全部负面结果保留；详细reports/M12_details.md；复用来源逐行记录，无模型/曝光缩量。
