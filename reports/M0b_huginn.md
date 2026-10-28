# M0b Huginn v2：官方资源恢复与评测协议

状态：**官方权重 4/4 SHA-256 校验通过，真实模型已成功加载；smoke 因 CUDA OOM 未通过。完整复现、确定性验收和噪声曲线均未完成，全部 GPU 尝试已退出。** M0b 不能标记为验收通过，不阻塞仅使用 CODI 的后续审查。M0 v1 代码和结果未覆盖。

## 资源与可追溯性

仅使用官方 [`tomg-group-umd/huginn-0125`](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c)，固定 revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`。官方源码 checkout 是 `third_party/huginn`（commit `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`）。

恢复命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/m0b_huginn_download.py configs/M0b_huginn_download_v2.yaml
```

运行：`runs/20261028T111806531496Z_c0fb096f3f3e_1f3fea`，日志：`reports/evidence/M0b_huginn_download_v2.log`。使用 HF Hub 官方 `snapshot_download(resume_download=True, max_workers=4)`，关闭 Xet 嵌套并行；没有启用旧自写 range 下载。缓存是 `/home/me/models/huggingface`。原有 `/home/me/models/Huginn-3.5B/` 自写 range 残片不作为模型输入。启动时磁盘空余约 115 GiB。

API 返回 revision 与配置一致。四个权重分片共 15,645,609,096 字节。下列官方 LFS 值已由下载 run 逐文件 SHA-256 实测全部匹配；`verified_manifest.json` 已生成，评测入口强制要求该文件。

| 分片 | 字节 | 官方 LFS SHA-256 |
|---|---:|---|
| 00001 | 4,771,970,936 | `5b461584b74249e87eabec34b59a13e188791dd8a7ccf2c7be95bb7fa8383e43` |
| 00002 | 4,744,780,096 | `44bac332b204abb687b2820eb0fadcb5795fdb0cfc06b3b6f2ac5241eec5c591` |
| 00003 | 4,744,737,616 | `07160dce9da01918d0b0bf2159ffe11d007e1fbe661d687ad4a7b0e65bde4e84` |
| 00004 | 1,384,120,448 | `40ce4f159938857dbbfa1020962045e1b96303ff902cfa76051ce54d2d51dea4` |

## 官方协议与 v1 差异

[官方 README 的复现命令](https://github.com/seal-rg/recurrent-pretraining#reproducing-benchmark-scores) 对带系统提示的 GSM8K 使用 `gsm8k_cot`、chat template、fewshot-as-multiturn，以及系统提示 `You are a helpful assistant that can assist users with mathematical reasoning.`。原生任务含 8 个固定 CoT 示例。模型为 bf16、mean_recurrence=32。推理 eval 模式的官方 `randomized_iteration_sampler` 返回固定 32 步；初始递归状态仍由官方随机初始化产生，所以固定 RNG seed 是协议的一部分。

v2 准备使用独立环境 `.venv/huginn-eval`（继承主环境 Torch 2.5.1；transformers 4.52.4；lm-eval 0.4.8）。官方项目未锁定论文时期 lm-eval 的精确版本，本次固定 0.4.8 是显式假设，不能称为完全相同的历史软件环境。独立环境安装记录：`reports/evidence/M0b_huginn_environment_v2.log`；实际导入版本与依赖快照见 `reports/evidence/M0b_huginn_versions_v2.json` 和 `reports/evidence/M0b_huginn_requirements_v2.txt`。

| 项目 | M0 v1 Huginn 接口 | M0b v2 原生 harness |
|---|---|---|
| 问题格式 | 手写 Question/Answer，零样本 | 原生 gsm8k_cot 的 Q/A + 8-shot + chat + system |
| 解码 | 手写逐 token greedy | 官方模型 `generate` + HFLM 原生 stopping criteria |
| 输出长度 | 256 | HFLM 原生默认 256 |
| 终止 | eos + Question 字符串 | 原生任务 Q:、`</s>`、`<\|im_end\|>` + 模型 EOS |
| 答案抽取 | CLR 数值答案抽取 | 原生 strict-match 与 flexible-extract 同时保留 |
| batch | 1 | 1，显式取代官方命令的 auto 调度，不改变提示或答案规则 |

因此 v1 手写零样本协议不能直接作为官方带系统提示成绩复现的证据。v2 不改 v1，也不把两种协议结果混为一条曲线。

## 已准备的 smoke

```bash
PYTHONPATH=. .venv/huginn-eval/bin/python scripts/m0b_huginn_eval.py configs/M0b_huginn_smoke_v2.yaml
```

实际 GPU smoke 配置已改为冻结清洁 v2 val 的按 ID 排序前 2 题（2/300≈0.67%，`data/M01_v2/val.json`、`data/splits/gsm8k_M01_v2.json`），硬性 900 秒超时。相同输入和 seed 执行两遍，逐步比较答案位置的 BF16 递归状态，保存最大逐元素差和答案一致性；所有前向次数计入成本。测试集不用于 smoke。任务构造的 CPU 检查已通过，确认原生八个示例、Q/A 模板、目标数值保留；另运行现有相关单测，4 passed / 6 deselected（日志 `reports/evidence/M0b_huginn_tests_v2.log`），官方 tiny architecture 的 HF generate 同 seed 双次逐 token 一致。它们不是真实权重的模型 smoke 通过。运行前必须全部权重通过 SHA-256、重新检查 GPU；配置候选 GPU5 与主线 GPU3 分开。若可用显存不够将报错，不自动换设备、精度、深度、上下文或模型。

每次评测保存原生/实际 task config、数据 ID 和输入文件哈希、代码快照、环境、预测原文及官方两种答案过滤结果。准确率的区间按题 bootstrap 10,000 次，不把确定性 seed 重复当独立样本。新增统计单测 3 passed（`reports/evidence/M0b_huginn_stats_tests_v2.log`），验证两个官方过滤器不重复计算题数、漏题/重复题报错及 bootstrap 边界和可复现性。

## 当前结果与阻塞

| 项目 | 结果 |
|---|---|
| 官方 revision/LFS 元数据核实 | 已完成 |
| 权重 SHA-256 实测 | 4/4 通过 |
| 完整模型加载 | GPU5、bf16、r=32 成功 |
| 原生评测 smoke | 未通过：第一遍2题完成，第二遍CUDA OOM |
| GSM8K 复现 / 95% CI | 无结果 |
| 噪声曲线 / 答案多样性 | 无结果 |
| 已记录模型阶段 GPU 小时 / 源模型前向次数 | 0.0728161 / 340；recurrent core 调用 10880 |

下载速度低且有波动曾是阻塞，现已解决。2026-10-28 11:30 UTC 时四片各约 1 GiB；近期每片约 0.2 MiB/s，余下可能需要数小时，不能保证最初估计的约 1.5 小时。下载不占 GPU。完整复现和噪声实验仍被未通过的 smoke 门槛阻止。没有采用第三方 checkpoint、降低递归深度或把小模型结果当作真实 Huginn 结果。噪声配置与正式数据面板尚未冻结，不能把当前 smoke 配置直接当作完整实验配置。

### 2026-10-28 11:43 UTC 下载事件

分片 `model-00002-of-00004.safetensors` 的官方 CDN `us.aws.cdn.hf.co` 返回 `Read timed out`；HF Hub 自动输出 `Trying to resume download...`。此时四片各约 1.08 GB，进程仍运行。这是官方客户端的内建恢复，不是切换模型或自写并发下载。完整错误保留于下载日志；尚不能宣称任何分片 SHA 校验成功。

2026-10-28 11:46 UTC，分片 `00003` 也发生官方 CDN `Read timed out`，官方客户端继续自动断点恢复。

### 2026-10-28 11:53 UTC：首片校验与真实 tokenizer 协议检查

分片 00004 已完成下载；独立 `sha256sum` 实测为 `40ce4f159938857dbbfa1020962045e1b96303ff902cfa76051ce54d2d51dea4`，与官方 LFS 相符。其余三片尚在下载，整体加载门槛仍未满足。

CPU 协议运行 `runs/20261028T115251348952Z_da3a5f18684d_62c6c5` 已用当前固定 revision 的真实 tokenizer 渲染两道 smoke 题；每题保留系统提示、8 个示例和目标问题，输入长度分别为 872、817 token。提示原文、token ID、原生 generation kwargs 和文件哈希保存在该 run，GPU 小时/前向次数均为 0。这是 tokenizer/提示的实测结果，不是模型准确率结果。

```bash
PYTHONPATH=. .venv/huginn-eval/bin/python scripts/m0b_huginn_protocol_render.py configs/M0b_huginn_protocol_render_v2.yaml
```

额外协议细节：原生 HFLM 0.4.8 不识别 RavenConfig 的 `block_size=4096`，而 tokenizer 的 model_max_length 为无限哨兵值，因此未手工覆盖时 HFLM 原生 max_length 为 2048、max_gen_toks 为 256。这里保留原生官方命令路径的行为，并记录实际值；上述 smoke 提示不会触发 1792-token 输入裁剪。不能把模型架构的 4096 直接当作本次 harness 的有效上下文长度。

论文对照目标已从 [arXiv v2 Table 2](https://arxiv.org/html/2502.05171v2#S5.T2) 重新核实：r=32、8-shot GSM8K CoT、有系统提示的 strict/flexible 分别为 34.80% / 42.08%。这是论文参考值，不是本机复现值；真实结果及偏差分析仍待完整测试评估。

### 清洁 v2 数据边界

主线完成去重后，Huginn GPU smoke / 噪声改用冻结清洁 val300。CPU 提示检查已对清洁数据重跑并通过：`runs/20261028T120458140308Z_31e13ac0ebaa_88bcba`，原始 run 保留。完整论文复现仍使用官方 GSM8K test 1319 题，仅作论文 benchmark 对照；后续 CLR 使用清洁 test1312，可从既有 benchmark 预测只读筛选统计，不重跑或调参。GPU3 由主线授权使用，设备变更均通过显式 override 记录；当前加载门槛已依据真实OOM提高为18 GiB，具体尝试见下表。

真实 GPU 复现明确使用 lm-eval 0.4.8 的默认种子组：Python=0，NumPy/Torch/fewshot=1234；不沿用早期仅 CPU 提示检查的 seed11。原因是 Huginn 递归初态确实含随机性，种子属于官方命令默认协议。当前没有 GPU 结果被更换种子覆盖。

### 2026-10-28 12:17 UTC：官方恢复成功，启动真实 smoke

下载 run `20261028T111806531496Z_c0fb096f3f3e_1f3fea` 成功完成，耗时 3580.07 秒，4 个 LFS SHA-256 全部匹配；GPU 小时及源模型前向次数均为 0。网络超时由官方客户端自动恢复，未替换 checkpoint 或采用自写 range。

GPU smoke run `20261028T121812416441Z_1b90f6844702_7ba16c` 已启动，实时显存预检及环境保存在该 run。命令：

```bash
PYTHONPATH=. .venv/huginn-eval/bin/python scripts/m0b_huginn_eval.py configs/M0b_huginn_smoke_v2.yaml device=cuda:3
```

### GPU3 显存预检失败与显式 GPU5 调度

首次 GPU3 smoke `20261028T121812416441Z_1b90f6844702_7ba16c` 预检 CUDA 空闲 6.81 GiB；同配置重试 `20261028T122358879576Z_1b90f6844702_d86203` 空闲 7.34 GiB。均低于既定 14 GiB，直接报错并由 recorded 写入 BLOCKERS，模型未加载，GPU 模型运行小时/前向次数均为 0。

UUID 核对确认 CUDA3 与物理 GPU3 相同（`ef23fa9f-5ab6-08fc-1fca-aef1b2b787cc`），不是编号映射错误。另一次同步探针又读到 CUDA 可用约 23.0 GiB，自身 CUDA 上下文仅 384 MiB，说明不能把差异归因于本程序上下文固定消耗 16 GiB；当前证据符合共享 GPU 负载造成的瞬时显存变化。没有忽略 CUDA 预检或降低阈值。

原分配 GPU5 的 CUDA 探针为 14.49 GiB，达到门槛，因此第三次尝试明确使用 `device=cuda:5`，所有模型、bf16、r=32、seed、提示和数据设置不变。设备调度变更不隐藏前两次失败。

## 最终可审查状态：真实加载成功，smoke 未通过

| Run | 实际结果 | GPUh | 源 forward |
|---|---|---:|---:|
| `20261028T121812416441Z_1b90f6844702_7ba16c` | GPU3预检6.81 GiB < 14 | 0 | 0 |
| `20261028T122358879576Z_1b90f6844702_d86203` | GPU3预检7.34 GiB < 14 | 0 | 0 |
| `20261028T122556758574Z_ea9f53c485ac_bb82a7` | GPU5通过预检；Accelerate要求禁用RTX4000不支持的P2P/IB | 0.00004984 | 0 |
| `20261028T122800941040Z_0be72c22d642_1ce21f` | 官方模型真实加载；首遍2题生成完成；第二遍CUDA OOM | 0.07276624 | 340 |
| `20261028T124143649075Z_6dd3a5e7057c_d4515c` | 新GPU3预检6.55 GiB < 18 | 0 | 0 |

GPU 型号为运行时报告的 NVIDIA GeForce RTX 4090。GPUh 是模型加载/推理阶段墙钟小时，未按共享 GPU 的 SM 利用率折算；硬件探针不做源模型前向。0.0728161 GPUh 为已记录模型阶段总和，下载和CPU协议检查均为0。

真实运行的原始 OOM 保存在 `runs/20261028T122800941040Z_0be72c22d642_1ce21f/traceback.txt`：模型进程约占14.83 GiB，分配额外2 MiB失败。首次14 GiB预检低估了运行峰值，因此后续预检提高到18 GiB；没有降低模型精度、递归深度、输入长度或换checkpoint。GPU3最新尝试仍没有满足该门槛。

OOM处理时还有次生故障：默认写入编码为ASCII，`clr.utils.records` 追加中文BLOCKERS时发生UnicodeEncodeError，使旧 `metrics.json` 的错误类型记录为次生错误。旧metrics与原始traceback均保留，`failure_causes.json`明确主因/次因。不能断言是lm-eval主动改了locale；Python层未找到显式setlocale调用，来源尚未定位。后续配置明确要求 `PYTHONUTF8=1`，入口按YAML在需要时重新启动Python启用UTF-8，不修改Idea A冻结日志代码。NCCL_P2P_DISABLE=1和NCCL_IB_DISABLE=1也明确入配置，仅解决单GPU初始化兼容性。

首遍两题预测未在OOM前及时落盘，因此**不能补写其准确率或CI**；也没有完整确定性最大差结果。后续入口已改为首遍结果立即保存，并在异常退出时保存比较进度。这是记录机制修复，不能使已经失败的run变为成功。

基于首遍两题生成耗时141.68秒（70.84秒/题）的粗略外推，官方1319题约26.0 GPUh；若噪声采用M0的5个sigma、清洁val300、K=16、3个seed，共72000次生成，约1416.8 GPUh。样本仅2题、共享GPU负载波动大，估计不确定性很高，**不是已消耗成本或已获实验结果**。没有据此自行缩小数据、K、seed数或r。

当前完整复现配置 `configs/M0b_huginn_repro_v2.yaml` 的 `smoke_run` 仍为空，入口会拒绝未通过smoke的完整实验。噪声协议/面板执行尚未完成，不存在可推荐的Huginn sigma。未来需要稳定的足够显存及通过真实smoke；完整噪声的资源规模需要明确安排，不能拿小结构单测或CPU提示检查代替。

后续仅重试既定smoke的配置驱动命令（GPU3必须实际满足18 GiB预检）：

```bash
PYTHONPATH=. .venv/huginn-eval/bin/python scripts/m0b_huginn_eval.py configs/M0b_huginn_smoke_v2.yaml device=cuda:3
```

UTF-8/NCCL启动环境由YAML记录和入口应用。此前失败run保持原配置与原始错误，不覆盖。

最终最小测试：`reports/evidence/M0b_huginn_tests_final_v2.log`，8 passed / 6 deselected（10.40秒）。包含原有相对噪声/官方小结构Huginn检查、按题bootstrap与去重统计、UTF-8启动后切换C locale仍可写中文错误记录的回归测试。它们不替代真实模型smoke。源forward统计含报错的调用尝试；所有GPU实验进程已退出。

协议记录澄清：已运行run的 `effective_harness_settings.json` 中 `padding_side=right` 是 tokenizer 静态默认属性，不是HFLM生成时的方向。原生 `tok_batch_encode` 默认临时使用left并在结束后恢复属性；batch=1没有实际补齐。后续记录分别命名 tokenizer_default_padding_side 和 harness_generation_padding_side，避免混淆，不修改旧run文件。
