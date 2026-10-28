# M0b Huginn v2：官方资源恢复与评测协议

状态：下载运行中；尚未完成真实模型加载、准确率复现或噪声实验。此报告不构成 M0b 通过。M0 v1 的代码和结果未覆盖。

## 资源与可追溯性

仅使用官方 [`tomg-group-umd/huginn-0125`](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c)，固定 revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`。官方源码 checkout 是 `third_party/huginn`（commit `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`）。

恢复命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/m0b_huginn_download.py configs/M0b_huginn_download_v2.yaml
```

运行：`runs/20261028T111806531496Z_c0fb096f3f3e_1f3fea`，日志：`reports/evidence/M0b_huginn_download_v2.log`。使用 HF Hub 官方 `snapshot_download(resume_download=True, max_workers=4)`，关闭 Xet 嵌套并行；没有启用旧自写 range 下载。缓存是 `/home/me/models/huggingface`。原有 `/home/me/models/Huginn-3.5B/` 自写 range 残片不作为模型输入。启动时磁盘空余约 115 GiB。

API 返回 revision 与配置一致。四个权重分片共 15,645,609,096 字节。下列是官方 LFS 声明值，**不是已通过校验的声明**；成功运行后才会产生 `verified_manifest.json`，评测入口强制要求该文件。

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

配置仅取冻结 v1 val 的按 ID 排序前 2 题（2/250=0.8%），硬性 900 秒超时。相同输入和 seed 执行两遍，逐步比较答案位置的 BF16 递归状态，保存最大逐元素差和答案一致性；所有前向次数计入成本。测试集不用于 smoke。任务构造的 CPU 检查已通过，确认原生八个示例、Q/A 模板、目标数值保留；另运行现有相关单测，4 passed / 6 deselected（日志 `reports/evidence/M0b_huginn_tests_v2.log`），官方 tiny architecture 的 HF generate 同 seed 双次逐 token 一致。它们不是真实权重的模型 smoke 通过。运行前必须全部权重通过 SHA-256、重新检查 GPU；配置候选 GPU5 与主线 GPU3 分开。若可用显存不够将报错，不自动换设备、精度、深度、上下文或模型。

每次评测保存原生/实际 task config、数据 ID 和输入文件哈希、代码快照、环境、预测原文及官方两种答案过滤结果。准确率的区间按题 bootstrap 10,000 次，不把确定性 seed 重复当独立样本。新增统计单测 3 passed（`reports/evidence/M0b_huginn_stats_tests_v2.log`），验证两个官方过滤器不重复计算题数、漏题/重复题报错及 bootstrap 边界和可复现性。

## 当前结果与阻塞

| 项目 | 结果 |
|---|---|
| 官方 revision/LFS 元数据核实 | 已完成 |
| 权重 SHA-256 实测 | 待下载结束 |
| 完整模型加载 | 未运行 |
| 原生评测 smoke | 未运行 |
| GSM8K 复现 / 95% CI | 无结果 |
| 噪声曲线 / 答案多样性 | 无结果 |
| GPU 小时 / 源模型前向次数 | 当前 0 / 0 |

下载速度低且有波动，是当前实际阻塞。2026-10-28 11:30 UTC 时四片各约 1 GiB；近期每片约 0.2 MiB/s，余下可能需要数小时，不能保证最初估计的约 1.5 小时。下载不占 GPU。完整复现和噪声实验必须等待下载校验及 smoke 门槛。没有采用第三方 checkpoint、降低递归深度或把小模型结果当作真实 Huginn 结果。噪声配置与正式数据面板尚未冻结，不能把当前 smoke 配置直接当作完整实验配置。

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
