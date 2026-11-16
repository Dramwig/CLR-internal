# M10e：CVPR27-CLR 论文修改交接

实验已完成并再次冻结。本文件供另一个对话修改其持有的新版论文；本对话按用户后续指令未编辑、未重新编译论文，因此尚未验证改稿后的页数。保持既有实验数字、模型及阈值不变，仅加入以下内容。

## 1. 表 1，`sec/5_results.tex` 的 `tab:main`

以下是三行完整数字，列顺序为 GPT-2 Q1 NLL / Q2 AUROC / Q3 acc / Q4 acc，再到 LLaMA 相同四列。Q3/Q4 为百分数。替换已有对应 TODO 行；若已有 B2 行，仅补 LLaMA 四格并核对 GPT-2 旧值，避免重复插行。

```latex
HSRM-style~\cite{hsrm} & 0.559 & 0.834 & -- & -- && 0.625 & 0.834 & -- & --\\
CLUE~\cite{clue} & 0.648 & 0.575 & -- & -- && \textbf{0.545} & 0.793 & -- & --\\
B2 linear probe & 0.553 & 0.836 & 53.4 & 50.1 && 0.615 & 0.764 & 54.0 & 50.1\\
```

统一口径为 test、clean、已见模板，t 内按实例聚合、各 t 等权。HSRM 和 B2 是固定 seed 11/22/33 均值；CLUE 为一次确定性评估，不复制成三 seed。HSRM/CLUE 不回答 Q3/Q4，`--` 表示不适用。

表注在原有“mean of 3 seeds”附近补充 `CLUE is deterministic.`，避免将 CLUE 误写成三 seed 结果。按未四舍五入的原始值重新比较每列，NLL 取低、其他列取高；移除原来已不再最优的加粗，完全相等时同时加粗。候选最优见下表，最终以新版表实际保留的行核对：

| Source | Column | Best raw mean | Model(s) |
|---|---|---:|---|
| gpt2 | Q1 | 0.50671172000113 | B4 |
| gpt2 | Q2 | 0.857292721226276 | B3-prefix |
| gpt2 | Q3 | 0.767827529021558 | B5c |
| gpt2 | Q4 | 0.764088901689709 | CLR-lite-multi |
| llama | Q1 | 0.544663269075889 | CLUE |
| llama | Q2 | 0.852119384835572 | B3-prefix |
| llama | Q3 | 0.749727552712627 | CLR-lite-multi |
| llama | Q4 | 0.765000960061444 | CLR-lite-multi |

模型名对应：`CLR-lite-single` 是 `\clr{} single-step`，`CLR-lite` 是 `\clr{}`，`CLR-lite-multi` 是 `\clr{} joint (2 sources)`，`B4` 是 LTO-style latent RM，`B5c` 是 LatentQA-style LoRA verbalizer。以上最优候选仅覆盖服务器现有归档表中的既有行和本次新增行；若新版另有 Patchscope 等行，仍须纳入最终逐列比较。

对服务器现有归档表，LLaMA Q1 的加粗应从 B4 的 0.551 移至 CLUE 的 0.545，其余七列最佳行保持上表所列。加粗仅表示点估计最优，不表示差异具有统计显著性。

## 2. `sec/4_setup.tex` 的两句话

用以下两句话替换 HSRM/CLUE 的 TODO 描述。句内无破折号。可放在现有 LTO 介绍之后；不改其他基线的已冻结协议。

```latex
Our HSRM-style reader~\cite{hsrm} applies a two-layer Transformer encoder with width 256 and four attention heads to the frozen latent prefix, then uses masked mean pooling, layer normalization, and a scalar head trained on Q1 trajectory outcomes, with separate temperature scaling for Q1 and Q2 on cal$_A$.
CLUE~\cite{clue} represents each prefix by $z_t-z_0$, compares its Euclidean distances to success and failure centroids formed from complete reader-training trajectories, and calibrates the distance difference separately for Q1 and Q2 by logistic regression on cal$_A$.
```

实现边界：HSRM 使用原文第 3.2 节架构，GPT-2/LLaMA 分别 1,777,153 / 2,104,833 参数；按用户要求用 Q1 独立 Bernoulli BCE 监督，原文主损失为组内成对排序，因此应标作 HSRM-style。CLUE 沿用原文双质心和欧氏距离，按用户指定只读取冻结末层的 latent prefix delta，原文读显式推理起止点的全层 delta；不得称作完全照搬原论文实验。

B2 沿用 M3 `Probe('B2', qtype)`、训练归一化、模板/选项特征、AdamW、学习率网格和 8 epoch；LLaMA 仅适配到 2048 维，保留原 Q3 双线性打分缩放。全部训练及选择只读 reader_train/reader_val，所有模型和配置 hash 统一锁定后才打开 test，未修改任何旧 conformal 阈值。

## 3. `number_audit.md` 追加内容

日期：2026-11-12（北京时间）。M10e 新增 HSRM 两源三 seed、CLUE 两源确定性、LLaMA B2 三 seed。所有方法均完成，无需删除相应表格行或 setup 句子。

| Source | Method | Q | Metric | Raw mean | 95% CI | Display |
|---|---|---|---|---:|---|---:|
| gpt2 | HSRM | Q1 | nll | 0.55877782206363 | [0.53768978, 0.579799428] | 0.559 |
| gpt2 | HSRM | Q2 | auroc | 0.834324408829717 | [0.816724832, 0.85227497] | 0.834 |
| llama | HSRM | Q1 | nll | 0.624759218713756 | [0.605133261, 0.644317614] | 0.625 |
| llama | HSRM | Q2 | auroc | 0.83429526093743 | [0.814161027, 0.853242278] | 0.834 |
| gpt2 | CLUE | Q1 | nll | 0.648457870913454 | [0.634354247, 0.662721285] | 0.648 |
| gpt2 | CLUE | Q2 | auroc | 0.57534399358661 | [0.552145368, 0.599366656] | 0.575 |
| llama | CLUE | Q1 | nll | 0.544663269075889 | [0.523876149, 0.566058055] | 0.545 |
| llama | CLUE | Q2 | auroc | 0.79265882543742 | [0.776386134, 0.808108114] | 0.793 |
| llama | B2 | Q1 | nll | 0.614866817500227 | [0.593459009, 0.6358585] | 0.615 |
| llama | B2 | Q3 | accuracy | 0.540298507462686 | [0.523004027, 0.557972045] | 54.0 |
| llama | B2 | Q4 | accuracy | 0.501440092165899 | [0.494335637, 0.508424539] | 50.1 |
| llama | B2 | Q2 | auroc | 0.763974259801513 | [0.744227088, 0.783579494] | 0.764 |
| gpt2 | B2 | Q1 | nll | 0.553237608500889 | reuse M3 point estimate | 0.553 |
| gpt2 | B2 | Q2 | auroc | 0.836248038120458 | reuse M3 point estimate | 0.836 |
| gpt2 | B2 | Q3 | accuracy | 0.533522869859423 | reuse M3 point estimate | 53.4 |
| gpt2 | B2 | Q4 | accuracy | 0.501008067812238 | reuse M3 point estimate | 50.1 |

accuracy 的原值和区间是概率，Display 列转为百分数。新结果统一沿用既有 `scripts/m9_export.py` 从保存的 logits 以 float64 导出的口径；CPU汇总v1曾因在线float32近0.5的舍入差触发一致性检查，v2保留原预测/温度，分别重放原生精度和表1导出精度后通过。最终数值来自 `scripts/m10e_summarize_v2.py`，不从在线float32 statistics的舍入值代填。

新数字来源（服务器 `/home/me/PaperField/CalReader/`）：

- `data/M10e_v1/table_cells.json`、`summary.csv`：表格原值、10,000 次按题 bootstrap 的 95% CI；保持每次 bootstrap 的题目抽样在 t 和 seed 间一致，区间条件于已拟合模型。
- `data/M10e_v1/final_{gpt2_HSRM,llama_HSRM,gpt2_CLUE,llama_CLUE,llama_B2}.json`：索引真实 run；各 run 有逐实例 `*_test.parquet`、`*_reader_val.parquet`、`statistics.json`、`calibration.json`。
- `data/M10e_v1/training_lock.json`：测试前统一模型/配置/代码 SHA256 锁；`trained_*.json` 指向训练 run 和 `selection.json`。Q1 的 reader_val clean/traj 分列在 `summary.csv`。
- GPT-2 B2 仅复用 M3 原结果，未重训或推断；来源及 hash：`data/M10e_v1/legacy_gpt2_B2.json`。
- `data/M10e_v1/upstream_before.json` / `upstream_after.json`：相关旧科学输入字节核验；`aggregation_checks.json`：独立逐 seed/t 汇总一致性。
- `data/M10e_v1/freeze.json`：本次代码、配置、训练、校准、预测、统计与交接材料冻结清单。新增成本 0.942317 GPUh（RTX 4090，包含 smoke/训练/校准/最终推断的保守 wall 口径），源模型前向 0；CLUE 的 CPU 时间不计作 GPU 时间。

## 4. 文献条目与编译验收

两篇原文已下载阅读并保留 hash：`data/M10e_v1/literature/sources.json`。若新版的 `hsrm`/`clue` 引用仍有 `TODO-VERIFY`，替换为：

```bibtex
@article{hsrm,
  title={HSRM: Hidden-State Reward Models for Test-Time Verification},
  author={Li, Xianzhi and Zhu, Xiaodan},
  journal={arXiv preprint arXiv:2608.30841},
  year={2026}
}
@article{clue,
  title={Clue: Non-parametric Verification from Experience via Hidden-State Clustering},
  author={Liang, Zhenwen and Li, Ruosen and Zhou, Yujun and Song, Linfeng and Yu, Dian and Du, Xinya},
  journal={arXiv preprint arXiv:2510.01591},
  year={2025}
}
```

在新版论文环境重新编译，确认正文仍 ≤8 页，检查表格无溢出、引用无缺失、各列加粗正确、表格和 setup 无遗留 TODO。若超页，先只压缩本次新增说明或表格排版，不删除负面结果、不改变数字。把编译结果追加到 number_audit.md。本文件不声称已通过此次改稿的编译或页数验收。
