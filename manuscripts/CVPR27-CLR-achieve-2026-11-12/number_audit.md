# Number audit — 指定 TODO 替换

日期：2026-11-12（北京时间）。修改对象：本地 `sec/5_results.tex` 和 `fig/{architecture,transfer,search}.tex`；服务器数据只读。
来源主机：`4090`；仓库根目录：`/home/me/PaperField/CalReader`。本次核对的 HEAD：`9f0a9975378fd91807149c183599ff818853b52a`。
CSV 采用 `test` 冻结汇总行；Q1 NLL/Q2 AUROC 保留三位小数，Q3/Q4 的概率乘 100 后保留一位小数。未重跑实验、重校准或更改冻结数据。

## 1. 数字替换清单

| 位置 | 原值 → 写入值 | CSV 原始 mean | 来源文件（服务器仓库相对路径及行号） |
|---|---|---|---|
| tab:main / LLaMA / B0 question-only / Q1 nll | `\TODO{}` → `0.692` | 0.691724614833074 | `paper/derived/t1_gsm8k.csv:66` |
| tab:main / LLaMA / B0 question-only / Q2 auroc | `\TODO{}` → `0.509` | 0.509004192330581 | `paper/derived/t1_gsm8k.csv:138` |
| tab:main / LLaMA / B0 question-only / Q3 accuracy | `\TODO{}` → `39.1` | 0.391471215351812 | `paper/derived/t1_gsm8k.csv:71` |
| tab:main / LLaMA / B0 question-only / Q4 accuracy | `\TODO{}` → `50.1` | 0.500672043010753 | `paper/derived/t1_gsm8k.csv:73` |
| tab:main / LLaMA / B3 MLP probe / Q1 nll | `\TODO{}` → `0.612` | 0.611703345581023 | `paper/derived/t1_gsm8k.csv:74` |
| tab:main / LLaMA / B3 MLP probe / Q2 auroc | `\TODO{}` → `0.851` | 0.851368919434858 | `paper/derived/t1_gsm8k.csv:139` |
| tab:main / LLaMA / B3-prefix / Q1 nll | `\TODO{}` → `0.606` | 0.60615229743764 | `paper/derived/t1_gsm8k.csv:82` |
| tab:main / LLaMA / B3-prefix / Q2 auroc | `\TODO{}` → `0.852` | 0.852119384835572 | `paper/derived/t1_gsm8k.csv:140` |
| tab:main / LLaMA / B3-prefix / Q3 accuracy | `\TODO{59.2}` → `59.3` | 0.592513622364368 | `paper/derived/t1_gsm8k.csv:87` |
| tab:main / LLaMA / B3-prefix / Q4 accuracy | `\TODO{60.3}` → `60.3` | 0.603230606758832 | `paper/derived/t1_gsm8k.csv:89` |
| tab:main / LLaMA / B6 attention probe / Q1 nll | `\TODO{}` → `0.632` | 0.63181077521184 | `paper/derived/t1_gsm8k.csv:90` |
| tab:main / LLaMA / B6 attention probe / Q2 auroc | `\TODO{}` → `0.824` | 0.823777205097489 | `paper/derived/t1_gsm8k.csv:141` |
| tab:main / LLaMA / B6 attention probe / Q3 accuracy | `\TODO{57.8}` → `57.8` | 0.578109452736318 | `paper/derived/t1_gsm8k.csv:95` |
| tab:main / LLaMA / B6 attention probe / Q4 accuracy | `\TODO{66.6}` → `66.6` | 0.665754608294931 | `paper/derived/t1_gsm8k.csv:97` |
| tab:main / LLaMA / \clr{} single-step / Q1 nll | `\TODO{}` → `0.627` | 0.626804175272813 | `paper/derived/t1_gsm8k.csv:114` |
| tab:main / LLaMA / \clr{} single-step / Q2 auroc | `\TODO{}` → `0.813` | 0.812945213964628 | `paper/derived/t1_gsm8k.csv:144` |
| tab:main / LLaMA / \clr{} single-step / Q3 accuracy | `\TODO{}` → `74.7` | 0.747311063728974 | `paper/derived/t1_gsm8k.csv:119` |
| tab:main / LLaMA / \clr{} single-step / Q4 accuracy | `\TODO{}` → `75.5` | 0.754944316436252 | `paper/derived/t1_gsm8k.csv:121` |
| 5.2 / N=100 / gpt2->llama / transfer / Q3 | `TODO (not supplied)` → `39.1\%` | 0.390713101160862 | `paper/derived/f1_transfer.csv:12` |
| 5.2 / N=100 / gpt2->llama / transfer / Q4 | `TODO (not supplied)` → `54.3\%` | 0.543154761904762 | `paper/derived/f1_transfer.csv:13` |
| 5.2 / N=100 / gpt2->llama / b3 / Q3 | `TODO (not supplied)` → `36.6\%` | 0.365979625681118 | `paper/derived/f1_transfer.csv:10` |
| 5.2 / N=100 / gpt2->llama / b3 / Q4 | `TODO (not supplied)` → `50.0\%` | 0.499903993855607 | `paper/derived/f1_transfer.csv:11` |
| 5.2 / N=100 / gpt2->llama / lite / Q3 | `TODO (not supplied)` → `35.2\%` | 0.352049277422412 | `paper/derived/f1_transfer.csv:14` |
| 5.2 / N=100 / gpt2->llama / lite / Q4 | `TODO (not supplied)` → `50.4\%` | 0.503768241167435 | `paper/derived/f1_transfer.csv:15` |
| 5.2 / N=100 / llama->gpt2 / transfer / Q3 | `TODO (not supplied)` → `36.6\%` | 0.366453447050462 | `paper/derived/f1_transfer.csv:42` |
| 5.2 / N=100 / llama->gpt2 / transfer / Q4 | `TODO (not supplied)` → `55.4\%` | 0.553763440860215 | `paper/derived/f1_transfer.csv:43` |
| 5.2 / N=100 / llama->gpt2 / b3 / Q3 | `TODO (not supplied)` → `30.9\%` | 0.309073679222933 | `paper/derived/f1_transfer.csv:30` |
| 5.2 / N=100 / llama->gpt2 / b3 / Q4 | `TODO (not supplied)` → `50.0\%` | 0.500480030721966 | `paper/derived/f1_transfer.csv:31` |
| 5.2 / N=100 / llama->gpt2 / lite / Q3 | `TODO (not supplied)` → `31.8\%` | 0.317934138829661 | `paper/derived/f1_transfer.csv:44` |
| 5.2 / N=100 / llama->gpt2 / lite / Q4 | `TODO (not supplied)` → `50.1\%` | 0.501224078341014 | `paper/derived/f1_transfer.csv:45` |

其中 B3 LLaMA Q3/Q4 原有的 58.9/62.7 与 T1 一致，已核对但未修改。B3-prefix 的 Q3 从反推值 59.2 改为 T1 原值四舍五入后的 59.3；其 Q4 60.3、B6 Q3/Q4 57.8/66.6 虽显示值相同，也均重新依据原始 mean 核对并删除对应 TODO。

## 2. 缺失项：LLaMA B4

**未替换 B4 的 LLaMA Q1 NLL / Q2 AUROC 两格，继续保留 `\TODO{}`。**
当前 T1、`paper/derived/t1_gsm8k.csv` 及哈希锁定的 `data/M10b_v1/paper_inputs/main.csv` 均没有 B4；M10 原始输入 `data/M10_v1/paper_inputs/main.csv` 和 M9 的 main.csv 也未匹配到 B4。
进一步检查 M6/M10 训练入口：M6 单源族为 b3/lite；M10 补充控制族为 B0、B3-prefix、B6、CLR-lite-single、CLR-per-type，没有找到 LLaMA B4 的可追溯配置或结果。
在 `runs/` 选定 config/metrics 文件、`configs/` YAML 和 `reports/` MD/CSV 中扫描 4039 个元数据文件，没有发现同时提及 B4/LRM 和 LLaMA 的文件；跳过 4 个超过 1.5 MB 的文件，不声称已证明从未运行。详见证据快照 `B4_search.json`。
B4 Q3/Q4 的 `--` 表示该标量基线不回答这些问题，按原稿保留。未用 B3、GPT-2 或其他基线数字代填。

## 3. 审计模型

`Qwen3 \TODO{size}` → `Qwen3 4B`。
来源：`configs/M71/source_hash.yaml:7` 和 `configs/M7/data.yaml:7`，型号目录为 Qwen3-4B，checkpoint revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`。
`clr/sources/kv_relay.py:1` 的说明及 `KVRelay.__init__/encode_texts/score` 证明 sender/receiver 顺序共用同一个冻结 backbone，均为 Qwen3-4B。未将服务器路径写入论文。

## 4. 图片和 caption

- `fig/F1_transfer.pdf`：从服务器 `paper/figures/f1_transfer.pdf` 原样复制；哈希与 `paper/build_records/f1_transfer.json` 一致。
- `fig/F2_search.pdf`：从服务器 `paper/figures/f2_rq1.pdf` 原样复制；哈希与 `paper/build_records/f2_rq1.json` 一致。
- `fig/fig_arch.pdf`：本次新绘的矢量架构示意图，依据现有 `sec/3_method.tex` 架构描述，及只读 `clr/models/{clr_v1,m6}.py`。不包含新实验值；生成脚本 `fig/draw_architecture.py`，可独立生成 `outputs/fig_arch.pdf`。
- 删除上述三处 caption 的 TODO，保留其余原有措辞；同时删除 tab:main 表注中的 TODO。
- `fig/` 的 PDF 是论文输入素材；编译输出 PDF 统一写入 `outputs/`。

## 5. 按用户要求保留措辞，需注意的现有问题

1. F1 冻结图包含 (a) 新类型、(b) GPT-2→LLaMA、(c) LLaMA→GPT-2 三个面板；原 caption 只写 (a)/(b)，正文也只引用 b。依照“不改措辞”没有调整面板说明。
2. tab:main 中 LLaMA Q2 的 CLR 0.821 原有加粗，但补入的 B3/B3-prefix 分别为 0.851/0.852，CLR 不是该列最优；本次只替换 TODO，没有修改加粗或其他措辞。
3. GPT-2 B6 Q1/Q2、Cost 表等其他现有 TODO 不在本次指定范围，保持不变。

## 6. 来源快照和哈希

小型证据快照位于 `evidence/number_fill_20261112/`，保留 CSV、构建记录、源配置、模型代码和查询结果。以下 SHA256 已核对构建记录：

| 本地证据或输入图片 | SHA256 |
|---|---|
| `T1_frozen_main.csv` | `dcb77fc2c0ed9854146461a5bfb19788f15eb7e423f29b81960a791bde93f85e` |
| `t1_gsm8k.csv` | `843a5b0fa74cdde8cce39d22880c4c0c475884738817dcd04f7cc1252b39e762` |
| `t1_gsm8k.tex` | `dfa9a0dbd992addd23a514eef07e9623bdf32c697c38f92671b5d00fb2554a80` |
| `source_transfer.csv` | `7dddfa2ec27dabb2b45d6817d2f2664029229fd30827645fea05f32ea39ee3be` |
| `f1_transfer.csv` | `7cb2558b510c651c91f6dcae00a3dec75e365f52d870e1db643944c7676887de` |
| `f1_transfer.pdf` | `cc46418f2ee5aa33cb3c04c60371baa501e4f18abb73a396fb9bfadbfb99f0ce` |
| `f2_rq1.pdf` | `89a0e22c027dd34ff6795366b36b847ae078a415d581548a1abbb8cad7c2792b` |

## 7. 编译与范围验证

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 成功，输出 `outputs/main.pdf`，共 8 页（参考文献从第 7 页开始）。
- 已检查全部 8 页渲染图；三张图片正常显示。
- 无缺失图片、未定义引用、未定义 citation 或 overfull 警告；仅保留两次原有 Times small-caps italic 字体回退警告。
- 范围核对：`sec/5_results.tex` 仅 8 行发生变化（5 个表格行、表注、迁移 TODO 所在段落、模型 TODO 所在段落）；GPT-2 数值与其余原文逐行保持不变。
- 三个图 caption 除删除指定 TODO 外逐字相同。F1/F2 原始 PDF 字节保持不变。
- 当前 results 文件仍有四个 TODO：LLaMA B4 两格缺少来源，GPT-2 B6 两格不在本次请求范围。

## 8. M10d：tab:main 最后四格（2026-11-12）

统一口径：GSM8K test、clean、已见模板，t 内按原样本数聚合、各 t 等权，seed 11/22/33 均值。

| 模型 | Q1 NLL | Q2 AUROC | 来源与运行 |
|---|---:|---:|---|
| GPT-2 B6 | 0.574010961648221 → 0.574 | 0.852715892250876 → 0.853 | `data/M10_v1/paper_inputs/main.csv`；查到并直接复用M10冻结三seed评估，无新推断 |
| LLaMA B4 | 0.551071084851158 → 0.551 | 0.805841717531682 → 0.806 | `data/M10d_v1/B4_summary.csv`；新跑M3标量协议，三seed统一锁定后最终评估 |

LLaMA B4 原始预测/温度/逐 t 指标：`/home/me/PaperField/CalReader/runs/20261112T024717093641Z_b3ebcf90e887_187be8/B4_Q{1,2}_{11,22,33}_test.parquet`、`calibration.json`、`statistics.json`。训练与选择：`/home/me/PaperField/CalReader/runs/20261112T024518010110Z_2c8d73cccc27_1dba42/selection.json`；锁定：`data/M10d_v1/training_lock.json`。

M6 6817题版本 reader_train/reader_val；B4仅输入当前 z_t，Q1轨迹Bernoulli不平均，Q2直接复用标量并独立cal_A温度。搜索完整沿用M3：LR={0.001,0.0003}、8epoch、AdamW WD=0.0001、batch4096；仅reader_val选优。

新增成本 `0.02894120` GPUh（RTX4090 GPU5，包含smoke/三seed两LR训练/校准/最终推断），源模型前向0；B6查表0 GPUh。

来源SHA核验：`data/M10d_v1/upstream_inputs.json`（4895项一致）。不改任何旧冻结数据、模型、预测或阈值。

仅替换四个TODO，并按列最优规则：LLaMA Q1新增B4加粗；LLaMA Q2加粗移至B3-prefix 0.852（原CLR 0.821加粗移除）。其余措辞与数字不变。

B6 raw three-seed evaluation references: `data/M10d_v1/B6_existing_evaluation_sources.json`; original `data/M10_v1/evaluated/gpt2_B6_{11,22,33}_full.json` points to frozen Q1/Q2 prediction/calibration files.

M10d重新编译：Tectonic 0.17.0 + 独立T1字体兼容wrapper（归档build/M10d，不修改main.tex），输出outputs/main_M10d_T1.pdf并复制outputs/main.pdf；确认8页，主表第5页，四数及加粗已视觉核查。无overfull或未定义引用/citation；保留原两处Times small-caps italic回退及lineno宏包内部UTF8注释警告。日志服务器reports/M10d_compile_T1.log。
