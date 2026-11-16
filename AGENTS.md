# CalReader 项目约定：每个新对话必读

## 强制入口（长期有效）

**本仓库的每个新对话、恢复会话及后续里程碑都必须遵守 [CLR 全局工程与审查约定](docs/GLOBAL_RULES.md)。开始工作前必须完整读取该文件，不能只读本页摘要或把历史偏离视作许可。**

开始顺序：
1. 读取 [工作区规则](/home/me/AGENTS.md)、[研究项目规则](/home/me/PaperField/AGENTS.md) 和 [全局约定](docs/GLOBAL_RULES.md)。
2. 通读 [Idea B 提案](docs/IdeaB_Calibrated_Latent_Reader.md)；其原文为 [proposal.md](docs/proposal.md)，预期效果与理论声明不等同实验结论。
3. 读取 [当前状态与接续索引](docs/STATUS.md)、[项目规划](docs/CLR_Codex_Plan.md) 及用户当前里程碑任务书，核实冻结版本后再工作。

用户后续明确修订须同步落盘；全局约定不会因换对话失效。历史记录只作证据，不覆盖当前规则。

## 必须牢记的执行约束

- CLR：`R(z, q, s) → p(a | z, q)`；类型为 choice / score / yes-no，输出校准概率及 split conformal 预测集。
- 代码放 `clr/`；实验只由 YAML 驱动，命令行只传 config 路径和覆盖项。每次运行保存 config、git commit、环境、seed、日志、指标和成本。
- 按 problem / episode ID 冻结划分。调参只用 val / cal；**smoke 和调试一律用 val 代理，不触碰 test；test 只在最终评估使用。**
- 每个里程碑先通过 ≤ 5% 数据、≤ 15 分钟的 smoke，再跑完整版本。模型加载/显存等失败直接报错并记录，禁止静默换模型或缩小设置。
- 如实报告负面结果，不改指标、换划分或挑 seed 美化结果。最小测试覆盖标签、conformal、选项顺序和划分无泄漏。
- 已完成代码与结果冻结；修改新增版本，不覆盖旧结果。研究声明须有证据，conformal 保证须说明假设。
- **从 M4 起唯一发给审查者的文件是 `reports/Mx_REVIEW.md`：≤ 8KB，固定 (a)–(f) 六段、≤ 3 张核心主表、3 位有效数字；具体格式见全局约定第 10 条。**详细报告和 CSV 留在仓库并索引。
- **EXPERIMENTS.md / BLOCKERS.md 每条新记录 ≤ 10 行**，长内容放报告链接；不复制历史阻塞，也不拆条规避长度限制。
- 所有状态更新持久化：本页维护简要状态，`docs/STATUS.md` 维护接续；实验、阻塞、变更写入 docs 下相应文档。根目录 EXPERIMENTS.md / BLOCKERS.md 是同一记录的链接。
- reader/所有基线不看源问题文本，只读观测状态/前缀、类型化模板与选项；不用加噪输入向量代替观测状态。并列报告 reader_val/test，Q1 在 reader_val 分 clean/traj。
- 复用 Idea A 通过 import，不复制或修改其冻结代码。共享数据/权重优先复用；环境独立实测，不照搬其他项目验收结论。
- 不因磁盘余量低主动停止：按授权和配重管理规则释放 `/home/me/tools/disk-reserve/` 空间，使用期间暂停自动补足；真实 I/O 错误必须记录。GPU 限制与资源须实时核实。

## 当前状态（2026-11-03，北京时间）

- M0.1、M1 v3 已获用户审查通过；M2–M3/G0 已获用户审查通过；M4 实验完成待 G1 审查，任务书见 [docs/M4_TASK.md](docs/M4_TASK.md)。M5–M9 尚未执行。
- M1 数据：`data/dumps/codi_gpt2_v3`；冻结清单 `reports/M1_completed_v3_manifest.json`。保留原 cohort 左 padding 与协议上下文，状态受 cohort 影响。
- M2–M3：`data/M23_v1a/freeze.json`（635 个文件）；核验 `data/M23_v1a/verification.json`。历史 smoke/test 暴露如实保留；用户已通过 G0，不抹去历史偏离；M4 严格使用 val 代理。
- Q1 已确认规则：每条轨迹最终正误只监督干净起点 z_t0 和其后观测状态；不监督更早干净前缀，独立 Bernoulli 样本不预先平均；终点分别使用 clean/traj 自身正误。
- M2–M3 校准划分：`data/splits/gsm8k_M23_cal_v1.json`，cal_A250 / cal_B250。后续使用冻结版本，变更须按版本管理与任务授权执行。
- CODI-LLaMA 复现已按现状接受，噪声留至 M6 前；Huginn/Coconut 暂停，B5b 延至 M6。M2–M3“不训练 CLR”的任务范围不是 M4 的永久禁令。
- 2026-10-29 临时 GPU 2 禁用已到期；新任务仍须实时检查资源，不能依据过期快照分配 GPU。

历史逐次进度原样归档于 [旧 AGENTS 快照](docs/history/AGENTS_20261103_before_global_rules.md)，完整状态历史见 [STATUS.md](docs/STATUS.md)。这些历史材料不覆盖本页与全局约定。

<!-- M4_LIVE -->
M4 v0实验完成，待G1审查；唯一审查文件 reports/M4_REVIEW.md。代码/数据/结果冻结于 data/M4_v0/freeze.json；未经用户G1审查不启动M5。
<!-- /M4_LIVE -->


M4控制器实时状态索引：[docs/M4_LIVE.json](docs/M4_LIVE.json)，每次任务完成自动持久化；详细进度从该文件指向的run/progress.json核实，不能把启动状态当完成。

<!-- M5A_LIVE -->
M5a全量val300/三seed及配对统计完成，待审查；oracle同预算S1增益32.8pp，95%CI[27.8,38]；B4/B3未见正收益。保守累计5.81GPUh≤10，不触碰test。仅S3 oracle限N16/b4全步（用户继续授权），其余网格完整。唯一审查reports/M5a_REVIEW.md；冻结data/M5a_v13/freeze.json，接续docs/M5a_LIVE.json；不代表G1通过。
<!-- /M5A_LIVE -->


## M4.1 当前任务（2026-11-05）
用户授权失效诊断与修复，G1暂不判定；[任务书](docs/M4.1_TASK.md)。先仅val/reader_val诊断，两个变体通过过拟合/合成后才重跑主比较；预算35 GPUh，M4冻结不改。

M4.1接续索引：docs/M41_LIVE.json（生成后）、docs/STATUS.md；tmux clr-m41-gates / clr-m41-pipeline。A诊断完成前不定根因，双候选学习能力关卡全部通过前不进入C。A/B已完成，C全量三seed训练中；M4.1尚未完成。

<!-- M41_LIVE -->
M4.1（2026-11-06T00:58:03+08:00）：M4.1实验完成，待G1审查；唯一审查文件reports/M4.1_REVIEW.md，冻结data/M41_v1/freeze.json。M4原冻结哈希全部一致。；状态索引docs/M41_LIVE.json。
<!-- /M41_LIVE -->

M4.1实验完成，待G1审查；唯一审查文件reports/M4.1_REVIEW.md，冻结data/M41_v1/freeze.json。M4原冻结哈希全部一致。

<!-- M5B_LIVE -->
M5b A–D/三seed、配对统计完成，待审查；累计16.8/25 GPUh。唯一审查reports/M5b_REVIEW.md；冻结data/M5b_v1/freeze.json，接续docs/M5b_LIVE.json。搜索结论见审查报告；A历史无效开发run提前test暴露保留并排除，B/C/D未触碰test。用户G1按Q3/Q4有条件通过；未启动M6。
<!-- /M5B_LIVE -->


<!-- M6_LIVE -->
M6实验与统计完成，待审查；唯一审查reports/M6_REVIEW.md，冻结data/M6_v1/freeze.json。
<!-- /M6_LIVE -->

M7 前置核验（2026-11-06）：Idea A M1.1 已完成，KV 契约已通过 M3 机制验收；Idea A M4 正式 val 仍待评测，CLR 集成须另验。归档对话与版本证据见 [docs/M7_IDEA_A_READINESS.md](docs/M7_IDEA_A_READINESS.md)；本次未启动 M7。

<!-- M7_LIVE -->
M7非自适应实验完成待审查；唯一审查reports/M7_REVIEW.md。冻结data/M7_v1/freeze.json；缩量/历史raw缺证据/1%无穷门限等限制见审查文件；未启动M8。
<!-- /M7_LIVE -->

<!-- M71_LIVE -->
M7.1足量非自适应审计完成待审查；C1/C2通过，三seed主比较/统计完成；唯一审查reports/M7.1_REVIEW.md，冻结data/M71_v1/freeze.json；接续docs/M71_LIVE.json，M8未启动。
<!-- /M71_LIVE -->

<!-- M8_LIVE -->
M8a固定500组评估与统计完成待审查；P0通过，三seed/E0–E4齐全；唯一审查reports/M8a_REVIEW.md，冻结data/M8a_v1/freeze.json。
<!-- /M8_LIVE -->

<!-- M8B_LIVE -->
M8b四种设计/三seed、固定500组E1/E2/E3评估与统计完成待审查；唯一审查reports/M8b_REVIEW.md，冻结data/M8b_v1/freeze.json。
<!-- /M8B_LIVE -->

<!-- M8C_LIVE -->
M8c固定test200组三seed E3主矩阵完成待审查；21GPUh内次要项完成范围见reports/M8c_REVIEW.md，冻结data/M8c_v1/freeze.json。
<!-- /M8C_LIVE -->

<!-- M9_LIVE -->
M9-A三seed联合训练与论文产物完成待审查；实验至此冻结。唯一审查reports/M9_REVIEW.md，冻结data/M9_v1/freeze.json。
<!-- /M9_LIVE -->

<!-- M10_LIVE -->
M10投稿补实验及论文产物完成待审查，实验最终冻结；B5c全量因预算NR，HiddenBench仅探索性。唯一审查reports/M10_REVIEW.md；冻结data/M10_v1/freeze.json。
<!-- /M10_LIVE -->

<!-- M10b_LIVE -->
M10b GPT2 B5c Q3/Q4三seed与论文复现完成待审查，实验永久冻结；唯一审查reports/M10b_REVIEW.md；冻结data/M10b_v1/freeze.json。
<!-- /M10b_LIVE -->

<!-- M10c_LIVE -->
M10c只读开销测量与T6/配对conformal汇总完成待审查；无训练/重校准，科学结果永久冻结。唯一审查reports/M10c_REVIEW.md；冻结data/M10c_v1/freeze.json。
<!-- /M10c_LIVE -->

<!-- M10d_LIVE -->
M10d四格补齐完成待审查；GPT2 B6复用0.574/0.853，LLaMA B4新增三seed0.551/0.806（Q1 NLL/Q2 AUROC）；0.0289GPUh≤2、源前向0，旧冻结不改；论文仍8页。唯一审查reports/M10d_REVIEW.md；冻结data/M10d_v1/freeze.json。
<!-- /M10d_LIVE -->

<!-- M10e_LIVE -->
M10e两源HSRM三seed/确定性CLUE及LLaMA B2 Q1–Q4三seed完成待审查，实验再次冻结；0.942GPUh、源前向0，旧科学输入SHA一致。论文按用户修订仅交付reports/M10e_PAPER_HANDOFF.md，未改稿/编译。唯一审查reports/M10e_REVIEW.md；冻结data/M10e_v1/freeze.json。
<!-- /M10e_LIVE -->

<!-- M11_QUAL_LIVE -->
M11-qual只读导出完成待审查；两源各12题、跨题300、兄弟40、审计600、M8c100组。唯一审查reports/M11_qual_REVIEW.md；交付data/M11_qual_v1/qual_export.zip，冻结data/M11_qual_v1/freeze.json。无训练/重校准/PGD；M8c仅已有分数。
<!-- /M11_QUAL_LIVE -->

<!-- M12_LIVE -->
M12组件消融完成待审查；5变体三seed，18.1/30GPUh；唯一审查reports/M12_REVIEW.md，冻结data/M12_ablation/freeze.json；未改论文。
<!-- /M12_LIVE -->

2026-11-16 目录维护：目录/Git 保留策略见 [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)；当前论文是 `manuscripts/CVPR27-CLR/` 独立 Git 仓库。补充统计在 `supplemental/stats_20261116_v1/` 独立冻结；维护不改变科学结果或审查状态。
