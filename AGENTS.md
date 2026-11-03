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
- 不因磁盘余量低主动停止：按授权和配重管理规则释放 `/home/me/disk-reserve/` 空间，使用期间暂停自动补足；真实 I/O 错误必须记录。GPU 限制与资源须实时核实。

## 当前状态（2026-11-03，北京时间）

- M0.1、M1 v3 已获用户审查通过；M2–M3/G0 已获用户审查通过；M4 已授权并开始，任务书见 [docs/M4_TASK.md](docs/M4_TASK.md)。M5–M9 尚未执行。
- M1 数据：`data/dumps/codi_gpt2_v3`；冻结清单 `reports/M1_completed_v3_manifest.json`。保留原 cohort 左 padding 与协议上下文，状态受 cohort 影响。
- M2–M3：`data/M23_v1a/freeze.json`（635 个文件）；核验 `data/M23_v1a/verification.json`。历史 smoke/test 暴露如实保留；用户已通过 G0，不抹去历史偏离；M4 严格使用 val 代理。
- Q1 已确认规则：每条轨迹最终正误只监督干净起点 z_t0 和其后观测状态；不监督更早干净前缀，独立 Bernoulli 样本不预先平均；终点分别使用 clean/traj 自身正误。
- M2–M3 校准划分：`data/splits/gsm8k_M23_cal_v1.json`，cal_A250 / cal_B250。后续使用冻结版本，变更须按版本管理与任务授权执行。
- CODI-LLaMA 复现已按现状接受，噪声留至 M6 前；Huginn/Coconut 暂停，B5b 延至 M6。M2–M3“不训练 CLR”的任务范围不是 M4 的永久禁令。
- 2026-10-29 临时 GPU 2 禁用已到期；新任务仍须实时检查资源，不能依据过期快照分配 GPU。

历史逐次进度原样归档于 [旧 AGENTS 快照](docs/history/AGENTS_20261103_before_global_rules.md)，完整状态历史见 [STATUS.md](docs/STATUS.md)。这些历史材料不覆盖本页与全局约定。

<!-- M4_LIVE -->
M4：核实ModernBERT-base、块掩码与资源；D数据补齐及读取器实现准备中，尚无M4实验结论。总预算80GPU小时，D上限6GPU小时。
<!-- /M4_LIVE -->

M4当前：源val smoke12题通过，完整D估算0.451 GPUh并已启动；CLR smoke通过（26.6训练样本/s、93.8评估样本/s、2.42GiB峰值、置换差0）。训练预算据此设4epoch、每qtype每epoch10000样本，优先四项必做，详见M4_TASK.md及后续预算报告。未见池172条13-gram跨划分重叠另作无重叠子集诊断，不改冻结划分。
