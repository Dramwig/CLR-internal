# 目录与 Git 保留策略

2026-11-16 整理。科学代码、配置、报告和冻结结果沿用原路径；不因目录整理改写冻结清单、删除失败证据或重新运行实验。

## 版本管理

| 材料 | Git 策略 |
| --- | --- |
| `clr/`、`configs/`、`scripts/`、`tests/` | 保留源码、配置及测试，包括历史版本 |
| `docs/`、`reports/` | 保留任务书、记录、报告、CSV、图和历史日志；日志可能是冻结证据，不统一忽略 `*.log` |
| `data/splits/*.json` | 保留冻结划分；不评估或重新选择测试集 |
| `data/M*/freeze.json` | 保留里程碑冻结清单；只读索引，不代表所有清单内产物都在 Git 中 |
| `data/M*/paper_inputs/*.csv`、`*.json` | 保留小型图表输入与来源；文件名含 `problem` 的大型逐题 CSV 留在本地 |
| `paper/` | 保留图表、汇总数据、来源清单和构建记录 |
| `supplemental/<version>/` | 保留脚本、配置、报告、记录及冻结清单；Parquet 观测缓存和嵌套 `runs/` 留在本地 |
| `manuscripts/` | 当前论文独立仓库不纳入主 Git；历史冻结快照的源文件保留，详见该目录 README |
| `.venv*/`、`.tools/`、`third_party/` | 本地环境、安装工具和外部依赖，不纳入 Git |
| `runs/`、`outputs/`、`build/`、其余 `data/` | 本地运行、数据和编译产物，不纳入 Git；忽略不等于可以删除 |

重新克隆时需另外恢复本地数据、运行产物、外部依赖和独立论文仓库，才能重现依赖这些输入的分析。源权重优先复用共享模型目录。

## 入口与兼容

`docs/` 是提案和规划的唯一正文位置；根目录 `IdeaB_Calibrated_Latent_Reader.md`、`CLR_Codex_Plan.md` 使用相对符号链接保留旧入口。根目录 `EXPERIMENTS.md`、`BLOCKERS.md` 继续链接到 `docs/` 下的同一份记录。

科学文件名称、旧配置版本和历史报告保持原样，避免破坏冻结 SHA 和脚本引用。当前状态从 `AGENTS.md`、`docs/STATUS.md` 和对应 `*_LIVE.json` 进入，不从旧 README 推断。

## 本次清理

只清理未冻结的 Python/pytest 缓存、五个无活动进程的空 `docs/*_LIVE.lock`、根目录临时编译包装文件和废弃空 `calreader/` 骨架。保留环境、工具、数据、模型、历史日志及论文文件。

清理前后核对冻结清单哈希、其引用文件的大小与 mtime，以及非运行文件的 SHA-256；核对独立论文仓库状态未变。机器可读记录见 [maintenance/20261116_cleanup.json](maintenance/20261116_cleanup.json)。记录中的五个原已缺失路径属于旧论文构建/审计路径，本次未恢复或改写其历史证据。
