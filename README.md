# Calibrated Latent Reader (CLR)

`R(z, q, s) → p(a | z, q)`：读取潜在思维/KV 消息，输出类型化答案概率、校准概率和 split conformal 预测集。

实验已推进至 M12 组件消融，结果完成待审查；最新补充统计位于 [supplemental/stats_20261116_v1/REPORT.md](supplemental/stats_20261116_v1/REPORT.md)。完成不代表审查通过，结论与限制以各阶段报告及冻结证据为准。

- 开始工作：[项目约定](AGENTS.md)、[全局规则](docs/GLOBAL_RULES.md)、[状态与接续](docs/STATUS.md)。
- 研究材料：[原始提案](docs/proposal.md)、[总体规划](docs/CLR_Codex_Plan.md)、[文档索引](docs/README.md)。
- 结果与溯源：[总结](reports/FINAL_SUMMARY.md)、`reports/M*_REVIEW.md`、[论文图表入口](paper/README.md)。
- 持久记录：[实验](docs/EXPERIMENTS.md)、[阻塞](docs/BLOCKERS.md)、[变更](docs/CHANGELOG.md)。

| 目录 | 内容 |
| --- | --- |
| `clr/` | 源接口、数据、模型、校准、搜索与审计代码 |
| `configs/`、`scripts/`、`tests/` | YAML 配置、执行入口和单元测试 |
| `docs/`、`reports/` | 规则、任务与接续记录；审查报告、明细及历史证据 |
| `paper/` | 冻结结果的图表、汇总数据与构建来源 |
| `manuscripts/` | 独立论文仓库及历史冻结快照，见其 [说明](manuscripts/README.md) |
| `supplemental/` | 独立版本的补充分析、配置、报告和冻结清单 |
| `data/`、`runs/` | 本地实验数据和实际运行产物；仅划分、冻结清单和小型论文输入进入 Git |

完整目录和保留策略见 [REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)。虚拟环境、安装工具、模型权重、逐题观测和运行目录不随 Git 分发；克隆仓库并不等于获得全部实验输入。本地环境为 `.venv*`，共享权重只读引用 `/home/me/models/`，不修改 Idea A 冻结代码。实验入口只接受配置路径与覆盖项。

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python scripts/m0_tests.py configs/M0_tests.yaml
PYTHONPATH=. .venv/bin/python scripts/m0_inventory.py configs/M0_inventory.yaml
```

源接口：`from clr.sources.codi import load; source = load(config); result = source.run(problem, perturb=None)`。每个源的latent定义、指定层观测和噪声插入点见对应docstring。小型随机模型测试不代表真实源复现。
