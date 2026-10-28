# CalReader 项目约定

## 目标与状态

通过类型化问题读取潜在推理状态与潜通信消息，输出校准概率和 conformal 预测集，用于过程价值估计与语义审计。

提案见 [docs/proposal.md](docs/proposal.md)。当前仅完成项目骨架；原文中的参考文献、新颖性、理论保证和预期效果尚未独立核验，不作为已有实验结论。

## 工作约定

- 遵循 [工作区规则](/home/me/AGENTS.md) 与 [研究项目规则](/home/me/PaperField/AGENTS.md)，沿用本项目目录。
- 研究代码放 `calreader/`；实验配置放 `configs/`，入口放 `scripts/`，关键行为测试放 `tests/`。
- 不编造实验结果。每次运行在 `runs/<run_id>/` 保存配置、代码版本、环境、种子、资源版本、日志与 `metrics.json`；汇总结果须可追溯。
- 固定数据划分，调参使用验证集，测试集用于最终评估；评测修正和影响写入变更记录。
- 训练、验证、校准和测试数据分别管理；conformal 覆盖率须说明可交换性等假设。跨模型迁移、分布偏移和自适应攻击下的表现均待验证。
- 优先复用共享数据和权重；项目派生数据、日志和 checkpoint 保留在本项目内。大型下载或训练前检查磁盘与 GPU。
- 环境和依赖版本在实现时确定并记录；不直接照搬 LatDefer 的环境验收状态或实验结果。
- 阻塞和必要的替代方案记录于 `docs/BLOCKERS.md`，运行记录见 `docs/EXPERIMENTS.md`，变更见 `docs/CHANGELOG.md`。
