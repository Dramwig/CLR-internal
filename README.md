# CalReader

**Calibrated Latent Reader**

通过类型化问题读取潜在推理状态与潜通信消息，输出校准概率和 conformal 预测集，用于过程价值估计与语义审计。

当前阶段：构想初始化，仅建立目录与文档，尚未实现模型、配置环境或运行实验。短标题用于项目目录；提案原有名称与缩写保留。

- [完整提案](docs/proposal.md)；[附件原文](IdeaB_Calibrated_Latent_Reader.md)。
- [文档索引](docs/README.md)；[项目约定](AGENTS.md)。
- 目录布局参照 [LatDefer](../LatDefer/)，按本项目模块调整。

```text
calreader/data/  潜对象、类型化问题与标签加载
calreader/adapters/  不同源模型的潜对象适配
calreader/reader/  共享读取器与类型化输出头
calreader/calibration/  概率校准与 conformal 预测集
calreader/search/  潜空间搜索与过程价值估计
calreader/audit/  潜通信一致性审计与攻击评测
calreader/eval/  准确率、校准、覆盖率与成本评测
calreader/utils/  种子、日志与运行记录
configs/       实验配置
data/          项目派生数据；共享数据通过路径复用
scripts/       训练、缓存与评测入口
tests/         实现后的验证测试
docs/          提案与研究记录
runs/          每次实验的配置、指标、日志与 checkpoint
outputs/       准备性产物
third_party/   按需引入的第三方代码
```

共享数据使用 `/home/me/datasets`，共享权重使用 `/home/me/models`。具体资源与依赖版本在实际选型后记录。
