# 变更记录

## 2026-10-28

- 短标题定为 CalReader，参照 LatDefer 建立项目目录骨架。
- 保留附件原文及 `docs/proposal.md` 完整副本。
- 添加项目说明、工作约定与文档索引；尚未实现研究代码或运行实验。

## 2026-10-28 — M0 实现（进行中）

- 按用户新约定采用 `clr/`，保留旧空骨架；增加CODI/Coconut/Huginn源接口和相对噪声，禁止模型/设备/大小静默回退。
- 初始化本项目Git；通过import只读复用Idea A的Run、日志、seed和YAML loader；Idea A KV模块仍为占位文件，未复制或修改。
- 核实官方模型与代码，记录精确revision；提供固定版本下载和SHA-256验证的并行传输。
- 官方数据文本最初误按三个`||`字段解析并报错；按上游`question || steps ## answer`修正。失败运行保留；未产生旧版划分或准确率，不影响既有结果。
- 冻结GSM8K问题级val/cal/test；测试仅用于最终源复现，噪声曲线在事先指定的64题val面板运行。
- 推理新增显式batch16，按等长token组执行，避免padding改变latent attention。单样本与批处理smoke分别留档，禁止静默替换。确定性抽查16题，batch与single漂移另计。
- Coconut实际batch32成本探针OOM；此前明确batch8的吞吐外推仅为粗略预算，不能当作精度实验或完整官方recipe预算。
