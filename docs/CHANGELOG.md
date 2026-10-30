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

### M0 CODI子结果交付

- 官方未修改test.py对照为43.594%，CLR等长接口为41.698%；差异完整保留并分析，未用测试结果调整checkpoint、sigma或指标。
- 64题固定val面板完成5个sigma × K16 × 3 seeds；报告负面准确率变化和batch数值差异。
- 15项最小测试通过；Idea A仓库保持未修改。
- M0整体仍有LLaMA授权、Coconut预算/显存、Huginn不完整权重等阻塞，不标记完成。

## 2026-10-28 — M0.1 v2（审查中）
- 新建CODI v2接口，显式记录左padding上下文、mask、position、张量布局及fp32/batch1候选；v1接口未修改。
- 按题10k bootstrap替换确定性评测的重复seed CI；保留旧报告不追改。
- 官方GSM8k-Aug归一化精确/13-gram去重，获准将原500验证池重划为清洁val300/cal193，test移除7题至1312；原v1划分不变。
- 新增18设置K16噪声与采样对照程序、V直方图和非退化门槛，smoke通过；完整噪声与M1受生成协议一致率门槛阻止。
- 新增M0b Huginn官方恢复/校验与原生harness入口，Coconut仅第三方元数据报告。

## 2026-10-29 — LLaMA镜像资源 v2
- 按用户指定使用ModelScope的同一Llama-3.2-1B-Instruct，锁revision并逐文件校验本地SHA与上游Git对象；没有HF模型文件下载。
- 新增配置驱动下载/离线格式验证入口，保留原M0 v1配置和结果。
- CODI作者checkpoint的获取渠道仍待范围澄清；GPT2完整噪声作业继续。

## 2026-10-29 — CODI作者checkpoint来源已获不限平台授权
- 作者公开checkpoint已下载并验证官方LFS SHA；ModelScope底模保持不变。
- 新建CODI-LLaMA v2协议、官方test.py只读封装和阶段流水线；源加载strict、smoke限时、确定性按题bootstrap。
- 既有M0 v1与正在运行的GPT2实验设置不改动，复现后仍须完成噪声及审查。

## 2026-10-29 — CODI-LLaMA v2真实复现结果
- 通过精确allowlist保留不参与官方推理的4个auxiliary张量，其余state_dict仍严格校验；未知键/缺失键测试通过。
- GPU5预检失败后显式调度GPU4并重跑matching smoke，完成官方1319题、CLR batch16与batch128控制。
- 保留27/14题不一致与论文参考+3.597pp偏差，未改指标或seed；原M0 v1不改动。噪声与M0b最终审查仍待完成。

## 2026-10-29T16:10:48.755456+08:00 临时资源约束

记录物理 GPU 2 至北京时间 2026-10-29 18:06 的禁用窗口（含 CUDA_VISIBLE_DEVICES 重映射），当前 GPT2 噪声任务核实位于物理 GPU 3，保持运行。未新增实验或修改冻结结果。

## 2026-10-30 — M0.1 收尾

新增收尾报告，核验71个冻结噪声文件及划分、重跑回归23 passed/1 skipped；更新EXPERIMENTS/BLOCKERS当前状态。M1等待用户任务书。释放磁盘配重10GiB并保持暂停，未改冻结协议或结果。

## 2026-10-30 — M1 数据管线启动

新增共享continuation policy、PrefixStates/KV上下文、候选划分准备、gold中间量解析、safetensors/Parquet writer+loader、200题smoke与val300关卡、预算后冻结脚本。保留M0/M0.1冻结源文件。当前没有reader/probe训练，也没有M1全量完成声明。
