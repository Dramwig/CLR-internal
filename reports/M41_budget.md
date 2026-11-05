# M4.1 吞吐与预注册预算

仅依据val代理吞吐，未打开test。v1保留原22层ModernBERT训练，非静默换成小模型；lite按用户任务为4层256随机初始化。

- batch256 v1：Q1/Q2/Q3/Q4约1045/1323/647/925条每秒；峰值见runs/20261105T023724970330Z_8182c962d12c_a96c51。
- batch1024 lite：约22174/20660/21190/23697条每秒；runs/20261105T023724287608Z_fa5962b8b0c3_70c091。上述不含序列化/搬运，不能直接作实际成本。
- 每epoch完整遍历QA：Q1 740228、Q2 140000、Q3 860228、Q4 279440。取消1万采样；每个epoch无放回遍历所有实例，以逆qtype频率加权loss实现等类型目标。该方式与原固定条数平衡采样不同，显式记录。
- 主比较三seed，最早checkpoint资格为8个完整epoch（不少于各类型B3最佳checkpoint实际曝光；不把两个独立LR搜索的累计曝光当单个checkpoint曝光）；最多10epoch、patience1，reader_val宏qtype NLL选模型。B3-multi也按完整曝光重训，B3/B0引用冻结M4最终评估，评估锁之前不加载其test预测。
- 按实测纯计算速率：v1主训练8–10epoch三seed约16.3–20.4 GPUh；lite约0.62–0.77。另保留数据加载、验证、B3-multi及评估约4–5小时；A/B诊断约2–3小时。
- 两者较好者由三seed reader_val NLL选择。留Q4预训练全量Q1–Q3，最多2epoch、patience1；N100/1k/10k仅Q4全量小面板最多20epoch、patience4；独立三seed。该预训练预算不同于主比较8epoch曝光下限，不声称迁移预训练也匹配逐类型B3曝光。预计额外≤4–5 GPUh。
- 合计计划约29–35 GPUh；全局35硬限制，含失败、smoke和诊断。真实吞吐偏慢或门槛不通过直接记录，不静默缩样本/降模型/改seed。主训练进入前必须两个候选全部通过A4/A5及真实训练入口smoke。


2026-11-05 预算修订：用户明确允许提高上限，采用总50 GPUh（原35），含全部既有失败/smoke成本；新主训练单run时限14h。模型、数据、三seed、曝光下限与评估协议不变。旧进程保留启动时预算快照，不篡改运行记录；若seed11触发旧9h时限，则在原控制器全部退出后从last.pt显式恢复，新run记录50h授权。接续configs/M41_budget50_resume.yaml；新pipeline为configs/M41_pipeline_budget50_v3.yaml。
