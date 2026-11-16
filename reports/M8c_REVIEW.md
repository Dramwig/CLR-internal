# M8c 审查

### (a) 状态一句话
M8c固定主test200组三seed E3完成待审查；次要项按21GPUh余量执行，实际完成数如下。

### (b) 关卡清单
- 通过：short direct-lookup val200有/无/错配消息准确率93/27/26%，95%组bootstrap CI分别为[89.5,96.5]/[21,33]/[20,32.5]%，达到≥80/≤35/≤35%准入。完整日志69.5/27.0/30.0%未通过，按预定顺序缩短日志。
- 通过：冻结synthetic v2三核心SHA；仅删除非目标Ticket日志、保留全部目标记录。D0沿用M71权重/温度，新cal_B1000重定阈值；R4新适配器、原主体冻结，train4000、val200、cal_A300/cal_B1000。
- 通过：R4三seed各≥4.22e+06问题曝光，checkpoint仅在达到曝光后按val NLL选。审计期望严格解析可见commitment；receiver不看commitment或私有原文。
- 通过：val20 PGD smoke、实际BF16逐层约束、保存/在线特征重放与协议测试；test先锁定配置/权重/ID。报告抽样独立于优化/停止/λ选择。
- 成本：累计21/21 GPUh，含加载/失败/smoke/训练/校准/评估，RTX4090按驻留墙钟计。矩阵与预算限制见(d)。

R4 val/test A1/A4并列报告；val全部200组，test固定200组，逐seed证据见详细表。
正式val20矩阵90分钟超时，仅19完整组用于描述性E3；不称val20矩阵完成。

### (c) 核心主表（最多3张）
三seed组内均值；95%CI为10000次group bootstrap，条件于这些seed/阈值/有限搜索。QA为clean，A1正负各半且覆盖三种假值；FPR由独立clean问题集报告。

| 读取器 | val A1/A4 % | test A1/A4 % | val/test clean FPR % |
|---|---|---|---|
| D0 | 77.3 [75.4, 79.1] / 95.7 [93.2, 97.8] | 77.5 [75.9, 79.1] / 94 [91.2, 96.3] | 0 [0, 0] / 5.17 [3.17, 7.5] |
| R4 | 94.2 [93, 95.4] / 99.8 [99.5, 100] | 91.7 [90.3, 93.1] / 99.5 [98.8, 100] | 10.5 [3.51, 19.3] / 3.67 [2.17, 5.5] |

E3：test固定分母200（来自冻结test500的哈希序前缀；val为超时run完整前19组的描述性结果），原本非目标→CF目标且在名义FPR5%阈值下接受；200步上限/EOT16，成功早停。主λ与附加网格严格遵循budget_lock，有限攻击给未知最坏失效的经验下界。

| ε | D0 val/test失效 % | R4 val/test失效 % |
|---|---|---|
| 0.01 | 91.2 [78.9, 100] / 96.5 [94, 98.5] | 87.7 [73.7, 98.2] / 81.7 [78.3, 85] |
| 0.1 | 91.2 [78.9, 100] / 96 [93.5, 98.2] | 71.9 [59.6, 82.5] / 72 [68.3, 75.5] |
| 1 | 93 [80.7, 100] / 95.3 [92.7, 97.5] | 64.9 [52.6, 77.2] / 64 [60.3, 67.7] |

最小扰动为有限PGD找到的成本，非认证数学最小。比值=规避读取器成本/仅改变receiver成本；原已目标、搜索失败/右截尾不当作零或删除。表中分位数条件于双侧搜索成功，全部截尾与独立报告面板接受率另见CSV。

| split/读取器 | 有限比值对数/全部 | P10/P50/P90 | 读取器/receiver截尾 |
|---|---|---|---|
| test/D0 | 45/45 | 1/1/1 | 0/0 |
| test/R4 | 45/45 | 1/1/2 | 0/0 |

次要项实际执行：{"minimum_test": "15/100完整组，预算停止或资源失败，部分结果保留", "test_blind": "未执行：前序未完成或预算耗尽", "test_sensitivity": "未执行：前序未完成或预算耗尽"}。仅完成组用于描述性比值分位数，不能冒称100组完成。

### (d) 本里程碑新增的偏离与阻塞
主test按用户最终授权固定200组、三seed、ε=.01/.1/1、λ=1、PGD200；原test500不外推，ε=.03/.3裁掉。
次要项仅主200的前100组，按最小扰动→E1/E2→λ敏感性顺序消耗余量；未完成项与部分组分别记录，不再申请预算。
此前15GPUh预算报告及60组提议保留为历史，60组从未运行；21GPUh含此前6.43GPUh、全部失败/中止成本。
最小扰动是有限PGD搜索发现成本，二分最多6次；有限成功对的分位数不是认证最小或100/500组总体保证。
seed22原GPU run完成189组后因8h单run时间上限失败；接续继承原部署随机key、只补缺失trial，原结果和8.00GPUh均保留。CPU合并精确1200行，不冒充源GPU运行；不根据test结果改设置。
原final_lock保持只读；运行补充execution_supplement.json记录接续代码/资源门槛，execution_secondary_supplement.json记录次要项资源预检15→8GiB；模型、精度和科学矩阵不变。
共享卡显存预检失败及一次GPU5模型加载OOM均留run/trace并计费；未换模型/精度。早期误用M8b runtime曾更新其LIVE，已恢复索引；冻结结果SHA复核不变。早期route-context诊断、低曝光R4只作开发证据，未进入主表。
Conformal阈值只在clean可交换假设下有边际误报保证；自适应攻击不享有该保证。依赖准入证明群体任务依赖消息，不能证明每次行为改变都必然改变可读语义。R4消费消息后才可检查，不能在消费前拒收。各ε独立有限PGD、未合并小半径候选，非单调曲线不表示更大扰动更安全。

### (e) 需要审查者决定的问题
无。

### (f) 证据路径
reports/M8c_details.md；data/M8c_v1/analysis_final21/（robustness_summary.csv、group_metrics.csv.gz、qa_summary.csv、minimum_summary.csv、minimum_ratios.csv、minimum_rows.json、costs.csv、epsilon_failure.pdf）；data/M8c_v1/（final_lock.json、budget_lock.json、training_exposure_v2/、calibration/、upstream_final.json、freeze.json）；docs/M8c_TASK.md、M8c_LIVE.json；reports/M8c_evidence/。
