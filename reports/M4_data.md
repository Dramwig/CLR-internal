# M4 D：未见题池补齐（数据生成已完成，M4整体仍运行）

生产run：`runs/20261103T063500842769Z_924ffcfc4ca2_e9f41f`；QA run：`runs/20261103T070034793210Z_f4a64da4e26f_7aefa4`；追加冻结ID核对的审计：`runs/20261103T071814616276Z_158d311376b0_81e1b6`。

## 规模与成本

| 项目 | unseen_train | unseen_eval |
|---|---:|---:|
| SVAMP题数 | 496 | 497 |
| ASDiv题数 | 500 | 500 |
| 总题数 | 996 | 997 |
| 保存观测状态行 | 42992 | 6979 |
| QA行（eval含两个held-out模板） | 98964 | 99603 |
| Q1行 | 37016 | 20937 |
| Q2行 | 6972 | 20937 |
| Q3行 | 42992 | 20937 |
| Q4行 | 11984 | 36792 |
| 公式解析失败题 | 139 | 120 |
| 无法配对Q4负例题 | 1 | 1 |

完整生产：0.419144562 GPU小时、117130次实际源模型forward调用（批处理一次调用计一次，不乘K），物理GPU2/RTX4090。此前12题GSM val代理smoke另计0.00402749 GPU小时、696次forward；完整D加smoke约0.423 GPU小时，低于6小时上限。只用于生产数据，没有训练reader；D统计不冒充三seed学习结果。

## 冻结协议

划分：`data/splits/M4_unseen_v0.json`，seed11，池内按problem ID随机二分。保留M3已有clean/早退标签；源模型使用M1的fp32/batch1、原cohort左padding、原position/attention/KV上下文。每题从相同上下文重算前缀，再接续唯一的 `clr.policy.rollout`；所有重算clean状态转fp16后与既存dump逐元素一致，否则生产程序直接报错。

train每题K8、均匀随机t0、sigma1.5、temperature0，保存实际输出观测状态；eval对每个t=0..5做K16同策略rollout，t6正确性来自clean greedy。数据：`data/M4_v0/unseen`；QA：`data/M4_v0/qa`；逐题完整上下文与方程解析：`data/M4_v0/unseen_plan.json`。

Q1只标clean起点和后续观测轨迹状态，未把更早clean前缀错标为策略样本。Q2复用clean早退；Q3四选项去重、标签对应该状态所属clean/轨迹最终答案；Q4从原始Equation/Formula安全AST解析，失败只跳过该题Q4。两题无可用匹配负例：train `asdiv:nluds-1847`、eval `asdiv:nluds-0325`，不丢弃其余类型。

## 审计与限制

`data/M4_v0/data_audit.json` 已通过：manifest ID精确等于冻结split、train/eval两两无交集、Q1起点规则、Q2早退标签、Q3去重与正确选项、Q4数值正负边际完全相同、cohort协议一致。Q3 train正确位置0/1/2/3计数10751/10831/10619/10791；eval（含重复模板）5352/5157/5196/5232。

SVAMP990/993、ASDiv744/1000题公式可解析；合计259题解析失败，另2题负例匹配失败。本次采用“逐题失败跳过Q4、池内成功题仍保留Q4”的解释，完整缺失列表保留，不把缺失记为负例。

虽然problem ID分离，172/997个eval题与新增train存在13-gram重叠（SVAMP164、ASDiv8），精确文本重复0。保留用户规定的冻结二分；最终同时报告825个无13-gram重叠eval题的诊断结果，不以该过滤子集调参。匹配证据：`data/M4_v0/unseen_train_eval_overlap.json`。与GSM train/test/Aug的去重及许可沿用M3已冻结材料；ASDiv的非商业许可限制也继续适用。

unseen_eval的reader指标与GSM cal_B阈值下的跨域覆盖率只在全部训练锁定后最终评估；本文不提前挑选数据或展示性能来调整训练。
