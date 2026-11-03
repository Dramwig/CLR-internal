# M4 当前任务书与预注册约束

2026-11-03：用户明确 M2–M3/G0 审查通过，授权 M4 CODI-GPT2 单源 CLR v0。此前报告与偏离保留；本次不得再以 smoke/调试触碰 test。

## A 模型

2层MLP适配768维观测状态，读取z_0..z_t前缀、步位置编码及源token。预训练双向encoder采用官方核实的ModernBERT-base。只输入类型化模板和选项；每个选项附加与B3相同的8维数值特征投影。所有层采用祖先块掩码、兄弟选项共享逻辑位置；模板不可读取选项，避免间接串扰。每隔k层插入cross-attention，k为验证集超参。共享模型训练Q1–Q4、qtype平衡采样，clean/traj遵循M23标签语义；reader_val各qtype平均NLL早停。Q1为二分类Bernoulli，不改为10-bin。

## B 校准与集合

冻结cal_A250、cal_B250。cal_A按qtype拟合温度；并对CLR/B3公平比较仿射校准。二分类对log-odds拟合正温度+偏置；Q3使用共享选项偏置（softmax抵消，等价温度），不得用选项位置偏置破坏置换等变性。cal_B按(qtype,t) Mondrian、每题一个实例；α=.05/.1/.2。沿用M23 Bernoulli NLL/Brier/ECE定义。

## C 实验顺序与预算

总预算≤80 GPU小时，保守计入D≤6 GPU小时、smoke、失败、调参及评估。先val smoke（≤5%数据、≤15分钟），再实测吞吐估算并冻结完整实验预算。模型选择仅reader_val，校准仅cal_A，阈值仅cal_B；锁定方案后才进入最终test/unseen_eval。

必须按优先级：
1. CLR vs逐类型B3/B4/B0，三seed11/22/33；Q1 NLL/Brier/MAE、Q2 AUROC、Q3 acc及相对B0提升、Q4 acc/AUROC，以及ECE、覆盖率/集合大小；reader_val/test并列，另报Q1 clean/traj。
2. B3-multi：共享MLP加qtype embedding，区分统一模型效果。
3. Q1–Q3预训练CLR后用N=100/1000/10000个Q4训练标签微调，对比相同N从零训练的B3，三seed，学习曲线；N为标签实例数，记录独立题数和嵌套取样，验证/测试题不参与取样。
4. 纯reader_train vs reader_train+unseen_train，在GSM8K test及unseen_eval评估；不以unseen_eval调参。
时间允许才做：无cross-attention（平均状态拼CLS）、单步vs前缀、无块掩码及置换、预训练vs同尺寸随机初始化；舍弃项须写明预算原因。即使不做无掩码消融，带掩码的置换/隔离测试仍是实现关卡。

## D 未见题池

复用M3 SVAMP993、ASDiv1000及clean/早退。各池按problem ID固定seed对半分为unseen_train/unseen_eval并冻结；奇数SVAMP训练496、评估497。保留原M3 cohort/padding/KV协议。train补K8随机起点π轨迹，eval每个干净t=0..5补K16π MC，π=相对高斯sigma1.5、temperature0，共用clr.policy.rollout。source调用计入独立≤6GPU小时预算。D smoke以GSM8K val代理，不用unseen_eval调试。使用原始Equation/Formula字段构造Q4，记录解析失败/跳过规则。用GSM8K cal_B阈值报告CLR/B3在unseen_eval的覆盖率，不重校准未见池。

## E 唯一审查交付

reports/M4_REVIEW.md：≤8000 UTF-8字节，固定(a)状态、(b)关卡通过/未通过+数字、(c)≤3张reader_val/test核心表（3位有效数字）、(d)本次新增偏离/阻塞、(e)需审查决定、(f)详细报告/CSV/run路径。在其中回答G1四问：CLR Q3扣B0/Q4是否明显优B3；同覆盖率集合是否更小；统一CLR是否不劣于B3且优于B3-multi；小N迁移是否优从零B3。另报仿射改善及补未见训练后的GSM8K落差变化。详细报告保留逐t/模板/状态及成本/CI，不在审查摘要堆大表。

“明显”以预先固定的配对按题bootstrap差异95% CI（10k次）报告，三训练seed全部保留；不将不显著等同已证明不劣，覆盖率不足时不能把小集合称为效率优势。没有预设非劣界时报告区间及是否足以支持结论，不事后选择界值。

## 首次正式训练前的预算修订（2026-11-03）
k4完整pilot：40000样本训练（含加载）0.3921446GPUh，202068行reader_val验证与保存另0.2463656GPUh，总0.6385102。比val smoke评估吞吐估计更快；尚未启动任何正式三seed训练、未查看test。
为提高四项必做的训练充分性并使早停真正生效，新 `configs/M4_train_v1.yaml` 取最大10epoch、每2epoch验证、patience2；每qtype每epoch10000样本及所有数据/指标/seeds/模型不变。CLR主模型、B3-multi、Q1–Q3预训练和增强训练都遵循该上限；few-shot仍20次小样本遍历。保留原4epoch配置/预算和两次pilot，不覆盖旧结果。重新估算含20%裕量约76GPUh，上限80；详见reports/M4_budget_v2.md。此修订只依据val吞吐与pilot学习不足，不依据test结果。
