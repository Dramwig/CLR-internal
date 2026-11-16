# M6 跨源读取与覆盖率

实验/设置锁：data/M6_v1/training_lock.json；所有模型与低优先级省略在最终reader test推断前锁定。

## 四项回答

CODI-LLaMA：Q3优势复现: Δacc=13.2pp [11.7, 14.8]；Q4优势复现: Δacc=11.5pp [9.81, 13.1]。配对区间完全大于0才支持优势复现；否则记为未建立。

小N按方向/类型/预算逐项判断：20/20个已执行配对区间下界>0；对照含b3,b5b,lite。B5b小N对照不全，仅可判断已执行N/方向/类型；不能推广到全部小N。

α=.1、Q3/Q4中6/32个设置标准覆盖率低于目标超过2pp（描述性界限），涉及dataset_gpt2_gsm_to_svamp、dataset_llama_gsm_to_svamp、source_llama_to_gpt2、step_gpt2_early_to_late；weighted对目标绝对偏差的中位改善=-0.0774pp。覆盖率变化的配对95%CI见analysis/coverage_recovery.csv；逐设置覆盖率CI及集合大小见shift/summary.csv。估计权重不构成偏移下的覆盖保证。

较小sigma仍未建立超过greedy的正收益；保留RQ1负面/分析结论。

## 协议与成本

LLaMA noise选择：sigma=2；门槛未达，按用户预登记降级使用。原四点失败保持不改，完整八点证据reports/M6_noise_selection_v2.md。reader_train=6817、reader_val=1000、val/cal/test沿用冻结problem ID。reader_train按smoke成本事前等比例截前缀；Q4配对不重采。

GPU墙钟包含smoke、失败与重启；各并行GPU作业直接求和。A=0.413、B=27.9、C+D=19.3，合计=47.6GPUh。源模型物理batch前向=1.91e+06次，每batch调用计1、不按K展开，不能视作等FLOPs。历史冻结模型成本另见上游报告，不混入M6新增预算。

CLR-lite：每源原生维度adapter与直接当前状态skip；迁移冻结全部donor参数，仅训练目标线性adapter；多源每源线性adapter+显式源token+共享lite。B3各类型独立LR/epoch由reader_val选定。

完整CLR沿用事先固定的10epoch上限、min_epochs=8、patience=1；仅epoch≥8的checkpoint参与选模。LLaMA早期reader_val NLL上升的过拟合迹象保留在history.json，未据此修改协议；不是跨全部早期epoch选择最小验证NLL。

小N为不同QA标签实例（同题/轨迹不假设独立），四类型平衡、固定嵌套100/300/1000/3000；完整train无标签归一化共用。reader_val固定128题选小N模型，完整模型用完整reader_val；不使用test选择。N只计目标训练标签，选模reader_val及所有模型共用的cal_A250题校准监督另计，不声称N为全部监督预算。按任务截取ID前缀并复用Q4原配对，子集不自动保留原完整划分的正负数值边际精确相等；实际TVD及计数见data_audit运行的audit.json，不重新配对以改变本次任务。

B5b只训练线性映射，将标准化z注入冻结Qwen3-1.7B，占位token后仅类型模板/选项；读Yes/No或A-D。单源与跨源均只重训目标映射，冻结decoder没有另一个可迁移主体。

## 统计口径

沿用M4.1：主比较clean，accuracy/Brier/NLL/MAE在每步按QA实例微平均、再平均步骤与三个固定seed；CI为10000次problem cluster bootstrap，保留每题每步的实例数权重。seed指标另存。ECE/AUROC在seed_metrics.csv为逐seed点估计，不伪称有配对CI。Q1 reader_val并列all/clean/traj；Q1 Bernoulli样本不在训练前平均。

Q1按源定义V：GPT2 πσ=1.5，LLaMA πσ=2；跨源Q1并非同一目标。LLaMA Q1：标签退化程度高，单列报告。

Q4数值边际TVD（seen模板）：reader_train=0.13、reader_val=0.198、cal=0、test=0；原配对完整复用，ID前缀非配对闭集，未重配。

C1差及C3学习曲线CI为逐对95%区间，多个N/方向/类型探索不能无修正宣称全网格显著；A四配置用同时区间。Q3主比较为clean；训练保留M2的轨迹Q3及其真实终点标签。噪声不改变Q3行为标签定义，但会改变所生成轨迹的终点及干扰项分布，不能将全部轨迹Q3数值标签称为与sigma无关。

同题Q3一致性比较预测数值字符串，按底层源模型干净终点正误分组，平均clean t=0..6和三个seed；不会因各源选项位置不同把位置一致误当答案一致。两源Q3候选集可能不同，一致性是各自候选集条件下的诊断，不能单独归因于表示差异。

## RQ4边界

每题每qtype固定一个clean实例；cal_A只拟合temperature/domain classifier，cal_B标签只定阈值。跨源比较C1 source-specific读出函数形成的score偏移，包含表示与函数差异；不能据此分离来源。域分类器为预测概率/步数/熵的summary covariates，不是已知真实密度比。

Mondrian source-only遇未校准源/数据集输出全集；另列使用目标cal_B的Mondrian，额外标签可用性必须区分。跨数据集目标cal来自既有unseen_train，仅作M6校准，不训练reader；LLaMA unseen已补：完整1993题；源数据和审计见data/M6_v1/llama_unseen/completion.json。既有unseen划分/重叠限制沿用M4证据。跨步t的支持集分离，缺少真实重要性加权所需的重叠，还可能有条件标签变化；截断估计权重只作经验诊断，全集带来的覆盖恢复须与集合大小一起解读。

## 执行缺口

LLaMA unseen已补：完整1993题；源数据和审计见data/M6_v1/llama_unseen/completion.json。

[
  {
    "name": "C2",
    "reason": "Cost-only estimate 7.963131282645044 GPUh plus mandatory reserve 6.2592591300268765 exceeds remaining CD budget"
  },
  {
    "name": "C4_N300",
    "reason": "Cost-only estimate 1.840911378091902 GPUh plus mandatory reserve 12.827073880530577 exceeds remaining CD budget"
  },
  {
    "name": "C4_N1000",
    "reason": "Cost-only estimate 2.7229366156049086 GPUh plus mandatory reserve 12.827073880530577 exceeds remaining CD budget"
  },
  {
    "name": "C4_N3000",
    "reason": "Cost-only estimate 5.24300872278493 GPUh plus mandatory reserve 12.827073880530577 exceeds remaining CD budget"
  },
  {
    "name": "C4_Nfull",
    "reason": "Cost-only estimate 171.46454191594984 GPUh plus mandatory reserve 12.827073880530577 exceeds remaining CD budget"
  }
]

可选Huginn未执行：启动无整张空闲≥24GB GPU；接续时虽出现空闲卡，历史计时外推完整复现约26.0GPUh已超过B剩余2.10GPUh，尚未含数据生成；估计不确定性高，未追加GPU实验。证据data/M6_v1/huginn_admission.json。

## 图与证据

![test learning curves](../data/M6_v1/analysis/learning_curves_test.png)

![reader_val learning curves](../data/M6_v1/analysis/learning_curves_reader_val.png)

所有完整数字：data/M6_v1/analysis/{metrics,seed_metrics,single_source_paired,multisource_paired,transfer_paired,same_problem_summary,coverage_recovery,costs}.csv；逐题Parquet；shift/{summary,seed_summary}.csv；RQ1报告reports/M6_rq1_closure.md；运行目录见evaluation_index.json与pipeline.json。