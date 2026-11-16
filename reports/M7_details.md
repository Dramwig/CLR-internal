# M7 潜通信与语义审计详细证据

主任务与用户commitment修订：docs/M7_TASK.md。审查文件为reports/M7_REVIEW.md。

|方法|reader_val 检测率% [CI] / FPR% / AUROC|test 检测率% [CI] / FPR% / AUROC|
|---|---|---|
|CLR|1.72 [0,4.02] / 1.72 / 0.502|0.575 [0,1.44] / 0.575 / 0.5|
|B3|1.72 [0,4.02] / 1.72 / 0.5|1.44 [0.287,2.87] / 1.44 / 0.5|
|linear|8.05 [4.6,12.1] / 7.47 / 0.507|6.32 [4.02,8.62] / 5.46 / 0.506|
|B0|2.3 [0.575,4.6] / 2.3 / 0.5|1.72 [0.575,3.16] / 1.72 / 0.5|
|norm|3.45 [0,8.62] / 3.45 / 0.499|3.45 [0.862,6.9] / 4.31 / 0.5|
|text|0 [0,0] / 0 / 0.589|0 [0,0] / 0 / 0.552|

|问答|reader_val CLR / B3 %|test CLR / B3 %|test held-out CLR / B3 %|
|---|---|---|---|
|A1|50 / 50|50 / 50|50 / 50|
|A2|50 / 50|50 / 49.9|50 / 49.9|
|A3|50 / 50.1|50 / 50|50 / 50|
|A4|23.8 / 26.7|26.6 / 26.3|28.2 / 26.7|

|自然承诺|reader_val n / 均值 [10%,90%]|test n / 均值 [10%,90%]|
|---|---|---|
|direct 正确|18 / 0.512 [0.512,0.513]|37 / 0.512 [0.512,0.513]|
|direct 错误|40 / 0.512 [0.512,0.513]|79 / 0.512 [0.511,0.513]|
|cot512 正确|25 / 0.512 [0.511,0.513]|59 / 0.512 [0.511,0.513]|
|cot512 错误|33 / 0.512 [0.512,0.513]|57 / 0.512 [0.512,0.513]|


统计：按group整体重采样10000次，取2.5%/97.5%分位作95%CI；先在每个seed/组内平均问题，再跨3seed平均。CI条件于这3个训练模型及冻结校准集，未包含重新训练与重新抽取校准集的不确定性。AUROC按同一组的clean/替换消息配对bootstrap；确定性文本/范数只拟合一次。校准用cal_A按问题类型温度缩放（独立组、seen模板、clean/CF预先随机混合）；cal_B每组一clean消息/问答定阈值；held-out两模板未用于训练、选模型或校准。QA集合控制只针对与cal_B抽样规则可交换的clean任务实例；CF/mismatch/held-out仅报告经验覆盖率。

一致性目标只由可见承诺与公开T、z约束推出；k个问题在4种seen改写中按固定seed随机抽取（有放回），并不独立，不声称指数检测界。random k使用同一16题序列的前k题，所有方法配对一致；2种held-out改写另做QA迁移，不用于主检测阈值。

审计z：Qwen3-4B同权重顺序sender/receiver，raw sender读x_pub+x_priv，receiver只读x_pub+全层KV；K已post-RoPE，prefix槽含mask/注入层/门控。审计仅保存末层35的每个有效token，拼K各头再V各头，fp16。适配器按token投影并以6个query读取，再复用冻结M4.1 Lite实现的4层256 reader；原M4.1源码未改。

训练：CLR/B0/B3/线性各seed11/22/33，完整曝光上限20 epoch、min4、patience3，reader_val四类平均NLL选checkpoint；损失按类型频率加权。B3均值池化KV与问题/选项交互MLP；线性为条件于问题/选项的线性状态分数。B0固定6个零潜向量，去掉长度信息。范数只读保存层K/V log-RMS；文本只读同格式承诺和receiver选择输出，不看x_pub/x_priv。
B0正式训练前移除完全未使用的长KV归一化/搬运；相同零前缀送入相同主体。原/新A1/A3/A4逐位输出相等、max误差0（reports/M7_evidence/b0_equivalence.log）；科学设置不变，含错配位置、统计边界及生成器独立哈希锚测试共16项通过（tests_v12.log）。
GPU2/GPU7从独立固定队列改为共享待办队列（configs/M7/balanced_training.yaml）；交接时已运行子作业继续正常结束。旧CPU调度器随后标记interrupted/superseded，成本0；这不是训练失败或重启。实际设备、启动代码、完整科学配置仍逐run保存。
最终控制器configs/M7/finish_v2.yaml等待全部训练及错配v2完成才校准锁；锁后reader_val评估与test生成在GPU7/GPU2并行，二者完成后才评估test。每次test访问前仍逐项核验最终锁。状态写入使用M7专属文件锁，避免并行阶段争用临时文件。

programmatic承诺三模板共享，类别间复用同一原episode文本；CF保持公开输入/无关日志，只交换目标记录中的团队字段来干预e。mismatch来自同split下一独立组；corrupted三控制分别截断全prefix前半、将保存层全置零、保存层相对L2=.1高斯噪声。receiver实际消费每种控制，审计读取其真实扰动KV（bf16再转fp16），不是用加噪输入代替源输出。噪声固定seed，非训练攻击。
原mismatch.jsonl的receiver对供体单条重编码，而审计保存供体与CF共同左padding的KV；开发580组中118个原供体有非零左pad。已保留旧版并创建mismatch_v2.jsonl：重新构造同一原/CF批次，选第一条时保留物理槽，逐张量/metadata精确比对后供receiver使用。修正仅在test前，主训练不含mismatch；正式文本拟合、检测与mismatch问答全部使用v2。全量修正证据data/M7_v1/mismatch_v2_complete.json，概率/选择变化见analysis/mismatch_v2_impact.json；最终test直接生成v2。噪声半径.1为目标值，bf16/fp16转换有舍入误差，不声称严格浮点上界。

QA主表为clean/CF等权；完整CSV另含mismatch，标签取自供体，A1以split内无同值的边际匹配排列生成假事实。破坏性corrupted不存在可靠的“信息仍包含”标签，因此只作为非语义检测控制。A1对真/反事实两消息使用互换的正负事实，取值边际精确对齐。当前A1只审计唯一四值语义变量e，不代表任意事实开放域读取。
mismatch不强制改变e或route，其中一部分仍支持原可见结论；QA按实际供体标记，检测类别按“替换了来源episode”定义，不能把全部mismatch都称为语义矛盾。供体在split内循环复用，其组bootstrap区间按固定供体面板解释，不作为独立重抽供体的置信保证。
manifest的commitments[1]记录CF来源episode自身的结论元信息；主counterfactual类实际公开文本始终是commitments[0]，与clean/mismatch/corrupted完全相同。

程序化承诺定义来自solve_v2_text原文求解，不向Qwen/reader传入label/e/meta；约束目标可从公开T,z和承诺计算。自然模型承诺另存source/<split>/natural_commitment.jsonl，主训练和阈值入口从未读取该文件；只在锁定后做附加分析，所有val/test基础episode的direct/cot512原始生成、预测和正误齐全。
模型自然承诺取冻结S2接口的direct或cot512最终选项打分argmax，经同一formatter转为短承诺；cot512完整token/原文另存，不把自由生成原文中的任意句子当作已核验的最终选项。

数据：新seed20261106直接import冻结m1-v2-r2生成器；保持原arithmetic_wide/priv_len12,48,128/四类事实/步骤0,2,4。每个基础组选择alpha1及按组索引轮换d_pub，不跨split，CF共享组ID。目标规模1层约179GiB，按15GiB规则缩为{'reader_train': 464, 'reader_val': 58, 'cal_A': 29, 'cal_B': 29, 'test': 116}。data/M7_v1当前10.4GiB，含真/CF实际KV与统计；另有run内checkpoint合计0.571GiB，不计入KV存储。未通过原8k/1k/1k/2k规模目标。

历史兼容核验不是借用Idea A结论：在.venv-comm重新加载4B，冻结历史8个episode/seeded messages/adapter，重建每层消息与历史receiver logits逐项核对；原始KV另与Idea A的冻结提取实现比较，误差0。历史raw中继张量未保存，所以该历史比对要求仍缺证据，不把新复放冒充历史输出。Idea A的学习式消息和简化KVTransfer没有冒充官方C2C。

真实KV的4个训练组过拟合诊断：CLR四类100%，B3四类最低95.3%；run=/home/me/PaperField/CalReader/runs/20261106T092709727360Z_071fe0592665_737311，完整指标见analysis/learnability_evidence.json。该诊断只验证路径具有学习能力，没有替换正式超参或选seed。正式source的clean/CF目标准确率与选择变化见analysis/source_diagnostics.json；干预只改私有语义变量，接收者公开输入不变。开发前上游冻结23800文件校验无差异，证据data/M7_v1/upstream_pretest_verification.json；收尾另重新核验。

私有干预的目标概率对比定义为[P_cf(y_cf)−P_clean(y_cf)]−[P_cf(y_clean)−P_clean(y_clean)]，按组bootstrap；在最终test前登记该附加因果统计。它隔离私有事实改变的影响，区别于同时移除/错配全部公开与私有KV的固定val80对照；val80限定alpha1、d_pub0、priv_len12，不外推其准确率到长日志。

与GSM8K的对应：M4.1冻结test上Q3 CLR−B3为+17.5pp [16.1,19.0]、Q4为+7.93pp [6.31,9.56]。M7的A4对应Q3、A1对应Q4，比较必须以analysis/qa_CLR_minus_B3.csv的配对组区间为准，不能把小样本中的方向当作重现。

成本7.73 GPUh，包含所有M7记录的源/训练/评估smoke、失败和natural诊断；逐进程按单设备驻留墙钟保守累加，GPU共享/进程重叠也不抵扣，不按利用率折算。controller/纯CPU统计为0GPUh；启动import失败无CUDA模型调用且未创建run，保留原始错误日志。source_forward_calls=59274含batched调用，forward_examples与prefill令牌另见run；不得把调用数当样本数。

证据：qa_per_seed.csv（按类别）、qa_combined_per_seed.csv（clean+CF，含二分类AUROC/ECE；单类别A3只有一个标签故AUROC为NA）、qa_combined_group_bootstrap.csv（含AUROC/ECE的非线性组重采样CI）、qa_core.csv、qa_group_bootstrap.csv、qa_CLR_minus_B3.csv；detection_group_bootstrap.csv、counterfactual_CLR_paired_differences.csv；natural_commitment_distribution.csv；causal_bootstrap.json、costs.csv。真实run索引trained/、evaluated/、final_lock.json；源码/环境在run快照。可复现入口configs/M7/下所有YAML和scripts/m7_*.py。

![Counterfactual detection versus question count](../data/M7_v1/analysis/counterfactual_detection.png)
图中阈值均来自cal_B clean、名义α=.05；实测FPR见主表，阴影为条件于固定模型和校准的95%组bootstrap区间。
