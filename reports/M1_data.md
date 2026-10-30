# M1：CODI-GPT2 单源数据生成（进行中）

本里程碑采用用户 2026-10-30 给出的新任务书，替代总规划原 M1 并提前完成 M2 的源模型调用部分。M0.1 已审查通过；CODI-LLaMA 现状获接受、噪声延至 M6 前；Huginn 暂停。没有训练 reader 或 probe。

## 当前交付状态

已实现候选划分、gold 方程式中间量解析、共享 `clr/policy.py::rollout`、早退、fp16 safetensors/Parquet 存储和 loader；200题 smoke 正在进行（第一版因工作量超过15分钟失败，按正式split分工修正后重跑）。v3 仅有候选池，**尚未最终冻结，也未生成全量数据**。候选生成 run：`20261030T011913127306Z_cb2983bfb3ec_109195`。smoke：`20261030T012126501342Z_c7c7a651e47e_6eef7a`。批处理关卡将在 smoke 成功后运行；未通过时不会启动全量。

| 候选 split | 题数 | gold CoT 解析失败（保留题目） |
|---|---:|---:|
| reader_train |20000|40|
| reader_val |2000|3|
| val |300|3|
| cal |500|9|
| test |1005|12|

reader_train/reader_val 属于源模型 GSM8k-Aug 训练分布，不称为源模型未见题。从按 M0.1 清洁 heldout 边界核验的训练ID登记中，再对题目文本归一化去重：91条重复，发现1条答案冲突，固定保留原文件首条并保留冲突证据。随后对problem ID排序、seed11 shuffle抽样。此处题数仍待smoke预算确认，若超60GPU小时或8GiB只缩reader_train。

val300 保留原顺序、ID和cohort上下文，来自官方 Internalize_CoT_Step_by_Step 固定commit的500题valid.txt（M0.1保守13-gram去重后剩493题，分val300/cal193）。v3将cal193与clean test1312合并，以seed11对ID排序后的池shuffle为500/1005；没有任何reader结果参与重分配。gold CoT来自相同commit的valid.txt/test.txt，URL和SHA在候选生成run/resources.json。原v2不修改。全层面板在预算确认后按固定hash选择至多800个reader_train，其余来自reader_val，合计1000题，以支持后续层探针的训练/验证分离。候选准备中的reader_val-only占位面板不会用于最终冻结；选择不依赖模型结果。

## 状态和策略语义

设固定cohort宽度为w：z0位于BOT位置w；zt（1..6）位于第t个连续输入位置w+t，保存该位置最后一层输出。共有7个观测状态；层0是embedding输出，层12是最终归一化层。

一次动作使用 `prj(z_t + sigma*||z_t||*epsilon/sqrt(d))`，其前向输出是z_(t+1)。故从zt出发，from_step=t 对应扰动之后t+1..T的转换。trajectory保存观测状态和实际加噪反馈两种张量；t0起点自身仍是干净观测，不把加噪输入与后继输出混为一谈。最终zT不再经过潜在步扰动。

共享策略接收包含batch1精确KV和前缀状态的PrefixStates。同题K路复制此KV，无额外padding；使用原协议的左padding、隐式绝对position、prefill mask、后续无mask。manifest记录cohort_id、left_padding、pad_to和protocol hash。M5必须调用同一rollout函数并保留同一上下文。

标签生成使用生产fp32状态；磁盘特征保存fp16，另设相对L2往返误差阈值0.001并报告最大绝对误差/RMSE。批处理验收区分原始fp32状态续算和fp16已保存状态续算，在相同原始KV上量化答案差异与z_(t+1)误差，不能用容差掩盖答案不一致。

## 预注册验收与成本

200题smoke：固定前100个候选reader_train和前100个val；每题clean、7个早退、K8轨迹，以及两策略各K16在一个均衡轮换t上。它用于代表性吞吐估计；全量MC仍做所有6个t，不把smoke当作完整MC。smoke设900秒硬截止。前三题三个seed的共享策略与M0.1 from_step3比对；全部200题早退T与正常推理比较。

smoke后val300验收：早退T必须完全一致；sigma0的K16续算对每题每个t记录答案一致性和最大逐元素/相对L2误差，列出全部不一致。sigma1.5/from_step3，K16、seed11/22/33，与M0.1同上下文同seed的batch1面板比较。M0.1三个seed准确率范围为0.208333pp，非退化比例范围为1.666667pp；预注册为各seed绝对差均不超过相应范围。未通过关卡不使用该批处理路径生成全量。

测试已通过：`PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_m1_policy.py tests/test_m1_data.py`，5 passed（tiny模型测试，不替代真实300题关卡）。全量预算、标签直方图、K64方差和最终规模均待实际运行，不填写估计为实测。

## 偏离与限制

- 候选准备首次因输出目录未创建失败，修复目录创建后重新运行；失败run保留。
- GPU4实际CUDA初始化OOM（运行前空闲显存发生变化），0次源前向；显式改GPU2，物理GPU2临时禁用窗口已在10月29日18:06结束。数值配置仍fp32/batch1，原协议文件不改。
- z_t依赖cohort组成/padding宽度；这是用户接受的已知限制，不尝试消除。
- fp16特征不是无损fp32状态，更不是KV。任何续算差异都需报告；全量是否可采用batch16尚未确认。
- 随机生成按seed11/22/33为轨迹或rollout分配独立确定性种子；固定K8/K16不擅自扩成K24/K48。三seed重复验收单独进行；最终标签统计按题bootstrap，不将相关轨迹当作独立题。

## 2026-10-30 实测更新：首轮smoke超时，修正版已启动

首轮GPU2 smoke完成161/200题后触发900秒截止，标记failed，23655次源前向、0.2500001 GPU小时；未通过、未进入val300验收。失败run `20261030T012126501342Z_c7c7a651e47e_6eef7a` 完整保留。原因是smoke对所有200题同时做traj和MC，超出正式split分工，并非源模型加载或指标失败。

显式修正为 `scripts/m1_smoke_v2.py`：仍为同样200题、全部7个早退；仅100道reader_train生成K8轨迹，仅100道val生成两种K16 MC（每题一个均衡轮换t，完整MC仍覆盖六个t）。K、sigma、温度、精度、batch、输入上下文均未减少或更换。预算公式使用各阶段实际题数100/200，不能继续按200除以轨迹或MC耗时。新run `20261030T013639323603Z_13ad784ef5d6_518bf2`，device=cuda:2，仍设900秒截止。

回归run `20261030T012839060341Z_6e2e124eba99_eb1749`：28 passed、1 skipped。后续量化阈值加强到每个向量的相对L2最大值，相关5项测试再次通过。gold解析列表是显式方程RHS集合（包含最终计算结果）；没有推断隐状态中是否已算出这些数值。

后台流水线仅在200题smoke、val300关卡和预算全部通过时启动全量；所有失败尝试的GPU小时计入60小时预算。若失败，生成 `reports/M1_results.md` 证据并停止，不擅自换精度/源模型/划分。

## 最终冻结前的上下文核验

val300已逐题核对M0.1噪声面板的padding宽度。对于cal/test池中原test1312题，最终v3会从M0.1逐题协议审计恢复原1319题队列中的cohort和padding，不能因删除7题或重划分而重新组cohort。候选准备时产生的clean-test重新组队上下文未用于任何cal/test推理；修正由独立context_audit记录，原候选文件不覆盖。其余来自cal193的问题此前没有完整生产推理上下文，使用候选阶段冻结的cal193队列上下文。
