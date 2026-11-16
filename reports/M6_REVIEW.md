## (a) 状态一句话

M6已完成已执行实验与统计，待审查；低优先级预算省略如(d)，可选Huginn未执行。

## (b) 关卡清单

通过：val代理smoke≤5%/15min、行为标签/选项/划分审计、三seed、训练hash锁后最终reader test、旧冻结哈希复核。LLaMA val300/K16/seed11：σ=2，下降1.99pp，非退化14.8%。预算 A/B/CD=0.413/27.9/19.3GPUh，总47.6/60。未通过（未执行）：Huginn；C2; C4_N300; C4_N1000; C4_N3000; C4_Nfull。

## (c) 核心主表（最多3张）

表1：各源test/reader_val，Δ为lite−B3的准确率pp及题bootstrap95%CI。

|源|集|Q3 lite/B3 %|Δ pp [95%CI]|Q4 lite/B3 %|Δ pp [95%CI]|
|---|---|---|---|---|---|
|gpt2|test|76.1/58.6|17.5 [16.1,19]|76.2/68.3|7.93 [6.35,9.52]|
|gpt2|reader_val|86.3/65.6|20.7 [19.7,21.7]|85.1/71.4|13.7 [12.7,14.7]|
|llama|test|72.1/58.9|13.2 [11.7,14.8]|74.2/62.7|11.5 [9.81,13.1]|
|llama|reader_val|80.1/65.9|14.2 [12.7,15.8]|81.1/62|19.2 [17.7,20.7]|

表2：小N迁移相对scratch，test；括号为胜出配对数/已执行数（Q3/Q4，N100/300，区间下界>0）。

|方向|vs B3|vs lite|vs B5b|
|---|---|---|---|
|gpt2→llama|4/4|4/4|2/2|
|llama→gpt2|4/4|4/4|2/2|

表3：test覆盖率偏差pp及平均集合大小范围，α=.1、Q3/Q4、两个reader/方向合并描述；完整CI在证据CSV。

|偏移|standard偏差|weighted偏差|standard size|weighted size|
|---|---|---|---|---|
|source|-4.78…3.38|-1.32…3.7|1.29…2.27|1.32…2.34|
|dataset|-3.78…4.47|-2.77…4.1|1.26…2.11|1.29…2.11|
|step|-2.7…4|9.97…10|1.23…2.29|2…3.99|

CODI-LLaMA：Q3优势复现: Δacc=13.2pp [11.7, 14.8]；Q4优势复现: Δacc=11.5pp [9.81, 13.1]。配对区间完全大于0才支持优势复现；否则记为未建立。

小N按方向/类型/预算逐项判断：20/20个已执行配对区间下界>0；对照含b3,b5b,lite。B5b小N对照不全，仅可判断已执行N/方向/类型；不能推广到全部小N。

α=.1、Q3/Q4中6/32个设置标准覆盖率低于目标超过2pp（描述性界限），涉及dataset_gpt2_gsm_to_svamp、dataset_llama_gsm_to_svamp、source_llama_to_gpt2、step_gpt2_early_to_late；weighted对目标绝对偏差的中位改善=-0.0774pp。覆盖率变化的配对95%CI见analysis/coverage_recovery.csv；逐设置覆盖率CI及集合大小见shift/summary.csv。估计权重不构成偏移下的覆盖保证。

较小sigma仍未建立超过greedy的正收益；保留RQ1负面/分析结论。

Q1按源定义V：GPT2 πσ=1.5，LLaMA πσ=2；跨源Q1并非同一目标。LLaMA Q1：标签退化程度高，单列报告。

## (d) 本里程碑新增的偏离与阻塞

未通过：八点无同时达两噪声门槛者；按用户预登记“门槛未达，降级使用”，并非历史关卡通过。

Huginn未准入：启动资源不足，后续复现成本估计26.0GPUh>B余量2.10GPUh，见huginn_admission.json；LLaMA unseen已补：完整1993题；源数据和审计见data/M6_v1/llama_unseen/completion.json。B按授权成本缩小train至6817题；Q4数值边际TVD（seen模板）：reader_train=0.13、reader_val=0.198、cal=0、test=0；原配对完整复用，ID前缀非配对闭集，未重配。source smoke目录/Cache API失败、RQ1递归包装失败保留并计成本，修复后独立重启、校验续用。A启动YAML含未用旧网格，实际协议固定在启动源码，未完全YAML化；事后审计rq1/configuration_audit.json不冒充启动记录。控制器交接修复D→C2/C4优先级，GPU训练未中断。CPU绘图库路径及独立审计PYTHONPATH已修复；首条失败log同名覆盖，同错误第二次log保留。预算省略见(b)及training_lock.json；完整原因在详细报告。 Mondrian缺目标stratum时全集；目标cal版本另列。weighted只估计summary密度比，跨步t支持不重叠，无无条件保证；跨源C1 score偏移不能分离表示与函数差异。

## (e) 需要审查者决定的问题

是否接受A的已记录配置偏离、本里程碑执行范围及四项结论。

## (f) 证据路径

详细：reports/M6_cross_source.md、reports/M6_rq1_closure.md、reports/M6_noise_selection_v2.md。统计/CI：data/M6_v1/analysis/*.csv、shift/summary.csv。配置/运行：configs/M6_pipeline_noise_v2.yaml、data/M6_v1/pipeline_noise_v2.json、evaluation_index.json、training_lock.json；源码/数据/结果冻结data/M6_v1/freeze.json；新增启动错误及限制docs/BLOCKERS.md。
