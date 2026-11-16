# M5a latent search feasibility

val300全量完成，待审查。 主要S3 oracle相对同题同seed、账单预算内S1：32.8 [27.8, 38] pp。其95%区间下界为正；这是该网格/策略的特权经验参考，非数学上界。MC成本单独列出的上限诊断增益为33.4 [28.6, 38.6] pp，不能当作已计费收益。B4/B3均未显示取得正收益；有好的价值函数时存在可用空间，现有探针尚未把它转为收益。

## Protocol and statistics

任务与预先固定的比较规则：docs/M5a_TASK.md；不读取test。源fp32/temperature0，固定原cohort左padding与完整KV。先扰动、源前向，再让probe读真实观测。B3/B4只读状态/模板0/冻结dummy选项，不接收原题或gold；oracle例外读取gold。N4/8/16、beam2/4、全步与t≤3。各N独立π轨迹池；同N的S1/S2复用候选，所有设置逐题CSV保留，不按accuracy挑配置。
300题，seed [11, 22, 33]；先在题内平均seed，再按题配对bootstrap 10000次（bootstrap seed 20261105），区间单位与均值一致，accuracy为百分比、差为pp。S0确定性只计一次。置信区间为逐比较区间，未多重比较校正；val300可行性观察不外推test。

## Primary actual-compute matches

|配置|accuracy % [95% CI]|均值前向|同预算S1 %|差 pp [95% CI]|
|---|---:|---:|---:|---:|
|S0/none/N1/b1/h0|42.3 [36.7, 48]|13.2|—|—|
|S3/oracle/N16/b4/h6|70.6 [65.6, 75.6]|4.56e+04|37.8|32.8 [27.8, 38]|
|S3/B4/N16/b4/h6|34.9 [30.1, 39.9]|343|37.1|-2.22 [-5.22, 0.889]|
|S3/B3/N16/b4/h6|36.4 [31.4, 41.7]|343|37.1|-0.667 [-3.78, 2.22]|
|S3/oracle（MC另计）|70.6 [65.6, 75.6]|343|37.1|33.4 [28.6, 38.6]|

每项搜索与同题同seed独立长投票池的预算内最长前缀配对。未用完余额均值3.95942、最大15.0068源前向；严格离散预算上限相同，不能保证逐lane完全相等。投票池覆盖所有搜索成本；该基线覆盖不是在单一全局均值预算取相同条数。

## Early search and terminal selection

|配置|accuracy % [95% CI]|均值前向|同预算S1 %|差 pp [95% CI]|
|---|---:|---:|---:|---:|
|S3/B4/N16/b4/h4|35.8 [30.9, 40.9]|223|37.7|-1.89 [-4.56, 0.778]|
|S3/B3/N16/b4/h4|36 [31.2, 41.1]|223|37.7|-1.67 [-4.11, 0.667]|
|S2/oracle/N16/b1/h6|52.2 [46.9, 57.7]|1.82e+03|38.3|13.9 [10.7, 17.2]|
|S2/B4/N16/b1/h6|32.3 [27.7, 37.1]|198|37|-4.67 [-7.44, -2.11]|
|S2/B3/N16/b1/h6|33.6 [28.8, 38.6]|198|37|-3.44 [-6, -1.11]|

早期beam保留分支之后继续π但不再扩展/中途筛选，终点再选一条。S1/S2两horizon没有中途搜索，仅作为独立随机面板。全部设置：data/M5a_v13/analysis/summary.csv。

## Fraction at the same allowed oracle budget

|成本口径/价值函数|相同允许预算下差 pp [95% CI]|取得oracle增益 % [95% CI]|预算使用率 %|
|---|---:|---:|---:|
|全计费/B4|-2.89 [-5.89, 0.222]|-8.81 [-19.6, 0.66]|0.753|
|全计费/B3|-1.33 [-4.56, 1.78]|-4.07 [-14.9, 5.02]|0.753|
|MC另计/B4|-2.22 [-5.22, 0.889]|-6.64 [-16.9, 2.58]|99.9|
|MC另计/B3|-0.667 [-3.78, 2.22]|-1.99 [-12.1, 6.54]|100|

三者固定N16/b4全步；每题/seed共同预算取三者实际成本最大值，MC另计仅扣oracle内部rollout，probe开销仍计入。这样小数probe开销和答案长度差不会强迫换成小beam。分母为oracle对共同预算S1的增益，分子为probe对同一个S1的增益；该参考可与主表各自实耗预算S1有少量差异，精确分母在fractions.json。预算规则不依正确率，固定配置不换N/b。比例不截断到[0,100]；分母95%CI含零则不解释比例。全计费口径probe无法利用oracle多出的全部预算，使用率明确欠用，不能冒称全部算力已等量花完。

## Accuracy–compute curves

![Actual billed cost](/home/me/PaperField/CalReader/data/M5a_v13/analysis/actual_compute.png)

原始配置点使用每题实际成本的均值作横轴；黑色星号是oracle逐题账单预算内的独立S1，不能将横轴均值误读为每题花相同固定前向数。oracle误差条为按题95%CI；点的数值及配对差在CSV。预设共同预算点最高32768，oracle beam若超出则不在下图硬凑点，由本图实际成本点与配对表呈现。

![Billed cost](/home/me/PaperField/CalReader/data/M5a_v13/analysis/accuracy_compute.png)

横轴含prefill、潜步骤、答案解码、probe FLOPs等价和全部oracle K16内部源前向。计算单位是一条lane一次源API前向；prefill按一次源调用计费，单列prefill token数。probe折算使用源线性FLOPs+固定pad_to+4注意力近似，未称精确延迟比；B3/B4非常小，完整参数与公式在search run/compute_profile.json。物理batch调用/实际lane在metrics中记录；解码压缩已完成lane，所有执行lane均收费。汇总各配置理论独立成本不等于工程执行成本（候选池复用、额外匹配投票、门槛均影响后者）。
共同允许预算点固定128..32768，S2/S3每题/seed选预算内最大(N*b,N,b)，无accuracy参与选择；horizon分面。曲线只有全部题/seed支持时才画点，缺覆盖标在common_budget_curves.csv，不用部分题替代整集。预算内策略可欠花成本，使用率已落盘。常规S1长池曲线不外推未生成轨迹；N4/8/16主面板仍在summary.csv。

![Unbilled diagnostic](/home/me/PaperField/CalReader/data/M5a_v13/analysis/oracle_unbilled_diagnostic.png)

该图仅诊断去掉oracle成本后的几何趋势；oracle_mc_separate.json给出相应S1配对数值，不用于回答同账单计算量收益。

## Costs, gates, deviations

GPU为NVIDIA GeForce RTX4090；全量/当前search run墙钟GPUh 5.42484336，源物理调用1077667，lane 91127113，prefill token 39552。GPU共享，GPUh按作业占用墙钟保守累计，不是独占活跃kernel时间。含加载、smoke、失败和中断保守预留的总charge 5.80961141 GPUh≤10；旧exit143保守预留0.080h，缺失最终源计数不当作零。逐run台账data/M5a_v13/analysis/cost_ledger.csv，旧版本lane门槛计数缺口在预检保留。CPU报告GPUh=0。
v13五题smoke101.869秒通过；测试：9 passed in 5.16s。原全量gate_passed=False，补充恢复run=runs/20261105T113203872032Z_8ac13d9a6bdd_79f616，组合证据逐题协议/标签/投票覆盖均通过；44400条结果，50配置键；原输入hash与启动记录逐项核对一致。test_access=false。用户继续授权后保留10GPUh，仅S3 oracle限N16/b4全步；B4/B3与S1/S2原网格全保留。K16有MC误差/自适应选择偏差，终点oracle读取gold后成为特权0/1选择。LoRA合并非逐位一致；批量iid改变seed寻址，保持同扰动分布。早期Q1特征错误已修复，旧smoke不作科研结果；OOM/资源失败/exit143及缺失成本见预检报告。原搜索gate=false：仅一S1池4096条仍欠556前向，已按原随机地址另存补长到4224条，原结果未覆盖；恢复逐项核验后通过。

## Reproduction and evidence

搜索配置configs/M5a_full_v13.yaml；本报告配置configs/M5a_report_v13.yaml；搜索run runs/20261105T060403949446Z_3a4adb8b01c7_fd463e；报告run /home/me/PaperField/CalReader/runs/20261105T113419960040Z_aef520d9b61a_52a5fe。
全部配置与逐题匹配、共同预算及选择日志、比例、图和成本：data/M5a_v13/analysis/。源码/环境/git/权重hash：search run/source_snapshot、input_sha256.json、environment.json、git.json；报告自身另存分析源码hash。旧预检reports/M5a_preflight.md及data/M5a_preflight_v0不覆盖。
