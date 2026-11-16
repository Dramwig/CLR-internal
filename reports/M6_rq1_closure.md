# M6 RQ1 收尾

val300、seed11/22/33；sigma0候选纳入S3每个父节点的8个子节点，b4，全6步；冻结sigma1.5下训练的CLR-lite/B3。相同噪声种子按父索引配对。pass@8为独立候选池（1clean+7pi轨迹）的特权上限，不等于beam终点候选数。

|sigma|value|accuracy|Δgreedy pp (95% simultaneous CI)|pass@8|
|---|---|---|---|---|
|0.3|B3|0.431|0.778 [-0.944, 2.5]|0.446|
|0.3|CLR-lite|0.424|0.111 [-1.23, 1.45]|0.446|
|0.75|B3|0.428|0.444 [-2.51, 3.4]|0.483|
|0.75|CLR-lite|0.417|-0.667 [-3.22, 1.89]|0.483|

较小sigma未在预设四项比较中建立相对greedy的正收益；RQ1保留为负面/分析性结论。无显著正收益不等同证明所有小sigma搜索必定无效。

CI按题cluster bootstrap10000次，平均三个固定训练seed，四设置用max标准化同时区间；不把rollout/步骤当独立题。每seed值和普通配对区间见CSV。所有新增计算只用val；完整成本在M6成本台账。

## 既有分析图（无新增模型计算）

![between vs within](../data/M6_v1/rq1/between_within_by_t.png)

来源：data/M5b_v1/analysis/B_summary.csv。跨题AUROC与同题同父候选的rho衡量不同排序任务，不能以高跨题AUROC推断搜索有效。

![pass@N](../data/M6_v1/rq1/pass_at_N.png)

来源：冻结C_summary.csv；仅使用既有N4/8/16完整6步池，不外推更大N。

![early exit](../data/M6_v1/rq1/historical_early_exit.png)

早退图逐字节复用reports/M3_figures/unseen_early_exit.png；其原池/旧协议限制沿用reports/M3_unseen_pools.md，不将之冒充M6新增LLaMA结果。

证据：runs/20261106T071548963602Z_eb6498d61086_716529；data/M6_v1/rq1/problem_results.csv、summary.csv；configs/M6_rq1_full.yaml。
