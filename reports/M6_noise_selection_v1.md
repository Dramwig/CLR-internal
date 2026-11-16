# M6 LLaMA 噪声选择：完整val300证据（v1）

计算完成，历史选择关卡未通过。seed11、K16、相对观测步t=0..5等权；不含test。原始run保留failed状态，因为selected=null触发硬关卡，不把它误写成只算了部分数据。

|sigma|accuracy %|drop pp [95%CI]|nondegenerate % [95%CI]|drop<5pp|历史nondeg≥30%|
|---|---|---|---|---|---|
|0.5|53.9|0.455 [0.0313,0.993]|3.06 [1.56,4.78]|通过|未通过|
|1|53.6|0.75 [0.0832,1.53]|6.28 [4.17,8.56]|通过|未通过|
|1.5|53.1|1.2 [0.246,2.29]|10.8 [8.17,13.6]|通过|未通过|
|2|52.3|1.99 [0.708,3.42]|14.8 [11.6,18.1]|通过|未通过|

clean accuracy=54.3%。四个sigma均满足下降<5pp；如本次仅以该约束定义可行集，sigma1.5满足用户优先规则。若沿用M0.1历史nondeg≥30%硬门槛，全部不合格，不能自动选择1.5或改变t聚合。当前实际执行为后者，并已在数据生成前停止。

当前请求澄清本次B1括号定义是否保留历史30%门槛。未启动正式LLaMA数据生成/训练/test reader预测；未偷偷扩sigma网格。低Q1标签非退化不直接证明Q3/Q4读取不成立。

CI为10000次按题cluster bootstrap，保留每题六个t和固定K16，条件于一个已授权seed；不是三seed区间。门槛原先按点估计判定，事后CI不改变规则。逐t描述在by_t.csv，不据此重新挑t。

运行：runs/20261106T064505704384Z_46346b5277a6_2acd02；原始noise_000..299.parquet、selection.json、config/启动源码/metrics；衍生data/M6_v1/noise/{summary,by_t,costs_at_scan_end}.csv；A独立完成见reports/M6_rq1_closure.md。
