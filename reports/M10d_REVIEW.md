# M10d REVIEW

四格已补齐；B6复用冻结结果，LLaMA B4按M3协议新训三seed。

|源/模型|Q1 NLL|Q2 AUROC|来源/方式|
|---|---:|---:|---|
|GPT-2 B6|0.574|0.853|data/M10_v1/paper_inputs/main.csv；查到，无重跑|
|LLaMA B4|0.551|0.806|data/M10d_v1/B4_summary.csv；新训练/最终推断|

test、clean、已见模板、各t等权、seed11/22/33均值。精确值/CI：data/M10d_v1/four_cells.json；B4原始run由data/M10d_v1/final.json索引，权重锁data/M10d_v1/training_lock.json。

新增成本：0.0289 GPUh≤2（RTX4090 GPU5，含smoke、三seed两LR训练、cal_A独立温度及最终评估）；B6查表0 GPUh；源模型前向0。旧冻结模型/结果/阈值未改。

论文四格/加粗及number_audit已更新；T1/Times编译通过，仍8页，主表视觉核查通过；正文措辞不变。证据：data/M10d_v1/manuscript_validation.json。
