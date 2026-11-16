# M10c REVIEW

## (a) 状态一句话
只读开销测量、冻结CSV配对汇总与T6复现完成，待审查；无训练、无新test推断、无重校准，原模型/科学结果永久冻结。

## (b) 关卡清单
- 通过：reader_val smoke 4/2000题、28.2s；正式固定64题，共享同一RTX4090 GPU4，batch1/64各三轮中位数。
- 通过：四模型×Q3/Q4×两batch共16格；计数展开前后全部argmax一致、最大logit误差0；冻结输入与Qwen资源匹配。
- 通过：cal_B α=.1原阈值、三个固定seed逐题逐step精确配对，未重新选温度/阈值。CI整problem bootstrap10000次，保留step及seed；不覆盖重训seed不确定性。
- 通过：7项相关单测；独立无torch/transformers的paper环境，48项TeX/CSV/PDF双目录逐字节一致。
- 未通过（缺证据）：B3独立训练耗时无账本；表中NA，原B0–B4全类型/三seed含校准评估共享运行0.534 GPUh。
- 未通过（未执行）：独占GPU延迟、端到端预处理延迟与TeX编译。共享GPU并发快照保留，延迟不能视为孤立硬件极限。

## (c) 核心主表
精度为冻结test三seed均值（95%CI见T6）；参数M，GFLOPs/源步比各为batch1/64。训练GPUh为三seed已完成resident训练总和，跨类型重复，不能相加。

|类型|模型|test acc %|可训/总参数M|GFLOPs/query 1/64|源步比 1/64|训练GPUh|
|---|---|---|---|---|---|---|
|Q3|B3|58.6|0.108/0.108|0.000861/0.000861|0.00296/0.00296|NA|
|Q3|B6|65.4|0.661/39.3|0.00726/0.00726|0.025/0.025|2.35|
|Q3|CLR-lite|76.1|17.2/17.2|0.254/0.311|0.874/1.07|6.03|
|Q3|B5c|76.8|18.9/1.74e+03|165/193|567/664|16.7|
|Q4|B3|68.3|0.108/0.108|0.000215/0.000215|0.000741/0.000741|NA|
|Q4|B6|72|0.661/39.3|0.00525/0.00525|0.0181/0.0181|2.35|
|Q4|CLR-lite|76.2|17.2/17.2|0.175/0.256|0.604/0.881|6.03|
|Q4|B5c|76.3|18.9/1.74e+03|76.6/121|264/416|16.7|

源一次cached潜在步约0.291 GFLOPs（含未合并LoRA、投影和词表头，无prefill）；CLR batch1成本约该源步的0.604–0.874，并非数量级更低。B6总参数含38.7M冻结选项embedding表；B3/B6可训参数为每类型，CLR/B5c共享。FLOPs为矩阵/注意力2 FLOP/MAC，不计lookup、norm、激活/softmax；B5c含冻结Qwen3完整前向。原生精度：B3 FP32，其他BF16 autocast。

|类型|模型|b1 ms/batch|b1 query/s|b64 ms/batch|b64 query/s|
|---|---|---|---|---|---|
|Q3|B3|0.581|1.72e+03|1.49|4.3e+04|
|Q3|B6|1.72|582|3.37|1.9e+04|
|Q3|CLR-lite|7.79|128|10.1|6.36e+03|
|Q3|B5c|152|6.58|802|79.8|
|Q4|B3|0.519|1.93e+03|1.63|3.93e+04|
|Q4|B6|1.83|545|5.4|1.19e+04|
|Q4|CLR-lite|7.69|130|8.87|7.22e+03|
|Q4|B5c|145|6.9|393|163|

每轮64查询×3内循环，2轮warmup；输入特征准备/分词/搬运在计时外，GPU同步包围前向。CLR对B5c的Q3/Q4 FLOPs降低648/437倍（batch64为621/472倍）；实测b1延迟低19.5/18.8倍，b64低79.7/44.3倍。

|split|类型|CLR覆盖率% [95%CI]|B5c覆盖率% [95%CI]|B5c−CLR集合size差 [95%CI]|
|---|---|---|---|---|
|reader_val|Q3|95.8 [95.3,96.2]|95.8 [95.3,96.2]|-0.0113 [-0.0178,-0.00498]|
|reader_val|Q4|94.7 [94.2,95.2]|93.6 [93,94.1]|-0.0485 [-0.0565,-0.0403]|
|test|Q3|91.6 [90.7,92.5]|92.1 [91.2,93]|-0.0169 [-0.0269,-0.00692]|
|test|Q4|89.7 [88.7,90.7]|90.2 [89.1,91.2]|-0.0194 [-0.0326,-0.00634]|

覆盖率不同，size配对差不是同覆盖率优势；绝对size及覆盖率配对差见T6 companion。精度CLR−B5c Q3 −0.659 [−1.51,0.190]pp，Q4 −0.0864 [−0.984,0.830]pp；“精度相当”为描述性，不证明等价。CLR/B6历史成本训练Q1–Q4（B6含两LR sweep），B5c仅Q3/Q4一遍，曝光/搜索量不同。

## (d) 本里程碑新增的偏离与阻塞
两次smoke因inference_mode下FLOP计数为0失败；改为与冻结评估一致的no_grad，未改权重/科学设置，失败保留计费。全部新增GPU成本0.0941h，源前向0。
B3独立账本缺失、GPU共用、计时不含预处理与未编译TeX如上。开始前已有M10b完成stdout追加、LIVE更正两项非科学哈希漂移，未修旧freeze；M10 pytest日志既有例外见data/M10b_v1/preexisting_log_drift.json，已按用户要求加入FINAL_SUMMARY。完整上游字节核验不通过，科学输入一致；没有新科学结果漂移。

## (e) 需要审查者决定的问题
无。

## (f) 证据路径
- 详细口径/原始run：reports/M10c_details.md；data/M10c_v1/full_measurement.json、cost_ledger.json。
- T6：paper/tables/t6_cost_precision.tex、t6_conformal.tex；data/M10c_v1/paper_inputs/*.csv；scripts/paper/make_t6_*.py。
- 配对与输入：data/M10c_v1/conformal_pairs.csv、summary_inputs.json、upstream_audit.json；configs/M10c/。
- 复现/归档/冻结：data/M10c_v1/reproducibility.json、m10b_paper_snapshot/mapping.json、freeze.json、completion_audit.json；paper/README.md；reports/FINAL_SUMMARY.md。
