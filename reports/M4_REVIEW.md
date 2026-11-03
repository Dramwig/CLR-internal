## (a) 状态

M4未完成：D数据补齐已完成，CLR的k=4/8验证试验正在运行，尚无M4最终test结论。

## (b) 关卡

- 通过：val-only源/reader/校准评估smoke；13项单元测试，含选项隔离/置换、观测前缀拼接和Bernoulli conformal覆盖率。
- 通过：D生成1993题，996训练题K8轨迹、997评估题每t K16 MC；完整生产0.419 GPUh、117130次forward；含源smoke约0.423 GPUh，低于6小时。冻结ID与标签审计通过。
- 未通过（运行中）：k选择、主模型/B3-multi/留一类型迁移/补充训练的三seed实验。
- 未通过（未执行）：最终cal_A温度/仿射、cal_B Mondrian与test/unseen_eval评估。正式评估须先通过全部训练锁。
- G1四问均未通过（缺最终证据）：Q3扣B0/Q4优势、同覆盖率集合效率、统一模型相对B3/B3-multi、少样本迁移优势。仿射改善与补充训练后的分布落差也尚未测得。

## (c) 核心主表

尚无完整三seed reader_val/test结果，不填估计值或以smoke结果冒充主表。完成后将替换为最多3张核心表。

## (d) 本里程碑新增偏离与阻塞

正式训练前依据完整val吞吐把最大4epoch显式修订为10、每2epoch验证、patience2；每qtype每epoch10000采样不变。估算含20%裕量约76 GPUh≤80，保留旧方案与pilot；可选消融暂未运行。
未见池259题公式解析失败、2题负例匹配失败跳过Q4；172条eval与新增train有13-gram重叠、精确0，保留规定划分并预先声明825题无重叠诊断子集。
控制器running识别、无限阈值JSON、严格YAML键错误均已修复并留存失败记录；k8预检查由4改为3.5GiB；GPU1另做同设置val smoke确认峰值2.42GiB后，将新run门槛显式设3.0GiB。模型/batch/精度未变。没有静默回退。

## (e) 需要审查者决定的问题

目前无：计算继续进行，尚不请求G1验收或进入M5。

## (f) 证据路径

任务与预算：docs/M4_TASK.md、reports/M4_budget_v2.md。D详细报告：reports/M4_data.md。当前状态：docs/M4_LIVE.json、docs/STATUS.md。
D run：runs/20261103T063500842769Z_924ffcfc4ca2_e9f41f；QA与审计见M4_data.md。
控制器日志：reports/M4_training_suite_v9.log、reports/M4_pipeline.log；运行中试验：runs/20261103T065137301904Z_23bf54104797_6e243a、runs/20261103T070515702150Z_e258073e121b_22aa90。
后续run索引将保存于data/M4_v0/training_suite.json、evaluation_suite.json；最终详细指标与报告尚未生成。
