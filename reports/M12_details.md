# M12 详细证据

完成时间：2026-11-13T06:51:34.247415+08:00；截止：2026-11-15T12:00:00+08:00。配置与组件定义：docs/M12_TASK.md。M41主模型真实配置复制到configs/M12/base.yaml；完整训练每epoch 2019896条QA，batch1024，无放回，宏NLL最早8/最多10/patience1。所有保留变体三seed，无test选择。

训练锁：data/M12_ablation/training_complete.json；逐run history/best/training_manifest可审计完整曝光。预算准入：admission.json；每个run config/git/env/seed/code_sha256/startup_source.tar.gz/日志/metrics/budget齐全。code快照包含未跟踪文件，git提交号不冒充覆盖它们。

统计：summary.csv为长表，每split/qtype/variant一行，seed_11/22/33、mean、95%按题cluster bootstrap10000次、variant-minus-CLR-lite配对差CI；固定三seed在每次抽样内平均，清洁状态/seen模板/t等权，每t内部按QA实例加权，Q2每t/seed AUROC后等权。CI条件于原校准集，不含重新抽校准集不确定性。reader_val_q1_clean_traj.csv另报Q1两类状态。主输出选择沿用M41，未用test调参。

Conformal：cal_A温度，cal_B按qtype/t Mondrian；每题/每t固定选一个实例，alpha.1；集合大小和经验覆盖率/CI见summary.csv及conformal_observations.parquet。覆盖保证依赖分组内校准/测试可交换性；clean与训练分布偏移下不额外承诺保证。复用行未重校准，逐stratum集合大小与原conformal.json对照误差≤1e-7。

置换：permutation_observations.parquet与permutation.csv；主/A1，三seed，在完整clean/seen Q3上逐行固定非恒等随机排列，按选项身份恢复后比较argmax，报告canonical接口及raw接口。A1只改变attention mask；原共享逻辑位置与规范化排序仍保留，因此零变化不表示选项隔离。raw浮点差异及最高分并列计数详情见permutation_complete.json所指run/details.json。

运行恢复：A4 seed11在共享GPU3首epoch一个优化步后OOM；失败run和0.0225GPUh保留，同模型/batch/bf16/曝光在GPU0从头重跑；A4 seed22/33不重启。接续控制器复用A1–A4完成回执，seed设备0/2/5。证据recovery.json、recovered_controller.json及原training_exception.json。

后处理环境恢复：全部最终评估完成后，置换入口在recorded/GPU初始化之前因Pillow/libLerc所需GLIBCXX_3.4.29缺失而退出。显式LD_PRELOAD现有latdefer/lib/libstdc++.so.6，依赖导入预检通过后仅接续置换/统计/交付；未修改锁定科学代码/配置，未重跑训练/评估。证据postprocess_environment_recovery.json、reports/M12_evidence/permutation.log及permutation_retry_environment.log。该失败未初始化GPU，不计作GPU作业，原CPU监督失败记录保留。

CPU统计提速：原高级索引bootstrap在2000题合成面板耗时41.1秒；改为相同RNG(11)整数抽样的计数矩阵乘法后0.398秒。固定10000次抽样、按题聚类/原步与seed聚合均不变；含缺失步骤、不等样本权重及三seed的合成对照中点估计/CI误差≤1e-14，抽样计数逐项相同。旧CPU统计进程被显式中止、日志和启动快照保留，新进程重新读取同一科学预测完成最终统计，无新GPU/推断/校准。证据statistics_runtime_recovery.json及statistics_counted_parity_tests.log；未改冻结旧实现。

论文只读核验：paper_readonly_audit.json；paper/及已存在的重命名归档manuscripts/CVPR27-CLR-achieve-2026-11-12共96文件与旧冻结SHA一致，未改稿/编译。

复用：source_registry.json逐seed指向原始结果，M41 lite/v1、M5b single、M10 per-type、M9联合两源；已有行不重训/重推理/重校准。仅新增任务所需主模型置换诊断。复用设计存在训练曝光/数据来源区别，不能解释为新增A1–A5的严格单组件对照。

数据：problem_metrics.parquet、auroc_observations.parquet、conformal_observations.parquet；原始预测evaluation_complete.json；源文件sha统计statistics_provenance.json；冻结输入核验upstream_audit.json。

成本：18.1463299 GPUh≤30，4090；完整GPU作业墙钟总和，含smoke、初始化、数据装载、失败与推断；CPU控制器/统计不计GPUh；源前向0，旧里程碑成本不重复计入。成本逐项costs.csv。

代码数值验证：TensorSerializer与冻结Serializer四种类型真实val batch逐tensor相同；控制类与原Lite训练梯度逐bit一致，mask/padding/排列等变、A3数值消融、A4无模板中转、split隔离、Q1标签和AUROC配对bootstrap均有最小测试。启动前/收尾日志reports/M12_evidence/unit*.log。训练smoke20训练题、协议proxy50个reader_val题，cal_A/B/eval为15/15/20互斥；规模≤5%、每阶段≤900秒。smoke.json保存实际耗时。
