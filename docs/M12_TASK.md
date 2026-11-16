# M12 CLR-lite组件消融
用户2026-11-12授权；北京时间2026-11-15 12:00硬截止，逾时项放弃且保留已完成项。总预算≤30 GPUh（含smoke/失败/评估），仅CODI-GPT2、源前向0，不改论文或旧冻结。

A1全注意力（保留padding屏蔽及原逻辑位置/规范化序列）；A2联合MLP替换为选项表示与标准化有效前缀均值经原adapter投影的点积+每输出头一个标量bias；A3删除两处数值通路（token数值投影及联合头数值输入），保留选项文本；A4所有文本token与潜token双向隔离以防模板中继，潜状态仅标准化有效前缀均值进入原联合头；A5同原批次采样/顺序，仅loss权重设1。A1→A5优先，成本不足从后放弃，不能缩曝光。

训练逐项沿用M41主lite真实启动配置：三seed11/22/33、batch1024、bf16、AdamW head_lr3e-4/weight_decay1e-4、梯度clip1、无放回全QA、每epoch完整reader_val宏NLL、最早epoch8选checkpoint、最多10epoch/patience1。A5早停仍宏NLL。所有保留变体训练完并锁checkpoint/代码/config hash后统一打开test。校准沿用M41 cal_A按类型temperature、cal_B每t/类型Mondrian、alpha0.1。smoke≤5%数据/900秒且仅reader_train及reader_val代理。

复用CLR-lite及CLR-v1(M41)、lite-single(M5b)、per-type(M10)、双源联合(M9)原始预测和原始校准，不重训/重校准/重评估。历史来源/曝光差异在表中标注。主表沿用clean/seen、t等权，三seed均值；按题cluster bootstrap10000次并使用相同抽样配对差；Q1 reader_val另报clean/traj。Q3/Q4 cal_B alpha.1集合大小遵循原每题/每t单实例选择。

置换诊断：主模型/A1，独立固定随机非恒等置换，对齐回原选项身份后计算argmax变化比例；同时公开canonical=True和关闭排序的raw结果，避免排序掩盖。完整reader_val/test clean seen Q3，三seed；新诊断允许对主模型推理，不重跑已有科学指标。

交付data/M12_ablation/summary.csv（含3seed均值、按题CI、与主模型配对差CI及来源）、permutation.csv、reports/M12_REVIEW.md≤2000 UTF8字节/六段/表格+成本；详细证据另索引。无额外审查门槛，完成后冻结data/M12_ablation/freeze.json。

实现核验补充：新TensorSerializer仅缓存原序列化的CPU张量，真实val四类型逐张量相同，CPU控制模型训练输出/梯度逐bit一致（tests/test_m12.py）。二分类头保留原二维调用形状以避免形状引起舍入差，修正后重新smoke。所有新文件在每个run启动保存源码tar/hash，明确区别于已有git提交，不依赖未跟踪patch复现。

配置来源说明：base由M41 seed11真实config复制，保留少数未被M12入口读取的历史键（例如budget_authorization旧50h文字、cross_grid、samples_per_type、gradient_accumulation、validation_epochs、gate_lock、retry_note）。M12有效预算仅total_budget_hours=30及deadline；lite不启用预训练encoder/checkpointing；实际训练用batch_size=1024、每epoch全QA、每epoch完整val。旧键不覆盖当前任务，不将历史50h当成本次授权。

运行恢复：A4 seed11共享GPU3 OOM后同科学配置在GPU0全新重跑；不改batch/精度/曝光，失败成本保留。接续控制器复用所有completed训练回执，显式seed设备0/2/5；原A4 seed22/33不重启。最终置换诊断沿用接续seed11设备，避免硬编码已拥挤GPU3。
