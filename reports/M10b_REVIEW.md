# M10b 审查

### (a) 状态一句话
GPT2 Q3/Q4 B5c三seed完整训练、最终评估和论文复现完成待审查；本里程碑后实验永久冻结。

### (b) 关卡清单
- 通过：20组train/20组reader_val训练smoke，76.7秒；15组reader_val校准/评估代理，10.6秒；无test访问。
- 通过：Qwen3-1.7B冻结基座、两层映射、LoRA r16/alpha32；每seed完整Q3 860228/Q4 279440条训练一遍，seed11/22/33。无少标签、少seed。
- 通过：全部权重SHA锁后才test；cal_A温度/cal_B按类型与step Mondrian，三seed并列reader_val/test。
- 通过：成本17.2/22 GPUh（加载/失败/校准/推断在账本）；独立环境44个产物逐字节一致；上游模型/数据/指标SHA一致。
- 未通过（既有非指标日志完整性）：M10 tests_final.log在M10冻结后、M10b启动前被改写；原字节缺失，不重造，预期/实际SHA与时间记录preexisting_log_drift.json。其余136268项一致，不影响本次模型/指标比较。
- 通过：审计改为类型化审计框架，正文并列CLR/B6；跨用途/HiddenBench只入附录，后者数据不足无法结论。
- 未通过（授权范围外、未执行）：B5c Q1/Q2、LLaMA源及审计任务，均NA/NR，不外推。

### (c) 核心主表
准确率/差值均为%，差值单位pp；固定三seed，整problem bootstrap10000次95%CI，保留组内step；不是重新训练seed的不确定性。

|split|类型|B5c准确率 [95%CI]|CLR−B5c [95%CI]|
|---|---|---|---|
|reader_val|Q3|86.8 [86,87.5]|-0.426 [-0.905,0.0643]|
|reader_val|Q4|85 [84.2,85.9]|0.062 [-0.526,0.635]|
|test|Q3|76.8 [75.5,78.1]|-0.659 [-1.51,0.19]|
|test|Q4|76.3 [74.8,77.7]|-0.0864 [-0.984,0.83]|

用户所问test差值：Q3 -0.659 [-1.51,0.19]pp；Q4 -0.0864 [-0.984,0.83]pp。

### (d) 本里程碑新增的偏离与阻塞
新运行账本source_snapshot首次把绝对config路径复制到自身而失败；原记录/traceback保留，修复后同配置smoke通过。一次入口未source环境导致import失败，没有模型/GPU调用。seed33在11月11日12:02因外部显存占用OOM，失败0.0377GPUh计费；原控制器停止至状态检查，曾启用CPU资源等待，后依用户修订取消所有显存阈值等待/提前退出，min_free_GiB=0直接运行；同模型/batch/标签量仅重跑seed33，保留11/22。取消阈值后GPU1首batch实际OOM，0.0431GPUh保留计费；22GPUh硬限不变。
M10实际B5c训练循环epochs=1，原smoke的min_epochs8未使用；本次明确一遍完整标签，未擅自变为八遍。其曝光低于CLR/B6，差距不能单独归因于架构或语言化设计。M10及更早冻结保持原字节，授权更新的paper/FINAL_SUMMARY有逐字节归档。

### (e) 需要审查者决定的问题
无。实验永久冻结，不再申请追加预算。

### (f) 证据路径
- 详细：reports/M10b_details.md；汇总：reports/FINAL_SUMMARY.md。
- 数据/配对：data/M10b_v1/paper_inputs/B5c_summary.csv、CLR_minus_B5c.csv、B5c_conformal.csv。
- 权重与run：data/M10b_v1/trained/、evaluated/、final_lock.json；配置configs/M10b/。
- 成本/复现/完整性（含上述已知例外）：data/M10b_v1/cost_ledger.json、reproducibility.json、upstream_audit.json、freeze.json。
