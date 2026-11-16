# M10b 接续（执行中，非完成）
先读GLOBAL_RULES/AGENTS/STATUS/M10b_TASK。M10及上游冻结只读；本任务只授权GPT2 B5c Q3/Q4完整数据一遍×seed11/22/33，硬预算22GPUh，含失败加载校准推断；不做LLaMA或审计，不削减seed/标签。

## 实际进程与入口
- tmux `clr-m10b`：`.venv/bin/python -u scripts/m10b_controller.py configs/M10b/controller.yaml`；PID需用ps实时核实，已记录data/M10b_v1/controller.pid（193638）。worker seed11最初193824，当前run在train_11.log。
- tmux `clr-m10b-watch`：独立CPU预算看门狗，data/M10b_v1/watchdog.json；21.97GPUh保护线，10秒采样，只终止本次M10b GPU lease pid。不得重复启动三seed。
- 日志data/M10b_v1/controller.log、train_{seed}.log、eval_{seed}.log；真实曝光/计时从各run/progress.json/history.json/budget.json核实，不能把启动当完成。
- 顺序GPU3：完整seed11→22→33→锁全部权重→cal_A温度/cal_B Mondrian→reader_val/test→scripts/m10b_finish.py CPU导出/纸面复现/上游哈希/报告/冻结。尚无final_lock、未开test。

## 已完成关卡
- 20训练组/20reader_val组smoke：Q3 845/Q4 280曝光，76.7秒含加载；同一Qwen/LoRA/mapping真实梯度。15组reader_val校准/评估代理10.6秒。test=false。
- 独立.venv-paper的T1 val代理导出和逐字节重建，详情export_smoke/；15项协议单测通过，data/M10b_v1/tests.log。
- 预算准入data/M10b_v1/budget_admission.json：三seed train15.1866、训练val.460818、预留2，预计17.6474/22 GPUh。完整训练counts Q3 860228/Q4 279440，未减少标签。
- M10 B5c实际循环epochs1，历史min_epochs8无效；本次min_epochs1明确设置，不能声称与CLR/B6八遍曝光匹配。审查报告必须说明此局限。
- 首次独立snapshot把绝对config复制到自身失败；同配置修复后smoke通过。原run保留，GPU计时缺失按3.6秒保守收费；一次未source环境import失败无GPU调用。docs/BLOCKERS有记录。

## 产物与冻结
- 不修改M10代码/结果，import clr.models.m10_b5c、clr.data.m10。新runtime/train/eval/controller/export/finish/watch为M10b版本。
- M10当前paper/FINAL_SUMMARY76文件已逐字节归档data/M10b_v1/m10_paper_snapshot/mapping.json。旧M9归档保留。当前summary/claims已按用户修订C3框架并列CLR/B6；跨用途与HiddenBench仅附录，后者数据不足无法结论。
- 新T1 renderer读冻结CSV，Q1/Q2以及LLaMA B5c全部NA；audit B5c仍NR。附录caption在新配置里标注。
- finish自动输出reports/M10b_REVIEW.md（≤8KB、(a)–(f)、并列val/test、CLR−B5c配对CI）/details、更新T1/README/manifest/FINAL_SUMMARY；独立环境两次全纸面重建逐字节对比；上游SHA按M9/M10两个archive版本恢复核验；最后M10b freeze/completion_audit。
- 代码最后一次编译/单测已通过，正式GPU评估前确认CPU val代理链路通过，且upstream_audit没有unexpected_errors。已知M10 tests_final.log在旧freeze后、本次启动前被改写，仅非指标pytest stdout；136268项其余文件一致。不得把passed:false的全字节审计写成无例外通过，见preexisting_log_drift.json。如controller出错，保持失败与成本，修复新版本入口后从已锁权重继续，不重训/不选择test结果。
- 原始完整训练三seed预计约15小时驻留GPU3；不要因前几分钟进度少而启动重复worker或改变epoch。硬预算触发则停止并如实报NA，不追加预算。

## 2026-11-11 17:00恢复调度（覆盖旧PID与ETA）
seed11/22已完成；33于12:02 GPU3外部占用导致OOM，0.0376743GPUh保留。原控制器已退出，未final_lock/未test，当前累计12.4975GPUh。
所有卡当前空闲<9GiB，不能启动batch512；tmux clr-m10b-recover运行m10b_resource_wait configs/M10b/resource_wait.yaml，CPU每30秒检查，连续3次空闲≥22GiB后自动选卡，运行controller_v2。等待不计GPUh。
controller_v2从已完成ref恢复，只重跑seed33，科学设置全同，仅device变化；完整33与全部cal/eval同卡。预算估计12.4975+6.87918+2=21.3767≤22。原controller.skip时把已完成11误算成两次剩余的projection不适于恢复，v2按真实缺失ref计数。
恢复后8–9小时加资源等待；原20:00–21:30 ETA已失效。新日志resource_wait.log、controller_resume_v2.log；失败原controller/train_33.log不覆盖。任何OOM与预算失败必须报告，不缩减设置。

## 用户取消显存阈值（2026-11-11 19:25）
已终止CPU resource_wait PID4095886；不再等待22GiB或内存guard提前退出。新增controller_v3，训练命令覆盖min_free_GiB=0，其余模型/batch/标签/seed相同。cuda:1按当前最大空闲选取，不以该值准入。
tmux clr-m10b-no-threshold，配置configs/M10b/controller_resume_v3.yaml，日志data/M10b_v1/controller_resume_v3.log；原12.5GPUh失败成本保留，硬预算watch继续。实际OOM才报错，不循环盲重试、不缩设置。

## 无阈值实际尝试结果（2026-11-11 19:27）
GPU1 run20261111T112434784211Z_beec019ccbd0_c9844b首batch实际OOM：申请280MiB时剩余176MiB，失败0.0430547GPUh计入，累计12.5406/22。controller_v3已退出；resource_wait已停止，不再阈值等待；22GPUh硬限watch继续。无成功新seed33、未开test，不得声称训练仍运行。

## 最新：启动前门槛，启动后不按显存停（2026-11-11 19:50）
用户最新明确恢复持续监控：tmux clr-m10b-startup-monitor，m10b_resource_wait_v2 configs/M10b/resource_wait_v2.yaml；CPU每30秒取一次，连续3次≥22GiB才启动controller_v3。子进程训练min_free_GiB=0；监控器在subprocess.run等待，不检查显存或发停止信号。预算看门狗仅22GPUh，保留。
若实际OOM，原run/成本保留，重新回CPU启动等待；非OOM错误或预算预计超限则停止如实记录。seed11/22不重复，33完整原batch/标签/模型不改；CPU等待0GPUh。日志startup_monitor.log、controller_monitored.log；状态resource_wait_v2.json。五项协议/调度单测通过tests_startup_monitor.log。
