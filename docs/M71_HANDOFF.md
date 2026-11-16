# M7.1 接续（2026-11-07 15:15 北京时间）

入口仍须完整读 GLOBAL_RULES/任务书。任务 docs/M7.1_TASK.md；独立 .venv-comm，source scripts/m7_env.sh 后运行，禁止修改 M7/M6/Idea A 冻结。

- 当前主任务是 M7.1，30 GPUh、全部新产物15GiB。统一三模板程序化 commitment。用户已授权 receiver cal_A/cal_B各300；natural reader_val100/test300，仅诊断。
- 初始选层 [7,16,25,34]，scan v2锚点25，投影A1/A4准确率点损失0/1.75pp；A4损失95%CI[-2.25,5.75]pp，A1 AUROC损失1.88pp [0.566,3.23]。不能声称总体损失上界<2pp。
- GPU1/2/4源原控制器 PID4053747，configs/M71/pipeline_pool_v3.yaml。GPU1/4已完成；GPU2 PID4054305在reader_val（最终237/348，15:15）。三片train合计8000组已生成。之后自动merge并在GPU1/4跑完整 C1 linear/B3（seed11）。
- source/log/progress的确切run路径均在docs/M71_LIVE.json。约3GPUh已用，预算以各run metrics求和。test尚未打开，data/M71_v1无test产物。
- 后续控制器 tmux clr-m71-continue-v2，PID4056753；配置configs/M71/continue_gates.yaml；日志reports/M71_evidence/continue_gates_v2.log。旧CPU watcher4055645已在无子任务、C1未产生时TERM并重载；源GPU未中断。
- 初次C1若失败，唯一备选固定critical逐token（同四层、同组和门槛），先清理失败pool数组缓存（SHA核验，manifest/权重/预测保留），再val12 smoke和train1000组投影检查、完整8000/1000 C1。二次未准入即scripts/m71_stop_report.py与freeze自动结束。备选投影失败也明确记C1未执行，不能写成完整跑过。
- 若C1通过，写admitted_representation.json；tmux clr-m71-main（configs/M71/main_pipeline.yaml）已等待此文件。自动reader smoke→500样本C2（≥98%）→B3/linear/B0/CLR/supervised三seed→cal特征和receiver→校准→最终锁→test→natural；CLR曝光匹配同seed B3实际曝光。C2不过主流程会报错停止，必须如实收尾。
- 主流程尚未实际运行。需要监控smoke/正式吞吐与预算，保持30h/15GiB；还有成功分支的统计/报告脚本尚未实现，必须在最终比较完成后补齐（不能以启动当完成）。若C1停止，停止报告与冻结已有实现。
- 最近新增：m71_continue_gates.py、m71_main_pipeline.py；m71_statistics.py给C1配对组bootstrap10000次；test访问标记在读取前落盘，live test标记只增不减；Serializer在CPU组装相同typed one-hot后整体搬运；C2另存reader_val诊断，不用于停止。
- 最近测试 reports/M71_evidence/tests_resume_v16.log：33 passed（12.55秒）。此前所有测试/失败日志保留，未触碰test。新模块均通过语法检查（最新controller也已成功启动等待）。
- 结束时必须复核上游M7 2912/M6 35688冻结文件全SHA；已有data/M71_v1/upstream_verification.json通过。冻结清单自身SHA：M7 0c045351d8fdf8aca3788dc074334d341c2c785e3fabcae625d6816df30ac5e8；M6 8354b1fa76abee3f7cce4f3d064fbad1b15014cb51aef71b12d3679df5a0da2d。
- 磁盘配重已0且保持暂停，15:08可用约5.2GiB，不因余量停。失败候选数组按任务授权可重算缓存清理；不删任何冻结数据。当前初始数据约5GiB，若备选失败不需生成cal/test。
- 唯一当前审查文件reports/M7.1_REVIEW.md（进行中，33项测试，非验收结论）；最后须≤8000bytes、(a)-(f)、≤3核心表、3位有效数字。详细报告/CSV另索引；各记录≤10行。M8未启动。

## 15:26 新进展（覆盖上述阶段快照）
初始pool C1通过，未执行critical备选，未删除任何pool特征。linear选epoch9 A4=.4975/A1AUROC=.7560875；B3选epoch3 A4=.456/A1AUROC=.758324625。C2第83epoch总体.982通过；val12诊断近随机，不能声称泛化。主流程已进入B3三seed训练；test仍封闭。准确run见data/M71_v1/pool/{C1,C2,reader_smoke}.json。

## 15:38 接续更新
B3三seed已完成，linear三seed正在训练；随后自动B0/CLR/supervised。新增成功分支scripts/m71_report.py及configs/M71/report.yaml：读取锁定预测，QA/检测/自然承诺/因果按组10000次CI，另有matched_text300公平面板；≤8000字节三表审查。最终预测改为gzip无损pt.gz，未改概率精度。
34项测试通过（tests_resume_v17.log）；所有M71入口语法通过。tmux clr-m71-finalize已等待主比较完成，之后自动上游SHA全核验→报告→冻结，配置configs/M71/finalize.yaml。需要最后检查主报告数值/图/8KB与冻结完整性，不能只相信启动。
主controller加载了初版main_pipeline函数，后续改源码不会自动更新内存；源/评估/报告子脚本在启动时会读最新代码。final_lock会锁所有当时现有m71脚本/config：test打开后不要改被锁文件；若纯报告bug须新增report版本并保留原代码/锁。

## 15:54 接续复核
B3/linear三seed完成，B0全量三seed第2epoch；test仍封闭。主/收尾控制器原进程继续，未重启。
报告新增层扫描峰值、CLR−B3配对CI方向、CLR−监督参照直接差值；report.yaml引用35项通过的tests_resume_v19.log。这些修改均在final_lock之前。
清理4个数月前遗留pip临时安装目录，实测释放817262592字节；无安装进程，未碰安装环境/科研产物。清理清单data/M71_v1/stale_pip_cache_cleanup.json。约4.43GiB剩余，配重为空且暂停；继续观察真实I/O，不能因余量主动缩量。

## 16:53 接续更新
B0三seed完成，CLR三seed运行（GPU1/2/4）；前者seed11/22/33分别5/6/4epoch。CLR当前第3–4epoch，前两seed验证A1/A4已约99%，A2/A3仍难，未据此改设置。累计约6.97GPUh；test仍封闭。
36项测试通过，最终report.yaml引用tests_resume_v20.log；新增QA ECE/AUROC组重采样与显式重复样本独立比较，精度一致。
报告脚本已补层扫描峰值、CLR−B3区间方向、CLR−监督参照直接差值；所有改动都在final_lock前完成。最终锁后切勿改已锁文件。
为完整cal/test留空间，清理了旧pip临时目录0.761GiB、2025年Xet传输分块1.42GiB和Conda索引缓存1.52GiB；仅可重建缓存，完整模型/数据/环境及冻结文件未动，清理清单均在data/M71_v1/。当前约5.91GiB可用，配重空且暂停。
主tmux clr-m71-main/PID4056842和收尾tmux clr-m71-finalize/PID4059006均继续；主训练结束后自动cal特征、receiver cal各300、校准、最终锁、test2000特征/receiver300、natural val100/test300 direct+cot512，接着统计/复核上游/冻结。
结束前必须检查最终报告、图、预算/字节、冻结verification、4个用户问题；不能只交付后台启动。当前仍无最终test结论。

## 18:22 接续更新
CLR三seed完成（11/22/33分别实际8/11/9epoch，选5/8/6epoch，合格所选曝光均≥同seed B3实际曝光），监督参照三seed刚启动；累计约10.0GPUh，test仍封闭。
CLR选中reader_val A1/A4约99%，A2/A3约50%；报告已明确k16聚合检测不单独证明A2/A3可读，此澄清在final_lock之前写入。完整主指标须等最终评估。
磁盘再清理了3个9天前的HF .incomplete残片2.32GiB，均无打开句柄/冻结引用，不含完整模型或数据；清单data/M71_v1/incomplete_download_cache_cleanup.json。共享盘仍有其他写入，当前约4.8GiB可用；不改完整规模，配重保持暂停。
主/收尾控制器继续；当前唯一待做仍是完整cal/receiver→校准/锁→test→natural→统计/最终上游SHA核验/冻结，以及结果审阅。reports/M7.1_REVIEW.md此时仍是进行中版本，不能当完成结果发回。

## 18:52 接续更新
主比较五类模型各三seed全部训练完成；cal_A500/cal_B1000特征已全部生成/合并，数据总计约6.20GiB。累计11.1GPUh，test仍封闭，source_full_completed只表示分片状态，不能据此称最终完成。
receiver cal_A/cal_B各300固定哈希组正在GPU1运行（pool_cal_receiver.yaml，log同名；当前cal_A约47/300，约1.5秒/组），之后自动校准并final_lock，再test。当前磁盘约2.87GiB，预计test特征约2.05GiB，仍保持全量；若出现真实I/O错误须记录并修复，不能预先缩量。
无需重跑训练或已有smoke/36项测试；等待自动主/收尾控制器，最后必须核验报告数值、预算/字节、图、上游冻结和M71冻结。final_lock之后不能改锁定代码/配置；纯报告bug需新增版本。

## 19:17 最终锁后接续
所有训练、cal_A/cal_B receiver各300、15个模型校准和norm/text校准完成，累计11.7GPUh。final_plan.json/final_lock.json已生成且逐SHA核验通过，首次test2000特征生成已在GPU1/2/4启动；live test_opened=true永久保留。
此后不得修改final_lock中已有代码/配置/checkpoint/校准文件；只做固定最终推断、诊断与统计。纯报告bug须新增版本并保留原锁定代码，不能覆盖。
test源分片之后receiver test300→五模型各三seed最终val/test→norm/text→natural val100/test300 direct+cot512→三seed A3诊断→主stage complete→finalize自动上游SHA核验/报告/冻结。
当前可用约2.66GiB，预计test特征约2.05GiB，后续预测gzip存储；若真实I/O失败需原样记录、保留run并修复，不得静默缩量或覆盖冻结。

## 19:42 最终推断接续
test2000全特征已生成合并，test receiver300也完整完成；累计约12.4GPUh，data/M71_v1约8.25GiB，磁盘余约0.60GiB，后续预测为gzip小文件。
现已启动B3三seed最终reader_val/test推断，随后linear/B0/CLR/supervised、norm/text、natural。final_lock149文件此前已逐SHA核验；不修改锁定代码配置，不按test重调阈值。
主/收尾控制器仍正常，尚未完成整体里程碑；最后应读reports/M7.1_REVIEW.md正式版，核验analysis/表/图、成本、冻结与上游哈希，再回答用户问题。

## 20:25 诊断调度与统计接续
主比较五方法三seed、receiver/norm/text最终推断均完成；final_lock149文件保持原SHA，test_opened=true。GPU6固定val首batch2 direct/cot512复放完全一致，误差0，耗时43.7秒，与GPU1接近。
为缩短墙钟，保留原GPU1进程完成val100；test原批次配对按batch_index%3分片至GPU5/6/7，每片先重复val一致性门槛再执行test，不改变300组/批次配对/解码。新入口scripts/m71_natural_partition_v2.py，配置natural_partition_{0,1,2}_v2.yaml。test natural_commitment.jsonl空文件为分片专属占位；原进程完成val后会因FileExistsError保护自然结束并完整记录成本（预期调度退出）。不得删空文件以使旧进程重跑test。
原收尾controller4059006已在无子任务时TERM；新tmux clr-m71-natural-finish-v2等待分片与val完成→严格核对300组与原顺序→合并→三seed自然A3→上游核验→冻结。统计tmux clr-m71-statistics-v2通过m71_report_wait_natural_v2.py并行计算锁定主指标，到natural统计处等待；完成后加调度说明并写report_wait_natural_v2_completed.json供冻结barrier。
仍需实际审阅报告/图/成本/字节及最终verification。所有旧锁定源码和配置未改，新增v2仅调度及报告等待；旧失败run和日志保留。

## 21:06 最终完成（覆盖上述所有运行中状态）
M7.1已完成并冻结、待审查；无M71后台任务。唯一审查reports/M7.1_REVIEW.md，6317bytes、固定六段、三主表；36项测试通过。最终freeze/verification为data/M71_v1/{freeze,verification}.json，22830文件零不一致，9.65432GiB、15.9381GPUh。
最终有效统计data/M71_v1/analysis_v3，完整详细reports/M7.1_details.md。原analysis统计及CPU报告失败run全部保留；失败原因为relative/absolute当前run比较，新增m71_report_v3.py规范化路径，m71_report_resume_v3.py复用已落盘统计，不改模型/阈值/样本。freezer同类比较在m71_freeze_v2.py另版修复，未执行旧错误freezer。最终审查十进制三有效数字由m71_review_freeze_v5.py处理并留旧render。
主结果：名义α1/5%，CLR CF83.9/93.1%，test实测FPR.933/5.07%；α5 B3 24.0%、norm6.50%，text300为0%；同300组CLR/B3/norm/text=93.4/25.9/7.00/0%。CLR−B3 A1=31.3pp[30.4,32.1]，A4=53.8pp[52.5,55.2]，与GSM8K Q4/Q3同向。CLR−当前监督参照88.2pp[87.4,89.1]，后者未学会A3，不能作oracle上界。
receiver test300有/无/错配28.7/23.0/24.7%；私有事实概率干预+13.8pp[9.31,18.7]；有私有依赖证据，但历史冻结raw逐张量对照仍缺artifact，独立验收不能记全通过。A2/A3近随机、held-out A1=88.4%、natural正确/错误A3分布重叠，负面结果全部保留。
上游最终核验M7 2912/M6 35688文件均一致。M71 final_lock149文件仍不可修改；本次所有新增结果/代码均已冻结，不覆盖。后续如需修改另建版本。M8未启动；下一步只等用户审查。

## 冻结后完成审计（2026-11-07）
11项执行要求逐项核验通过；61,222个去重冻结文件SHA全一致，12,500组/35,500条特征序列全部实际解压核验，78个模型阈值从cal_B clean重算一致。审计未调用模型，未修改主冻结、149文件最终锁或审查报告。审计结果docs/M71_COMPLETION_AUDIT.json，附加SHA清单docs/M71_COMPLETION_AUDIT_FREEZE.json；准确run见结果文件。
实验执行已完成；监督参照未学会，不构成绩效上限，不能将CLR与其差距归因于避免特定攻击训练的代价。A2/A3、投影区间和历史raw缺证据限制继续保留。无后台任务，M8未启动；待用户审查。
