# M10 接续（2026-11-10；未完成）

先读 AGENTS.md / docs/GLOBAL_RULES.md / STATUS.md / M10_TASK.md；M9及上游冻结结果只读。当前goal仍为完整M10，不能把启动/部分训练写作完成。

## 真实进程与预算
- 运行 `.venv/bin/python scripts/m10_status.py configs/M10/status.yaml`（先 source scripts/m7_env.sh）读取 /proc 实际进程和预算，不能只信 LIVE。
- P1 GSM控制器 PID 2984723，exec session 95681；配置 full_gsm_queue.yaml，日志 full_gsm_queue_retry2.log；GPU4保留给该队列。
- P1 audit控制器 PID 2902400，child 2902601；exec session 70918；配置 full_audit_queue.yaml，日志 full_audit_queue.log；GPU0保留给该队列。它启动早于新增GPU锁，不能在0上另开M10。
- 上述PID须每次用 /proc/ps 复核，job未终止不得重复启动。新run有fcntl逐GPU锁，禁止本任务同卡并行。
- 最新状态 data/M10_v1/status_observation.json；预算从 runs/config.json + metrics.json/budget.json累计。上次P1约1.04GPUh，P2-4约0.101GPUh，test=false（不是永久快照）。硬限P1=12，P2-4=12，P2-5=12，P2-6=4，总40。

## 已完成与代码
- 7个GSM初始smoke（20组）和audit B6 smoke通过；GSM/audit评估smoke仅val代理通过；tests_v5.log为6 passed。
- LLaMA B0及B3-prefix三seed正式训练完成；single训练中，之后队列自动训练B6两源三seed与每类型CLR两源三seed。
- audit B6每task独立，hidden896=4866177参数>=B3每task4719876；两LR/6epoch，返回checkpoint曝光>=B3总曝光（5/5/6epoch）。4任务均训练，主报A1/A4；检测用原M71全任务随机16问公共期望。
- GSM入口 scripts/m10_train_gsm.py、m10_evaluate_gsm.py；audit B6入口 m10_train_audit_b6.py、m10_evaluate_audit_b6.py。
- 每类型CLR曝光靶标 data/M10_v1/exposure_targets.json 来自冻结best epoch与真实train_counts；均8个完整epoch（不是旧exposure.json的过时40000估计）。full训练会自动核对counts与曝光。
- eval最终需 final_lock.json 中 all_training_locked + checkpoint_sha256；当前未创建锁，不得开test。应先锁定所有最终比较（含P2的预算取舍和训练）。

## 本轮偏离/失败，必须报告
- LLaMA reader_val真实1000组；补充吞吐run 20261110T094237091635Z_bb9fab37a1bf_56ff0b用了100组=10%，超5%。原20组正式smoke合规；偏离不是test暴露。已保留，补跑50组run 20261110T100152255016Z_fb1f73635d46_97f871通过。GSMDataset已加自动5%组比例检查。
- B5c batch64先GPU3外部占用OOM，后误与P1共用GPU4造成B5c和prefix22都OOM。已坦诚报告、计费、保留traceback；prefix22同科学配置重跑通过。新GPU锁防再发生。
- B5c最初强制body.eval导致HF激活checkpoint不生效；修正后dropout=0且基座仍requires_grad=False，GPU2真实checkpoint smoke通过。旧失败/成功记录保留，不混成正式模型。

## P2-4 B5c：预算准入失败，已决定不跑全量
- clr/models/m10_b5c.py：两层映射+Qwen3-1.7B全线性层r16/alpha32 LoRA（FP32 LoRA、bf16基座），Yes/No或A-D单位置logits。
- 成功smoke：batch16、真实checkpoint64与512；64->512吞吐不增。最快实测仅GPT2 Q3/Q4一遍全训练集×3seed=19.212GPUh，未含LLaMA/audit/cal/test，超过12。
- 预算脚本 scripts/m10_b5c_budget.py / configs/M10/B5c_budget_admission.yaml；精确data/M10_v1/B5c_budget_admission.json。
- 不擅自少seed/少标签替代。不产生B5c test比较；T1/T4显式NR（预算），不能声称CLR优于训练式语言化基线。P2-4未完成属任务授权预算取舍，写REVIEW未通过（预算）。不再重复优化或申请预算。

## 下一步：HiddenBench P2-5
- 只做了格式核实：data/M10_v1/hiddenbench_format_audit.json；原始路径 /home/me/datasets/HiddenBench/1e3c25b1fd798c6717f4df0463edd3825c8e37f9/benchmark.json，SHA 2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3。
- 共65case；其中3个evacuation案例公开description完全相同，必须同组，因此63独立组。源数据已在划分前做schema审计，尚无HiddenBench冻结split/source/模型实验。
- 63<99，无论怎样切分都不能给finite 1% split-conformal阈值；1%必须报告infinity/拒绝率0和不可辨识性，不捏造99个组。5%需要至少19个clean校准组。
- 后续需要先冻结组划分。可考虑train13/val20/cal_A4/cal_B20/test6独立组以保证val1组smoke<=5%且5%校准finite；这只是草案，未锁定，不得当已执行设置。所有65case尽量使用，不复制改写当独立场景。
- 私有信息是自由文本hidden_information，答案大多3选项，无M71的可逆公开T。不能从结论唯一推回私有事实。一个可行的协议草案是程序化commitment显式声明被审计fact并附原结论，所有clean/mismatch同模板风格；期望仅解析commitment；QA为A1事实包含、A4四个原子事实候选，未构造单事实CF则只mismatch。需要明确这是HiddenBench的协议适配，不能冒充与synthetic完全同任务。
- 保留M71特征函数、只读projection.pt、选层[7,16,25,34]、tail8、postRoPE、Qwen3-4B型号和mask契约。clr/sources/kv_relay.py支持encode_texts；record_metadata可读x_pub/x_priv对象；KVFeatures/geometry/persisted等只读import。
- 原M71 Serializer/Reader的team one-hot仅适用4个TEAMS，不能强行套自由文本；自由文本可用clr.audit.m7_reader.Serializer（必要时新版本提高序列上限，禁止静默截断）+同Reader架构；B3需真正state×option MLP（文本embedding），B6需generic fact query，不能将自由文本误解析成TEAMS[0]。
- Norm从同层raw KV统计；text只读programmatic commitment+receiver输出；source新forward和磁盘预算都要记。test source只在最终锁后生成，smoke只val代理。

## P2-6 代码已准备，未运行
- clr/audit/m10_transfer.py：嵌套A1/A4标签面板；M9 multi_{seed}主体严格SHA核对后初始化Reader.body，新FeatureAdapter随机；与scratch同seed适配器相同。主体和适配器共同微调（是初始化比较，不是冻结主体对齐）。
- scripts/m10_train_transfer.py；configs/M10/transfer_smoke.yaml（20组、N100、2epoch、一seed、两初始化）；full为N100/300/1000/full、三seed，small32epoch/full4epoch，val100组选择，最终仍须完整reader_val/test。
- 这些配置未测吞吐，不能直接宣称<=4GPUh；先smoke估算。训练full将一次载入共享源以避免24次重复加载；每模型独立checkpoint、完整曝光和history。
- 尚未写transfer最终评估；只需A1/A4学习曲线及配对CI。不要用新test选N/epoch。

## 仍缺的收尾
- P1所有训练完成核对曝光/参数/checkpoint；P2-5与P2-6smoke/预算/训练；全M10训练和取舍final_lock后再最终test。
- 各设置最终calibration/reader_val/test、Q1 clean/traj、三seed按problem/CF-group bootstrap10000；对照冻结旧模型输出，不能用smoke指标当主表。
- reports/M10_REVIEW.md<=8000字节固定(a)-(f)/<=3表、M10_details、CSV、成本、BLOCKERS与状态；回答用户5问，未做B5c如实NR。
- 更新paper时先保留M9原paper/FINAL_SUMMARY的逐字节归档和映射，因为M9 freeze含原路径SHA；不得改M9 freeze或破坏原字节。M10产物新版本再更新当前paper/FINAL_SUMMARY，注明授权supersession。
- 纸面新增B6、缺失格、per-type对照、transfer曲线、HiddenBench；每产物独立脚本/CSV SHA，干净.venv-paper逐字节重建；更新manifest/freeze/FINAL_SUMMARY主张证据。冻结前不得把M10标完成。

## 2026-11-10 继续执行补充（以实时状态覆盖上文旧快照）
- P1原两个控制器仍在；GSM当前single33，audit B6当前seed11/A4第二LR，A4 val已学会（约97.5%中间值），不可沿早期随机水平下结论。
- P2-6 smoke和eval-smoke通过，4GPUh准入3.84（训练估算含20%余量），full PID3072515 / GPU5 / session19572运行；日志transfer_full.log。N100/300已完成，N1000进行中，full随后。
- HiddenBench协议已落盘data/M10_v1/hiddenbench/protocol.json：13/20/4/20/6组；全源65case、63组；fact-panel适配，只有mismatch，1%为+∞。
- 新模块clr/audit/m10_hiddenbench.py及source/train/eval脚本；source须.venv-comm（4.57.1），reader仍.venv；source非test完成，source/reader/eval smoke通过。
- HiddenBench控制器PID3089352 / session1353；训练PID3093059 / GPU2；正式CLR三seed后B3/B6三seed，日志hiddenbench_reader_full.log。预算准入5.95/12GPUh。
- 源smoke两次失败（误用主环境dtype和漏return）保留计费，retry3通过。训练smoke按≤5%QA标签、val1/20；train组1/13粒度限制见BLOCKERS。
- tests_v7.log为9 passed。旧paper/FINAL_SUMMARY66文件已保留于data/M10_v1/m9_paper_snapshot，mapping记录原SHA。尚未更新paper。
- 已启动final_controller PID需实时查，session93268，配置final_controller.yaml；仅等全部训练，之后自动lock→最终评估→CPU exports。它是已授权动作，不要重复启动或覆盖log。
- final_lock目前尚不存在；未开test。最终HB源在GPU2，和HB reader顺序；避免GPU7外部波动。P1评估GPU4/0、transferGPU5互不叠卡。
- 已写m10_export_gsm/audit/transfer、render_m10、m10_paper_prepare/repro，但均未最终执行；这些脚本仍须核验，不能当完成。final_controller只做到exports，不负责paper/review/freeze。
- 仍缺：真实全部结果、导出bug核验、paper_prepare+独立repro、可视核查、M10审查/FINAL_SUMMARY/README/manifest/上游SHA/最终冻结。最终lock的P2-6预留eval0.4GPUh，若准入失败先实测核验，不越4GPUh。

## 最新收尾代码（未当作已执行）
- 全部GSM评估分支smoke完成 data/M10_v1/evaluation_branches_smoke_completed.json，11项tests_v9通过；upstream_pre_paper.json核110122文件、0差异。
- HiddenBench公开契约v2已生成与逐QA一致性验证；最终norm/text已改成M71拟合定义（norm未标注train，text在cal_A每episode一次C1 logistic），final_protocol_smoke通过。
- 全量迁移一epoch约180秒；scratch_full_11已完成4epoch，M9_full_11训练中。预算仍有余。
- final_controller PID3123380/session93268；paper_controller新启动/session88897（PID须实查），等待exports_completed后paper_prepare→独立repro→上游核验→reports→README/manifest→再上游核验。它不自动freeze。
- 新增m10_write_reports/document_artifacts/freeze/completion_audit入口，均尚未执行；报告生成若>8000会缩短(d)，仍超则需人工精简；不要跳过格式门槛。
- 最终paper_reports_ready后需看PDF预览、核CSV与五问/主张措辞，再m10_freeze configs/M10/freeze.yaml，之后m10_completion_audit configs/M10/completion_audit.yaml。通过后才goal complete。
- HB最终source test已改GPU2（HB训练结束后有空）；final lock同时记录original与public_v2协议SHA。final_lock配置新增所有GSM评估分支smoke关卡。

## 本轮最近实际状态
HiddenBench训练已完成：15checkpoint（CLR共享A1/A4三seed，B3与B6各task三seed），训练0.655GPUh；非test完成标志已写。
当前P1审计seed22/A2、GSM seed11/Q3，transfer M9_full_11；正式GSM Q3第一epoch336秒、第二115秒，选项冷缓存开销记录，不更改协议。
paper_controller PID3211120/session88897等待exports；原final_controller PID3123380仍等全训练。所有最终评估尚未启动。
已释放磁盘配重25GiB、auto_refill_paused=true，见reports/M10_evidence/disk_reserve_release.json；不得随即resume。
新增m10_completion_audit可核冻结字节/三seed矩阵/每类型曝光/B6参数；冻结前须先可视核查和5问/主张人工核验。

## P1调度现已更新，旧PID不可再用
旧GSM控制器2984723已SIGTERM，只终止纯CPU父进程；原GPT2 B6 worker3151269继续GPU4。
新GSM接续控制器3339970/session70395：scripts/m10_resume_gsm_parallel.py configs/M10/resume_gsm_parallel.yaml。
它启动GPU2队列3340152（LLaMA B6三seed，worker3340460）和GPU3队列3340153（两源每类型CLR六run，worker3340459），等原worker结束后跑GPT2 B6余22/33于GPU4。
科学配置/数据/seed/曝光全不变，新增_gpu*.yaml仅device；补充data/M10_v1/gsm_execution_supplement.json，记录于CHANGELOG。
原full_gsm_queue的全部21run完成后才写gsm_training_completed.json，final_controller仍等待该门槛；不要重启旧队列。
审计原控制器2902400仍在，worker3134154 seed22/A3附近；迁移3072515在full22附近。HB已全部完成，无HB活worker。
其余final_controller3123380与paper_controller3211120仍等待，不要重复启动。test未开，final_lock不存在。

## 释放GPU接续（新增）
已启动scripts/m10_opportunistic_training.py两控制器，依赖transfer_full完成→GPU5、audit三seed完成→GPU0。
配置configs/M10/released_gpu{0,5}_controller.yaml；日志reports/M10_evidence/released_gpu{0,5}_controller.log。
它们仅帮助原GPU3队列尚未启动的每类型任务；新训练入口加case/seed任务锁、已完成精确协议跳过，避免重复运行/计费。
原GPU2/3/4队列、GSM resumer、final/paper controller均不变；当前活任务不移动。
释放GPU补充：GPU5依赖迁移完成后先GPT2 B6 seed33，再每类型任务；GPU2依赖LLaMA B6三seed完成后助每类型任务。GPU0依赖审计三seed完成。
三控制器session96404/85378/90715；不能重复启动。配置released_gpu{0,2,5}_controller.yaml与对应queue；按case/seed任务锁确保一次训练。
新增task复用协议测试后12项通过，reports/M10_evidence/tests_released_gpu.log；final lock新增核所有M10 run test_access/metrics标记及terminal状态。
