# M8b 接续（2026-11-08）
当前目标仍为完成用户M8b，预算25GPUh，不能把启动当完成。未创建最终test lock，未打开本次test。

## 已完成
- 强制入口已读；M8a/M71/M7/M6共62363文件哈希全一致（data/M8b_v1/upstream_preflight.json）。
- `.venv-comm` + source scripts/m7_env.sh，transformers4.57.1，Idea A只读import。
- source5500组：train4000 / val200 / cal_A300 / cal_B1000，原/CF二元cohort，R4真实receiver答案前末8token、4层，含R2/R3充分统计，17.5GiB。
- 原R4三seed完成；val A4约56%–60%；原R2三seed完成但A4随机。R3原版训练中，seed11 A4=91.5%，其余seed可能早停未学好。
- 7项单测通过，真实val20重放R1/R2/R3/R4 fp16值逐位一致（replay_smoke_v2_part0.json）；R4训练后val20 PGD smoke112秒通过。
- 原小样本warm R2过拟合诊断100组100epoch未充分学好，保留，不作为主模型。

## 当前运行与队列（需实时ps/tmux核实）
- tmux clr-m8b-development：原R3三seed → 原全部calibration，GPU5。
- tmux clr-m8b-r2-warm：R2 v2三seed，原适配器初始化、24epoch/patience8，GPU7，folder warm_training_v2。
- tmux clr-m8b-r3-warm：等待原全部校准结束后，R3同规则v2三seed，GPU5。
- tmux clr-m8b-selected：等待R2/R3 v2与原全部calibration → scripts.m8b_select → selected calibration → selected_smoke（val20，全设计/三seed，eps.03 λ1，50/200步，≤900秒）。
- 选择规则已在TASK预先落盘：R2/R3分别按三seed平均best val NLL整体选v1/v2，不逐seed拼接；R4不变。原索引不覆盖，输出selected_training、selected_calibration、variant_selection.json。

## 代码/协议注意
- clr/audit/m8b_*.py，scripts/m8b_*.py，configs/M8b/。所有实验有recorded config/env/code snapshot/cost。
- 新设计冻结对应M71主体，仅训练adapter。R2/R3输入128维，R4 512维；容量差异需明示。
- R2 8/36层，逐记录边界±1，P128×Bernoulli(.9)/.9；现场与缓存同fp16充分统计。随机部署master key由OS抽取并存run/private_report_rng.json，不给优化接口；EOT独立。
- E1/E2固定原M8a f盲层/池化核；R3自身未读层为空不作经验成功。E2检查实际旧f fp16逐位相同。
- E3每设计/seed自适应，fresh EOT k16，独立selection2面板；目标命中且selection均值≤较小门限(5%)才早停。候选按目标命中、再selection NLL最小、再目标logp；report score不参与选择。
- R1最终k1/4/16/64，共用总体期望NLL攻击，分别cal_B阈值；理论虚线为独立单问veto参考，均值NLL+conformal无自动指数定理。
- calibrated keys由config `training_folder`/`calibration_folder`指定；selected smoke已用selected文件夹。
- actual sender必须encode([e,CF(e)])再one(prefix,0)，保持冻结cohort。
- 明确主failure=clean非目标→CF目标且未报警，分母固定组数；附目标且接受、clean-correct条件指标。

## 已记录故障/成本
- LIVE tmp并发冲突已加fcntl锁，一次overfit失败；freeze字段files/sha256兼容修复。
- R2_33孤儿run无终态，原metrics不改；actual GPUh未知，data/M8b_v1/orphan_cost_upper.json保守加10.1分钟，runtime total_cost计入。同设置GPU5重启已完成。
- fp16匹配率float.mean假阴性(全部相同仍0.99999994)已改整数计数；重放v2全部通过。R2 jitter长度inplace破坏梯度问题在test前修好并加raw梯度单测。
- 当前GPUh需运行total_cost实时核实（此前约3–4，后续训练累计）；磁盘约40GiB可用，reserve=0且暂停。不得恢复配重。

## 剩余必要工作
1. 等selected_smoke完成，检查replay与实际步数/吞吐，核实三seed读取QA和cal。
2. 按测量预算锁正式矩阵，优先保留test500（三seed/4设计/E1/E2/E3）；eps .01/.03/.1/.3，50/200，λ .1/1/10是初始计划，可按val估算在test前缩矩阵并明示。user M8b没强制λ网格，但要足够强的自适应评估。预算留余量，不能越25。
3. 生成final_*.yaml、lock config，scripts.m8b_lock（需projected_gpu_hours参数、smoke marker），final_lock前不test。锁包括所有m8b模块/模型/校准/核心configs，之后不随意改锁中代码。
4. scripts.m8b_evaluate final val/test，scripts.m8b_overhead val20；建议tmux detached，普通exec有生命周期风险。可实时资源允许时2GPU分part，成本加总。
5. scripts.m8b_statistics（已写，尚未运行）：config evaluation_prefix='full_'、analysis_name，匹配full_reader_val_part*.json/full_test_part*.json。需在正式前用val代理验证统计程序（不可用test smoke）。可能需新增版本修统计bug，不能改已锁核心m8b模块。
6. 并列reader_val/test；正式valPGD数量按预算锁（训练/QA val200，可另跑PGD val20并清楚标规模）。QA额外200离线能用已有source，无需Qwen。
7. 完整详细报告/CSV/曲线/overhead/cost；review≤8000 UTF8字节、固定(a)–(f)、≤3表、3位有效数字。回答哪设计降低白盒失效最多及代价，若差异无证据须如实说。
8. 上游最终哈希复核、完整性检查、冻结data/M8b_v1/freeze.json，更新AGENTS/STATUS/LIVE/EXPERIMENTS/BLOCKERS/CHANGELOG。新日志每条≤10行，原记录不覆盖。

## 2026-11-09 最新覆盖信息
- 原全部训练/校准及R2/R3 warm v2完成；val整体规则选择两者v2。selected_training已生成。累计约6.7GPUh（实时再查）。
- 主比较已统一所有设计A1/A4各半，matched_questions=true；R1按此重定阈值。旧M8a四任务仅历史参考。
- selected全模型smoke20曾764秒通过；去掉50/200早停重复report、共享R1/R3确定性特征后约416秒。
- 一次试图仅计算R2抽中的8层产生数值差，已回退为计算36层充分统计再选8层。当前严格重放仍要求相同设备下逐位一致。
- 进一步发现CPU/GPU随机边界计算会有1ulp差；SourceBank.refresh新增device参数，R2新校准在GPU计算，输出selected_calibration_v3。R1/R3/R4数值不变复用。不要使用旧selected_calibration作最终R2。
- tmux clr-m8b-gpu-v3：GPU7重做R2 cal_A/B3seed → gpu_matched_smoke_v3（20组、严格same-device replay）。
- tmux clr-m8b-prelock-val：等待新smoke后，执行final_reader_val.yaml（val20、4eps、λ1、E1/E2/E3、3seed四设计、50/200）。此时尚未final lock，仍不test。
- tmux clr-m8b-prelock-sensitivity：等上述val后，执行prelock_val_sensitivity.yaml（val20、ε.03、E3 λ.1/10）。用于更精确估算test成本，不能用test选设置。
- QA200已单独scripts.m8b_qa/m8b_qa_r1跑了（需核实reader_val_QA200*.json和metrics）；新设计全部clean+CF，主QA表只筛clean，R1只clean。
- final_*.yaml已预建，尚未锁、尚未test。拟主test500四epsλ1 + ε.03 λ.1/10额外敏感性500；须按两份val实测（×25）+余量检查能否≤25，若不够，在test前明确删低优先敏感性，不悄悄改设置。
- scripts/m8b_lock.py可用，final_lock.yaml目前projected_gpu_hours=16.5只是初始估计，执行前须用实际val修正。
- final_pipeline.yaml目前含val任务（val即将已跑，锁前需移除此重复任务）、test主、test敏感性、overhead。不要重复打开同一x日志/marker。
- stats val smoke已通过（analysis_smoke_v1），全test统计尚未执行。主stats pattern只读full_test_part*，不会自动读full_test_sensitivity_part*；敏感性另版分析或将markers纳入独立combined analysis，按selection_nll而非report score选候选。
- 所有锁中m8b模块在test后不能随意编辑，报告/统计若需修正新增版本脚本。最终review/冻结/上游复核仍未完成。

## 2026-11-09 接续补充
严格GPU同路径smoke20完成432秒、四设计fp16逐位一致；QA200全三假值补报完成，7项单测通过。正式val20四ε矩阵运行，之后仍有λ敏感性val20用于预算锁。overhead已在GPU5完成（val20），不应重跑。
已预建三分区final_test_part0/1/2与final_sensitivity_part0/1/2（GPU7/6/5），尚未执行；最终是否全部运行由完整val实测预算决定。所有分区均以groups=500固定母集、index%3分派，不缩ID集合。
新增scripts/m8b_sensitivity_v1.py与m8b_finalize_v1.py及相应YAML；尚未运行，需先val统计smoke。最终final_lock仍不存在、test未打开。

## 2026-11-09 final lock之后
final_lock.json已写，test已打开。核心代码/模型/校准/final YAML/docs/M8b_TASK.md均锁定，禁止直接修改锁中内容。
val20主1552秒、敏感性810秒；含20%余量完整敏感性500会预计27.4GPUh，故额外λ敏感性预先缩为102组，主500完全保留。预算锁data/M8b_v1/budget_lock.json，ID文件sensitivity_ids.json。总含余量预计22.0GPUh。
主test三分区分母3；额外敏感性使用分区0/1/2、分母15，合计102组，GPU7/6/5。tmux clr-m8b-final-p0/p1/p2已启动；clr-m8b-final-finish等待六个marker后调用m8b_finalize_v1。
val主统计与lambda统计smoke均通过，analysis_val_prelock_v1及sensitivity_val_smoke_v1。finalizer将生成main analysis_v1、sensitivity_v1、上游复核、review、freeze及最终hash。尚无完成结论。
