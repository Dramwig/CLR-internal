# CalReader 项目约定

## 目标与状态

通过类型化问题读取潜在推理状态与潜通信消息，输出校准概率和 conformal 预测集，用于过程价值估计与语义审计。

提案见 [docs/proposal.md](docs/proposal.md)。当前 M0.1 已获用户审查通过；M1 v3 已获用户审查通过；M2–M3合并里程碑已完成，待用户审查；未训练CLR。最新接续状态见 [docs/STATUS.md](docs/STATUS.md)。提案中的参考文献、新颖性、理论保证和预期效果不自动视为已验证实验结论。

## 工作约定

- 遵循 [工作区规则](/home/me/AGENTS.md) 与 [研究项目规则](/home/me/PaperField/AGENTS.md)，沿用本项目目录。
- 研究代码放 `clr/`；实验配置放 `configs/`，入口放 `scripts/`，关键行为测试放 `tests/`。
- 不编造实验结果。每次运行在 `runs/<run_id>/` 保存配置、代码版本、环境、种子、资源版本、日志与 `metrics.json`；汇总结果须可追溯。
- 固定数据划分，调参使用验证集，测试集用于最终评估；评测修正和影响写入变更记录。
- 训练、验证、校准和测试数据分别管理；conformal 覆盖率须说明可交换性等假设。跨模型迁移、分布偏移和自适应攻击下的表现均待验证。
- 优先复用共享数据和权重；项目派生数据、日志和 checkpoint 保留在本项目内。大型下载或训练前检查磁盘与 GPU。
- 环境和依赖版本在实现时确定并记录；不直接照搬 LatDefer 的环境验收状态或实验结果。
- 阻塞和必要的替代方案记录于 `docs/BLOCKERS.md`，运行记录见 `docs/EXPERIMENTS.md`，变更见 `docs/CHANGELOG.md`。

## 历史 GPU 资源限制（2026-10-29，已到期）

- 即日起至北京时间 2026-10-29 18:06（UTC+08:00），4090 服务器物理 GPU 2（UUID `GPU-88350653-3384-889c-17b1-94083fb3c657`）留给他人；本项目不得在其上运行训练、推理、评测或 smoke，包括经 `CUDA_VISIBLE_DEVICES` 重映射后的逻辑设备。不得自动回退到该卡。
- 当前 CODI-GPT2 完整噪声实验继续使用物理 GPU 3，不更改研究协议。截止后本限制失效；新任务仍须实时检查资源。

## 状态持久化与当前接续点（2026-11-03）

- 用户要求所有状态更新落入本文件及 `docs/`：本文件维护简要当前状态，`docs/STATUS.md` 维护完整接续索引；运行、阻塞、变更分别追加到 `docs/EXPERIMENTS.md`、`docs/BLOCKERS.md`、`docs/CHANGELOG.md`。不要只在聊天中报告。
- M1 最终产物：`data/dumps/codi_gpt2_v3`，23,805题；配置 `configs/M1_data_v3.yaml`；冻结清单 `reports/M1_completed_v3_manifest.json`。生成和报告均完成，用户已审查通过。原v1/v2保留，不重启已完成任务，不覆盖冻结结果。
- M1已知限制：中断部分未保存量化误差聚合；reader_train/test恢复后的误差汇总只覆盖各自新生成部分。状态受原cohort左padding影响，后续继续沿用同一上下文。
- CODI-LLaMA复现已按现状获用户接受，噪声延至M6前；Huginn、Coconut暂停。M2–M3已获授权并开始；M4及后续尚未执行。
- 不以低磁盘余量主动终止生成：按授权通过 `/home/me/disk-reserve/reserve.sh release <整数GiB>` 提前释放配重，并在使用期间保持自动填充暂停；真实I/O错误必须记录，不静默降级。

## M2–M3全局约束（2026-11-03起）

- reader及所有基线（含B5）不得看到源问题文本；仅观测状态/前缀、类型化模板与选项。读取states，不读取加噪feedback。
- 所有指标并列reader_val与test；Q1另分clean/traj。Q1干净标签只来自轨迹起点，每条轨迹独立Bernoulli；不标更早前缀；t=T分别用clean/traj的确定正误。报告每步缺标签比例。
- cal_A250温度缩放、cal_B250 Mondrian，冻结文件 `data/splits/gsm8k_M23_cal_v1.json`；不训练CLR，不生成额外轨迹。
- 当前配置 `configs/M23.yaml`，QA输出 `data/M23_v1a`。首轮QA smoke干扰项不足失败已保留，修正使用固定训练集答案池后通过；全量QA、标签审计与基线smoke已通过；完整三seed基线和B5a已完成，结果待审查。
- 源调用例外按明确子任务执行：B5a、C官方500题、D≤5GPU小时侦察、E量化复算。B5b可选推迟；其他源仍暂停。


<!-- M23_LIVE -->
M2–M3已完成，待用户审查；未训练CLR。冻结索引 data/M23_v1a/freeze.json。
<!-- /M23_LIVE -->

2026-11-03 M23收尾验证：全套单元测试38 passed、1 skipped（reports/M23_tests.log）。官方500题79.8%，同题M1为79.4%；SVAMP993题43.001%，ASDiv1000题54.9%，两池完整评估合计0.410463 GPU小时。量化抽查200题的重算fp32转fp16与保存值完全一致。完整三seed基线与B5a仍在运行；不宣称里程碑已完成。

M23标签覆盖/域诊断：reader_train t0–5的clean Q1缺标签比例22.99%–23.675%，reader_val22.15%–25.2%，未补样本。域分类器各t三seed均值AUROC约0.729–0.783；含cohort/padding/长度混杂，不能独立解释为记忆化。完整基线/B5a仍运行。

2026-11-03完整运行进度（当时快照，最终状态见M23_LIVE）：{"baseline_run": {"baseline": "B2", "qtype": "Q1", "seed": 22}, "patchscope_run": {"qtype": "Q2", "split": "test"}}。收尾检查要求51个checkpoint、216份基线预测、16份B5a预测及QA哈希齐全后才冻结。

2026-11-03 M23：B0/B1全部三seed完成，B2运行中；正在用require_complete=false验证报告生成路径，输出仍明确标为部分报告。最终完整报告只在所有必需run完成后生成。

2026-11-03报告路径验证通过（run 20261103T051328880380Z_1dab4bbeb533_eff0ec，明确milestone_complete=false），未见题池早退图已视觉核查；SVAMP最终接近现有test、ASDiv更容易，均低于源训练分布。仅侦察，无额外训练轨迹。

2026-11-03收尾完整性：M1冻结清单6个协议/代码/报告文件SHA全部一致；M23未覆盖这些冻结文件。完整基线和B5a继续运行，磁盘监控可按授权释放配重，不因低余量主动停跑。

2026-11-03补充关键测试：Q1聚合K16目标与逐Bernoulli展开的NLL/Brier/ECE/accuracy一致性；B5a prompt排除原题/gold解答。完整测试结果见reports/M23_tests.log，B2推进至Q3，B5a仍在Q3读取。

2026-11-03完整运行：B2全部三seed完成，进入B3；新增测试后全套40 passed、1 skipped。训练配置/划分/seed保持冻结，无基于test的调整。

2026-11-03已完成B0/B2预汇总（clean、seen、t等权、三seed均值；其余基线仍运行）：B2 Q3 reader_val约58.4%、test53.4%，Q4两组约50%；B0 Q3 test35.9%高于25%随机，表明选项先验可被利用。仅诊断，不据此修改后续已冻结训练设置。最终95% CI见完整报告。

2026-11-03：B3已完成Q1/Q2，Q3继续；B5a的Q3较大状态集继续，无错误。冻结范围已覆盖候选池dump、来源材料、报告计算产物；完成状态仍须等待所有必需run和产物核验。

2026-11-03：Q4线性结构、Q1标签精度差异、conformal依赖、域诊断混杂、候选池排除及历史成本缺口已集中写入docs/BLOCKERS.md的M2–M3已知限制节。完整运行继续。

2026-11-03 B3三seed完成：clean/seen/t等权均值，Q3 reader_val65.65%、test58.59%；Q4 reader_val71.42%、test68.26%。Q4联合MLP优于线性探针，与状态-数值交互作用相符；不将其当作CLR结论。B4/B5a继续。完整CI由报告生成。

2026-11-03 B0–B4完整运行完成：run 20261103T045628621237Z_2964c31f2153_11a348，51个checkpoint、1860行指标、1008行test conformal结果，0.534301 GPU小时、源前向0。B5a仍运行；正在验证包含完整基线的报告/reader_val conformal汇总，尚未宣布整个里程碑完成。

2026-11-03 B5a吞吐预检：旧batch8完整运行继续保留；新v2计划在固定reader_val50题面板比较8/64，预设max概率差≤1e-4且argmax一致100%才采用64。使用物理GPU3（检查free5484MiB），不改变模型、fp32、prompt/指标，不使用test选择批大小。

2026-11-03 B5a batch门槛通过：run 20261103T053137666964Z_b807ffac4af3_0d5c8e；四类各128条固定reader_val实例，答案一致100%，max概率差2.27094e-5<1e-4；max原始logit差0.00120163。batch64 smoke运行中，旧batch8完整运行继续，未替换最终结果。导入顺序失败已解决且记录，未模型/精度降级。

2026-11-03完整基线报告路径验证通过：run 20261103T052935319542Z_1dab4bbeb533_484ebc，仍milestone_complete=false（缺B5a）。reader_val/test conformal汇总与可靠性图已生成并视觉核查Q1图。batch64虽数值门槛通过，是否采用仍取决于smoke吞吐，不改模型选择/指标；旧batch8继续运行。

2026-11-03 B5a Q3 reader_val已完成（source_forward_calls累计69253）；batch64 smoke完成但耗时78.75秒，原batch8为55.72秒，未见吞吐收益（不同共享GPU，不能归因为batch本身），因此不采用v2替换完整run。保留gate及smoke结果、额外成本；最终协议仍fp32/batch8。

M23审计：早期smoke触及10题test及中止run部分test预测，未用于调参，但违反严格test-only-final约定；详见docs/BLOCKERS.md。后续smoke必须用val代理，不再触及test。α=.1的B2/B3平均test覆盖约.89–.92，不声称所有层达标。B5a继续。

2026-11-03：B5a Q3两组均已完成，累计71892次源前向、1.12089 GPU小时，进入最后Q4。其余计算完成。最终交付标为完成待审查；smoke测试集提前访问的偏离未获用户审查接受，不能写作协议无偏离。

2026-11-03层分析图视觉核查通过：仅1000题源训练分布，13个观测层（含embedding）、三seed95%区间；不据此选择未见test最优层。B5a Q4继续，最终冻结同时保留所有已关闭的M23实验日志（监控自身活动日志除外）。

2026-11-03 B5a已完成的Q3结果：test clean/seen/t等权准确率0.2455，接近四选一随机；记录为当前零样本注入方式的负面结果，不换prompt/模型，不据test调参。Q4继续运行。

2026-11-03汇总核验：conformal 2016行，覆盖reader_val/test、α=.05/.1/.2、两类模板，覆盖率/集合大小全部范围合法。Q4边际匹配仅保证原始split（含整体cal），cal_A/B抽样后未独立重配，不保证半集精确边际相同，报告已澄清。

2026-11-03测试审计：40 passed、1 skipped；跳过来自tests/test_m0b_huginn_eval.py的pytest.importorskip(lm_eval)，环境find_spec确认该可选依赖缺失。Huginn已按用户暂停，M23测试全部通过。B5a Q4继续，无新运行错误。

2026-11-03：B5a Q4 reader_val完成，累计82802次前向，1.286236 GPU小时；仅余Q4 test，预测文件14/16。B0–B4 checkpoint51/51、预测216/216齐全。最终报告待最后两份B5a预测生成。

2026-11-03所有必需实验完成：B5a run 20261103T043008727615Z_3d1392832851_85503d completed，88010次源前向，1.36440353 GPU小时。最后完整报告/按题bootstrap/冻结核验进行中；未训练CLR。


M2–M3问题库、B0–B5a、层/域诊断、官方500题、两候选未见池与量化抽查均完成，待用户审查。B5b推迟M6，未训练CLR。报告：reports/M2_questions.md、M3_baselines.md、M3_shift.md、M3_unseen_pools.md；冻结索引 data/M23_v1a/freeze.json。

2026-11-03 B5a最终Q4 test clean/seen/t等权准确率0.503744，接近随机。所有源模型计算结束，最终报告bootstrap/汇总进行中，不再生成数据或训练模型。


## M2–M3最终核验（2026-11-03）

实现、全部必需实验、四份报告及图已完成，待用户审查；未训练CLR，B5b按任务许可推迟M6。Q1按用户确认的起点/后续观测状态Bernoulli规则执行。
冻结清单 `data/M23_v1a/freeze.json` 的635个文件SHA全部一致；核验记录 `data/M23_v1a/verification.json`。51个基线checkpoint、216份基线预测、16份B5a预测齐全。测试40 passed；1个已暂停Huginn模块因可选lm_eval缺失跳过。最终报告run `runs/20261103T055223163924Z_faee26776ad3_cea0d0`。
已记录成本下界2.6075 GPU小时、202641次源前向，历史中断缺失成本未记成零。D两池含smoke合计0.417348 GPU小时，低于5小时。
必须随结果保留的审查限制：早期smoke触及10题test，旧中止run保存过部分test预测，未用于调参但不符合严格test-only-final约定；不得声称该偏离已获接受。Q4边际匹配在原split保证，cal_A/B半集没有独立重配。conformal依赖假设、来源未核实候选池及成本缺口见BLOCKERS；不更换划分/seed掩盖问题。
B3 test Q3准确率58.59% ± 0.19pp，Q4 68.26% ± 1.09pp（三seed95% t区间、clean/seen/t等权）；B5a Q3约24.55%、Q4约50.37%，如实保留负面结果。M3_baselines.md末尾已逐项回答三项审查问题。
