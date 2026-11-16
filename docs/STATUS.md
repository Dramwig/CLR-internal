# CLR 当前状态与接续索引

最后更新：2026-11-16（北京时间）。此文件按历史追加，最新状态见末尾及各里程碑 LIVE 索引；历史失败、重启及恢复记录保留在 EXPERIMENTS/BLOCKERS/CHANGELOG。

## 2026-11-05 当前 M5a 接续

M5a全量val300/三seed及配对统计完成，待审查；oracle同预算S1增益32.8pp，95%CI[27.8,38]；B4/B3未见正收益。保守累计5.81GPUh≤10，不触碰test。
用户“允许 继续完成”后按10GPUh方案执行：仅S3 oracle限N16/b4全步，其余S1/S2/B4/B3原网格不变。两项研究问题的估计/限制见唯一审查reports/M5a_REVIEW.md。详细reports/M5a_search.md；逐题CSV/图/成本data/M5a_v13/analysis；冻结data/M5a_v13/freeze.json；接续docs/M5a_LIVE.json。专项授权不代表G1通过，完整M5/M6未启动。

## 里程碑

|项目|当前状态|
|---|---|
|M0.1 CODI-GPT2|用户审查通过；接受bf16复放验收与fp32/batch1生产协议区别，G-σ按预设均值规则通过|
|M1 CODI-GPT2数据|全量生成及报告完成，用户已审查通过|
|CODI-LLaMA|复现按现状获接受；噪声留至M6前|
|Huginn / Coconut|按用户指令暂停；Coconut不训练、不使用第三方checkpoint|
|M2–M3|用户已通过G0；冻结索引data/M23_v1a/freeze.json|
|M4|已授权，准备模型与未见数据；当前任务书M4_TASK.md|
|M5–M9|尚未执行|

## M1完成产物

- 数据：`data/dumps/codi_gpt2_v3`；reader_train20,000 / reader_val2,000 / val300 / cal500 / test1,005，共23,805题。包含clean、早退、训练K8轨迹、held-out两种策略K16 MC、50对K64检查、1,000题全层状态。
- 冻结划分：`data/splits/gsm8k_M1_v3.json`；生产协议：`configs/protocol_codi_v2.yaml`；最终恢复配置：`configs/M1_data_v3.yaml`。
- 最终生成run：`runs/20261102T031137055963Z_24b03138a255_a2c9f2`，status=completed。
- 报告run：`runs/20261102T034305056543Z_f31fce96a804_d66753`，status=completed。
- 报告与图：`reports/M1_data.md`、`reports/M1_results.md`、`reports/M1_figures/`。
- 冻结清单：`reports/M1_completed_v3_manifest.json`；数据SHA清单见最终生成run的 `artifacts_sha256.json`。2026-11-03复核770个数据文件、6个冻结文件，无哈希不一致。
- 产物3,248,059,671字节（约3.03GiB）。生成累计22.615373 GPU小时，包含失败/重启重算，不含预检；源前向2,757,716次。预检与失败smoke另见预算run `20261030T022507631329Z_ae2889e005b8_1fe957`，不将生成成本当作整个项目成本。

## 保留的限制和接续规则

1. 原中断阶段未持久化fp32→fp16误差聚合；reader_train/test恢复后的误差统计仅覆盖新生成部分。原写入执行了误差阈值检查，但不能反推缺失的全量聚合。
2. z_t依赖原cohort组成和左padding；clean、早退、rollout、后续搜索必须共享该上下文；manifest记录cohort/padding/协议hash。
3. 官方bf16复放与fp32生产协议的逐题差异获用户明确接受，不能宣称两者完全等价。
4. 原v1/v2失败或重启run的running快照不是实时状态。v3已完成，不重跑；冻结代码/结果有修改需求时新建版本。复用硬链接文件也不得原地编辑。
5. 低磁盘空间时按用户授权释放整数GiB配重，不主动以余量阈值停止。先前小数参数错误已修复；真实I/O失败仍记录。
6. 下一步是M1结果审查及后续任务；不把“生成完成”写成“用户验收通过”，不自行启动reader/probe训练。

## 文档入口

- 项目约定：`../AGENTS.md`
- 实验记录：`EXPERIMENTS.md`（根目录同名文件为入口）
- 阻塞与偏离：`BLOCKERS.md`
- 变更历史：`CHANGELOG.md`
- 原始方案：`proposal.md`；总体规划：`CLR_Codex_Plan.md`


## 2026-11-03 — M1审查通过，M2–M3启动

用户授权合并里程碑，当前不训练CLR。输入隔离源题目文本，状态只用模型实际输出。Q1规则经用户明确：clean仅轨迹起点独立Bernoulli标签及clean终点，traj起点之后用自身观测及终点；不监督更早clean前缀。reader_val报告clean/traj分别指标和各t样本覆盖。cal_A/cal_B250/250已固定。

QA首轮smoke因少量题不足4种答案失败，未覆盖；固定训练集数值池补足后run `20261103T042122100178Z_172c4808908a_81b8b5`通过，输出 `data/M23_v1a/smoke`。全量QA任务tmux `clr-m23-questions`。Q4成对交换正值作为对方负值，保证保留问题上数值边际精确一致；未匹配/解析失败单独统计。全量不得使用失败smoke。

源调用范围冲突按具体要求解释为B5a/C/D/E例外，其他仅读M1；不新增轨迹。GPU当前均忙，显式使用cuda3训练轻量probe、cuda1量化复算，负载与实际OOM如实记录；没有替换模型或降精度。D候选许可和谱系正在核实；尚无完整基线结论或M2–M3完成声明。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "running", "asdiv_run": "running", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "running", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。


<!-- M23_LIVE -->
M2–M3已完成，待用户审查；未训练CLR。冻结索引 data/M23_v1a/freeze.json。
<!-- /M23_LIVE -->

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "running", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

2026-11-03 M23收尾验证：全套单元测试38 passed、1 skipped（reports/M23_tests.log）。官方500题79.8%，同题M1为79.4%；SVAMP993题43.001%，ASDiv1000题54.9%，两池完整评估合计0.410463 GPU小时。量化抽查200题的重算fp32转fp16与保存值完全一致。完整三seed基线与B5a仍在运行；不宣称里程碑已完成。

M23标签覆盖/域诊断：reader_train t0–5的clean Q1缺标签比例22.99%–23.675%，reader_val22.15%–25.2%，未补样本。域分类器各t三seed均值AUROC约0.729–0.783；含cohort/padding/长度混杂，不能独立解释为记忆化。完整基线/B5a仍运行。

2026-11-03完整运行进度（当时快照，最终状态见M23_LIVE）：{"baseline_run": {"baseline": "B2", "qtype": "Q1", "seed": 22}, "patchscope_run": {"qtype": "Q2", "split": "test"}}。收尾检查要求51个checkpoint、216份基线预测、16份B5a预测及QA哈希齐全后才冻结。

2026-11-03 M23：B0/B1全部三seed完成，B2运行中；正在用require_complete=false验证报告生成路径，输出仍明确标为部分报告。最终完整报告只在所有必需run完成后生成。

2026-11-03报告路径验证通过（run 20261103T051328880380Z_1dab4bbeb533_eff0ec，明确milestone_complete=false），未见题池早退图已视觉核查；SVAMP最终接近现有test、ASDiv更容易，均低于源训练分布。仅侦察，无额外训练轨迹。

2026-11-03收尾完整性：M1冻结清单6个协议/代码/报告文件SHA全部一致；M23未覆盖这些冻结文件。完整基线和B5a继续运行，磁盘监控可按授权释放配重，不因低余量主动停跑。

2026-11-03补充关键测试：Q1聚合K16目标与逐Bernoulli展开的NLL/Brier/ECE/accuracy一致性；B5a prompt排除原题/gold解答。完整测试结果见reports/M23_tests.log，B2推进至Q3，B5a仍在Q3读取。

2026-11-03完整运行：B2全部三seed完成，进入B3；新增测试后全套40 passed、1 skipped。训练配置/划分/seed保持冻结，无基于test的调整。

2026-11-03已完成B0/B2预汇总（clean、seen、t等权、三seed均值；其余基线仍运行）：B2 Q3 reader_val约58.4%、test53.4%，Q4两组约50%；B0 Q3 test35.9%高于25%随机，表明选项先验可被利用。仅诊断，不据此修改后续已冻结训练设置。最终95% CI见完整报告。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

2026-11-03：B3已完成Q1/Q2，Q3继续；B5a的Q3较大状态集继续，无错误。冻结范围已覆盖候选池dump、来源材料、报告计算产物；完成状态仍须等待所有必需run和产物核验。

M23运行状态：`{"question_run": "completed", "baseline_run": "running", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

2026-11-03：Q4线性结构、Q1标签精度差异、conformal依赖、域诊断混杂、候选池排除及历史成本缺口已集中写入docs/BLOCKERS.md的M2–M3已知限制节。完整运行继续。

2026-11-03 B3三seed完成：clean/seen/t等权均值，Q3 reader_val65.65%、test58.59%；Q4 reader_val71.42%、test68.26%。Q4联合MLP优于线性探针，与状态-数值交互作用相符；不将其当作CLR结论。B4/B5a继续。完整CI由报告生成。

M23运行状态：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

2026-11-03 B0–B4完整运行完成：run 20261103T045628621237Z_2964c31f2153_11a348，51个checkpoint、1860行指标、1008行test conformal结果，0.534301 GPU小时、源前向0。B5a仍运行；正在验证包含完整基线的报告/reader_val conformal汇总，尚未宣布整个里程碑完成。

2026-11-03 B5a吞吐预检：旧batch8完整运行继续保留；新v2计划在固定reader_val50题面板比较8/64，预设max概率差≤1e-4且argmax一致100%才采用64。使用物理GPU3（检查free5484MiB），不改变模型、fp32、prompt/指标，不使用test选择批大小。

2026-11-03 B5a batch门槛通过：run 20261103T053137666964Z_b807ffac4af3_0d5c8e；四类各128条固定reader_val实例，答案一致100%，max概率差2.27094e-5<1e-4；max原始logit差0.00120163。batch64 smoke运行中，旧batch8完整运行继续，未替换最终结果。导入顺序失败已解决且记录，未模型/精度降级。

2026-11-03完整基线报告路径验证通过：run 20261103T052935319542Z_1dab4bbeb533_484ebc，仍milestone_complete=false（缺B5a）。reader_val/test conformal汇总与可靠性图已生成并视觉核查Q1图。batch64虽数值门槛通过，是否采用仍取决于smoke吞吐，不改模型选择/指标；旧batch8继续运行。

2026-11-03 B5a Q3 reader_val已完成（source_forward_calls累计69253）；batch64 smoke完成但耗时78.75秒，原batch8为55.72秒，未见吞吐收益（不同共享GPU，不能归因为batch本身），因此不采用v2替换完整run。保留gate及smoke结果、额外成本；最终协议仍fp32/batch8。

M23运行状态：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

M23审计：早期smoke触及10题test及中止run部分test预测，未用于调参，但违反严格test-only-final约定；详见docs/BLOCKERS.md。后续smoke必须用val代理，不再触及test。α=.1的B2/B3平均test覆盖约.89–.92，不声称所有层达标。B5a继续。

2026-11-03：B5a Q3两组均已完成，累计71892次源前向、1.12089 GPU小时，进入最后Q4。其余计算完成。最终交付标为完成待审查；smoke测试集提前访问的偏离未获用户审查接受，不能写作协议无偏离。

2026-11-03层分析图视觉核查通过：仅1000题源训练分布，13个观测层（含embedding）、三seed95%区间；不据此选择未见test最优层。B5a Q4继续，最终冻结同时保留所有已关闭的M23实验日志（监控自身活动日志除外）。

M23运行状态：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "running", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

2026-11-03 B5a已完成的Q3结果：test clean/seen/t等权准确率0.2455，接近四选一随机；记录为当前零样本注入方式的负面结果，不换prompt/模型，不据test调参。Q4继续运行。

2026-11-03汇总核验：conformal 2016行，覆盖reader_val/test、α=.05/.1/.2、两类模板，覆盖率/集合大小全部范围合法。Q4边际匹配仅保证原始split（含整体cal），cal_A/B抽样后未独立重配，不保证半集精确边际相同，报告已澄清。

2026-11-03测试审计：40 passed、1 skipped；跳过来自tests/test_m0b_huginn_eval.py的pytest.importorskip(lm_eval)，环境find_spec确认该可选依赖缺失。Huginn已按用户暂停，M23测试全部通过。B5a Q4继续，无新运行错误。

2026-11-03：B5a Q4 reader_val完成，累计82802次前向，1.286236 GPU小时；仅余Q4 test，预测文件14/16。B0–B4 checkpoint51/51、预测216/216齐全。最终报告待最后两份B5a预测生成。

M23运行状态：`{"question_run": "completed", "baseline_run": "completed", "patchscope_run": "completed", "layers_run": "completed", "quant_run": "completed", "official_run": "completed", "svamp_run": "completed", "asdiv_run": "completed", "audit_run": "completed"}`；详见 `data/M23_v1a/pipeline_status.json`。

2026-11-03所有必需实验完成：B5a run 20261103T043008727615Z_3d1392832851_85503d completed，88010次源前向，1.36440353 GPU小时。最后完整报告/按题bootstrap/冻结核验进行中；未训练CLR。


M2–M3问题库、B0–B5a、层/域诊断、官方500题、两候选未见池与量化抽查均完成，待用户审查。B5b推迟M6，未训练CLR。报告：reports/M2_questions.md、M3_baselines.md、M3_shift.md、M3_unseen_pools.md；冻结索引 data/M23_v1a/freeze.json。

2026-11-03 B5a最终Q4 test clean/seen/t等权准确率0.503744，接近随机。所有源模型计算结束，最终报告bootstrap/汇总进行中，不再生成数据或训练模型。


## M2–M3最终核验（2026-11-03）

实现、全部必需实验、四份报告及图已完成，待用户审查；未训练CLR，B5b按任务许可推迟M6。Q1按用户确认的起点/后续观测状态Bernoulli规则执行。
冻结清单 `data/M23_v1a/freeze.json` 的635个文件SHA全部一致；核验记录 `data/M23_v1a/verification.json`。51个基线checkpoint、216份基线预测、16份B5a预测齐全。测试40 passed；1个已暂停Huginn模块因可选lm_eval缺失跳过。最终报告run `runs/20261103T055223163924Z_faee26776ad3_cea0d0`。
已记录成本下界2.6075 GPU小时、202641次源前向，历史中断缺失成本未记成零。D两池含smoke合计0.417348 GPU小时，低于5小时。
必须随结果保留的审查限制：早期smoke触及10题test，旧中止run保存过部分test预测，未用于调参但不符合严格test-only-final约定；不得声称该偏离已获接受。Q4边际匹配在原split保证，cal_A/B半集没有独立重配。conformal依赖假设、来源未核实候选池及成本缺口见BLOCKERS；不更换划分/seed掩盖问题。
B3 test Q3准确率58.59% ± 0.19pp，Q4 68.26% ± 1.09pp（三seed95% t区间、clean/seen/t等权）；B5a Q3约24.55%、Q4约50.37%，如实保留负面结果。M3_baselines.md末尾已逐项回答三项审查问题。

## 2026-11-03 — 全局约定持久化

新增 [GLOBAL_RULES.md](GLOBAL_RULES.md)，完整保存十条工程/审查约定；AGENTS.md 强制索引，适用于每个新对话。
从 M4 起唯一审查文件 ≤ 8KB、固定六段、≤ 3 张主表、3 位有效数字；实验/阻塞新条目每条 ≤ 10 行。
smoke/调试仅用 val 代理，不触碰 test；旧 AGENTS 原样归档，提案指定路径链接到现有 proposal.md。
本次仅修改文档与索引，未启动实验，未修改冻结代码/结果；M2–M3 仍待审查。

## 2026-11-03 M4启动

用户通过G0并授权M4，当前协议见[M4_TASK.md](M4_TASK.md)。encoder选择官方ModernBERT-base；只在val做smoke/调试。GPU全部共享繁忙，源计划cuda:2、reader计划cuda:3，加载前再检查。已申请释放40GiB，但实测配重已为0，本次未获得新增空间；当前约15.6GiB可用，保持自动补足暂停并继续工作。M4尚无结果。

M4当前：源val smoke12题通过，完整D估算0.451 GPUh并已启动；CLR smoke通过（26.6训练样本/s、93.8评估样本/s、2.42GiB峰值、置换差0）。训练预算据此设4epoch、每qtype每epoch10000样本，优先四项必做，详见M4_TASK.md及后续预算报告。未见池172条13-gram跨划分重叠另作无重叠子集诊断，不改冻结划分。

M4接续：reader模型/训练/仿射校准最小测试6通过；k4试验tmux `clr-m4-reader`，D `clr-m4-unseen`。预算冻结 `reports/M4_budget.md`；训练控制器将仅用reader_val选k，所有训练完成后再解锁最终评估。

M4更新：校准/阈值测试共8通过；控制器running状态误判已修复，原训练未中断。D完成1625/1993（当时快照）。最终评估入口要求训练checkpoint hash锁及val-only校准评估smoke；尚未运行test。

M4评估smoke发现极小val代理层的conformal无限阈值不能写严格JSON；修复为null+infinite标志（内部保持∞/全集合），新增边界测试，原失败run保留。此修复未读取test。

M4：9项测试通过，val代理校准评估smoke v2通过 `runs/20261103T065541614647Z_cf0498aecef6_cfafa6`；无限阈值以null+标志落盘，内部全集合不变。B3对照有效batch32与CLR一致，此调整在相关训练启动前冻结；控制器重载，正在运行的pilot未改。

M4 D完成：`runs/20261103T063500842769Z_924ffcfc4ca2_e9f41f`，1993题、0.419145GPUh、117130前向；QA `runs/20261103T070034793210Z_f4a64da4e26f_7aefa4`，审计 `runs/20261103T070131960428Z_158d311376b0_8ad0ee`。训练996题98964行QA，评估997题99603行（含heldout）；Q4解析失败259题、额外负例匹配失败2题。Q1标注/选项去重/Q4边际完全相同/cohort协议审计通过。
D结束后物理GPU2用于k8；k4继续物理3，均原模型未降级。后续seed11/22/33固定GPU1/2/3并行，合计GPU小时仍按作业求和。控制器 `clr-m4-suite` 日志 `reports/M4_training_suite_v4.log`，试验 `clr-m4-reader`/`clr-m4-pilot8`。尚未打开M4最终test。

M4 k8首次预检查退出（非模型OOM）；新case pilot_k8_preflight_v2门槛3.5GiB，模型/batch等保持，旧run不覆盖。控制器改读显式pilot case映射。

M4接续：测试10通过；k4约33600/40000、k8约10400/40000（当时快照）。自动接续已部署：训练锁→最终评估→报告→测试/冻结；每个完成事件写 `docs/M4_LIVE.json`，AGENTS已有索引。未见池D已完成；正式test仍未打开。

M4更新：k4已完成40000样本训练，进入完整reader_val；k8继续。增加观测轨迹前缀拼接及Bernoulli conformal覆盖率测试，总12项；D审计另核对冻结split精确ID集合。磁盘当前约35GiB（外部余量变化，非本任务声称释放）。

M4待完成审查摘要已写 `reports/M4_REVIEW.md`（明确未完成、不填虚构结果），D完整数据报告 `reports/M4_data.md`。自动最终报告会在锁定评估完成后更新当前未冻结摘要。k4完整验证中，k8接近结束训练。

M4：k4/k8均完成40000样本训练、现完整reader_val评估中；共享MLP置换测试另通过，总13项。当前待完成审查摘要已同步测试数；实际最终结果尚未产生。

M4 k4 pilot完成：reader_val宏qtype原始NLL0.769895，Q1/Q2/Q3/Q4=0.421176/0.559212/1.363840/0.735353。Q3接近均匀四选一NLL1.386，Q4差于常数0.5的0.693；保留负面结果，尚非最终校准比较，不改变已冻结训练预算。k8验证中。

M4预算v2（正式训练尚未启动/test未打开）：完整k4验证0.246GPUh显著快于smoke估计，采用 `configs/M4_train_v1.yaml`，10epoch/每2epoch验证/patience2，保留旧4epoch方案；约76GPUh含20%裕量，优先必做四项。控制器新日志 `reports/M4_training_suite_v8.log`。

M4：k锁定4（val0.769895 vs k8的0.795120）；seed22/33已正式训练，seed11 GPU1预检查失败（未训练）。先在GPU1以相同模型/batch/精度、显式3.0GiB预检查重做val smoke验证实际容纳能力；不静默换卡或缩模型。新增低成本B3_matched诊断在test之前锁定。

GPU1同设置smoke通过（0.01956GPUh，峰值2.41778GiB）；以M4_train_v2配置min_free3.0显式重启main seed11。新控制器session `clr-m4-suite-v2`，manifest `data/M4_v0/training_suite_v2.json`；旧控制器仍等待其seed22/33子进程完成，禁止kill旧session以免中断训练。旧controller因首次seed11失败最终会退出，属已知已恢复事件；新controller从现存run接续不重跑22/33。pipeline改监视新session，并在最终评估前补B3_matched锁。

M4：14项单元测试通过（reports/M4_tests_v6.log）。三个main seed现均实际训练中；11恢复run `runs/20261103T074705242247Z_5aff9999085b_a3ec49`，22 `runs/20261103T073907451420Z_899a42fad462_cc925a`，33 `runs/20261103T073906424755Z_51393c3ef0a3_649863`。新pipeline日志 `reports/M4_pipeline_v2.log`；旧controller不杀，其失败退出已由新controller接管。匹配曝光B3按CLR最佳epoch、相同采样顺序训练，保留原始B3作为主参考。

M4最新核实：main seed11/22/33都在运行，分别已训练24800/40800/37600个样本（22进入epoch2，其他epoch1）；无M4 test访问。新增混合来源reader前缀批处理只合并已保存观测状态，维持行/标签顺序，减少增强训练的双forward开销；测试总15项。源模型cohort/rollout协议未改。后台接续覆盖原始B3、匹配曝光B3、B3-multi、迁移、增强、校准、最终评估、报告与冻结。

## M4 时间预估更新（2026-11-03 22:50，北京时间）
已核实主模型3seed完成（耗时3.67/3.70/5.15 GPUh），B3-multi 3seed完成；留一类型预训练3seed处于epoch5–6，已运行约1.89小时。控制器与pipeline均存活，最终test尚未评估。累计约20.2 GPUh。
剩余墙钟估计：留一类型预训练0.5–2.5小时；小样本迁移/对照约2–4小时；增强训练约4–6小时；匹配曝光对照、最终校准评估与汇总约2–4小时。考虑阶段重叠和早停，整体预计再需10–16小时，即11月4日09:00–15:00；不是保证完成时间，共享GPU吞吐/错误可能改变范围。不调整实验设置或80GPUh上限。
证据：data/M4_v0/training_suite_v2.json、docs/M4_LIVE.json、当前三个leave_q4_out run的progress.json；该估算按墙钟时间，不把GPU小时直接当经过时间。

## M4 实时核实（2026-11-04 09:05，北京时间）
M4（11月4日09:05北京时间）：原训练队列31/32完成；主模型、B3-multi、留一类型预训练及N=100/1000/10000迁移/从零B3三seed均完成。增强训练seed11/22完成，seed33已结束epoch10训练、正在最后Q4验证。累计约45.7 GPUh/80。控制器与接续pipeline存活；接下来匹配曝光B3、校准/最终评估、统计报告与冻结。预计还需3–6小时（今日12:00–15:00，视共享GPU及汇总耗时）；尚无最终test/G1结论。
证据：training_suite_v2.json完成31项；augmented_33 run `runs/20261103T195318687671Z_9ba0fc3828b4_a91323` 最终验证Q4已处理16000/27944；预算工具汇总45.659GPUh。最终训练锁及评估尚未生成，不以训练完成比例代表整个里程碑完成比例。

M4 v0实验完成，待G1审查。唯一审查文件 `reports/M4_REVIEW.md`；完整报告 `reports/M4_clr_v0.md`；冻结 `data/M4_v0/freeze.json`。M1/M23冻结哈希复核无变化，测试通过；不自动启动M5。

## 2026-11-05 M4.1启动
用户暂缓G1判定，授权A诊断→B两变体修复→C主比较。任务书M4.1_TASK.md；M4结果不改，诊断不访问test。首先核实信息通路与原训练曝光，根因尚未确定。

M4.1初诊：80批FP32梯度/干预完成（0.02095GPUh，test_opened=false），每层cross可见性测试通过；通路非完全断开。原CLR最佳checkpoint各类型4/4/6万曝光，B3 Q3为430/516/516万、Q4各224万。B3的val300 1k实例过拟合达98%，v0及合成任务继续。pending审查文件 reports/M4.1_REVIEW.md；根因尚未定稿。

M4.1合成基准控制：保留原16值/RMS1/std0.08及后续低噪声、4值控制失败记录；最终单位L2线性编码、16值、std0.001的同一任务上B3 Q3/Q4均达98%独立验证门槛（索引data/M41_v1/synthetic_unit）。v0正在此同协议上核验；不能把B3也未达标的任务当作CLR架构失败证据。
候选实现已新增clr/models/clr_v1.py（尚未宣布修复成功）：v1保留22层encoder，增加非零cross残差尺度与显式状态/数值/选项联合非线性头；lite为4层256。置换、padding梯度和选项隔离2项测试通过；val吞吐实测v1 batch256约647–1323条/s，lite batch1024约2.07万–2.37万条/s（不含数据加载，正式预算须保留余量）。未进入C。

M4.1接续已部署：tmux clr-m41-gates（A完成→根因→B门槛），clr-m41-pipeline（双门槛→val smoke→C→最终报告）。候选路径/隔离/置换与按题统计单测4通过。旧v0原配置1k Q4训练100epoch为81.2%，降低LR续训正在提高；单位L2合成Q3原v0已过98%，Q4仍运行，不能概括为架构完全不能学。当前仍A，未访问test。

M4.1 A诊断完成，根因结论已写reports/M4.1_REVIEW.md(b)，完整证据data/M41_v1/diagnosis_complete.json。继续B两候选学习能力关卡；尚未进入C。

M4.1 A根因已落盘（≤3KB）：无完全断路证据；v0单位L2合成Q3/Q4验证98.6%/99.6%，原配置1k Q4拟合81.2%，低LR续训97.0%，曝光差37–129倍。B首次运行出现GPU2共享显存OOM及lite日志器AttributeError，已记录并显式恢复到GPU1/5、相同模型/batch/精度。tmux clr-m41-gates-v2，日志reports/M41_gate_controller_retry_v2.log。代码066ba79；C pipeline仍等待双门槛。M4冻结12946文件SHA全部一致。

M4.1 A诊断完成，根因结论已写reports/M4.1_REVIEW.md(b)，完整证据data/M41_v1/diagnosis_complete.json。继续B两候选学习能力关卡；尚未进入C。

M4.1（2026-11-05T10:59:22+08:00）：A根因已落盘；两个候选1k Q4拟合均超过99%，lite合成Q3/Q4通过，v1合成仍在运行；C尚未启动；状态索引docs/M41_LIVE.json。

M4.1 B关卡完成，both_variants_passed=True；见data/M41_v1/gate_lock.json。C只有两者全部通过才可启动。

M4.1（2026-11-05T11:01:05+08:00）：A诊断已完成；v1/lite的1k Q4拟合99.2%/99.3%，合成Q3均100%、Q4为98.9%/99.2%，六项门槛全通过。正式入口val smoke正在运行，随后全量三seed训练；test未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T11:20:46+08:00）：seed11启用activation checkpointing，等价性及全量长度审计对应的48-token压力smoke通过；seed22/33原训练继续，模型/batch/精度/曝光不变；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T11:33:20+08:00）：A/B完成，三seed全量训练运行；bf16补充80批均有非零z梯度/置换效应，7项测试通过。seed11 checkpointing恢复正常；最终test未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T17:39:16+08:00）：C训练完成1项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T19:16:55+08:00）：C训练完成2项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T19:36:45+08:00）：CLR v1 seed22/33完成，seed11已完成8epoch并在第9epoch；lite seed33第9epoch、seed22第2epoch。累计约27.1/35 GPUh；剩余主训练、迁移及评估预算偏紧，预计墙钟仍需4–6小时且不保证预算内全部完成；test未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T19:37:31+08:00）：C训练完成3项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T19:39:54+08:00）：用户授权提高预算：总50 GPUh，新训练单run14h；现有训练不打断，旧9h时限若触发则从epoch检查点新run恢复。预算50接续控制器已启动，科学设置不变；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T19:57:13+08:00）：C训练完成4项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T21:24:13+08:00）：C训练完成5项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T21:45:24+08:00）：C训练完成6项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T21:45:43+08:00）：C训练完成7项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-05T23:42:31+08:00）：C训练完成8项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:02:29+08:00）：C训练完成9项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:24:53+08:00）：C训练完成10项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:25:36+08:00）：C训练完成11项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:26:08+08:00）：C训练完成12项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:27:07+08:00）：C训练完成13项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:27:28+08:00）：C训练完成14项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:27:36+08:00）：C训练完成15项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:28:28+08:00）：C训练完成16项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:29:04+08:00）：C训练完成17项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:30:40+08:00）：C训练完成18项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:30:52+08:00）：C训练完成19项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:31:27+08:00）：C训练完成20项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:34:42+08:00）：C训练完成21项，最终test尚未打开；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:40:13+08:00）：最终评估完成1/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:40:58+08:00）：最终评估完成2/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:42:48+08:00）：最终评估完成3/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:43:34+08:00）：最终评估完成4/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:44:32+08:00）：最终评估完成5/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:44:34+08:00）：最终评估完成6/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:45:18+08:00）：最终评估完成7/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:45:32+08:00）：最终评估完成8/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:46:07+08:00）：最终评估完成9/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:46:18+08:00）：最终评估完成10/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:46:36+08:00）：最终评估完成11/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:46:53+08:00）：最终评估完成12/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:47:07+08:00）：最终评估完成13/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:47:37+08:00）：最终评估完成14/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:47:57+08:00）：最终评估完成15/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:48:30+08:00）：最终评估完成16/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:49:20+08:00）：最终评估完成17/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1（2026-11-06T00:50:10+08:00）：最终评估完成18/18项，训练已锁定；状态索引docs/M41_LIVE.json。

M4.1实验完成，待G1审查；唯一审查文件reports/M4.1_REVIEW.md，冻结data/M41_v1/freeze.json。M4原冻结哈希全部一致。

M4.1（2026-11-06T00:58:03+08:00）：M4.1实验完成，待G1审查；唯一审查文件reports/M4.1_REVIEW.md，冻结data/M41_v1/freeze.json。M4原冻结哈希全部一致。；状态索引docs/M41_LIVE.json。

## 2026-11-06 M5b 启动与实现复核
用户按Q3/Q4有条件通过G1并授权M5b；预算25GPUh，A预先指定消融可test，B/C/D禁止test。任务书docs/M5b_TASK.md，实时索引docs/M5b_LIVE.json。
A正式控制从头重训：B3-prefix均值/当前拼接、lite-single仅当前；保持原训练协议/三seed。早期冻结权重输入干预有归一化混用且已test暴露，run保留并全排除，不用作结论/选超参。B3全量GPU缓存OOM后显式移至CPU缓存，科学设置未变。
A train/val smoke、B/C五题全网格smoke通过（26.1秒/题/seed，test=false），5项单测通过；预算含余量估计21.2GPUh。A训练、B全量运行，C等待B完成，D仅smoke；尚无正式完整结论。

## 2026-11-06T05:13:28.693528+00:00 — M5b完成待审查
M5b A–D/三seed、配对统计完成，待审查；累计16.8/25 GPUh。唯一审查reports/M5b_REVIEW.md；冻结data/M5b_v1/freeze.json，接续docs/M5b_LIVE.json。搜索结论见审查报告；A历史无效开发run提前test暴露保留并排除，B/C/D未触碰test。用户G1按Q3/Q4有条件通过；未启动M6。
详细reports/M5b_details.md，全部CSV/曲线/成本data/M5b_v1/analysis；run索引data/M5b_v1/completion.json。M41/M5a共4154个冻结文件SHA一致。

## 2026-11-06T05:16:33.248867+00:00 — M5b完成待审查
M5b A–D/三seed、配对统计完成，待审查；累计16.8/25 GPUh。唯一审查reports/M5b_REVIEW.md；冻结data/M5b_v1/freeze.json，接续docs/M5b_LIVE.json。搜索结论见审查报告；A历史无效开发run提前test暴露保留并排除，B/C/D未触碰test。用户G1按Q3/Q4有条件通过；未启动M6。
详细reports/M5b_details.md，全部CSV/曲线/成本data/M5b_v1/analysis；run索引data/M5b_v1/completion.json。M41/M5a共4154个冻结文件SHA一致。


<!-- M6_LIVE -->
M6实验与统计完成，待审查；唯一审查reports/M6_REVIEW.md，冻结data/M6_v1/freeze.json。
<!-- /M6_LIVE -->


## 2026-11-06 — M7 前置 Idea A 与归档对话核验
已找到本仓库归档对话「搭建 CLR 研究项目骨架」；对话链接、源码与 run 证据见 [M7_IDEA_A_READINESS.md](M7_IDEA_A_READINESS.md)。
Idea A M1.1/synthetic v2 已完成，三份核心生成器源码哈希与冻结一致；M3 KV 契约和机制已验收。
Idea A M4 正式 direct 训练完成、正式 val 评测待执行；不写成整体训练效果已验收。
两项目 transformers 版本不同（CLR 4.52.4 / Idea A 4.57.1）；CLR 尚无 LatentMAS/C2C source，需 M7 自行验证集成、传信及 commitment。
本次只核验既有记录并新增接续文档，未启动 M7 实验，未修改 Idea A 或任何冻结产物。

<!-- M7_LIVE -->
M7非自适应实验完成待审查；唯一审查reports/M7_REVIEW.md。冻结data/M7_v1/freeze.json；缩量/历史raw缺证据/1%无穷门限等限制见审查文件；未启动M8。
<!-- /M7_LIVE -->

## M6 2026-11-07
M6实验与统计完成，待审查；唯一审查reports/M6_REVIEW.md，冻结data/M6_v1/freeze.json。

<!-- M71_LIVE -->
M7.1足量非自适应审计完成待审查；C1/C2通过，三seed主比较/统计完成；唯一审查reports/M7.1_REVIEW.md，冻结data/M71_v1/freeze.json；接续docs/M71_LIVE.json，M8未启动。
<!-- /M71_LIVE -->

<!-- M8_LIVE -->
M8a固定500组评估与统计完成待审查；P0通过，三seed/E0–E4齐全；唯一审查reports/M8a_REVIEW.md，冻结data/M8a_v1/freeze.json。
<!-- /M8_LIVE -->

## 2026-11-08 — M8b启动
用户授权设计比较，优先R4→R2→R1→R3，独立预算25GPUh。任务书docs/M8b_TASK.md；实时docs/M8b_LIVE.json。
R4 receiver状态提取smoke（train100/val20）完成，4项协议单测通过；小样本适配器尚未学好A4，保留为诊断。
正在提取固定train4000/val200/cal_A300/cal_B1000；不改上游冻结，不打开test。最终审查reports/M8b_REVIEW.md。

## 2026-11-08 — M8b接续更新
提取5500组完成（17.5GiB、1.42GPUh）；上游62363项SHA一致。R4三seed训练完成，val A4约56%–60%；R2接近随机，初始化诊断负面结果保留。
R4训练后val20 PGD smoke约112秒通过；原始实现与修正版本均存run快照，未打开test。
R2 seed33孤儿run实际GPU成本缺失，原记录保留；保守上界10.1分钟单列，同设置GPU5重启完成。tmux clr-m8b-development继续R3训练/校准。
接续docs/M8b_LIVE.json、reports/M8b_evidence/；最终test须先执行M8b final lock。

## 2026-11-09 — M8b训练完成、test前预算核验
R4原三seed、R2/R3两种初始化均完成，按三seed平均val NLL整体选R2/R3 v2；R3 val A4约94.5%–96.5%，R2约37.8%–47.8%。
统一A1/A4分布；R2随机读取CPU/GPU舍入已定位，正在GPU同路径重校准并严格重放。所有旧记录保留，累计成本实时见M8b runtime。
后台GPU7：同设备smoke→val20完整ε矩阵→λ敏感性预算核验，尚未final lock/尚未test。接续docs/M8b_HANDOFF.md、M8b_LIVE.json。

## 2026-11-09 M8b最终评估已启动
val20正式主矩阵1552秒、λ敏感性810秒，严格重放及统计smoke、7项单测通过。累计7.54GPUh后锁定final_lock.json。
主500/三seed/四设计/E1–E3/四ε/50与200预算/λ1保留；附加λ=.1/10在ε=.03固定102组，预算含20%余量预计总22.0GPUh，硬上限25。
GPU7/6/5三分区tmux clr-m8b-final-p0/p1/p2；完成后自动接统计/上游复核/审查与冻结，tmux clr-m8b-final-finish。尚未完成，不能把启动当结论。

<!-- M8B_LIVE -->
M8b四种设计/三seed、固定500组E1/E2/E3评估与统计完成待审查；唯一审查reports/M8b_REVIEW.md，冻结data/M8b_v1/freeze.json。
<!-- /M8B_LIVE -->

## 2026-11-09 M8c启动
任务书docs/M8c_TASK.md；独立15GPUh，先4B完整日志direct-lookup val20/val200，失败按授权缩日志、升型号，仍失败即报告。M8a/M8b冻结不变，未打开test。

<!-- M8C_LIVE -->
M8c固定test200组三seed E3主矩阵完成待审查；21GPUh内次要项完成范围见reports/M8c_REVIEW.md，冻结data/M8c_v1/freeze.json。
<!-- /M8C_LIVE -->

## 2026-11-10 M9 启动
用户授权最后补实验仅M6 C2（GPT2+LLaMA联合CLR-lite、3seed、10GPUh），其余不新增实验。论文产物和主张核验见docs/M9_TASK.md。
两源各5训练/val题smoke完成（70.7秒），训练及校准/预测val代理均通过；正式3seed顺序训练，全部权重锁后再test。
上游105515个唯一文件SHA核验全部一致（data/M9_v1/upstream_audit.json）。论文脚本仅消费冻结CSV，缺失对照保留NA。
<!-- M9_LIVE -->
M9-A三seed联合训练与论文产物完成待审查；实验至此冻结。唯一审查reports/M9_REVIEW.md，冻结data/M9_v1/freeze.json。
<!-- /M9_LIVE -->

## 2026-11-10 M9 接续
seed11原worker保留，原纯CPU顺序控制器停止；22/33按同一配置在GPU0/4并行，累计10GPUh账本不变。
接续scripts.m9_resume_pipeline configs/M9/resume_pipeline.yaml；调度补充data/M9_v1/execution_supplement.json。
已生成静态表图并核查；论文CSV哈希门槛与AUROC/单调包络测试通过。独立.venv-paper依赖直连安装成功，主环境未改。
正式训练仍运行；全部权重锁后才test。主表缺失对照保留NA，强主张按四贡献/两局限核对。

## 2026-11-10 M9 收尾接续
当前实验控制器PID1560935，保留seed11、22/33并行，训练配置不改；所有权重锁后才test。
新增CPU收尾控制器PID1763304：scripts.m9_finish configs/M9/finish.yaml，日志reports/M9_evidence/finish.log；等待experiment_complete后统计→CSV冻结→独立环境制图/逐字节重建→上游复核→审查/冻结。
旧格式预测已按冻结QA恢复选项，逐项对照原conformal值；已有模型CSV导出PID1690664仍运行，无模型调用。
脚本测试4项通过，无warning；PDF已核查无raster图像对象且嵌入TrueType字体。最终协议附录采用v2；原开发v1保留。

<!-- M10_LIVE -->
M10投稿补实验及论文产物完成待审查，实验最终冻结；B5c全量因预算NR，HiddenBench仅探索性。唯一审查reports/M10_REVIEW.md；冻结data/M10_v1/freeze.json。
<!-- /M10_LIVE -->

## 2026-11-10 — M10 P1执行中
用户授权投稿前补充实验≤40GPUh（P1≤12、B5c≤12、HiddenBench≤12、跨用途≤4），完成后最终冻结。
7项GSM设置及审计B6通过val-only smoke；LLaMA缺失格/B6/每类型CLR正式训练队列已启动，test仍关闭。
配置入口configs/M10/full_gsm_queue.yaml、full_audit_queue.yaml；接续docs/M10_LIVE.json，预算按runs/*/budget.json实测，不把启动写为完成。
M9及上游冻结只读；M10新建data/M10_v1。HiddenBench原始文件仅65场景组，独立组校准的有限样本约束仍待处理。

## 2026-11-10 M10执行接续
P1缺失格三seed训练已完成；GSM B6/每类型及审计B6仍在队列；跨用途小N完成、full训练中；HiddenBench CLR完成后训练B3/B6。
GSM所有评估分支、审计/迁移/HiddenBench及公开契约v2 val-only smoke通过；11项单测；上游110122文件SHA一致。
B5c全量准入失败：仅GPT2 Q3/Q4一遍×三seed实测下界19.2GPUh>12，保留NR；不声称优于训练式语言化基线。
HiddenBench63组，程序化事实+结论+公开one-of-panel约束；13/20/4/20/6组，只有mismatch，1%阈值+∞。局限与早期失败见BLOCKERS。
最终控制器等待全部训练后自动lock/eval/export；当前未开test。旧paper/FINAL_SUMMARY66文件已归档。尚缺最终结果、paper/review/复现/冻结；详见接续文档。

## 2026-11-10 M10接续调度与关卡
HiddenBench非test源及15checkpoint已完成；B5c最快实测19.2GPUh是条件于当前吞吐的单轮估算，并非理论下界。
B6两源与审计、每类型CLR及迁移full继续；12项单测通过。final lock新增逐run test访问标记与terminal状态检查。
依赖已完成队列释放GPU0/2/5接续未启动任务，case/seed任务锁避免重复训练；科学配置仅device变化，仍在原子预算。
当前未开test；接续docs/M10_EXECUTION_CONTINUATION.md、M10_LIVE.json；完成后再paper复现/审查/最终freeze。

<!-- M10b_LIVE -->
M10b GPT2 B5c Q3/Q4三seed与论文复现完成待审查，实验永久冻结；唯一审查reports/M10b_REVIEW.md；冻结data/M10b_v1/freeze.json。
<!-- /M10b_LIVE -->

<!-- M10c_LIVE -->
M10c只读开销测量与T6/配对conformal汇总完成待审查；无训练/重校准，科学结果永久冻结。唯一审查reports/M10c_REVIEW.md；冻结data/M10c_v1/freeze.json。
<!-- /M10c_LIVE -->

## 2026-11-12 — M10d完成待审查
M10d四格补齐完成待审查；GPT2 B6复用0.574/0.853，LLaMA B4新增三seed0.551/0.806（Q1 NLL/Q2 AUROC）；0.0289GPUh≤2、源前向0，旧冻结不改；论文仍8页。唯一审查reports/M10d_REVIEW.md；冻结data/M10d_v1/freeze.json。
任务书docs/M10d_TASK.md；接续docs/M10d_LIVE.json；证据reports/M10d_details.md及data/M10d_v1/four_cells.json。
论文同步目录manuscripts/CVPR27-CLR；sec/5_results.tex只替换四格及最优加粗，number_audit.md仅追加。输出outputs/main.pdf；独立Tectonic+T1兼容构建，8页/第5页主表核查通过。

<!-- M10e_LIVE -->
M10e两源HSRM三seed/确定性CLUE及LLaMA B2 Q1–Q4三seed完成待审查，实验再次冻结；0.942GPUh、源前向0，旧科学输入SHA一致。论文按用户修订仅交付reports/M10e_PAPER_HANDOFF.md，未改稿/编译。唯一审查reports/M10e_REVIEW.md；冻结data/M10e_v1/freeze.json。
<!-- /M10e_LIVE -->

<!-- M11_QUAL_LIVE -->
M11-qual只读导出完成待审查；两源各12题、跨题300、兄弟40、审计600、M8c100组。唯一审查reports/M11_qual_REVIEW.md；交付data/M11_qual_v1/qual_export.zip，冻结data/M11_qual_v1/freeze.json。无训练/重校准/PGD；M8c仅已有分数。
<!-- /M11_QUAL_LIVE -->

## 2026-11-12 M12启动
用户授权CODI-GPT2 CLR-lite A1–A5单组件消融，三seed沿用M41曝光/早停/校准；预算30GPUh、北京时间11-15 12:00截止，按优先级放弃未完项。
任务docs/M12_TASK.md；仅新代码/结果，M41等冻结只读，不改论文。先val-only smoke，所有保留训练hash锁后再统一test。

M12 23:07：修正版训练/协议smoke通过（训练约6.8分钟、协议约1.8分钟），12项单测及20项复用val格式/conformal回放核验通过。A1三seed在GPU3/2/5实际训练，未访问test；后续A2–A5按成本优先级准入。自动接续PID见data/M12_ablation/controller_pid；控制器reports/M12_evidence/controller.log，逐run从jobs/及progress.json核实。

M12持续核验：A1三seed进程存活（seed11/33至epoch8、22至epoch6），约2.75GPUh，未开test。收尾增强完整曝光/原配置/早停历史核验与未准入项显式NA；paper/和已有CVPR归档共96文件SHA匹配。统计前测试通过，实际最新进度以docs/M12_LIVE.json/jobs下progress.json为准。

M12 2026-11-13 00:30：A1三seed全部训练完成（11/22/33实际10/9/9epoch、选中9/8/8epoch），完整曝光及checkpoint选择回执核验通过；A1训练合计3.51GPUh。预算准入后A2三seed实际启动，含全部smoke累计约3.79GPUh；test仍关闭。完成回执核验data/M12_ablation/completed_training_receipt_audit.json；后续A3–A5按成本继续。

M12 2026-11-13 00:50接续：A1三seed完成，A2三seed均至epoch3（11/22/33 batch约1660/240/1960），进程持续推进；累计4.77/30GPUh，test仍关闭。现有start/pipeline PID2845222/2867996保持运行，勿重启已完成训练。最终评估/统计/冻结仍待所有保留变体训练锁；本轮核对收尾代码未改科学设置。

M12 01:23接续：A2 seed11/33至epoch8、22至epoch6，累计6.40GPUh，test仍关闭。新增最终论文96文件SHA复核及置换结果三seed/支持集/身份对齐一致性核验，8项单测通过（reports/M12_evidence/completion_audit_tests_v3.log）；只改交付核验，不改训练/评估科学设置。

M12 01:34：A2 seed11/33完成9epoch，均选epoch8，完整曝光/选点与hash回执核验通过；seed22仍运行（epoch7末，进程存活）。累计约6.86/30GPUh，test未开；A3–A5仍等待现有控制器继续准入，未重启已完成run。

M12 01:52：A2三seed均完成9epoch/选择8epoch，完整曝光与checkpoint SHA回执核验通过，现累计6个完成run。A3通过成本准入（三seed预测4.89GPUh+最终预留2GPUh），实际三进程已启动；累计约7.18/30GPUh。test仍关闭；既有控制器PID2867996继续A3–A5。

M12 03:22：A3三seed训练完成（11/22/33实际9/10/9epoch，选择8/9/8epoch），完整曝光/选点及checkpoint SHA回执通过。A1–A3累计9个完成run；A4通过成本准入，三进程已实际启动，test仍关闭。累计约10.8/30GPUh，原控制器继续A4–A5，详见docs/M12_LIVE.json及jobs。

M12 03:44恢复：A4 seed11在GPU3首epoch一个优化步后共享显存OOM，失败0.0225GPUh保留；相同模型/batch1024/bf16/完整曝光在GPU0全新重跑PID3255842，已到epoch1。22/33原worker继续。CPU监督PID3256871等待旧pipeline及重跑结束后复用A1–A4回执接续A5/评估；配置configs/M12/recovery_supervisor.yaml，证据data/M12_ablation/recovery.json。累计11.5/30GPUh，test未开。

M12 04:51：A4三seed最终完成，均9epoch/选8epoch；完整曝光/选点/hash回执通过（12个完成run）。A4/11原OOM保留，GPU0同设置重跑成功。原controller终止后恢复监督复用A1–A4回执，A5三seed已实际启动，设备0/2/5；新PID见data/M12_ablation/recovered_controller.json。累计约14.2/30GPUh，test未开。

M12 06:02：A1–A5全部15个训练run完成，完整曝光/早停/hash回执通过；06:02:09统一锁定checkpoint/config/code后才开启最终test评估，三卡0/2/5按seed处理。累计约17.5/30GPUh；校准、置换、配对统计和最终冻结仍在执行。接续docs/M12_LIVE.json。

## M12完成
M12组件消融完成待审查；5变体三seed，18.1/30GPUh；唯一审查reports/M12_REVIEW.md，冻结data/M12_ablation/freeze.json；未改论文。

## 2026-11-16 — 目录整理与版本归档
按用户授权整理目录、清理缓存并完善 `.gitignore`，归档现有未提交的代码、配置、报告及冻结索引；未启动训练、评估或重校准。
根目录重复提案/规划改为兼容链接；冻结材料、旧稿路径及独立论文仓库保持原样。补充统计另见 `supplemental/stats_20261116_v1/REPORT.md` 与其 `freeze.json`。
目录分工与本地保留范围见 [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md)，清理核验见 `docs/maintenance/20261116_cleanup.json`。本次维护不代表任何里程碑审查通过。
