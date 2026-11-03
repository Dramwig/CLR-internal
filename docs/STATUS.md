# CLR 当前状态与接续索引

最后更新：2026-11-03（北京时间）。此文件描述当前状态；历史失败、重启及恢复记录保留在 EXPERIMENTS/BLOCKERS/CHANGELOG。

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
