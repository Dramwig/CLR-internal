# 变更记录

## 2026-10-28

- 短标题定为 CalReader，参照 LatDefer 建立项目目录骨架。
- 保留附件原文及 `docs/proposal.md` 完整副本。
- 添加项目说明、工作约定与文档索引；尚未实现研究代码或运行实验。

## 2026-10-28 — M0 实现（进行中）

- 按用户新约定采用 `clr/`，保留旧空骨架；增加CODI/Coconut/Huginn源接口和相对噪声，禁止模型/设备/大小静默回退。
- 初始化本项目Git；通过import只读复用Idea A的Run、日志、seed和YAML loader；Idea A KV模块仍为占位文件，未复制或修改。
- 核实官方模型与代码，记录精确revision；提供固定版本下载和SHA-256验证的并行传输。
- 官方数据文本最初误按三个`||`字段解析并报错；按上游`question || steps ## answer`修正。失败运行保留；未产生旧版划分或准确率，不影响既有结果。
- 冻结GSM8K问题级val/cal/test；测试仅用于最终源复现，噪声曲线在事先指定的64题val面板运行。
- 推理新增显式batch16，按等长token组执行，避免padding改变latent attention。单样本与批处理smoke分别留档，禁止静默替换。确定性抽查16题，batch与single漂移另计。
- Coconut实际batch32成本探针OOM；此前明确batch8的吞吐外推仅为粗略预算，不能当作精度实验或完整官方recipe预算。

### M0 CODI子结果交付

- 官方未修改test.py对照为43.594%，CLR等长接口为41.698%；差异完整保留并分析，未用测试结果调整checkpoint、sigma或指标。
- 64题固定val面板完成5个sigma × K16 × 3 seeds；报告负面准确率变化和batch数值差异。
- 15项最小测试通过；Idea A仓库保持未修改。
- M0整体仍有LLaMA授权、Coconut预算/显存、Huginn不完整权重等阻塞，不标记完成。

## 2026-10-28 — M0.1 v2（审查中）
- 新建CODI v2接口，显式记录左padding上下文、mask、position、张量布局及fp32/batch1候选；v1接口未修改。
- 按题10k bootstrap替换确定性评测的重复seed CI；保留旧报告不追改。
- 官方GSM8k-Aug归一化精确/13-gram去重，获准将原500验证池重划为清洁val300/cal193，test移除7题至1312；原v1划分不变。
- 新增18设置K16噪声与采样对照程序、V直方图和非退化门槛，smoke通过；完整噪声与M1受生成协议一致率门槛阻止。
- 新增M0b Huginn官方恢复/校验与原生harness入口，Coconut仅第三方元数据报告。

## 2026-10-29 — LLaMA镜像资源 v2
- 按用户指定使用ModelScope的同一Llama-3.2-1B-Instruct，锁revision并逐文件校验本地SHA与上游Git对象；没有HF模型文件下载。
- 新增配置驱动下载/离线格式验证入口，保留原M0 v1配置和结果。
- CODI作者checkpoint的获取渠道仍待范围澄清；GPT2完整噪声作业继续。

## 2026-10-29 — CODI作者checkpoint来源已获不限平台授权
- 作者公开checkpoint已下载并验证官方LFS SHA；ModelScope底模保持不变。
- 新建CODI-LLaMA v2协议、官方test.py只读封装和阶段流水线；源加载strict、smoke限时、确定性按题bootstrap。
- 既有M0 v1与正在运行的GPT2实验设置不改动，复现后仍须完成噪声及审查。

## 2026-10-29 — CODI-LLaMA v2真实复现结果
- 通过精确allowlist保留不参与官方推理的4个auxiliary张量，其余state_dict仍严格校验；未知键/缺失键测试通过。
- GPU5预检失败后显式调度GPU4并重跑matching smoke，完成官方1319题、CLR batch16与batch128控制。
- 保留27/14题不一致与论文参考+3.597pp偏差，未改指标或seed；原M0 v1不改动。噪声与M0b最终审查仍待完成。

## 2026-10-29T16:10:48.755456+08:00 临时资源约束

记录物理 GPU 2 至北京时间 2026-10-29 18:06 的禁用窗口（含 CUDA_VISIBLE_DEVICES 重映射），当前 GPT2 噪声任务核实位于物理 GPU 3，保持运行。未新增实验或修改冻结结果。

## 2026-10-30 — M0.1 收尾

新增收尾报告，核验71个冻结噪声文件及划分、重跑回归23 passed/1 skipped；更新EXPERIMENTS/BLOCKERS当前状态。M1等待用户任务书。释放磁盘配重10GiB并保持暂停，未改冻结协议或结果。

## 2026-10-30 — M1 数据管线启动

新增共享continuation policy、PrefixStates/KV上下文、候选划分准备、gold中间量解析、safetensors/Parquet writer+loader、200题smoke与val300关卡、预算后冻结脚本。保留M0/M0.1冻结源文件。当前没有reader/probe训练，也没有M1全量完成声明。


## 2026-11-01 — M1 v2 显式断点恢复

用户要求继续且不因磁盘余量停止。新增 `scripts/m1_generate_v2.py`、`clr/data/resume.py` 和 `configs/M1_data_v2.yaml`：在新目录 `data/dumps/codi_gpt2_v2` 复用完整分片，原结果不覆盖。核验连续分片、张量索引、冻结ID前缀、生产cohort上下文、7步clean、每题8条完整轨迹及早退日志后才续跑。未落盘尾部重新按原种子计算；重复计算成本保留。

取消低余量主动退出，改为低于20GiB时调用配重release补至约25GiB；已提前释放30GiB并保持自动填充暂停。实际I/O失败仍如实报错；60GPU小时（包含前次9.291732小时和预检）及8GiB产物预算保持。恢复逻辑及原policy/data测试6项通过，首次测试发现可变列表共享已修复。

已知统计限制：原中断运行没有持久保存fp32→fp16误差聚合，不能从fp16文件反推；原分片写入时已执行误差阈值检查，恢复后误差汇总仅涵盖新生成状态，报告必须标明，不冒充全量误差统计。尚未完成M1，不训练reader/probe。


## 2026-11-02 — 服务器重启后 v3 恢复

用户确认服务器重启并要求续跑。原v2进程已消失，metrics残留running是重启前快照，不代表仍在执行。重启前test798/1005、v2成本12.800449GPU小时/1606828次forward；原失败生成9.291732GPU小时也计入预算。四个已完成split复用，不重跑。test已持久化768题/5376状态行、9216条MC记录和44条K64记录，通过冻结前缀、cohort、完整步骤和行为日志核验；第769题起重算，共剩237题。

新代码与配置v3、新输出 `data/dumps/codi_gpt2_v3`，原v1/v2不覆盖。已完成文件用硬链接复用，生成器不修改复用文件；本次记录SHA和祖先run成本。重启恢复与policy/data测试8项通过。物理GPU2当前空闲，显式继续使用；release20GiB成功且自动填充保持暂停。自动配重调用已使用整数参数。

命令：`PYTHONPATH=. PYTHONUTF8=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_finish_job_v3.py configs/M1_data_v3.yaml device=cuda:2`。tmux `clr-m1-reboot-v3`，状态 `reports/evidence_M1/full_job_v3.json`。保持原协议、划分、seed及全部实验设置；中断test段的量化误差聚合未持久化，恢复后的test误差统计仅覆盖新生成部分，限制与reader_train一致，不能称为全量统计。


## 2026-11-03 — 完成状态与文档同步

M1 v3数据及报告均已完成，待用户审查。共23,805题，产物约3.03GiB，生成累计22.615373GPU小时（另计预检），770个数据文件及6个冻结文件SHA复核通过。当前接续索引为 `docs/STATUS.md`，`AGENTS.md` 已同步，后续状态变化必须同时持久化。保留reader_train/test中断部分量化误差聚合缺失、cohort依赖等限制；无新训练任务启动，原冻结结果未改动。


## 2026-11-03 — M1审查通过，M2–M3启动

用户授权合并里程碑，当前不训练CLR。输入隔离源题目文本，状态只用模型实际输出。Q1规则经用户明确：clean仅轨迹起点独立Bernoulli标签及clean终点，traj起点之后用自身观测及终点；不监督更早clean前缀。reader_val报告clean/traj分别指标和各t样本覆盖。cal_A/cal_B250/250已固定。

QA首轮smoke因少量题不足4种答案失败，未覆盖；固定训练集数值池补足后run `20261103T042122100178Z_172c4808908a_81b8b5`通过，输出 `data/M23_v1a/smoke`。全量QA任务tmux `clr-m23-questions`。Q4成对交换正值作为对方负值，保证保留问题上数值边际精确一致；未匹配/解析失败单独统计。全量不得使用失败smoke。

源调用范围冲突按具体要求解释为B5a/C/D/E例外，其他仅读M1；不新增轨迹。GPU当前均忙，显式使用cuda3训练轻量probe、cuda1量化复算，负载与实际OOM如实记录；没有替换模型或降精度。D候选许可和谱系正在核实；尚无完整基线结论或M2–M3完成声明。

M23结构审查修正：中止基线run20261103T044101470059Z_19e9238e5441_14ac1a（先前已保存尝试保留不纳入最终）。B0/B1/B3改为对z+模板特征+选项数值特征联合两层MLP，Q3共享逐选项打分；避免旧加法结构无法表达Q4状态×数值交互。新增交互测试，7项测试通过。此为实现修正，未按test结果选择模型或seed；超参网格/样本量保持。

2026-11-03 B5a吞吐预检：旧batch8完整运行继续保留；新v2计划在固定reader_val50题面板比较8/64，预设max概率差≤1e-4且argmax一致100%才采用64。使用物理GPU3（检查free5484MiB），不改变模型、fp32、prompt/指标，不使用test选择批大小。

B5a v2预检首次入口在创建run前触发libLerc/GLIBCXX_3.4.29动态库错误，尚未加载源模型或前向。原因是新增入口先导入pyarrow，修正为与现有已运行入口相同的torch-before-pyarrow导入顺序；无依赖/模型回退。首次日志reports/M23_patchscope_batch_gate.log保留，重试使用独立日志。


## M23测试集访问协议偏离（审计发现，2026-11-03）

早期smoke使用了固定test的10题子集并计算指标；被中止的早期完整基线尝试也已产生部分test预测。这不满足“test仅最终评估”的严格要求。未使用这些指标选择超参、模型结构、seed或划分；Q4修正源于代码交互结构检查，batch优化只用reader_val数值门槛与运行耗时。保留所有记录，不删除曝光历史，不更换测试集掩盖此问题。最终结果须带此限制交审查；后续milestone smoke一律使用val代理最终评估路径。


M2–M3问题库、B0–B5a、层/域诊断、官方500题、两候选未见池与量化抽查均完成，待用户审查。B5b推迟M6，未训练CLR。报告：reports/M2_questions.md、M3_baselines.md、M3_shift.md、M3_unseen_pools.md；冻结索引 data/M23_v1a/freeze.json。


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

## 2026-11-05 — M5a独立预检与预算分支
用户授权M5a作为G1前M5的专项例外，仅val300、不触碰test；原M1/M23/M4冻结实现和结果未改写。
新增`clr/search/m5a.py`、YAML搜索/预检报告入口、测试和唯一审查文件；保存每版启动源码/环境/资源/权重hash。
预检与两套待选择配置已落盘；原网格预算不通过，未启动全量，接续`docs/M5a_LIVE.json`。

## 2026-11-05T06:03:47.954732+00:00 — M5a授权版本v13
保留旧预检快照，新增独立N轨迹池、probe FLOPs成本、v13 smoke/full配置及授权预算。
本轮按用户继续授权保留10GPUh，明确S3 oracle范围与同预算分析规则；不修改M4/M41冻结结果。

## 2026-11-05T11:34:44.657867+00:00 — M5a v13完成待审查
完成M5a授权v13搜索、全部配置CSV、同预算配对、按题bootstrap、actual/allowed compute曲线与唯一审查。
新增统计/收尾/接续代码；M5a输入与旧预检快照保留，M1/M23/M4冻结结果不改；未触碰test。

## 2026-11-06 M5b
新增A重训混杂控制、B同父兄弟MC诊断、C含sigma=0的全前缀搜索与D排序头入口；冻结M4.1/M5a文件不改。A早期无效test干预保留且排除，后续原任务训练设置预先锁定；B3显存失败改CPU缓存。接续docs/M5b_LIVE.json。

## 2026-11-06T05:13:28.693528+00:00 — M5b最终统计与冻结
新增逐qtype同覆盖率配对CI、兄弟诊断、全部搜索网格及同时CI、D配对比较、成本总账与唯一审查。
正式A模型在test评估前锁定；初始无效开发test暴露未抹去。M4.1/M5a冻结文件哈希全部一致。

## 2026-11-06T05:16:33.248867+00:00 — M5b最终统计与冻结
新增逐qtype同覆盖率配对CI、兄弟诊断、全部搜索网格及同时CI、D配对比较、成本总账与唯一审查。
正式A模型在test评估前锁定；初始无效开发test暴露未抹去。M4.1/M5a冻结文件哈希全部一致。

## 2026-11-06 M6授权持久化
接受M5b背景结论、后续只用CLR-lite；任务/预算/优先级见M6_TASK.md。新增M6 LLaMA continuation与预算运行器，冻结上游不改。

## M6执行协议补全（2026-11-06）
新增不可变8题块/三GPU生成与校验合并；同一M1 cohort上下文及sigma/K/bf16不变，GPU墙钟相加。新增成本smoke、val覆盖率smoke、CPU统计/审查文件与冻结流程；C3/C2/C4以成本估算准入，预留3GPUh最终推断。

## M6固定成本优先（2026-11-06）
B生成顺序改为固定reader_val/val/cal/test后reader_train；仅按实际GPU成本复核训练前缀，保留plan_initial.json，sigma/K/bf16和固定split不变。CPU控制器在等待noise时计划重启，审计见runs/20261106T074310687653Z_0a05b2e7b9b2_165fa8/termination_audit.json；噪声进程未动。12项测试通过。

## M6统计口径与收尾关卡（2026-11-06）
恢复沿用M4.1的clean主比较、逐步实例权重与按题bootstrap；Q1 clean/traj分报。同题/轨迹的不同QA实例不宣称iid。合成统计fixture验证C1/C2/C3/三seed及Arrow统一N字段；14项测试通过，未读取正式test预测。

## M6 B1用户预登记v2（2026-11-06）
用户明确保留旧4点失败，授权3/4/6/8扩展、8点统一规则及无达标点的明确降级；扫描含smoke/失败≤2GPUh，预算仍B30/总60。登记data/M6_v1/noise_extension_preregistration.json先于新run。新增独立扩展控制器与源内Q1报告标记，旧选择null不改；有效GPU按实时显存选并记录，模型/批量不变。

## M6 2026-11-06：八点证据与推断预算预留
扩展扫描仍在运行；新增完成后统一八点CI报告，原四点报告/失败不改。
正式C3/C2训练准入增加按完整C1验证吞吐及冻结划分规模计算的待推断预留，含已有/拟新增检查点；不读取test指标或修改科学设置。
验证reports/M6_tests_noise_v4.log：M6及split共16 passed；CPU编译通过。

## 2026-11-06 — M7 独立管线与 commitment 协议
新增 docs/M7_TASK.md，用户授权 M7 与 M6 独立并行；源码/环境只读复用 Idea A。
独立 .venv-comm 复用 latdefer 只读基础依赖，实测 transformers4.57.1/tokenizers0.22.2/peft0.17.1；不改主环境。
按用户修订，所有主类别统一程序化 commitment（≥3共同变体，manifest来源标记）；模型输出仅 natural_commitment 附加诊断。
原 direct/cot512 val 诊断保留；正式源固定 GPU2、reader固定GPU7，避开当前M6配置GPU0/1/4/5。

## 2026-11-06 — M7 训练并行交接与最终锁
正式源train/val/cal生成完成后复用GPU2训练seed22/33，GPU7原seed11继续；仅暂停本任务旧调度父进程，作业不重复，结束后自动恢复汇总。
训练设置/三seed/划分未改；运行证据data/M7_v1/parallel_handoff.json、configs/M7/parallel_training.yaml。
最终锁补记复用的CLR源码、环境脚本、预先存在的split与source manifest；测试生成仍要求全部checkpoint与阈值锁定。
报告补充真实KV过拟合关卡、独立上游23800文件预核验、CF的私有变量干预诊断，以及数据/模型磁盘口径。
test前登记附加因果统计：[P_cf(y_cf)−P_clean(y_cf)]−[P_cf(y_clean)−P_clean(y_clean)]及按组bootstrap CI；仅分析已有receiver输出，避免把公开/私有KV同时错配的增益全部归因于私有事实。

## 2026-11-06 — M7 B0计算等价优化
B0尚未正式启动前，去掉其完全不使用的长KV归一化/搬运；仍向相同Lite主体输入6个固定零潜槽，样本、seed、batch、优化器和选择规则不变。
原/新forward逐位输出对比与选项/长度测试见reports/M7_evidence/b0_equivalence.log、tests_v8.log；不修改CLR/B3/线性的计算路径。

## 2026-11-06 — M7 剩余作业均衡调度
原GPU2/GPU7固定队列交接为两设备共享待办队列，配置configs/M7/balanced_training.yaml；已运行的B3/22与B0/11子进程继续至正常结束。
仅暂停旧CPU调度器，保留所有日志与run；在其子作业均结束后退役该调度器并标记interrupted/superseded，非训练失败/重启。
剩余7个作业不重复启动；每项科学设置沿用train.yaml，实际设备写入各run；最后恢复原汇总进程。证据data/M7_v1/balanced_handoff.json。

## 2026-11-06 — M7 错配管线v2
test前发现并修正单条供体重编码丢失原/CF批次padding的位置不一致；旧receiver错配输出保留，新建mismatch_v2.jsonl。
v2对每条实际供体KV与保存的审计KV作逐元素和metadata等值断言；重算开发split的错配receiver输出，最终test直接采用v2。
仅影响错配receiver/文本基线及其拟合数据；来源事实、clean/CF主QA训练、超参/阈值规则不改。BLOCKERS保留偏离。

## 2026-11-06 — M7 最终阶段并行
校准前用finish_v2替换尚无子作业的CPU等待器，旧run记录interrupted/superseded、0GPUh；不终止训练。
全部训练与错配v2完成→校准锁→reader_val评估(GPU7)与test生成(GPU2)并行→test评估→报告/冻结。
两路均不调参；test生成仍逐项核验最终锁，未提前访问。配置configs/M7/finish_v2.yaml，交接data/M7_v1/finish_handoff_v2.json。

## 2026-11-06 — M7 合并问答与统计
clean+CF合并后补充每seed完整QA指标，A3 AUROC不再局限于单标签类别；新增ECE/AUROC的配对组bootstrap10000次CI。
验证完美/无信息/混合预测的已知结果及样本顺序不变性，tests_v11.log共15项通过；统计定义、主比较与阈值规则不变，尚未生成test。
随后为3份生成器源码加入CalReader侧固定SHA锚，改变代码及其自述freeze清单也会拒绝；tests_v12.log共16项通过。

## M6 2026-11-06：覆盖率恢复的配对区间
报告按具体偏移列出标准覆盖率低于目标超过2pp的描述性设置；weighted−split的恢复量增加按题/固定三seed的配对bootstrap95%CI，不用test调参。
未来正式报告入口实施，reports/M6_tests_recovery_v1.log共16 passed；当前B源生成作业不变。

## M6 2026-11-06：后续产物空间
实时可用35.3GiB，按已有授权通过reserve.sh release 20额外释放20GiB配重，自动补足保持暂停；用于后续数据、模型检查点和推断产物，不停止实验。

## M6 2026-11-06：Q4子集边际审计
新增已指定Q4配对在LLaMA ID子集上的正负数值TVD及计数；不重新配对或改标签，报告明确完整split边际约束不会自动传给前缀。
CPU train/val审计smoke通过，reports/M6_tests_q4_audit_v1.log共16 passed；首次独立入口漏设PYTHONPATH已保留且更正。

## M6 固定数据完成与成本锁定（2026-11-06）
固定reader_val1000/val300/cal500/test1005全部生成合并；B累计12.7689731GPUh，成本复核维持reader_train6817题，预计B29.5803596GPUh。
原始官方cohort按LLaMA tokenizer复核无宽度偏差；仅核对输入元数据，未生成reader test预测。
报告补充同题Q3候选集差异与轨迹Q3真实终点解释，去除对独立M7任务状态的断言；科学设置不变。
接续docs/M6_LIVE.json，成本run runs/20261106T134256492042Z_a8a5f6304bcc_e4a435。

## M6 cohort 输入审计（2026-11-06）
新增YAML驱动CPU元数据审计：冻结ID前缀、原cohort、LLaMA tokenizer宽度和left padding；包含已从划分移除的原官方cohort成员。
覆盖全部计划9,622题，0偏差；17项M6/split测试通过。
证据runs/20261106T144645586083Z_b79ee1e1153d_5304e7/audit.json、reports/M6_tests_context_audit_v1.log；无源前向/reader预测，运行中生成不变。

## 2026-11-07 — M6 可选 unseen 成本证据
最终reader test前登记LLaMA unseen完整池成本核查，配置configs/M6_llama_unseen_budget.yaml；仅用source smoke/C1验证计时及冻结行数。若实际剩余预算可容纳则阻止提前完成，须补执行；不足须持久化量化理由。源生成/现有控制器未重启，训练及统计设置未改。

## 2026-11-07 — M6 D优先级修复
新增可选LLaMA unseen源数据编排，复用完整M4输入cohort、冻结ID半集和Q4配对；准入安排在C3之后、C2/C4之前，额外推断始终保留。全部reader最终推断仍在训练/哈希锁之后。18项测试通过；GPU训练队列未中断，CPU控制器在C1批次结束后交接。

## M6 最终推断并行调度（2026-11-07）
仅把锁定后的独立reader推断改为三GPU队列；完整评估索引原子更新，全部完成后才进入shift/report。训练/选模/校准/预测数学设置不变，20项测试通过。
旧CPU控制器SIGINT只在既有C3全量线程池排空后退出，GPU子进程未发送信号；接续configs/M6_parallel_eval_resume.yaml，证据data/M6_v1/controller_handover_parallel_eval.json。

## M6 最终预测存储与状态收尾（2026-11-07）
空间降至约8.9GiB后按授权追加release10GiB，维持自动补足暂停；不删除数据/权重。完成时同步STATUS的M6块、审查状态及实际成本/题数；冻结排除仍开放的最终控制器stdout。
Huginn资源与预算复核见data/M6_v1/huginn_admission.json；科学设置不变。

## M6 收尾报告核查（2026-11-07）
完成索引计数按evaluation_index/training_lock分别生成；审查表3并列standard/weighted平均集合大小范围，统计方法与训练设置不变。
20项检查通过：reports/M6_tests_report_sizes_v8.log；GPU最终推断继续，未重训或新增test设置。

## M6 C3 补充学习曲线（2026-11-07）
用锁定统计补齐Q1/Q2、两个方向、reader_val/test曲线；不新增GPU工作。Q1按源定义V并标注LLaMA高退化，Q1/Q2不呈一致迁移优势。
配置configs/M6_auxiliary_curves.yaml；图已视觉核查，说明data/M6_v1/analysis/auxiliary_learning_curves.md，索引auxiliary_learning_curve_index.csv。

## M6 2026-11-07
新增M6双源数据、lite适配器迁移、跨偏移统计与预算控制器；旧冻结未改。M6实验与统计完成，待审查；唯一审查reports/M6_REVIEW.md，冻结data/M6_v1/freeze.json。

## M6 完成后附加核验（2026-11-07）
本次35688文件、上游23806文件SHA零差异，96个模型及三seed完整；REVIEW 4451字节、六段、三表，主/补充曲线均已视觉核查。
首次附加核验误将3个上游复用checkpoint未列本次清单当作不一致，现直接SHA核对训练锁全部通过；原核验保留，不涉及实验文件变更。
证据data/M6_v1/verification.json及verification_attempt1.json；科学结果与冻结清单不改。

## 2026-11-07 — M7.1压缩表示与准入规则
新增docs/M7.1_TASK.md、features.py、独立M71配置/运行/成本；M7与Idea A冻结不改。
特征为日志记录均值+末8token、固定正交投影512维；f保持raw KV梯度。
层扫描按1000训练组的800/200内部划分，固定规则选4层，配对比较投影前后损失。
CUDA确定性cumsum失败保留，改为同一线性池化的矩阵实现后val smoke通过。

## 2026-11-07 — M7.1附加协议与资源核验
用户授权cal_A/B各300 receiver文本校准、natural val100/test300；均预固定组SHA子集，test仍须最终锁。
GPU2相同设置val12 smoke通过25.4秒；源生成采用GPU1/2/4三张卡按组SHA分片，保持独立clean/CF batch2。
首轮资源smoke解析阶段缩进错误已修复/保留日志；全入口编译与M71/M7共27项测试通过。
M7 2912、M6 35688冻结文件及生成器三源码SHA一致；成本与扫描接续docs/M71_LIVE.json。

## 2026-11-07 — M7.1恢复与C1分支编排
核对并保留GPU1/2/4源任务；31项M71/M7测试通过，未打开test。
首次C1前固定唯一备选critical逐token；失败候选数组逐SHA核验后可释放，manifest/预测/权重/配置保留，第二次C1失败即停止归档。
配置configs/M71/continue_gates.yaml；控制器tmux clr-m71-continue；上游冻结不改。

## 2026-11-07 — M7.1 test暴露边界与统计
新增按原/CF组配对C1 bootstrap10000次；33项测试通过。test读取前持久标记、live标记只增不减；当前test仍封闭。
CPU后续控制器在尚无子任务时重载到clr-m71-continue-v2，源GPU作业不受影响；备选投影门槛失败也按停止分支如实归档。
typed序列改CPU组装后整体搬运，字段与数值不变；证据reports/M71_evidence/tests_resume_v16.log。

## 2026-11-07 — M7.1接续复核
主比较仍在训练、test封闭；报告补充层扫描峰值、CLR−B3区间方向和CLR−监督参照配对差值。
35项相关测试通过（reports/M71_evidence/tests_resume_v19.log）；不改模型、阈值或划分。

## 2026-11-07 — M7.1磁盘缓存维护
清理4个数月前的pip临时安装/下载目录，释放0.761GiB；未改安装环境、模型、实验数据或冻结文件。
清理时间、路径、字节及前后可用空间见data/M71_v1/stale_pip_cache_cleanup.json。配重保持暂停。

## 2026-11-07 — M7.1统计回归验证
新增配对episode重采样的QA ECE/AUROC独立验证，与显式重复样本计算一致；36项相关测试通过。
证据reports/M71_evidence/tests_resume_v20.log；未读取真实test或新增模型调用，正式协议未改。

## 2026-11-07 — M7.1传输缓存维护
清理24个2025年遗留Xet下载分块，释放1.42GiB；保留所有完整模型/数据、环境及科研产物。
证据data/M71_v1/stale_xet_cache_cleanup.json；任务离线运行，配重继续暂停。

## 2026-11-07 — M7.1软件索引缓存维护
用conda clean --index-cache清理可重新下载的repodata索引，未删软件包或环境。
具体释放量和命令结果见data/M71_v1/conda_index_cache_cleanup.json；配重继续暂停。

## 2026-11-07 — M7.1未完成下载缓存维护
清理3个9天前遗留的HF .incomplete下载残片，释放2.32GiB；均无打开句柄/冻结引用，未碰完整权重或数据。
证据data/M71_v1/incomplete_download_cache_cleanup.json；配重保持为空且暂停。

## 2026-11-07 — M7.1最终统计空间维护
清理剩余旧pip临时目录和HTTP下载缓存，均无打开句柄；安装环境、完整模型/数据和冻结结果未改。
释放量与清单data/M71_v1/remaining_pip_cache_cleanup.json；所有科学设置仍受原final_lock约束。

## 2026-11-07 — M7.1完成待审查
足量特征管线、三seed审计/监督参照、锁定校准/最终test与natural诊断完成；上游冻结保持只读。
唯一审查reports/M7.1_REVIEW.md；完整证据reports/M7.1_details.md、data/M71_v1/analysis_v3/。

## 2026-11-08 — M8启动
按用户任务书新增P0可见信息核验、冻结权重输入干预和记录段定位；主协议不改M7.1。任务docs/M8_TASK.md。
# 2026-11-08 M8a 范围修订
用户将未完成M8大矩阵修订为M8a前置核验与鲁棒性评估，预算50→25GPUh；旧任务/P0运行保留并计成本，不执行防御重训矩阵。P0公开期望一致，无需重训或重定阈值；新增独立M8a代码/config与val20扰动smoke，M7.1冻结只读。任务docs/M8a_TASK.md，状态docs/M8a_LIVE.json。


## 2026-11-08 — M8b 新增版本
新增M8b独立源码/config/数据与记录，不修改M8a/M71/M7/M6/Idea A冻结。冻结核验62363项全部通过。
R4训练4000组读取答案前4层状态；R2随机f与R3全层训练新适配器，R1原读取器随机问集。调参及修正全在train/val/cal。
开发修复：LIVE加进程锁、freeze字段兼容、现场/训练同fp16充分统计、随机读取梯度保护；失败/孤儿成本单列。协议docs/M8b_TASK.md。

## 2026-11-09 M8b冻结
新增鲁棒读取设计、三seed适配器、公开期望校准、锁定自适应评估及审查证据；上游冻结只读，完整协议docs/M8b_TASK.md。

## 2026-11-09 M8c 用户预算修订持久化
用户改主E3为三ε、最大可行n且至少200；次要前100组顺序执行。已写入docs/M8c_TASK.md，test解锁新增预算通过/n≥200硬检查。
新增CPU吞吐预算审计与未完成审查报告，保留失败/中止原记录及成本补差；未修改上游冻结，未读取test。

## 2026-11-09 M8c 21GPUh用户最终决定
固定主200组，取消最大可行n扩张；次要前100顺序改最小成本→E1/E2→λ敏感性，超预算停止不再申请。
更新YAML、成本监视/锁定/统计/审查代码，报告R4 val/test A1/A4；此前15GPUh关卡和报告快照保留。

## 2026-11-09 M8c
新增消息依赖direct-lookup、曝光匹配R4、新校准、锁定PGD/二分、三seed统计与审查证据；未修改上游冻结。

## 2026-11-10 M9收尾冻结
新增两源联合CLR-lite三seed、冻结预测CPU统计、论文TeX/PDF/复现清单与六条证据主线；上游冻结不变。
派生conformal从旧QA元数据恢复并对照原结果；不新增M9-A以外模型实验。

## 2026-11-10 — M10新增版本
新增M10任务、独立运行账本、GSM维度适配控制、逐类型单层注意力探针、精确曝光单类型CLR及val-only smoke/正式训练队列。
M9冻结结果及上游源码不修改；test评估仍需所有P1训练checkpoint哈希锁。后续论文产物更新另存M10版本证据。

## 2026-11-10 M10投稿补实验接续
新增HiddenBench独立组/fact-panel协议与真实KV源提取、通用文本CLR/B3/B6、跨用途M9初始化训练和锁定评估入口。
旧paper与FINAL_SUMMARY在任何更新前已逐字节保留66文件，映射data/M10_v1/m9_paper_snapshot/mapping.json；M9冻结未改。
当前仍仅训练/val代理smoke，未打开M10 test。细节docs/M10_EXECUTION_CONTINUATION.md。

## 2026-11-10 M10 P1调度并行接续
实时GPU2/3可用后，停止纯CPU旧GSM队列控制器2984723，保留worker3151269；未启动的LLaMA B6移GPU2、每类型CLR移GPU3，GPT2 B6余两seed仍GPU4。
不改模型/数据/seed/曝光/超参/预算，配置新增_gpu*.yaml；防重由实际worker保留、完成reference核验和每GPU锁共同保证。
证据data/M10_v1/gsm_execution_supplement.json；控制器scripts/m10_resume_gsm_parallel.py configs/M10/resume_gsm_parallel.yaml；无test访问。

## 2026-11-10 M10释放GPU接续
P1每类型任务可在迁移GPU5/审计GPU0完成后接续；源配置仅device变化，不改数据、seed、曝光、超参。
新增按case/seed串行任务锁，在分配GPU前核相同科学配置，已完成任务跳过且不新增run/计费；当前活worker保持。
证据data/M10_v1/released_gpu_scheduling.json；入口configs/M10/released_gpu{0,5}_controller.yaml。

## 2026-11-10 M10最终产物
新增P1投稿控制、HiddenBench/跨用途证据和论文补表/曲线；上游冻结SHA核验，授权paper supersession逐字节可恢复。
实验最终冻结；B5c缺失及HiddenBench小样本不能升格为正结论。

## M10b 独立版本
只读复用M10模型/数据实现，新增M10b独立账本、训练/评估/CPU收尾与T1入口；不改M10冻结。
按用户修订FINAL_SUMMARY/claims：审计为类型化框架并列CLR/B6，跨用途/HiddenBench仅附录。旧字节归档后授权supersession。

## 2026-11-11 M10b资源恢复版本
新增controller_v2按实际缺失ref计剩余预算，resource_wait CPU检查显存后选设备，仅失败seed33同科学设置接续。
原失败/成本保留，未打开test、不改冻结M10；旧ETA失效，状态/审查摘要已更正。

## 2026-11-11 M10b取消显存阈值（用户修订）
停止CPU显存等待队列，controller_v3以min_free_GiB=0直接尝试seed33；仅实际OOM报错，原模型/batch/标签/seed不变。
22GPUh硬上限与异常记录保留；设备cuda:1，执行证据data/M10b_v1/execution_resume_v3.json。

## 2026-11-11 M10b只在启动前检查显存（最新用户修订）
新增resource_wait_v2 CPU后台监控，每30秒采样、连续3次≥22GiB启动；controller_v3运行时min_free_GiB=0，监控器不按显存停进程。
真实OOM保留计费后回启动等待，预算预计超限/22GPUh硬保护不变；五项调度/协议单测通过，tmux clr-m10b-startup-monitor。

## M10b 最终产物
新增GPT2 B5c三seed Q3/Q4、配对CI与T1；审计主张为框架，跨用途/HiddenBench仅附录。
M10及上游代码结果只读；当前paper与FINAL_SUMMARY为用户授权supersession，原字节归档。

## M10c 只读开销产物
新增冻结权重推理计数/测速和已有CSV配对汇总，T6及C1成本措辞；原模型/科学结果保持永久冻结。
旧paper/总结先归档，M10日志漂移与M10b行政文件例外显式披露；未训练或重校准。

## 2026-11-12 — M10d
用户授权旧冻结只读，缺失LLaMA B4新增训练/校准/评估至data/M10d_v1；GPT2 B6直接复用。
新增clr/baselines/m10d.py及YAML入口、最小协议测试；不修改M3/M6/M9/M10冻结代码与产物。
论文同步目录manuscripts/CVPR27-CLR，仅results四格及列最优加粗、追加number_audit；原文备份data/M10d_v1/manuscript_before。

## 2026-11-12T15:34:59.142298+08:00 — M10e
只新增外部基线、LLaMA B2与证据；旧代码/数据/结果/阈值不改。完成后冻结data/M10e_v1；用户最新要求仅交付论文修改Markdown。

## 2026-11-12 — M11-qual
新增YAML驱动只读导出器、12项验证、冻结数据与ZIP；不修改论文和既有科学结果。接续`M11_qual_LIVE.json`。

## M12
新增五项单组件消融与复用汇总、置换诊断、三seed/按题CI及配对差；旧冻结和论文不改。
任务授权及细节docs/M12_TASK.md；证据reports/M12_details.md。

## 2026-11-16 — 目录与 Git 维护
整理 `.gitignore`，排除本地环境/工具/运行产物，修复冻结划分例外，纳入冻结清单及小型论文输入。
删除未冻结缓存、空锁和废弃骨架；重复文档改兼容链接，更新目录说明与接续入口，保留科学文件原路径。
归档现有 M5–M12 及补充分析源码/配置/报告；独立论文仓库状态不变。核验见 `docs/maintenance/20261116_cleanup.json`，未新增实验或改动审查状态。
