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
