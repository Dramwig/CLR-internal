# M6：跨源读取、偏移覆盖率与 RQ1 收尾

授权：2026-11-06 用户当前任务书。M5b 结论按用户接受的背景继续；CLR 主体仅 CLR-lite，不再推进 CLR-v1。冻结 M1/M23/M4/M41/M5a/M5b 均不修改。

## A（val300，≤3 GPUh）
候选 sigma={0.3,0.75}，包含 sigma=0；S3 N8/b4/全部6步；价值为原 sigma1.5 的 lite/B3。报告相对 greedy 差、pass@N。复用历史跨题AUROC/组内rho逐t、pass@N随N、早退图，不新增该分析计算；reports/M6_rq1_closure.md。

## B（≤30 GPUh，优先）
CODI-LLaMA1B 沿用已审计 bf16/官方权重、M1 problem ID/cohort 协议（各源 tokenizer 独立计算原 cohort 的 padding）。val300 sigma={.5,1,1.5,2}，相对t策略、K16、一seed；drop<5pp内非退化比例最大，1.5满足则优先。M0.1 非退化门槛30%同时报告，若无合格项明确阻塞。
reader_train=GPT2前10000 ID、reader_val前1000；val/cal/test全量。clean/全部早退、训练分布K8轨迹、heldout K16 MC（pi及pi_T，M1规则）；Q1仅clean起点及后续观测；Q4逐题值配对直接复用GPT2，子集保留对应正/负数值，不重新配对。smoke≤5%/15min；若估算>30GPUh按比例缩小reader_train，不改heldout/K/精度。
Huginn仅整张空闲≥24GB GPU时复现后生成；不可用写BLOCKERS，不阻塞。

## C/D（共≤27 GPUh）
3seed={11,22,33}。C1每源单独lite/B3，test+reader_val；可复用GPT2冻结模型作为已完成单源参考。C2共享lite主体、每源adapter+source token。C3双向冻结主体、只训练目标adapter，标签N={100,300,1000,3000,全量}，与同N scratch B3/lite比较；N按不同QA实例计（同题/轨迹不假设独立），固定嵌套且类型平衡面板，各模型共用。C4冻结Qwen3-1.7B、仅线性z映射，读Yes/No或A-D logits；两源及只重训目标映射。C5同题Q3一致性及两源都对/一对一错/都错分组。
D：跨源 A cal_B→B test；GSM cal_B→SVAMP/ASDiv unseen_eval（GPT2现有；LLaMA预算许可）；t≤2 cal→t≥4 eval。标准split、按源/数据集Mondrian、weighted（域分类器只用cal数据）。每题每qtype独立固定一实例；校准目标/偏移目标及Mondrian是否使用目标cal须区分。alpha={.05,.1,.2}；coverage-target及size，两模型。估计权重不构成无条件保证。

## 执行/交付
总≤60GPUh，含smoke/失败/重启，各GPU作业墙钟求和，不按利用率打折。优先级B→C1→C3→D→C2→C4→C5→A。所有运行YAML/config、启动代码快照、commit/环境/seed/成本；smoke和调试仅val代理，不读test。所有训练及方法锁定后最终test。预算内不够的低优先项如实标未执行，不冒充完成。
唯一审查reports/M6_REVIEW.md，≤8000字节，固定(a)–(f)，最多3主表，3有效数字。详细报告/CSV/图留仓库；最后回答用户四问题：LLaMA Q3/Q4复现、小N迁移对B3/B5b、偏移失覆盖及weighted恢复、小sigma是否改变RQ1负面结论。状态docs/M6_LIVE.json；新实验/阻塞单条≤10行。

## 成本执行细则（不改科学设置）
固定reader_val/val/cal/test先生成并审计合并，再只按已用GPUh复核reader_train前缀；保留plan_initial.json，train开始后不变更该计划。生成用固定8题块、三GPU，所有GPU作业墙钟求和。主比较沿用M4.1 clean/逐步micro再平均与按题cluster CI，Q1另列traj；预算准入只依赖成本，CI/结果不决定设置。

## 用户预登记修订：B1 扩展扫描（2026-11-06，扩展运行前）
原sigma={0.5,1,1.5,2}全部未达nondeg≥30%，原结果/选择null不改。
扩展sigma={3,4,6,8}，val300、K16、相对t的π、seed11、同协议；扩展扫描含smoke/失败/重启≤2GPUh，计入B30/总60。
对全部8点统一选择：若有drop<5pp且nondeg≥30%，取nondeg最大（并列sigma小）；否则取drop<5pp中nondeg最大（并列sigma小），标“门槛未达，降级使用”并写BLOCKERS；若新4点全drop≥5pp，回到原4点按该降级规则。
本修订替代本任务书先前的sigma1.5优先/不合格即停止；历史版本保留在旧run启动快照，不改旧四点失败。
Q2/Q3/Q4行为标签定义不改；Q1标签与扰动状态使用最终源内π。若降级，LLaMA Q1结果单列“标签退化程度高”；跨源Q1说明π噪声尺度按源区分，不视为同一个V目标。
扫描完成立即确定，随后按已授权预算生成数据；不得扫描中再变规则、挑t或seed。

## 可选 LLaMA unseen 成本核查（最终 reader test 前登记）
C1/C3完成后、C2/C4之前，按 `configs/M6_llama_unseen_budget.yaml` 用已锁定 source smoke 的 heldout 秒/题、C1 reader_val 实测推断秒/QA、冻结 unseen 两池题数/QA行数估算完整1993题及六个C1 reader推断；只依赖成本、不依赖结果。生成留20%余量、推断留50%及每模型90秒加载余量、CD额外保留0.2GPUh。
从CD余量先扣除已承诺核心reader最终推断预留，再判断能否容纳完整补充。可容纳则执行源数据/QA/审计，并给C1 LLaMA最终推断增加unseen两池；不足则将量化理由存 `data/M6_v1/llama_unseen_budget.json` 并在报告/BLOCKERS说明，不直接无估算声称预算不允许。C2/C4准入继续保留已承诺的额外推断成本；全部reader训练锁定后才最终推断。此执行顺序落实用户D→C2→C4优先级，不改变任何训练设置或 test 指标选择。

C4成本复核：预算准入时可补一项15题reader_train/15题val代理的Qwen吞吐smoke（val为5%，≤15分钟，计CD；不读test）；正式训练参数不变。用冻结QA的模板标志/ID计数与固定128题验证面板计算工作量，不读取状态、答案、标签或logits。后续完整六模型族完成后可按最大实测训练/验证秒/QA更新成本估计与推断预留，始终保留20%余量及每模型90秒启动开销；不使用准确率/NLL决定准入。旧配置的粗总行数上界保留为历史估计，不再代替已知精确规模。
