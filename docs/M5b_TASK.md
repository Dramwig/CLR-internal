# M5b：CLR 混杂对照与搜索价值函数诊断（2026-11-06）

## 范围与冻结

- A 可在 smoke 后使用冻结 test 做预先指定的混杂对照；B/C/D 只使用冻结 val300/reader_train，禁止读取 test。
- 搜索源协议沿用 M5a：CODI-GPT2、fp32、temperature=0、原 cohort 左 padding、sigma=1.5、实际观测状态和完整 KV。
- A 最初的冻结权重输入干预方案已作废（原方案为均值/当前混合）；正式对照为下文预先锁定的从头重训版本。
- B 的每个 val 题、每个 t=0..5 生成 16 个 sigma=1.5 子候选及 1 个 sigma=0 子候选；每个候选 K=16 π rollout 得到 V̂。V̂ 读取 gold 仅用于 oracle 标签；价值函数不读取题目原文或 gold。
- B 报告跨题 AUROC、同父兄弟 Spearman、|ΔV̂|≥0.25 成对排序准确率，以及价值函数选择相对 sigma=0 的 V̂ 增益，按 t 分层。
- C 所有候选集合包含 sigma=0 子候选；S0/S1、S2 Best-of-N、S3 beam(N=4/8/16,b=2/4)，全步和 t≤3；B4、B3、CLR-lite、CLR v1 及 D 版本，oracle/pass@N 仅作参考。
- D 在 reader_train 生成至少 3,000 题×2 个 t 的 1 clean+7 noisy 兄弟（K=4），在 CLR-lite 上以 BCE+加权 pairwise ranking loss 训练，温度校准保留。

## 计算与统计

- 总上限25 GPUh，含 smoke、失败和报告；先 smoke（≤5%数据、≤15分钟）再按 A→B→C→D 估算决定完整范围。
- A 三 seed、reader_val/test 并列；test 只用于本预先指定的 A 对照。确定性指标按题 bootstrap10,000；CP 按 qtype 报告覆盖率/集合大小和同覆盖率配对 CI。
- B/C/D 的 seed 由配置固定为11/22/33；先按题平均 seed，再按题 bootstrap10,000。D 训练数据只来自 reader_train，模型选择只看 reader_val/val smoke。
- 主图横轴为实际 source-forward 等价次数，probe FLOPs 换算并计入；oracle K16 单列并标明特权 gold/MC 误差。所有原始 rows、成本和输入 hash 留在 `data/M5b_v1/`。

## 输出

唯一审查文件 `reports/M5b_REVIEW.md`，≤8KB、固定(a)–(f)、≤3张主表、3位有效数字。详细报告、CSV、图、run 和冻结清单索引于审查文件。

## 实现复核修订（2026-11-06；test 结果不用于选择设置）

- A 正式对照从头重训：B3-prefix 输入为 `concat(mean(normalize(z_0..z_t)), normalize(z_t))`，隐藏层128、每类型8epoch×两LR、batch4096，与原B3一致；增加输入维数导致参数数增加，另列成本。
- CLR-lite-single 仅输入当前观测（单一位置0），从头重训，沿用M4.1全量遍历、逆类型权重、batch1024、LR3e-4、WD1e-4、8epoch后允许选模、最多10epoch/patience1。
- 早期冻结权重输入干预run的B3混合了未归一化均值与已归一化当前状态，且无重训；该run全部作为无效开发结果保留，不用于结论。test已暴露这一历史明确保留；后续模型/超参全部由本节预先锁定，不按该test结果调整。
- C 中 N 为每父状态候选总数，含1个sigma=0；S2 clean lane全程无噪声，其余轨迹在选定范围内π、范围外sigma=0，终点选；S3每次扩展含clean，至范围终点选一条，余段sigma=0。范围4代表只扰动/扩展t=0..3。
- S1 保留N条独立π多数投票以便与M5a对照，另列含clean的S1-clean。C 的oracle仅列终点S2；温度0下其K16期望可解析为gold指示，等于pass@N，不执行16次重复终点解码。
- B raw候选、K16 Bernoulli与观测前缀保存后供D复评分；组间AUROC跨题池化，不把兄弟内部AUROC冒充跨题AUROC。
- D 正式微调固定LR=1e-4、5epoch、每batch32个兄弟组、排序系数1；loss=`0.5*(原Q1 Bernoulli CE + 新兄弟K4独立Bernoulli CE)+按|ΔVhat|加权pairwise softplus`。新兄弟训练3000题×2t，额外300题仅训练侧监测（M4.1预训练见过，不能当独立验证）；checkpoint按reader_val固定750题Q1 NLL选择，另外250题只做温度校准。B/C val300标签不参与选模。原Q1重放排除这300个训练侧监测题；不因预训练见过它们而声称独立泛化。
- 资源执行可按seed并行，科学设置/总计算不变；控制器CPU等待不计GPUh，各实验作业的完整墙钟保守求和计GPUh。
- A 的逐qtype同覆盖率比较同时纳入冻结CLR-v1与B3，以补齐M4.1原先合并口径；不新增test预测或模型选择。同覆盖率曲线使用评估标签匹配90%覆盖率，仅为描述性效率诊断，不冒充独立cal_B阈值的部署保证。
