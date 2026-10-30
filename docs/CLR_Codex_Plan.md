# Calibrated Latent Reader：Codex 实现计划（M0–M9）

> 用法：每次开新的 Codex 会话，先粘贴 **§0 全局约定**，再粘贴当前里程碑的 prompt。每个里程碑完成后，把 `EXPERIMENTS.md`、`BLOCKERS.md` 和该里程碑的 `reports/Mx_*.md` 发回来审查，审查通过再进入下一个。
>
> 总体顺序：先把潜在推理这条线（自读、单源）跑通，因为它便宜、模型现成；在 G1 关卡确认 CLR 相对 probe 有实质优势后，再扩展到跨模型和潜通信审计。潜通信部分（M7 起）复用 Idea A 的 KV 消息基础设施。

---

## §0 全局约定（每个会话都先粘贴这一段）

```
你在实现一个研究项目 "Calibrated Latent Reader (CLR)"。提案见 docs/IdeaB_Calibrated_Latent_Reader.md（请先通读）。
CLR 的定义：R(z, q, s) → p(a | z, q)，z 是来自源模型 s 的潜对象（潜在思维隐状态或 KV 消息），
q 是类型化问题（choice / score / yes-no），输出校准概率，并用 split conformal 给出预测集。

工程约定：
1. 代码放在 clr/ 包内，结构：
   clr/sources/    每个源模型一个文件（codi.py, coconut.py, huginn.py, latentmas.py, c2c.py），统一接口
   clr/data/       潜对象 dump、问题库、标签生成、splits
   clr/models/     adapter、reader 主体、输出头
   clr/calib/      温度缩放、split / Mondrian / weighted conformal
   clr/baselines/  question-only、线性探针、MLP 探针、LTO 式标量、Patchscope 式读取
   clr/search/     潜空间搜索
   clr/audit/      潜通信审计
   clr/attacks/    攻击
   configs/ scripts/ tests/ reports/
   所有实验用 config 文件驱动（yaml），命令行只传 config 路径和覆盖项。
2. 可复现：固定 seed；每次运行写 runs/<run_id>/ 下的 config 快照、git commit、环境信息、指标 json。
3. 不允许静默回退：模型加载失败、checkpoint 名称不对、显存不够等，直接报错并写进 BLOCKERS.md，
   不要偷偷换成别的模型或更小的设置。如需降级，写明原因并在报告里标注。
4. 数据划分一律按 problem / episode ID，不按样本或步骤划分；splits 存成 data/splits/*.json 并冻结。
   测试集只在最终评估时使用，所有调参只用 val / cal。
5. 每个里程碑先跑一个 smoke 版本（≤ 5% 数据，≤ 15 分钟），通过后再跑完整版本。
6. 每个里程碑结束时：
   - 在 EXPERIMENTS.md 追加一节：做了什么、命令、结果表（均值 ± 95% CI，至少 3 个 seed）、和预期的偏差、负面结果。
   - BLOCKERS.md 记录所有未解决问题、所做的假设、降级。
   - reports/Mx_<name>.md 给出该里程碑的完整报告和图。
   - 已完成里程碑的代码和结果冻结；后续需要改动时新建版本（如 v2），不覆盖旧结果。
7. 如实报告负面结果。不要为了让结果好看而改指标定义、换划分或挑 seed。
8. 记录计算成本：GPU 型号、GPU 小时、源模型前向次数。
9. 每个模块写最小单元测试（tests/），尤其是：标签生成、conformal 覆盖率、选项顺序不变性、划分无泄漏。
```

---

## M0：环境盘点与源模型复现

**目标**：确认所有源模型能加载、能跑、准确率与论文报告一致。不写任何 CLR 代码。

```
M0 任务：
1. 盘点环境：GPU 型号/数量/显存、磁盘、Python/torch/transformers 版本，写进 reports/M0_env.md。
   检查 Idea A 仓库中可复用的部分（KV 消息生成、Qwen3 加载、日志工具），列出打算复用的函数和路径，
   复用方式是 import，不复制代码，不修改 Idea A 的已冻结代码。
2. 找到并核实以下源模型的官方代码与 checkpoint（自己去 HF / GitHub 核实确切名称，不要猜）：
   - CODI：GPT-2 版本和 LLaMA-3.2-1B 版本（GSM8K-Aug 训练）
   - Coconut：GPT-2，GSM8K 和 ProsQA。若无官方 checkpoint，用官方 recipe 训练；先估算 GPU 小时，
     超过 40 GPU 小时则写 BLOCKERS 并暂缓，先推进其他源。
   - Huginn-3.5B（recurrent depth）
3. 为每个源实现 clr/sources/<name>.py 的统一接口：
   - load(config) -> model
   - run(problem, perturb=None) -> {answer, latent_steps: List[Tensor], meta}
     latent_steps 的含义按源定义并写进 docstring：
       CODI/Coconut：每个连续思维步的隐状态（默认最后一层，同时支持指定层）
       Huginn：每次循环迭代后答案位置的递归状态
   - perturb 参数用于后面的 MC rollout 和搜索：perturb = {from_step, sigma, seed}，
     在 from_step 之后的每个潜在步加相对高斯噪声 z <- z + sigma * ||z|| * eps / sqrt(d)。
4. 复现准确率：CODI-GPT2、CODI-LLaMA1B 在 GSM8K test；Coconut 在 GSM8K / ProsQA；Huginn 在 GSM8K。
   与论文数字对比，偏差 > 2pp 要分析原因。
5. 确定性检查：同一输入跑两次，latent_steps 逐元素差值的最大值记录下来（bf16/fp16 下的非确定性要量化）。
6. 噪声敏感性曲线：sigma ∈ {0, 0.01, 0.03, 0.1, 0.3}，from_step=0，报告准确率和答案多样性
   （K=16 次 rollout 中不同答案的个数）。这条曲线用来在 M2 选定 MC rollout 的 sigma。

交付：reports/M0_env.md、reports/M0_repro.md（复现表 + 噪声曲线图），EXPERIMENTS.md / BLOCKERS.md 更新。
```

**审查要点**：复现数字是否对得上；噪声曲线上是否存在一个 sigma，使准确率下降不多（< 5pp）但答案有足够多样性（平均 ≥ 2 个不同答案）。没有这样的 sigma，MC 价值标签就退化成 0/1，需要先讨论。

---

## M1：潜对象 dump 与存储格式

**目标**：把所有源的潜对象一次性 dump 成统一格式，后续所有实验只读 dump，不再反复跑源模型。

```
M1 任务：
1. 划分（冻结，写 data/splits/）：
   - reader_train / reader_val：来自源模型训练集的问题（GSM8K-Aug train 的子集，规模按显存和磁盘决定，
     目标 ≥ 20k 题；ProsQA 用其 train）
   - cal / test：来自 benchmark test 集，按 problem ID 对半划分（GSM8K test 1319 题 → 约 660 / 659）
   - 写测试检查任何 problem ID 不会出现在两个 split 中。
2. dump 格式：data/dumps/<source>/<split>/shard_*.safetensors（fp16）+ manifest.parquet
   manifest 每行：problem_id, source, step, layer, shard, offset, dim, gold_answer, model_answer(sigma=0), correct
   同时保存每个问题的元数据：问题文本、gold 解（GSM8K-Aug 的方程式 CoT）、中间量列表、ProsQA 的图结构。
3. 默认 dump 最后一层；额外对一个 1k 题子集 dump 所有层，供 M3 做层选择分析。
4. 写 clr/data/loader.py：按 (source, split, step, layer) 高效读取，支持 batch 和 memory map。
5. 统计报告：每个源每个 split 的题数、步数、维度、磁盘占用、dump 用时。

交付：dump 数据、loader、tests/test_splits.py、reports/M1_dump.md。
```

---

## M2：类型化问题库与标签

**目标**：构造训练和评估 CLR 用的 (z, q, a) 三元组。关键原则：**标签只能来自行为（rollout 结果）或程序（题目结构），不能来自我们对潜状态"应该编码什么"的假设。**

```
M2 任务：
1. MC 价值标签：
   策略 π_σ 定义：从第 t 步的潜状态 z_t 出发，对 t+1..T 步施加 M0 选定的 sigma 噪声，贪心解码答案。
   V(z_t) = P(正确 | z_t, π_σ)，用 K=16 次 rollout 估计（只在 reader_train 的一个子集上做全量 rollout，
   子集大小按 GPU 预算定，并报告标签的估计方差；cal / test 全量做）。
   这个 π_σ 的定义必须与 M5 搜索里生成候选所用的扰动完全一致，写进 config 并共用同一函数。
2. 问题类型（每类单独一个 question template 文件，模板文本有 ≥ 5 种改写以防模板过拟合）：
   Q1 score：「从当前状态继续，最终答对的概率？」 标签 = V(z_t)（软标签，分 10 个 bin 或回归）
   Q2 yes-no：「如果现在强制输出答案，是否正确？」 标签 = 在 step t 早退解码的正误（行为标签）
   Q3 choice：「模型最终会输出下列哪个答案？」 选项 = {sigma=0 下模型最终答案, gold, 2 个干扰项}，
      干扰项来自其他 rollout 的答案或数值邻近值；去重后不足 4 个时用数值扰动补足；选项顺序随机。
   Q4 yes-no：「数值 v 是否是本题标准解中的中间量？」 正例来自 gold 方程式 CoT，负例为同量级的随机数
      或其他题的中间量。（这是题目层面的语义问题，用 question-only 基线检测先验泄漏）
   Q5 （ProsQA）yes-no：「节点 v 是否在从起点到答案的路径上？」「节点 v 是否在第 t 步可达集合内？」
      标签由图结构程序生成。
3. 每个 (z_t, 问题类型) 采样若干问题实例；保存为 data/qa/<source>/<split>.parquet：
   problem_id, step, qtype, template_id, question_text, options, label, label_soft
4. 标签分布报告：每类问题的正例率、各 step 的 V 分布、Q3 中「模型答案 = gold」的比例。
   若某类问题的标签极度不平衡（正例率 < 5% 或 > 95%），报告并提出重采样方案，不要自行修改问题定义。
5. 单元测试：标签生成器在手工构造的 10 个样例上结果正确；Q3 选项中正确答案位置均匀分布。

交付：问题库、标签数据、reports/M2_labels.md（含 rollout 成本统计）。
```

---

## M3：基线读取器

**目标**：先弄清楚「线性能读出多少」。这决定 CLR 的卖点能不能成立。

```
M3 任务：实现并评估以下基线（全部在 reader_train 训练、reader_val 选超参、cal 上做温度缩放、test 上报告）：
B0 question-only：不看 z（z 置零），只看问题和选项。用来度量问题先验。
B1 shuffled-z：z 换成另一道题同一步的 z。用来度量「任意潜状态」带来的伪信号。
B2 线性探针：每个问题类型 × 每个模板一个 logistic / softmax 探针（Q3 用 z 与选项文本 embedding 的双线性打分）。
B3 MLP 探针：2 层 MLP，同样按问题类型分别训练。
B4 LTO 式标量 LRM：z_t → P(最终正确)，只回答 Q1/Q2。
B5 Patchscope / SelfIE 式读取：把 z 投影后注入一个冻结 LLM（源模型自身，或 Qwen3-1.7B）的某个占位 token，
   用问题 prompt 读出答案 token 的 logits，作为 LatentQA/UAV 一类「语言化后分类」方法的廉价代理。
   如有时间，加一个训练过 decoder 的 LatentQA 式版本（注入 + LoRA 微调），标注为 B5b。

指标（按源 × 问题类型 × step 报告）：
- 准确率 / AUROC（yes-no）、top-1 准确率（choice）、Brier 和 MAE（score）
- ECE（15 bins）温度缩放前后
- 层选择分析：在 1k 全层子集上，B2 线性探针的准确率随层变化的曲线

交付：clr/baselines/*、reports/M3_baselines.md。
必须在报告末尾回答：B2/B3 在各问题类型上离「完美 + 已校准」还有多远？B0 的先验有多强？
```

**审查要点（G0）**：如果 B0 在某类问题上已经很高，说明该问题有先验泄漏，要先修问题设计；如果 B2 在所有问题上都接近完美且校准良好，自读单源场景就没有 CLR 的空间，重心要提前转向跨模型和潜通信。

---

## M4：CLR v0（单源自读）

```
M4 任务：
1. 架构（clr/models/）：
   - 源适配器：每个源一个线性或 2 层 MLP，把 z（d_s 维）投影到 reader 维度；输入可以是单步 z_t，
     也可以是 z_1..z_t 的序列（默认序列，加步位置编码）。另加一个可学习的源 token。
   - Reader 主体：双向 encoder，起步用 0.1B 级（如 ModernBERT-base 或 deberta-v3-base 初始化，自行核实可用性），
     规模扫描到 0.4B 左右。问题 token 和选项 token 用块注意力掩码：每个选项只看问题和自身，选项之间互不可见，
     保证对选项顺序不变（参考 Narrow Decision Models 2610.25553，读原文确认掩码细节）。
     通过 cross-attention 读取适配后的潜对象（每层或每隔 k 层插入）。
   - 输出头：choice 头（每个选项一个 logit，softmax）、yes-no 头、score 头（10-bin 分类，输出分布也可算期望）。
2. 训练：多问题类型混合训练，按类型平衡采样；early stopping 看 reader_val 的平均 NLL。
3. 校准：在 cal 的一半上按问题类型做温度缩放；另一半做 split conformal（Mondrian：每个问题类型单独定阈值）。
   为保持可交换性，conformal 校准时每个 (problem, qtype) 只取一个问题实例。
   α ∈ {0.05, 0.1, 0.2}，报告 test 上的经验覆盖率和平均集合大小；做 100 次随机 cal 重划分，报告覆盖率分布。
4. 消融：
   - 无 cross-attention（把 z 平均池化后拼到 [CLS]）
   - 单步 z_t vs 序列 z_1..z_t
   - 去掉块注意力掩码（普通全注意力）；同时报告选项随机置换后预测的变化（应为 0）
   - shuffled-z 和置零 z 对照
5. 与 M3 全部基线同表对比；计算成本：reader 一次前向的 FLOPs / 延迟 vs 源模型一步前向。

交付：reports/M4_clr_v0.md，含主表、可靠性图、覆盖率图、消融表。
```

**审查要点（G1，关键关卡）**：CLR 必须在准确率不低于最好 probe 的前提下，至少在以下一项上明显占优：校准（ECE）、conformal 效率（相同覆盖率下集合更小）、一个读取器服务多个问题类型的样本效率（少量标签迁移到新问题模板/类型）。如果都不占优，就讨论是否把论文重心改成「跨模型 + 潜通信」，单源推理部分只作为分析。

---

## M5：RQ1 潜空间搜索

```
M5 任务：
1. 候选生成：在第 t 步，用 π_σ 的同一扰动函数从 z_t 生成 N 个候选（N ∈ {4, 8, 16}）。
2. 搜索方法：
   S0 greedy（sigma=0 单次）
   S1 多数投票：N 条独立扰动轨迹，最终答案投票（计算量匹配的主要基线）
   S2 Best-of-N + 价值函数：用 V 选最终轨迹，价值函数分别取 B4（LTO 式）、B2、CLR
   S3 Beam search：每步保留 top-b 个候选，按价值函数排序；价值函数取 B4 / CLR / MC 真值（oracle，作为上限）
   S4 CLR + conformal 提前停止：当 Q3（最终答案 choice）的 conformal 集合缩为单元素时停止扩展。
3. 计算量匹配：横轴为「源模型潜在步前向次数 + reader 开销换算成等价前向」，画 accuracy-compute 曲线。
   所有方法在同一组 compute 预算点上比较。
4. 数据：GSM8K（CODI-GPT2、CODI-LLaMA1B、Coconut）、ProsQA（Coconut）、Huginn 的循环深度维度；
   如时间允许加 Countdown / ProntoQA。
5. 分析：价值函数的校准误差（ECE）与搜索增益的关系（为理论中「次优差距随 ε 线性有界」提供经验图）：
   人为对 CLR 的输出加不同程度的温度失配，看增益如何下降。

交付：reports/M5_search.md，主图为 accuracy-compute 曲线。
```

**审查要点（G2）**：在至少一个源 × 数据集上，CLR 引导的搜索在相同计算量下显著超过多数投票和 LTO 式标量。否则 RQ1 降级为分析性结论。

---

## M6：RQ2 / RQ4 跨源读取与分布偏移下的覆盖率

```
M6 任务：
1. 多源训练：一个 reader 主体 + 每源适配器 + 源 token，在 CODI-GPT2、CODI-LLaMA1B、Coconut、Huginn 上联合训练。
   对比：每源单独训练的 CLR；多源 CLR。
2. 留一源（leave-one-source-out）：在 3 个源上训练，对第 4 个源只训练适配器（reader 主体冻结），
   适配器训练标签数 ∈ {100, 300, 1k, 3k, 全量}，画学习曲线；对比同样标签数下从零训练的线性探针 / MLP 探针。
3. 跨模型基线：B5（Patchscope 式）用一个固定的冻结 LLM 读其他源的 z，加线性对齐层；作为
   「语言化类方法跨模型」的对照（呼应 2509.13316 的负面结果）。
4. 覆盖率在偏移下（RQ4）：
   - 偏移类型：跨源（cal 在源 A，test 在源 B）、跨数据集（GSM8K → 其他数学集，若可得）、跨 step（cal 用前几步，test 用后几步）
   - 方法：标准 split conformal、Mondrian（按源）、weighted conformal（用一个源/域分类器估计似然比）
   - 报告经验覆盖率偏离 1−α 的程度和集合大小
5. 分析报告：「跨模型能读多少」—— 按问题类型列出哪些能跨源迁移、哪些不能。

交付：reports/M6_cross_source.md。
```

---

## M7：潜通信管线与审计数据

**依赖**：Idea A 的 synthetic v2（M1.1）生成器和 KV 消息代码已稳定。

```
M7 任务：
1. 通信管线（clr/sources/latentmas.py, c2c.py）：
   - LatentMAS 风格：sender（Qwen3 系列）把带私有信息的上下文编码成 KV / 潜在思维，receiver 直接消费。
   - C2C：用官方代码和 fuser checkpoint（核实可用的模型对），sender→receiver 跨模型。
   - 复用 Idea A 的 KV 消息生成代码（import），并在报告里说明两边的消息格式是否一致。
   每条消息的潜对象 z = sender 产生的 KV（按层取 key/value，或取 fuser 输出）；记录可见的文本 commitment
   （sender 同时输出的一句简短文本结论）。
2. 任务：
   - HiddenBench（核实可得性和格式）及其变体
   - Idea A synthetic v2：私有事实已知、可程序生成，用作主力审计数据集
3. 审计问题库（标签全部由生成器给出）：
   A1 yes-no：「消息是否包含事实 F？」（F 为真私有事实 / 同类型的假事实）
   A2 yes-no：「消息是否支持结论 C？」
   A3 yes-no：「消息与可见 commitment 是否一致？」
   A4 choice：「消息中某私有变量的取值是？」
4. 自然负例：
   - 错配消息：另一个 episode 的 KV 配上本 episode 的 commitment
   - 反事实消息：只改一个私有事实后重新生成的 KV，commitment 保持原样
   - 截断 / 部分层置零的消息
5. dump 与 M1 相同格式，按 episode ID 划分 reader_train / val / cal / test。
6. 基础检查：receiver 用 / 不用消息时的任务准确率（确认消息确实传递了信息，参考 2607.26773 的因果检验）。

交付：reports/M7_comm_data.md。
```

---

## M8：RQ3 审计与攻击

```
M8 任务：
1. 审计器：在 M7 数据上训练 CLR（可以从 M6 的多源 reader 初始化，比较从零训练）。
   一致性分数 = 对随机抽取的 k 个审计问题，CLR 答案与 commitment / 已知约束一致的平均对数概率；
   用 cal 上的干净消息做 conformal 阈值，保证误报率 ≤ α。
2. 非自适应攻击：
   - 随机噪声、范数约束的 KV 扰动
   - 复现 When Latent Agents Lie（2606.28958）的 KV 篡改；复现 2605.28214 的 latent attack
     （读原文和代码，复现不了的部分写 BLOCKERS）
   - M7 中的反事实消息替换（最贴近「文本正常、潜内容被换」的威胁模型）
3. 自适应白盒攻击：攻击者知道 CLR 权重和问题分布。
   用 PGD 优化 KV 扰动 δ：最大化 receiver 的目标行为改变，同时最小化被 CLR 判为不一致的概率
   （对问题分布取期望，用 EOT），约束 ||δ|| ≤ ε（相对范数），扫描 ε 和攻击步数。
4. 防御设计的消融：
   - 固定问题集 vs 每次随机抽题（k ∈ {1, 4, 16, 64}），画检测率随 k 的曲线，与「独立性假设下指数下降」的理论曲线对比
   - 因果一致性检查：比较 receiver 用 / 不用消息时 CLR 读 receiver 状态的变化
   - 基线：范数检测、线性探针审计、文本审计（只看 commitment 和 receiver 输出）
5. 指标：检测率 @ FPR 1% / 5%、AUROC；攻击成功率（行为改变且未被检测）；攻击成本（PGD 步数、需要的 ε）。
   明确报告白盒自适应攻击下的上界，不要回避 CLR 被绕过的情况。

交付：reports/M8_audit_attacks.md。
```

---

## M9：跨用途迁移与论文产物

```
M9 任务：
1. 跨用途迁移：推理数据 + 通信数据联合训练的 CLR vs 各自单独训练，比较两边的指标；
   以及只在推理数据上训练、通信数据只训练适配器的迁移效果。
2. 理论的数值验证：覆盖率（M4/M6 已有）、检测率随问题数的下降（M8）、校准误差 vs 搜索增益（M5），
   整理成统一的图。
3. 汇总所有主表，生成 paper/tables/*.tex 和 paper/figures/*.pdf，每张图表的生成脚本可一键重跑。
4. 写 reports/FINAL_SUMMARY.md：每个 RQ 的结论、证据、负面结果、尚未解决的问题。
```

---

## 关卡汇总

| 关卡 | 时机 | 通过条件 | 不通过时 |
|---|---|---|---|
| G-σ | M0 后 | 存在一个 sigma，使准确率下降 < 5pp 且有答案多样性 | 换扰动方式（dropout、温度采样答案）再讨论 |
| G0 | M3 后 | 问题先验（B0）不强；线性探针未饱和 | 修问题设计；或重心提前转向跨源 / 通信 |
| G1 | M4 后 | CLR 在校准 / conformal 效率 / 多类型样本效率至少一项明显优于 probe | 论文重心改为跨模型 + 潜通信 |
| G2 | M5 后 | 计算量匹配下 CLR 搜索 > 多数投票和 LTO | RQ1 降为分析 |
| G3 | M8 后 | 非自适应攻击检测率高；自适应攻击需要显著更大代价 | 定位为「提高攻击成本」，如实报告上界 |

## 每次发回审查时请附上

`EXPERIMENTS.md` 新增部分、`BLOCKERS.md`、`reports/Mx_*.md`，以及主表对应的 `runs/<run_id>/metrics.json`。
