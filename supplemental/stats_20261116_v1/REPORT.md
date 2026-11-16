# CLR 论文补充统计与 LLaMA 组件消融


生成时间：2026-11-16T02:56:43+08:00。所有差值为配对点估计 [区间下界, 上界]。除 TOST 特别列出的 90% 区间外，均为 95% 区间。


## 1. 结论摘要

- **A1：**40 个变体×指标比较中，唯一“显著更好”是单类型模型的 Q3 集合大小：-0.0158 [-0.0253, -0.00649]；其他表面改善均不显著。

- **A2：**Q3/Q4 accuracy 及集合大小全部通过预先固定的 ±2.0pp / ±0.10 等价性检验；accuracy 差值分别为 -0.659 [-1.51, 0.19]pp、-0.0864 [-0.984, 0.83]pp。

- **A3：**有 v1/lite 的预定义 validation 二选一规则；没有 single/default/joint 三行统一择优规则。最终修复架构是在 v0 test 已被观察之后设计，不能称为未见 test 的预注册配置。

- **A4：**相对实测最强单类型 reader，GPT-2/LLaMA 的 Q1 NLL 分别高 0.0415/0.0706，Q2 AUROC 分别低 0.0277/0.0308，四项差距均显著。

- **B：**LLaMA 三 seed 实验完成，Q3 的 dot-product / late fusion / single-step 差值依次为 -3.7 [-4.91, -2.48]pp；+3.6 [2.45, 4.75]pp；+2.6 [1.59, 3.62]pp；新增 3.67 GPUh，源前向 0。


## 2. 统计实现核对

直接调用原 `clr/paper/statistics.py → scripts/m6_report.py::cluster_estimate` 复现；新扩展仅暴露同一批抽样的 90% 分位数。10,000 次、NumPy `default_rng(11)`，按 test problem 有放回抽样，保留题内所有 step 和三个固定训练 seed。先在 step 内按 QA 数微平均，再对 step 等权、固定 seed 平均。AUROC 在每次抽样内重新计算，处理并列分数，不能用逐题 AUC 或两个独立区间相减。区间条件于固定权重及原校准集，不覆盖重新训练的 seed 不确定性。CLUE 为一次确定性评估，每次使用同一题抽样与 CLR 配对，不制造独立三 seed。

|CLR 减比较对象|类型|本次复现（pp）|任务给定/历史记录（pp）|
|---|---|---|---|
|CLR-per-type|Q4|2.51 [1.45, 3.6]|2.51 [1.45, 3.6]|
|CLR-per-type|Q3|-0.535 [-1.31, 0.261]|-0.535 [-1.31, 0.261]（点值本次恢复）|
|B3|Q3|17.5 [16.1, 19]|17.5 [16.1, 19]|
|B3|Q4|7.93 [6.35, 9.52]|7.93 [6.31, 9.56]|
|B5c|Q3|-0.659 [-1.51, 0.19]|-0.659 [-1.51, 0.19]|
|B5c|Q4|-0.0864 [-0.984, 0.83]|-0.0864 [-0.984, 0.83]|


最大差异 0.0384pp，低于 0.1pp 停止阈值，核对通过。上述为 M10/M10b 原导出的复放；verbalizer 的精确核对值来自原 M10b 记录（任务仅要求其区间含零）；任务给定 CLR−MLP Q4 为 7.93 [6.31, 9.56]pp，本次为 7.93 [6.35, 9.52]pp，保留这个小差异。计数矩阵加速与原函数最大差异 <3×10⁻¹⁷；A1 全部 45 格与冻结 M12 汇总最大误差 4.44e-15。

**数字差异审计：**Table 1/2 共 123 个已报告数字，在各自显示精度上全部一致；没有需要替换的表值。完整精度、123 格审计及所有输入 SHA 保存在本目录。所有“显著”均为未校正的逐比较 95% 区间判断；A1 的 40 项及 B 的 25 项消融比较均属探索性分析。


## 3. A1–A4 的结果表


### A1. 消融的双向配对区间

下表均为 **变体−CLR**。Q3/Q4 accuracy 差值为百分点；NLL、AUROC、集合大小用原尺度。基准 CLR 的绝对值为 0.548 / 0.830 / 76.1% / 76.2% / 1.49。Q1/集合大小越低越好，其余越高越好。

|变体|Q1 NLL|Q2 AUROC|Q3 acc（pp）|Q4 acc（pp）|\|C_Q3\||
|---|---|---|---|---|---|
|CLR|0 [0, 0]；不显著|0 [0, 0]；不显著|0 [0, 0]；不显著|0 [0, 0]；不显著|0 [0, 0]；不显著|
|w/o block mask|+0.0202 [0.0134, 0.0273]；显著更差|-0.00713 [-0.0139, -0.000526]；显著更差|-1.05 [-1.91, -0.199]；显著更差|-0.768 [-1.67, 0.132]；不显著|+0.0468 [0.0361, 0.0579]；显著更差|
|dot-product head|+0.0354 [0.0253, 0.0453]；显著更差|-0.0104 [-0.0201, -0.000746]；显著更差|-3.24 [-4.2, -2.3]；显著更差|-1.86 [-2.84, -0.859]；显著更差|+0.0903 [0.0783, 0.103]；显著更差|
|w/o numeric feat.|+0.00127 [-0.00645, 0.00881]；不显著|+0.00395 [-0.00253, 0.0106]；不显著|-1.15 [-1.89, -0.412]；显著更差|-0.758 [-1.54, 0.0408]；不显著|-0.00227 [-0.0114, 0.00678]；不显著|
|late fusion|+0.00831 [-0.0034, 0.0194]；不显著|+0.008 [-0.00193, 0.0179]；不显著|-2.13 [-3.09, -1.16]；显著更差|+0.54 [-0.494, 1.57]；不显著|+0.0442 [0.0317, 0.0569]；显著更差|
|pretrained encoder|+0.0157 [0.00738, 0.0239]；显著更差|+0.000689 [-0.00659, 0.008]；不显著|+0.109 [-0.739, 0.938]；不显著|-1.15 [-2.07, -0.238]；显著更差|+0.0238 [0.0139, 0.0338]；显著更差|
|single step z_t|+0.016 [0.00682, 0.0251]；显著更差|-0.0103 [-0.0194, -0.00112]；显著更差|-2.28 [-3.19, -1.36]；显著更差|-1.37 [-2.3, -0.454]；显著更差|+0.121 [0.109, 0.134]；显著更差|
|w/o type weighting|+0.0264 [0.019, 0.0339]；显著更差|-0.000916 [-0.00783, 0.00613]；不显著|-0.242 [-0.933, 0.464]；不显著|+0.394 [-0.42, 1.22]；不显著|+0.0114 [0.0028, 0.0204]；显著更差|
|one model per type|+0.00924 [-0.000357, 0.019]；不显著|-0.00611 [-0.016, 0.00347]；不显著|+0.535 [-0.261, 1.31]；不显著|-2.51 [-3.6, -1.45]；显著更差|-0.0158 [-0.0253, -0.00649]；显著更好|


**所有显著更好的格子：**仅 one model per type × |C_Q3|，-0.0158 [-0.0253, -0.00649]。去掉数值特征及 late fusion 的 Q2 点值更高，但区间跨零。集合大小比较使用各自冻结阈值，不能解释为等覆盖率下的效率优势；pretrained/single/per-type 为历史复用对照，不能全当成严格单组件替换。


### A2. 与 1.7B verbalizer 的等价性检验

**计算之前已写入本报告的固定容差：accuracy ±2.0pp；集合大小 ±0.10，此后未调整。**采用双单侧 α=.05 的 bootstrap 区间版 TOST：90% 区间须严格位于容差开区间内。

|类型/指标|CLR−verbalizer：95% CI|CLR−verbalizer：90% CI|预设容差|等价|最小容差阈值¹|
|---|---|---|---|---|---|
|Q3/accuracy（pp）|-0.659 [-1.51, 0.19]|-0.659 [-1.38, 0.0521]|±2.0|是|1.37882|
|Q3/set size|+0.0169 [0.00692, 0.0269]|+0.0169 [0.00843, 0.0252]|±0.1|是|0.0252073|
|Q4/accuracy（pp）|-0.0864 [-0.984, 0.83]|-0.0864 [-0.85, 0.667]|±2.0|是|0.849654|
|Q4/set size|+0.0194 [0.00634, 0.0326]|+0.0194 [0.00854, 0.0305]|±0.1|是|0.03053|


¹ 阈值为 max(|90%下界|, |90%上界|) 的下确界；严格判等价需略大于该值。六位有效数字仅为显示，未用来调整预设容差。完整精度见 a2.json。这里是 bootstrap 近似检验，不是额外的正态 t 检验。

四项均等价。集合大小仍存在小而显著的差异：Q3 +0.0169 [0.00692, 0.0269]、Q4 +0.0194 [0.00634, 0.0326]，CLR 更大；“统计显著不同”与“在实用容差内等价”可以同时成立，不能称两者完全相同。


### A3. 默认配置的选择依据

下表为各设计原有 **reader_val、clean/seen、step 等权、三 seed** 验证结果；准确率为 %，其余原尺度。均为点估计，95% 区间见 a3_validation.json。

|源|配置|Q1 NLL|Q2 AUROC|Q3 acc（%）|Q4 acc（%）|\|C_Q3\||
|---|---|---|---|---|---|---|
|gpt2|single step z_t|0.41|0.85|83.5|82.6|1.42|
|gpt2|CLR|0.42|0.859|86.3|85.1|1.32|
|gpt2|CLR, joint|0.41|0.859|86.8|85.3|1.31|
|llama|single step z_t|0.384|0.843|82.5|82.6|1.38|
|llama|CLR|0.361|0.85|80.1|81.1|1.5|
|llama|CLR, joint|0.379|0.834|83.3|84.2|1.38|


**规则与边界：**M4.1_TASK 预先规定 v1/lite 完整训练，以 reader_val 平均 NLL 选择较好变体。实现取三 seed 各自 eligible best checkpoint 的未校准、四类型宏平均 NLL 再平均；最早 epoch 8、最多 10、patience 1。记录为 lite=0.558942、v1=0.618356，选 lite。该选择分数含原训练验证的 Q1 clean/traj，不能从上表的已校准 clean 指标反推。M5b single-step 和 M9 joint 在以后才产生，未找到把这三行放在同一预定义 validation 规则下选择默认行的记录；上表也显示默认行不在每项上最佳。

**test 时间线：**M4 v0 主 seed11 最终评估于北京时间 2026-11-04 10:14:42 完成；11-05 的 M4.1 失效诊断与修复任务之后才设计最终 v1/lite 架构。M4.1 内部确实在本轮 test 前以 reader_val 选 checkpoint 并锁定，但它复用了已经观察过 v0 结果的同一 test。故“最终配置在从未看过 test 的条件下确定”不成立；“本轮调试只用 val、权重锁后最终评测”与这段历史必须同时披露。


### A4. Q1/Q2 与单类型 reader 的差距

下表均为 **默认 CLR−比较 reader**，NLL/AUROC 原尺度。按 Table 1 的六个已指定 reader，在各源各指标上标记实测最优者；这是对已观察 test 表的描述性选取，未做“挑最优 reader”后的同时区间校正。

|源|比较 reader|Δ Q1 NLL [95%CI]|Δ Q2 AUROC [95%CI]|
|---|---|---|---|
|gpt2|MLP probe|+0.0255 [0.0145, 0.0367]|-0.0165 [-0.0261, -0.00716]|
|gpt2|MLP probe, prefix|+0.0327 [0.0217, 0.0439]|-0.0277 [-0.0378, -0.0182]|
|gpt2|Attention probe|-0.0258 [-0.0379, -0.0132]|-0.0231 [-0.0329, -0.0136]|
|gpt2|LTO reward model|+0.0415 [0.0291, 0.0538]|-0.0154 [-0.0267, -0.00425]|
|gpt2|HSRM-style reader|-0.0106 [-0.0215, 0.00056]|-0.00476 [-0.0162, 0.00638]|
|gpt2|CLUE|-0.1 [-0.129, -0.0718]|+0.254 [0.226, 0.282]|
|llama|MLP probe|+0.00359 [-0.00228, 0.00945]|-0.03 [-0.04, -0.0203]|
|llama|MLP probe, prefix|+0.00914 [0.003, 0.0153]|-0.0308 [-0.0405, -0.0214]|
|llama|Attention probe|-0.0165 [-0.0235, -0.00946]|-0.00243 [-0.014, 0.00898]|
|llama|LTO reward model|+0.0642 [0.0354, 0.0914]|+0.0155 [-0.000314, 0.031]|
|llama|HSRM-style reader|-0.00947 [-0.0154, -0.00367]|-0.013 [-0.026, -0.000118]|
|llama|CLUE|+0.0706 [0.0512, 0.09]|+0.0287 [0.0141, 0.043]|


Q1 最强者：GPT-2 为 LTO（0.507），LLaMA 为确定性的 CLUE（0.545）；对应 CLR 差值分别 +0.0415 [0.0291, 0.0538]、+0.0706 [0.0512, 0.09]。Q2 两源最强者均为 MLP-prefix（0.857 / 0.852），差值分别 -0.0277 [-0.0378, -0.0182]、-0.0308 [-0.0405, -0.0214]。

**可直接用于论文：**“在两个 CODI 源上，默认 CLR 相比各指标实测最优的单类型 reader，Q1 NLL 高 0.0415–0.0706，Q2 AUROC 低 0.0277–0.0308，配对 95% 区间均排除零；CLR 的优势主要体现在状态与选项关联的 Q3/Q4。”不能把这些结果写成已证明与最强 specialist 等价。


## 4. B：LLaMA 关键消融

开跑前估计 **7.52 GPUh**：按历史 M12 同变体三 seed 耗时 × LLaMA/GPT-2 QA 数比 0.340932 × 1.7 原生维度/共享 GPU 余量，另留 1.50 GPUh 给 smoke/校准/评估/失败。低于 25 GPUh，故执行全部缺失设计。训练为原 M12 协议：batch1024、bf16、AdamW 3×10⁻⁴ / weight decay10⁻⁴、clip1、每 epoch 全 QA 无放回、类型加权、完整 reader_val 宏 NLL 早停，eligible epoch8、最多10。仅把源维度改为 LLaMA 2048，复用其冻结 normalization；未调参。

w/o block mask、dot-product、late fusion 新增各三 seed；单类型模型复用 M10 已完成的三 seed、每类型精确 8 epoch / matched exposure，12 个类型×seed checkpoint 与曝光均已核验；single-step 按任务复用原结果。所有九个新训练完成后统一锁定权重/代码/config，才打开最终 test。cal_A 温度、cal_B 按(qtype,t) Mondrian、α=.1，与 GPT-2 相同。

完整训练曝光（按 seed11/22/33 顺序）：w/o block mask 实际 9/10/9 epoch、选中 8/9/8；dot-product head 实际 9/9/9 epoch、选中 8/8/8；late fusion 实际 9/9/9 epoch、选中 8/8/8。每个 epoch 无放回遍历全部四类型 QA，未缩量。

**test 绝对结果**（clean/seen、step 等权、三 seed；accuracy 为 %）：

|设计|Q1 NLL|Q2 AUROC|Q3 acc（%）|Q4 acc（%）|\|C_Q3\||
|---|---|---|---|---|---|
|CLR|0.615|0.821|72.1|74.2|1.64|
|w/o block mask|0.619|0.811|64|71.9|1.91|
|dot-product head|0.627|0.786|68.4|73.3|1.78|
|late fusion|0.628|0.819|75.7|76.4|1.45|
|one model per type|0.634|0.818|72.6|71.4|1.64|
|single step z_t|0.627|0.813|74.7|75.5|1.5|


**变体−CLR 的双向配对区间**（Q3/Q4 为 pp，其余原尺度）：

|设计|ΔQ1 NLL|ΔQ2 AUROC|ΔQ3 acc|ΔQ4 acc|Δ\|C_Q3\||
|---|---|---|---|---|---|
|w/o block mask|+0.0034 [0.000647, 0.00611]；显著更差|-0.0103 [-0.0173, -0.00335]；显著更差|-8.15 [-9.55, -6.74]；显著更差|-2.31 [-3.48, -1.11]；显著更差|+0.276 [0.254, 0.3]；显著更差|
|dot-product head|+0.0116 [0.00686, 0.0165]；显著更差|-0.0356 [-0.0463, -0.0251]；显著更差|-3.7 [-4.91, -2.48]；显著更差|-0.914 [-2.07, 0.264]；不显著|+0.143 [0.124, 0.162]；显著更差|
|late fusion|+0.0124 [0.00809, 0.0167]；显著更差|-0.00277 [-0.0139, 0.00788]；不显著|+3.6 [2.45, 4.75]；显著更好|+2.23 [1.11, 3.38]；显著更好|-0.189 [-0.206, -0.173]；显著更好|
|one model per type|+0.0186 [0.0134, 0.0239]；显著更差|-0.00316 [-0.0111, 0.00479]；不显著|+0.474 [-0.531, 1.46]；不显著|-2.82 [-4.02, -1.6]；显著更差|+0.00431 [-0.0114, 0.0206]；不显著|
|single step z_t|+0.0115 [0.00825, 0.0147]；显著更差|-0.0084 [-0.017, 0.00031]；不显著|+2.6 [1.59, 3.62]；显著更好|+1.3 [0.314, 2.29]；显著更好|-0.133 [-0.149, -0.118]；显著更好|


**reader_val 结果**（同样 clean/seen、step 等权；各项区间见 b_results.json）：

|设计|Q1 NLL|Q2 AUROC|Q3 acc（%）|Q4 acc（%）|\|C_Q3\||
|---|---|---|---|---|---|
|CLR|0.361|0.85|80.1|81.1|1.5|
|w/o block mask|0.37|0.837|72.5|78.7|1.75|
|dot-product head|0.377|0.82|76.2|80.1|1.59|
|late fusion|0.383|0.84|82.3|81.2|1.36|
|one model per type|0.411|0.852|79.9|76.9|1.51|
|single step z_t|0.384|0.843|82.5|82.6|1.38|


Q1 reader_val 的 clean/traj 分项 NLL（点估计）：

|设计|clean|traj|
|---|---|---|
|CLR|0.361|0.363|
|w/o block mask|0.37|0.374|
|dot-product head|0.377|0.371|
|late fusion|0.383|0.389|
|one model per type|0.411|0.412|
|single step z_t|0.384|0.384|


**跨源一致性：**下列比较点估计方向，同时注明两源均显著的项；不把单源不显著与另一源显著自动当成已证明的交互效应。

- w/o block mask：同向 Q1、Q2、Q3、Q4、|C_Q3|；反向 无；两源均显著且方向相同：Q1 显著更差、Q2 显著更差、Q3 显著更差、|C_Q3| 显著更差。

- dot-product head：同向 Q1、Q2、Q3、Q4、|C_Q3|；反向 无；两源均显著且方向相同：Q1 显著更差、Q2 显著更差、Q3 显著更差、|C_Q3| 显著更差。

- late fusion：同向 Q1、Q4；反向 Q2、Q3、|C_Q3|；两源均显著且方向相同：无。

- one model per type：同向 Q1、Q2、Q3、Q4；反向 |C_Q3|；两源均显著且方向相同：Q4 显著更差。

- single step z_t：同向 Q1、Q2；反向 Q3、Q4、|C_Q3|；两源均显著且方向相同：Q1 显著更差。

新增实测成本 **3.67 /25 GPUh**（RTX 4090，按各 GPU run 墙钟求和，含加载、smoke、校准、验证和最终推断；CPU 统计为 0 GPUh）。源模型前向 0。Q1 reader_val clean/traj 分项保存在 b_reader_val_q1_clean_traj.json。


## 5. 与论文现有说法的对照

1. **“CLR 与 1.7B verbalizer 在 Q3/Q4 上 statistically tied”——需改写。**accuracy 的 95% 区间均跨零，且在预先固定 ±2.0pp 容差内通过等价性检验；集合大小也在 ±0.10 内等价，但 CLR 显著大约 0.017/0.019 个元素。建议明确写“accuracy 在 ±2pp 实用容差内等价”，避免用 tied 暗指完全相同或同等训练成本。B5c 一遍训练，CLR 为多遍，曝光亦不同。

2. **“CLR 在 Q1/Q2 上与 specialist readers competitive”——需改写。**四项相对最强者的配对区间都显示显著差距（A4）。可保留具体差距描述，不支持“与最强者无差别”或无容差的“性能相当”。

3. **“dot-product head、late fusion、single step 各使 Q3 下降 2–3 个点”——需改写。**GPT-2 的三个差值依次为 -3.24 [-4.2, -2.3]pp；-2.13 [-3.09, -1.16]pp；-2.28 [-3.19, -1.36]pp，其中 dot-product 下降超过 3pp。LLaMA 对应为 -3.7 [-4.91, -2.48]pp；+3.6 [2.45, 4.75]pp；+2.6 [1.59, 3.62]pp，须分别陈述，不能直接外推 GPT-2 结论。

4. **“联合训练使 Q4 提升，Q3 不变”——需改写。**这里应明确指 GPT-2 的多类型联合训练相对 matched-exposure 每类型独立模型，并非 Table 1 的双源 joint。GPT-2 多类型−单类型：Q4 +2.51 [1.45, 3.6]pp；Q3 -0.535 [-1.31, 0.261]pp。应写“Q4 提升，Q3 未检出显著差异”，未给 Q3 另设等价性容差，不能写已证明不变。LLaMA 多类型−单类型：Q4 +2.82 [1.6, 4.02]pp；Q3 -0.474 [-1.46, 0.531]pp；应分别报告两个源。

5. **“pretrained encoder 增加参数但不提升准确率”——需改写。**本次实现 trainable 参数为 v1 125M / lite 17.2M；GPT-2 Q3 差值 +0.109 [-0.739, 0.938]pp，Q4 -1.15 [-2.07, -0.238]pp。准确表述为“本次预训练 v1 未检出 Q3 收益，Q4 显著下降”；它与 lite 同时在深度/参数化等处不同，不能归因预训练本身，更不能外推所有 encoder。


## 6. 复现信息

仓库根目录：`/home/me/PaperField/CalReader`；启动 git commit：`9f0a9975378fd91807149c183599ff818853b52a`。工作区已有未提交材料；复现以每 run 启动保存的源码 tar、SHA/config、环境与 seed 为准，不把 git commit 冒充全部新脚本的版本。新脚本和产物均在 `/home/me/PaperField/CalReader/supplemental/stats_20261116_v1`，未修改已有代码、预测、结果或论文。

**输入文件及来源：**

- A1：`/home/me/PaperField/CalReader/data/M12_ablation/{problem_metrics.parquet,auroc_observations.parquet,conformal_observations.parquet,summary.csv}`；原始 run/校准来源见该目录 source_registry.json / statistics_provenance.json。

- A2：上项 CLR，加 `/home/me/PaperField/CalReader/data/M10b_v1/paper_inputs/B5c_{problem_metrics,conformal_problem}.csv`；与 `/home/me/PaperField/CalReader/data/M10c_v1/paper_inputs/conformal_paired.csv` 独立核对方向与区间。

- A3/A4：`/home/me/PaperField/CalReader/data/M10_v1/paper_inputs/{main,main_problem_metrics,conformal}.csv` 及 paper_cache；`/home/me/PaperField/CalReader/data/M10e_v1/{summary,problem_metrics_*,final_*}`；`/home/me/PaperField/CalReader/data/M10d_v1/{problem_metrics.csv,final.json}`；GPT-2 LTO 原始 run 由 data/M4_v0/evaluation_suite.json 的 B4_11/22/33 索引。

- B：`/home/me/PaperField/CalReader/data/M6_v1/qa`、`/home/me/PaperField/CalReader/data/dumps/codi_llama_M6_v1`、`/home/me/PaperField/CalReader/data/M6_v1/llama_normalization.npz`；每类 train QA=252399/47719/293301/95228。新 run 索引为 `/home/me/PaperField/CalReader/supplemental/stats_20261116_v1/trained_*.json`、`evaluated_*.json`；复用单类型来源 data/M10_v1/trained/llama_CLR-per-type_*_full.json。

**脚本及实际命令**（在仓库根目录，沿用 `source scripts/m7_env.sh` 设置；Python 为 `/home/me/anaconda3/envs/latdefer/bin/python`，每条入口只传 YAML）：

```bash
python -m supplemental.stats_20261116_v1.reproduce supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.analysis_a supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.audit_tables supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.validate_a supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.provenance supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.b_pipeline supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.analysis_b supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.write_report supplemental/stats_20261116_v1/config.yaml
python -m supplemental.stats_20261116_v1.finalize supplemental/stats_20261116_v1/config.yaml
```

最终核验：[final_audit.json](/home/me/PaperField/CalReader/supplemental/stats_20261116_v1/final_audit.json)；冻结清单：[freeze.json](/home/me/PaperField/CalReader/supplemental/stats_20261116_v1/freeze.json)。以上为实际运行命令；重跑时须复制到新版本目录并更新 YAML 输出路径，避免覆盖本次冻结产物。

B 配置最初由 prepare_b.py 生成并在训练前记录预算；b_train.py / b_evaluate.py 为 M12 入口的新版本，差异补丁随目录保留，模型仅新增 native 维度包装并 import 冻结实现。首次复现入口因 NumPy bool JSON 序列化退出，CPU 导出亦修正了旧里程碑索引/预测 schema 适配，原失败日志保留；均没有模型回退、重新校准旧结果或改变统计协议。

A 部分新增 GPUh=0；B 成本以 b_completed.json / jobs 下逐 run metrics 为准。协议 smoke：5 个 reader_val problem、四类型原生主模型逐位一致；训练每变体 reader_train20 / reader_val20 题，校准评估代理仅 reader_val30 题（≤3%），各 run ≤15 分钟；九项新训练全部锁定后才最终 test。旧科学输入已核验 776 个文件 SHA 一致，最终再次核验见 final_audit.json。输入全路径/SHA、完整精度结果、每 seed 曝光、全部运行命令/环境保存在本目录。

本补充分析不能消除历史 test 暴露；显著性和等价性均条件于当前固定模型、划分及原校准数据。冻结阈值下的经验集合大小不是分布偏移或自适应选择后的无条件 conformal 保证。
