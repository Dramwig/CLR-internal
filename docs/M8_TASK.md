# M8：自适应攻击与防御（潜通信审计）

2026-11-08 用户授权；预算含 smoke/失败/重启 ≤50 GPUh。M7.1 冻结只读。
先 P0，未通过则停下报告，禁止进入攻击。所有 smoke/调试仅 reader_val，20组，≤15分钟、≤5%。

## P0 前置核验
1. 逐问题解释一致性期望答案来源，只能 commitment 与公开 x_pub，不能真实私有事实。
   若有特权信息，新版本改为 commitment 推导期望答案；模型不重训，cal_B 按新分数重定阈值，在 test 重算 M7.1 主表。
2. 补报 z=0 的 B0、跨组 shuffled-z 在 A1/A4 的 QA 准确率；解释 seed33 检测偏低，提供训练曲线/早停点。
3. A4 比较仅保留私有值所在记录段与去掉该段，判断是否主要读取私有值 token。

## A 威胁模型
中间人修改传输 raw KV，原 commitment 不变。目标答案与该组 M7.1 CF 定义一致。
成功=receiver 输出目标且审计不报警；cal_B clean 定名义 FPR5%/1%，另报实际 clean FPR。
逐层 ||δ_l||≤ε||KV_l||，ε∈{.01,.03,.1,.3}，另报无约束上界。
最终攻击集 test 固定哈希抽样500组、冻结 ID；攻击调试/调参只在 val/cal。

## B 攻击
- B0：CF 替换、范数约束随机噪声；核实并复现2606.28958 KV篡改/2605.28214 latent attack；不能复现写BLOCKERS。
- B1：只动 f 未读取层，PGD 最大化 receiver 目标答案 logp。
- B2：只动读取层；投影 δ 到 f 线性映射零空间，保持审计特征精确不变，PGD 最大化目标 logp。
- B3：白盒已知 CLR/f/问题分布；目标 logp(receiver target)−λ·审计不一致分数，随机问题集 EOT。
  λ∈{.1,1,10}，PGD步骤∈{50,200}。
- B4：灰盒，未知 CLR 权重/seed；按相同分布训练替代 CLR，替代模型上 B3 后迁移。

## C 防御
- D0：M7.1 原审计器。
- D1：每消息随机 k∈{1,4,16,64} 问题，攻击者只知分布。
- D2：每消息保密随机层子集、记录边界抖动和投影矩阵；对应随机数据训练新适配器。
- D3：全层读取、池化投影，重训适配器；重点检验 B1。
- D4：receiver 消费 KV 后，答案位置前若干 token 的4层隐状态，A1/A4 CLR适配器。
  reader_train4000组 receiver prefill；读出私有值与 commitment 推导期望不一致报警。
各防御报告正常通信计算开销、clean FPR与检测率。训练相关3seed，攻击逐seed。

## D 评估与优先级
矩阵B0–B4×D0–D4，每格报告 ASR、receiver目标达成率、检测率@名义5%/1%。
ε–ASR、步数–ASR曲线；B3×D1另画 k–ASR 与独立性假设指数下降曲线。
优先 P0→B1/B2×D0→D4→B3×D0/D4→D2→其余；先20组val smoke后实测估算，严格≤50GPUh。
预算取舍写明未执行格，不能将未执行写通过。冻结模型/阈值/配置后最终test。

## 交付
唯一审查 reports/M8_REVIEW.md：≤8000字节，固定(a)–(f)、≤3主表、3位有效数字。
详细报告 reports/M8_audit_attacks.md，CSV/静态图/运行记录/成本、冻结清单与逐SHA验证。
docs/{STATUS,EXPERIMENTS,BLOCKERS,CHANGELOG}.md、AGENTS.md、docs/M8_LIVE.json 持久化。
末尾答：(1)期望是否只用接收方信息，93.1%是否需改；(2)B1/B2能否绕过D0并改变行为；
(3)抵抗B3最有效防御及代价；(4)D1的k是否显著增加攻击成本。

## P0 预登记实现口径（结果产生前）
额外报告已训练B0，以及冻结CLR置零输入干预，二者分开。shuffled-z按group_id SHA顺序循环供体，clean/CF类别保持。
A4记录定位同时报告：目标ticket中包含该消息真实team字面的记录（value-record）；生成器critical记录（多记录任务的完整支持集合）。
仅保留=只保留这些记录均值，移除全部其他记录与tail；去掉=删除这些记录均值及与它们token区间重叠的tail。
这是压缩特征段干预，不能单独证明token因果依赖；KV本身可携带前序上下文。
P0先val20组smoke，冻结实现后正式reader_val1000/test2000诊断；不按结果调模型/阈值。
