# M11-qual（2026-11-12）

用户授权仅推断/导出，不训练、不重新校准、不修改冻结结果；论文由审稿方绘图。
输出 `data/M11_qual_v1/` ≤300MB，并交付一个 `qual_export.zip`；审查 `reports/M11_qual_REVIEW.md` ≤2KB，保留全局(a)–(f)。

- 两源逐题：GSM8K test、gold方程可解析且冻结中间量≥3，按 sha256(problem_id+"qual") 前12；文本/gold/源最终/0–6早退、Q4全部中间量及等量负例、Q3概率和冻结cal_B α=.1集合、Q1/Q2和K16。
- GPT2跨题：同哈希test前300，t=0/2/4/6，原始fp16 z、Q1前向最后层潜状态池化、MLP隐藏表示及标签。
- M5b val300兄弟：同哈希前40，观测t=1/3，各17候选，三seed有现成预测则导出；包含σ0及K16。
- 审计test：同哈希group_id前600，每组clean/counterfactual，四读取层段均值512及逐层、KV范数、A4选项最后表示/概率、MLP隐藏、事实/commitment期望/冻结k16 Γ与报警。
- M8c固定test200组：同哈希前100，ε=.1已有白盒结果；只导出已有张量或分数，禁止重跑PGD。
- reader推断seed11；现成预测三seed全部保留；方法名question_only/linear_probe/mlp_probe/mlp_prefix/attention_probe/lto/hsrm/clue/latentqa（仅GPT2 Q3/Q4）/clr/clr_single/clr_joint。
- 每个数据文件配README；selection.json完整选择规则；checks.json抽查3题逐位比较冻结预测，报告含smoke/失败的GPUh。

执行细化（结果无关）：先reader_val/val代理smoke≤5%、≤15min再test。Q4固定已见template_id=0；冻结题库每题仅一对正负，扩展时保留其负例，再从其他题冻结Q4正值池按供体ID哈希/数值序补足，排除本题全部正值和重复值（沿用cross-problem真实中间量负例规则，不声称扩展后边际精确匹配）。所有新增概率标明inference，不替换冻结预测。Q3保留冻结每步seen模板/选项顺序，正确项指σ0源最终答案而非gold。审计报警并列α=.01/.05；Γ沿用冻结平均负对数概率，越高越不一致。
