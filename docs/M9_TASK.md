# M9 收尾实验与论文产物
2026-11-10 用户授权；实验到此冻结，只有A允许新增实验。上游全部冻结只读，M9新增版本。
A：补M6 C2，CODI-GPT2+CODI-LLaMA共享CLR-lite主体、各源适配器/source token，seed11/22/33；沿用M6数据/源内归一化及训练协议（10epoch、min8、patience1、batch1024、lr3e-4、全reader_val选NLL）。联合训练使用各源完整冻结训练集，按(source,qtype)平衡损失；与既有单源比较，记录曝光与参数差异。累计≤10GPUh，含smoke/失败/训练/校准/评估。先≤5%/≤15min train+val smoke；所有训练锁定后才最终test，cal_A温度/cal_B conformal，α=.1为论文主表。
B：仅已冻结CSV/已保存预测的CPU派生统计，T1 GSM8K、T2 conformal及配对size差、F1类型/跨源迁移、T3偏移、F2搜索分析、T4通信/定位、F3单调包络与R1 k、T5成本及噪声/协议/去重/标签/规模附录；每产物scripts/paper/make_<name>.py，YAML驱动，booktabs/矢量PDF统一样式。缺失实验标NA，禁止补跑。派生CSV另冻结后才渲染，不修改上游。
C：README列脚本/输入CSV/SHA；artifact_manifest汇总所有freeze和论文产物；独立干净环境重跑并核对数值/字节。PDF固定元数据，若字体/库差异说明。
D：FINAL_SUMMARY≤10000字节，按用户四贡献+两局限列主张→证据/CI→局限，并单列不能主张；M9_REVIEW≤8000字节、(a)–(f)、≤3表。
用户贡献原文要点：1类型化读取优于探针/小集合/类型迁移；2冻结主体跨模型小样本；3内容审计counterfactual约93%且端到端一致性差；4偏移conformal/weighted无增益。局限：跨题价值好但组内排序差，搜索无收益；白盒审计失效，应配合传输完整性。所有强措辞逐项以冻结证据核验，不能支持的列不能主张。

实现口径补充（2026-11-10，未改协议）：复用的M6 trainer仅在epoch≥min_epochs=8后保存候选best，epoch1–7仅诊断；不根据已见val曲线修改选模范围。联合/单源训练规模和权重不同必须列为混杂。
CPU派生统计调度：最终CSV导出按4个线程计算互相独立的模型AUROC bootstrap；每项仍用seed11、10000次重采样与原顺序，指标/区间算法不变。
