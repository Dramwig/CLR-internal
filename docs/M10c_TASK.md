# M10c：精度与开销（2026-11-12）

用户授权仅推理测量和已有 CSV/成本汇总；不训练、不重校准、不改冻结模型、划分或预测。
CODI-GPT2 Q3/Q4：CLR-lite、B3、B6、B5c，可训练/总参数，矩阵乘法及注意力 FLOPs（2 FLOP/MAC），相对一次 CODI 缓存潜在步，RTX4090 同卡 batch1/64 三轮中位延迟及吞吐。固定 seed11 权重作架构测速，精度/校准沿用固定三 seed。
先仅 reader_val 小面板 smoke（≤5%、≤15min），正式面板64个 reader_val problem；查询按冻结 ID 顺序与 t 轮换，所有模型共享。输入特征/序列化在计时外，B5c 的注入、LoRA、完整冻结 Qwen3 前向及答案头在计时内；报告长度与 GPU 竞争快照。
训练 GPUh 仅读历史账本；无法分配到某模型/类型时标 NA 和共享账本总成本，不按参数或样本比例伪造分摊。
cal_B alpha=.1 的 coverage/size、B5c−CLR 配对差只由冻结逐题 CSV 汇总，整 problem bootstrap10000，不拟合温度或阈值。
新增 T6 及对应脚本；更新 FINAL_SUMMARY C1（限定源/类型、明确 FLOPs 和延迟倍数；精度相当为描述性而非等价检验）、复现日志漂移说明；旧 paper/FINAL_SUMMARY 先归档，旧 freeze 不改。
输出 data/M10c_v1、reports/M10c_REVIEW.md（≤8000 bytes，a–f）、详细证据；实验冻结不解冻，不新增训练或科学评估。
