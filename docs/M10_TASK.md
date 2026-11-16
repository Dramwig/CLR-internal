# M10 投稿前补充实验
2026-11-10用户授权；总计≤40GPUh，含smoke/失败/加载/训练/校准/最终推断。P1≤12；P2-4≤12；P2-5≤12；P2-6≤4。先每项≤5%/15min smoke实测估算，超预算按编号优先，不做Huginn/Coconut。不改变旧数据/权重/结果，测试仅锁定模型后最终评估。
P1-1：CODI-LLaMA B0/B3-prefix/CLR-lite-single三seed；B0逐类型B3同协议（8epoch、两LR、batch4096）；prefix拼接均值/当前归一化状态；single仅当前状态、原位置0，batch1024、lr3e-4、10epoch/min8/patience1，完全沿用M5b A科学协议，只扩维2048。
P1-2：B6每qtype独立单层cross-attention、线性输出头，选项数值与冻结预训练文本embedding作query，状态序列作K/V；A1条件只由问题文字解析。无共享模板主体，无多类型混训。参数与所选曝光≥对应B3；reader_val NLL选超参。两个源Q1–Q4；synthetic A1/A4 QA与CF检测@5%/1%，三seed。
P1-3：每类型单独NativeLite，完整同架构、独立初始化；训练曝光精确匹配各源/seed已冻结主模型所选checkpoint该类型曝光，不按新test选停止点。
P2-4：Qwen3-1.7B冻结本体、状态序列映射+LoRA r16，读Yes/No或A–D logits；同源Q3/Q4及synthetic A1/A4，独立cal_A温度/cal_B Mondrian。
P2-5：核实真实HiddenBench原始资料，可解析的公开/私有事实用于新组划分、统一程序化commitment、M71固定特征/层。无法构造CF只做mismatch/A1/A4，不冒充synthetic；实际规模按15GiB和实测12GPUh准入。
P2-6：M9 joint主体初始化新审计适配器，与随机初始化相同架构配对；N100/300/1k/full QA标签嵌套面板、A1/A4曲线；三seed≤4GPUh。
产物：reports/M10_REVIEW.md≤8000bytes、(a)–(f)、≤3表，回答用户5问题；详细证据/CSV另存。仅新增M10实验，完成后最终冻结；M9原版保持可复现，paper与FINAL_SUMMARY用新版本路径/快照更新并记录版本关系，重跑干净环境复现。
统计：题/CF组bootstrap10000次，固定三个训练seed，test与reader_val并列，Q1 reader_val clean/traj分开。不能把注意力结构优势单独归因于类型化接口；真实负面/缺失项照实报告。
