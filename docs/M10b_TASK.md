# M10b：最后一个 B5c 基线
用户授权≤22 GPUh（含失败/加载/校准/推断），仅CODI-GPT2 Q3/Q4，seed11/22/33，全部冻结标签；LLaMA与审计不做。完成后实验永久冻结，不削减seed/标签。
只读复用M10 B5c：Qwen3-1.7B冻结bf16基座、两层映射、全部七类线性层LoRA r16/alpha32、相同typed模板/选项、学习率2e-4、batch512、checkpointing、完整训练一遍。M10 smoke的min_epochs8未被训练循环使用，本次明确epochs1/min_epochs1，不声称匹配八遍CLR曝光。
先val-only smoke≤5%/≤15min，再按实测准入。新结果data/M10b_v1；旧M10不修改。cal_A温度缩放，cal_B按qtype/step独立Mondrian阈值；test仅三权重全部锁定后最终评估。并列reader_val/test；固定三seed，按problem bootstrap10000次CLR−B5c配对95%CI。
更新GPT2 T1 B5c Q3/Q4，Q1/Q2明确NA（未执行）；审计正文同时列CLR/B6，主张“类型化审计框架（问题分解+公开期望+conformal阈值）”，不声称CLR检测更优。跨用途结果、HiddenBench只入附录，后者数据不足无法结论。新版本归档旧paper/FINAL_SUMMARY，保持旧freeze原字节可恢复。
最终reports/M10b_REVIEW.md≤8000bytes固定(a)–(f)，回答Q3/Q4差值和CI；重跑独立paper复现并freeze/audit。

## 用户运行修订（2026-11-11）
取消显存空闲阈值等待和提前退出；直接以相同模型/batch/完整标签尝试运行，实际OOM才报错。保留22GPUh硬上限及真实异常处理。

## 最新用户修订：只在启动前检查显存（2026-11-11）
持续CPU监控，显存足够后自动启动：每30秒采样，连续3次空闲≥22GiB。启动后min_free_GiB=0，不轮询显存来停进程，不因空余显存下降退出。实际OOM记录计费后可返回启动等待；预算预计超限或22GPUh硬保护仍有效。此条覆盖上一条取消启动等待的临时修订。
