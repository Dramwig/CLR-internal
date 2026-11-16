# M7 前置核验：Idea A 进度与消息接口

核验日期：2026-11-06（北京时间）。这是历史对话与现有产物的只读核验，不是 M7 实验或验收。
Idea A 为 `/home/me/PaperField/LatDefer`；本次 HEAD：`725d1606971684bdc11e9546b87a9d5655486261`。

## 找到的对话

- 本仓库已归档：[搭建 CLR 研究项目骨架](codex://threads/01a181bd-9f8d-7d93-bdea-da0553666450?hostId=remote-ssh-discovered%3A4090)。已读取全部三页历史；包含 M0 的 Idea A 复用盘点及后续 CLR 实验。该对话的早期 KV 占位判断是当时快照。
- Idea A 历史：[准备 LatDefer 研究环境 (3)](codex://threads/01a180f8-8739-7f62-952d-7e32dc9d6b39?hostId=remote-ssh-discovered%3A4090)。已读取其部分历史，最近收尾消息确认 M2 与 M3 机制验收完成、177 项测试通过；不冒称通读全部历史。
- Idea A 后续：[启动 M4 分阶段训练](codex://threads/01a18a59-0724-7ad0-98d0-72bf21e97fef?hostId=remote-ssh-discovered%3A4090)。最近收尾消息确认正式 direct 训练完成，正式 val 比较待执行。

## M1.1：已完成且核心生成器保持冻结

主数据版本 `m1-v2-r2`；路径 `/home/me/PaperField/LatDefer/data/synthetic/m1-v2-r2`。
train/val/test 分别为 48,000/3,600/4,800 行，独立组为 3,200/240/320；同组的 alpha/d_pub 变体必须保持在同一划分。

真实证据均位于 Idea A：

- 定义和结论：`docs/SYNTHETIC-V2.md`；冻结：`docs/v2-design-freeze.json`。
- 最终汇总：`runs/20261028T131140099734Z_714f3fc5a66c_fc4910/metrics.json`，`status=passed`、`acceptance.all_passed=true`。
- 独立审计：`runs/20261028T131616405767Z_236c10e837ef_c51607/metrics.json`，95 项测试通过、跨 split 实例重叠为 0。
- 本次重新计算 `latdef/data/synthetic_v2.py`、`synthetic.py`、`schema.py` 三个源码 SHA256，均与设计冻结清单一致。此项不是对冻结清单全部文件的重新核验。

可 import：`latdef.data.synthetic_v2.V2Config`、`load_v2_config`、`generate_v2_split`、`solve_v2_text`，结构为 `latdef.data.schema.Example`。
任务是 `y=T(z_pub,e_priv)`；生成器保存已知私有事实等标签依据。它证明输入事实是什么，不能单独证明某条实际 KV 已成功编码并传递该事实。

## KV：有明确契约并通过机制验收，CLR 集成仍待做

入口：`latdef.inject.cache.PrefixKV`、`prefix_forward`，`latdef.inject.adapter.KVAdapter`，`latdef.inject.transfer.extract_private_kv`、`KVTransfer`，以及 `latdef.handoff.base.S2Input`。

- `PrefixKV` 字段为 `layers`、`mask`、`active_layers`、可选 `gates`；每层 K/V 为 `[B,num_key_value_heads,S,head_dim]`，mask 为 `[B,S]`。
- K 为 **post-RoPE**；prefix 槽位置 `0..S-1`；真实 token 位置从 S 起，左 padding；层和门控信息必须随消息保留。
- 学习式消息：`MessageEncoder` 默认输出 `[B,16,64]`，经 `KVAdapter` 生成目标模型的 prefix；默认注入全部层。这与原始 sender 全量 token KV 是两种来源。
- 原始私有 KV：`extract_private_kv` 只编码私有原文，保留完整 token；`KVTransfer` 是 Qwen3 同族简化映射，不是 C2C 官方 fuser 的复现。
- 仅验收 Qwen3 全 attention、eager/SDPA；现实现拒绝滑窗及 S2 gradient checkpointing，同一 S2 实例不能并发 prefix forward。

契约见 Idea A `docs/SYSTEM2.md`，产物索引 `docs/m3-artifacts.json`。真实 4B 机制 run `runs/20261029T193142287175Z_142891a52ccc_f733c0` 及读回审计 `runs/20261029T194017540301Z_160b158023bb_4cfca2` 均 passed，未训练参数、未使用 test。机制验收覆盖 KV 注入、位置/padding、梯度、direct/cot512、教师强制和全量 KV 迁移。

本次 live 元数据：Idea A 为 transformers 4.57.1 / tokenizers 0.22.2 / peft 0.17.1；CalReader `.venv` 为 4.52.4 / 0.21.4 / 0.15.2；两边 torch 均 2.5.1。**尚未验证该 KV 实现能在 CLR 当前环境直接 import/运行**，M7 需固定版本并实测，不能照搬 Idea A 验收。
CalReader 当前没有 `clr/sources/latentmas.py` 或 `c2c.py`，也没有已验收的通信审计 dump/commitment 管线。

## Idea A 当前整体进度与有效性边界

M2 主 decoder 已收尾，encoder 支线未通过并停止；M3 机制已通过。
M4 过拟合和小规模关卡已通过；pilot 核心格子 alpha=1、d_pub=0 每条件仅 8 条，latent 100%、public-only 和 mismatched 各 25%，不能外推到所有难度。
正式 direct 已完成 48,000 条、6,000 步、8.83 小时，checkpoint 存在且本次 SHA256 与 metrics 一致。
`docs/m4-artifacts.json` 状态为 `M4_formal_direct_completed_evaluation_pending`；正式 run `runs/20261105T011932843823Z_7f1929290679_112322/metrics.json` 为 completed、evaluation_pending=true、test_used=false。完整正式 val direct/cot512 比较尚未执行，不能写成 M4 总验收通过。

## 对 M7 任务书的建议

M1.1 已完成，不构成等待条件。可固定 Idea A 版本，通过 import 复用生成器及需要的 KV 工具；M7 自己先做 val 代理上的格式/传信关卡，再扩量。
若主线要求官方 LatentMAS/C2C，可以继续使用 Idea A synthetic v2 数据，并独立接入官方通信源；无需等 Idea A M4 完整验收，但官方代码、模型对、checkpoint 可用性仍须 M7 核实，本次未验收。
任务书应明确潜对象取 sender 原始 KV、学习式 prefix 或 fuser 输出中的哪一种，补充 commitment 生成、审计标签规则、按 group/episode 冻结划分及 receiver 用/不用/错配消息的因果检查。Idea A 的简化 KVTransfer 不能标为官方 C2C。

历史对照：CLR `reports/M0_env.md`（2026-10-28）中的 KV/message 占位描述早于上述实现，保留原报告，不据此判断当前仍未就绪。
