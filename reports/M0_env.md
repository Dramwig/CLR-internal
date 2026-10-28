# M0 环境与复用盘点

日期：2026-10-28（Asia/Shanghai）。本报告来自 `runs/20261028T091921015475Z_cb8a4cbc8403_bc2b60`，不是 Idea A 的验收结果。

## 实测环境

- Python：`3.11.13 | packaged by conda-forge | (main, Jun  4 2025, 14:48:23) [GCC 13.3.0]`；解释器 `/home/me/PaperField/CalReader/.venv/bin/python`。
- PyTorch：`2.5.1`；CUDA runtime：`12.1`；可见设备：8。
- 实际 import 版本：`{'transformers': '4.52.4', 'peft': '0.15.2', 'tokenizers': '0.21.4'}`。
- 当前项目 `.venv` 使用 Idea A Python 3.11 的 system-site-packages，只在本项目覆盖 transformers=4.52.4、peft=0.15.2、tokenizers=0.21.4；没有改动 Idea A 环境。
- 依赖全表：`reports/evidence/environment.lock.txt`。CODI 官方 torch=2.7.1，当前 torch=2.5.1；Coconut 官方 transformers=4.46.2，当前4.52.4。这些是明确的环境差异，真实复现前须验证兼容性，不能视为官方环境等价。
- 磁盘：总量 13.86 TiB，空闲 147.05 GiB（快照，下载前后会变化）。

```csv
index, uuid, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], driver_version
0, GPU-48439f70-4469-efb6-2ad6-54000534f99d, NVIDIA GeForce RTX 4090, 49140 MiB, 45379 MiB, 3266 MiB, 535.171.04
1, GPU-ee568afe-d047-4f5a-27ec-2f5a9e119730, NVIDIA GeForce RTX 4090, 49140 MiB, 38710 MiB, 9935 MiB, 535.171.04
2, GPU-88350653-3384-889c-17b1-94083fb3c657, NVIDIA GeForce RTX 4090, 49140 MiB, 42453 MiB, 6193 MiB, 535.171.04
3, GPU-ef23fa9f-5ab6-08fc-1fca-aef1b2b787cc, NVIDIA GeForce RTX 4090, 49140 MiB, 42351 MiB, 6294 MiB, 535.171.04
4, GPU-4ca00c12-4fe0-af8f-6741-748c53ece6b5, NVIDIA GeForce RTX 4090, 49140 MiB, 38674 MiB, 9971 MiB, 535.171.04
5, GPU-456e3a58-d0a3-f38c-994d-fed0dc036cda, NVIDIA GeForce RTX 4090, 49140 MiB, 44192 MiB, 4453 MiB, 535.171.04
6, GPU-14e194dd-cd75-0b9a-dc6e-3998732471a9, NVIDIA GeForce RTX 4090, 49140 MiB, 43594 MiB, 5051 MiB, 535.171.04
7, GPU-c80588e1-d882-99e1-c9d4-2489986f617c, NVIDIA GeForce RTX 4090, 49140 MiB, 35378 MiB, 13267 MiB, 535.171.04
```

GPU 型号由驱动报告为 RTX 4090，每卡 49140 MiB，共8卡；不是通常零售规格的显存值，本报告保留实测值。所有卡已有外部负载，nvidia-smi 进程表未完整显示其他命名空间的进程。延迟和训练成本仅代表当时共享负载。选择设备必须使用显式 config.device，显存不足报错，不自动换卡或缩小设置。

## Idea A 只读复用

仓库：`/home/me/PaperField/LatDefer`，核实的 commit：`3fbf0889a6f5bb87e9867418ed2620c4efd8415a`。本轮未改动该仓库；每次 CLR Run 在 dependencies.json 保存其 commit、dirty 状态与 diff。

| 对象 | 精确路径与符号 | 本轮用途 |
|---|---|---|
| 运行记录 | `latdef/utils/run.py`: `Run`, `collect_environment`, `write_json`, `git_metadata` | 已通过 `clr.utils.idea_a` import；运行目录、config、版本、环境、metrics |
| 日志 | `latdef/utils/logging.py`: `JsonlLogger` | 已通过运行工具 import；逐问题/rollout JSONL |
| 固定种子 | `latdef/utils/seed.py`: `seed_everything` | 已 import；Python/NumPy/Torch、确定性开关 |
| YAML 去重 | `latdef/utils/config.py`: `UniqueKeyLoader` | 已 import；拒绝重复键 |
| Qwen3 加载 | `latdef/s2/batch_inference.py`: `BatchedQwen.__init__`, `encode`, `infer`, `length_batches` | 后续通信源计划直接 import；仅支持本地权重，当前分类输出协议不能冒充潜消息源 |
| KV adapter/cache | `latdef/inject/adapter.py`, `latdef/inject/cache.py` | 均为 M0 placeholder，无函数可复用 |
| 消息编码器 | `latdef/message/` | 尚无可复用 KV 消息生成实现；不复制、不改动冻结代码 |

Qwen3 共享资源可只读引用 Idea A 的 `configs/resources.lock.json`；该资源锁不能证明 CLR 源接口已通过测试。

## 命令

```bash
PYTHONPATH=. .venv/bin/python scripts/m0_inventory.py configs/M0_inventory.yaml
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python scripts/m0_tests.py configs/M0_tests.yaml
```

源核实、训练成本、未完成项目和复现状态见 `M0_repro.md` 与 `../docs/BLOCKERS.md`。
