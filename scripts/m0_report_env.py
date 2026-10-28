import json
from pathlib import Path
runs=[]
for p in Path('runs').glob('*/metrics.json'):
 d=json.loads(p.read_text());c=json.loads((p.parent/'config.json').read_text())
 if c.get('stage')=='inventory' and d['status']=='completed':runs.append(p.parent)
r=sorted(runs)[-1];e=json.loads((r/'inventory.json').read_text())
text=f'''# M0 环境与复用盘点

日期：2026-10-28（Asia/Shanghai）。本报告来自 `{r}`，不是 Idea A 的验收结果。

## 实测环境

- Python：`{e['python']}`；解释器 `{e['executable']}`。
- PyTorch：`{e['torch']}`；CUDA runtime：`{e['cuda_runtime']}`；可见设备：{e['visible_cuda_devices']}。
- 实际 import 版本：`{e.get('runtime_imports')}`。
- 当前项目 `.venv` 使用 Idea A Python 3.11 的 system-site-packages，只在本项目覆盖 transformers=4.52.4、peft=0.15.2、tokenizers=0.21.4；没有改动 Idea A 环境。
- 依赖全表：`reports/evidence/environment.lock.txt`。CODI 官方 torch=2.7.1，当前 torch=2.5.1；Coconut 官方 transformers=4.46.2，当前4.52.4。这些是明确的环境差异，真实复现前须验证兼容性，不能视为官方环境等价。
- 磁盘：总量 {e['disk']['total_bytes']/1024**4:.2f} TiB，空闲 {e['disk']['free_bytes']/1024**3:.2f} GiB（快照，下载前后会变化）。

```csv
{e['nvidia_smi']['stdout']}
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
'''
Path('reports/M0_env.md').write_text(text)
