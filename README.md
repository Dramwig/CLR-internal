# Calibrated Latent Reader (CLR)

`R(z, q, s) → p(a | z, q)`：读取潜在思维/KV消息，输出类型化答案概率，后续实现校准与conformal集合。

当前：M0部分完成。CODI-GPT2完成官方脚本复现、统一接口评测、三seed确定性与验证集噪声曲线；LLaMA授权、Coconut训练成本/显存、Huginn权重及真实评测仍待解决。尚未实现/验证Reader校准、搜索、通信审计和理论保证。

- [环境与Idea A只读复用](reports/M0_env.md)
- [M0复现报告与曲线](reports/M0_repro.md)
- [实验记录](docs/EXPERIMENTS.md) · [阻塞](docs/BLOCKERS.md) · [变更](docs/CHANGELOG.md)
- [原始提案](docs/proposal.md) · [官方资源锁](configs/resources.lock.json)

代码在 `clr/`，配置 `configs/`，入口 `scripts/`，测试 `tests/`。本地环境 `.venv`；共享权重只读引用 `/home/me/models/`。不修改Idea A冻结代码。所有实验入口只接受配置路径与覆盖项。

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python scripts/m0_tests.py configs/M0_tests.yaml
PYTHONPATH=. .venv/bin/python scripts/m0_inventory.py configs/M0_inventory.yaml
```

源接口：`from clr.sources.codi import load; source = load(config); result = source.run(problem, perturb=None)`。每个源的latent定义、指定层观测和噪声插入点见对应docstring。小型随机模型测试不代表真实源复现。
