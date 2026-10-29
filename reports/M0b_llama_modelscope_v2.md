# M0b CODI-LLaMA 底模来源修复 v2（2026-10-29）

已按用户指定从 [ModelScope LLM-Research/Llama-3.2-1B-Instruct](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct) 下载，未从Hugging Face下载模型文件。不是替代模型或量化版本。原M0 v1报告、配置、失败记录未改动。

## 锁定资源与同一性证据

ModelScope revision：`d3e551343d4d81508a0d226656b826c217e463cd`。上游对照为此前锁定的Meta仓库revision `9213176726f574b556790deb65791e0c5aa438b6`。本次HF仅获取公开tree元数据；底模、tokenizer、配置、许可证等9个文件均从ModelScope获取。

每个下载文件先校验ModelScope发布的SHA-256和字节数。普通文件另计算Git blob SHA-1与上游tree oid比较；模型权重的上游LFS SHA-256被元数据遮蔽，因此用ModelScope的SHA-256及大小重建标准135字节LFS指针，比较该指针的Git blob oid。结果为 `5daf23edd80cedb879a4a18e6049a19b2417a3f7`，与上游一致。再校验本地2,471,645,608字节权重的SHA-256，从而验证下载内容与这份上游LFS对象一致。没有声称直接读取了被遮蔽的上游SHA值。

| 文件 | 字节数 | 本地/ModelScope SHA-256 | 上游Git对象一致 |
|---|---:|---|---|
| config.json | 877 | 2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb | True |
| generation_config.json | 189 | 88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36 | True |
| special_tokens_map.json | 296 | 6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec | True |
| tokenizer.json | 9085657 | 79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4 | True |
| tokenizer_config.json | 54528 | 9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e | True |
| LICENSE.txt | 7712 | 0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35 | True |
| USE_POLICY.md | 6021 | 40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201 | True |
| README.md | 41742 | 18564977261167ff9f76d8ee3a94c8d1cc59c0e143ba054f1187744086004a93 | True |
| model.safetensors | 2471645608 | 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f | True |

总下载字节：2480842630（约2.48 GB）。下载前磁盘可用约37GB，配置另外保留10GiB；未动其他模型/数据，未占用GPU。

本地目录：`/home/me/models/Llama-3.2-1B-Instruct/modelscope-d3e551343d4d81508a0d226656b826c217e463cd`。来源清单在该目录的 `CLR_DOWNLOAD_MANIFEST.json`；公开元数据和同一性证据在 `reports/evidence/M0b_llama_modelscope/`。

## 实际运行与验证范围

下载run：`runs/20261029T010824516751Z_89dc7e0b7668_b742ff`；耗时218.771秒，GPU小时0，源前向0。格式验证run：`runs/20261029T011218552649Z_a2b3255c27bd_7e0397`，完成离线AutoConfig、AutoTokenizer及safetensors张量形状检查：model_type=llama、layers=16、hidden_size=2048、vocab=128256、张量数146。

这只是底模资源及格式验证，**不是CODI加载smoke或准确率复现**；未用CPU格式检查替代GPU实验，也不为下载/格式检查编造3seed准确率或CI。

```bash
PYTHONPATH=. PYTHONUTF8=1 .venv/bin/python scripts/m0b_modelscope_download.py configs/M0b_llama_modelscope_download_v2.yaml
PYTHONPATH=. PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0b_llama_base_validate.py configs/M0b_llama_base_validate_v2.yaml
```

## 仍需解决的独立文件

CODI作者训练后的 `zen-E/CODI-llama3.2-1b-Instruct` revision `b2c88ba224b06b12b52ef39b87f794b98a6eb1c8` 是另一份checkpoint，本地指定缓存路径不存在。ModelScope搜索尚未找到已核实的对应镜像，不能宣称镜像绝对不存在，也不能拿原始LLaMA底模代替训练后的CODI。

已向用户澄清“不用HF下载”是否仅针对Meta底模或也包括作者公开checkpoint。底模工作已完成；在该范围明确前未下载作者checkpoint、未启动CODI-LLaMA复现。CODI-GPT2完整噪声运行不受影响。
