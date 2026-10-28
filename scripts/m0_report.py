"""Render M0 from recorded evidence only. Pending sources never get fake curves."""
import argparse,json,math
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def read(run_id):
 p=Path('runs')/run_id;return json.loads((p/'metrics.json').read_text()),json.loads((p/'config.json').read_text())

def pm(stat,scale=1):return f"{stat['mean']*scale:.2f} ± {stat['ci95_halfwidth']*scale:.2f}"

def main(c):
 with recorded(c) as report_run:
    repro,rc=read(c['repro_run']);noise,nc=read(c['noise_run']);smoke,sc=read(c['smoke_run']);reference,refc=read(c['reference_run'])
    if repro['status']!='completed' or noise['status']!='completed':raise RuntimeError('Cannot publish incomplete CODI runs as full results')
    xs=[float(x) for x in noise['noise']];fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,key,title,scale in zip(axes,['accuracy','distinct_answers'],['Accuracy (%)','Distinct valid answers / 16 rollouts'],[100,1]):
        stats=[noise['noise'][str(x)] if str(x) in noise['noise'] else noise['noise'][str(int(x))] for x in xs]
        ax.errorbar(xs,[s[key]['mean']*scale for s in stats],yerr=[s[key]['ci95_halfwidth']*scale for s in stats],marker='o',capsize=3)
        ax.set(xlabel='Relative noise sigma',ylabel=title);ax.grid(alpha=.25)
    fig.suptitle(f'CODI-GPT2 · validation n={noise["problems"]} · K=16 · 3 seeds (95% t CI)')
    fig.tight_layout();dest=report_run.path/'M0_codi_noise.png';fig.savefig(dest,dpi=180);plt.close(fig)
    Path('reports/figures').mkdir(exist_ok=True)
    import shutil
    shutil.copyfile(dest,'reports/figures/M0_codi_noise_v1.png')
    delta=100*repro['aggregate']['accuracy']['mean']-43.7
    table='\n'.join(f"| {s} | {pm(v['accuracy'],100)} | {pm(v['distinct_answers'])} | {pm(v['invalid_fraction'],100)} |" for s,v in noise['noise'].items())
    sources=json.loads(Path('configs/resources.lock.json').read_text())
    text=f'''# M0 源核实、复现与噪声响应（v1）

2026-10-28。**M0 部分完成，未整体验收/冻结**：CODI-GPT2 已完成下表实验；其他源受授权、训练成本/显存和权重获取阻塞。N/A 不是零准确率。小型随机模型只用于单元测试或吞吐探针，不作为源模型替代。

## 官方来源与精确版本

| 源 | 官方代码 | checkpoint / revision | 状态 |
|---|---|---|---|
| CODI-GPT2 | [zhenyi4/codi](https://github.com/zhenyi4/codi) `{sources['codi_code']['commit']}` | [zen-E/CODI-gpt2](https://huggingface.co/zen-E/CODI-gpt2) `{sources['codi_gpt2']['revision']}` | 完整官方 state_dict，strict=True；模型卡标注 GSM8k-Aug |
| CODI-LLaMA | 同上 | [zen-E/CODI-llama3.2-1b-Instruct](https://huggingface.co/zen-E/CODI-llama3.2-1b-Instruct) `{sources['codi_llama1b']['revision']}` | 模型卡标注 GSM8k-Aug；基础模型 [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 配置/分词器请求401，未替换成非官方模型 |
| Coconut | [facebookresearch/coconut](https://github.com/facebookresearch/coconut) `{sources['coconut_code']['commit']}` | 未在官方 README/release 路径核实到已发布权重 | 官方 recipe 存在；训练暂缓，见成本段 |
| Huginn-3.5B | [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining) `{sources['huginn_code']['commit']}` | [tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) `{sources['huginn']['revision']}` | recurrent depth=32；真实 smoke 被显存预检阻塞；并行下载状态见 BLOCKERS |

HF API 元数据和模型卡快照保存在 `reports/evidence/`；完整文件列表、代码版本见 `configs/resources.lock.json`。未把检索返回的其他个人 Huginn/Coconut 上传当作官方 checkpoint。

## 协议与边界

- CLI 只接受 YAML 路径与 `key=value` 覆盖。每次运行保存 config、git、环境、依赖版本、资源和 selection 哈希、逐样本日志、metrics；不覆盖旧 run。
- 原官方 GSM8K valid=500 按规范化题目 SHA-256 排序，交错拆为 val/cal 各250；官方 test=1319 保留。冻结清单 `data/splits/gsm8k_M0_v1.json`。M0 未训练 CLR；源训练集与这些问题的重合尚未独立核验，不能据此主张后续 conformal 覆盖率。
- smoke 使用2道验证题（val 的0.8%）；通过后最终 test 全量复现。噪声曲线使用事先配置的64道验证题，按固定 ID 排序选取，不根据结果选题；只可用于探索性 M2 sigma 选择，不是整个 val 的估计。
- seeds={rc['seeds']}；sigma={nc['sigma']}；K={nc['rollouts']}；from_step=0。噪声为独立标准正态、相对向量范数缩放，fp32计算后转回源 dtype；不把 eps 归一成单位向量。
- CODI 默认读 final layer、projection 前的隐状态，每次真正消费的连续输入对应一个 latent；第0步由 BOT 产生。指定层只改变观测层，不改变官方 final-layer feedback。Coconut 同理。
- Huginn 返回每次 core recurrence 后、ln_f/coda 前的答案预测位置状态；按(token,recurrence)展开，并保存索引；噪声只作用该位置，from_step 按展开序列计数。该定义不是传统 decoder layer trace。
- 生成采用官方 greedy 路径、6个潜步骤、最多256个生成 token。批处理是显式配置：仅等长 token 组，禁止 padding；噪声每题16条并行。官方 CODI 脚本 batch128，本实现至多16。batch1/16 数值差异单独记录，未伪称 bitwise 等价。
- 数字答案按官方最后一个数字抽取，去逗号；无效输出计错。答案多样性为16次中**不同有效数值答案**个数，同时报告无效比例；不把16次采样当作16道独立题。
- 表中95% CI 为3个 seed 均值的 Student-t 区间。固定同一 checkpoint 的 greedy 重复不是3次独立训练；CI=0仅表示重复运行相同，不表示总体准确率没有不确定性。所有 source 仍为推理任务，不含 CLR 校准结果。

## 复现表

| 源 / 数据集 | 论文准确率 (%) | 本次均值 ± 95% CI (%) | 差值 pp | 状态 |
|---|---:|---:|---:|---|
| CODI-GPT2 / GSM8K test (1319) | 43.7 | {pm(repro['aggregate']['accuracy'],100)} | {delta:+.2f} | 完成；`{c['repro_run']}` |
| CODI-LLaMA1B / GSM8K | 51.9（论文A4的β=20配置，主图对应关系待进一步核对） | N/A | N/A | 基础模型访问401 |
| Coconut / GSM8K | 34.1 ± 1.5（论文原表的误差条，非本次95%CI） | N/A | N/A | 无已核实官方权重；训练阻塞 |
| Coconut / ProsQA | 97.0 ± 0.3（论文原表） | N/A | N/A | 同上 |
| Huginn r=32 / GSM8K，无system prompt | 28.05 / 28.20（strict/flexible） | N/A | N/A | 权重/显存阻塞；尚未核验完整 lm-eval prompt、stop、提取器等价性 |

论文依据：[CODI v3 Table 2/A3/A4](https://arxiv.org/pdf/2502.21074v3)、[Coconut Table 1](https://arxiv.org/html/2412.06769v2)、[Huginn Table 2](https://arxiv.org/html/2502.05171v2)。论文数字与本机实测严格分列。

CODI-GPT2 与论文偏差为 {delta:+.2f}pp。{'超过2pp：需要优先核查官方上传 checkpoint 与论文训练版本/seed、batch128 padding及旧attention mask、torch/transformers版本差异；目前不能把偏差归因于单一因素，不用测试集调参。' if abs(delta)>2 else '未超过预设2pp分析门槛。环境、batch大小和官方checkpoint训练seed信息不全仍是复现限制。'}

官方脚本三次对照为 **{pm(reference['accuracy'],100)}%**，与43.7%差 {100*reference['accuracy']['mean']-43.7:+.3f}pp。使用同一官方权重、相同1319道题、相同数值答案提取，仅替换官方数据加载函数为已下载的相同问题/答案；未修改上游test.py。附加hook只计数forward。运行：`{c['reference_run']}`。

因此，初始-2.002pp偏差中的约 **{100*(reference['accuracy']['mean']-repro['aggregate']['accuracy']['mean']):.3f}pp** 可由两种推理协议的联合差异解释：官方batch128左padding、未显式校正GPT-2位置和后续latent attention mask，CLR等长组不padding、batch≤16。没有独立拆分padding、位置编码与batch kernel舍入的因果贡献，不能只归因于其中一项。官方参考已接近论文，不能再将原始差异简单归因于错误checkpoint。本轮保留两种结果，不把官方协议替换进CLR来提高指标。未来若改变源语义，须新建接口/结果v2。

## 确定性

- bf16，固定 seed，关闭TF32，启用 deterministic algorithms；每个抽查问题重复完整输入2次。
- full reproduction 抽查前16个固定ID问题，每seed重复；最大逐元素差值：`{repro['max_latent_abs_difference']}`。
- noise run 的同类检查最大差值：`{noise['max_latent_abs_difference']}`。
- sigma=0 与无扰动单输入轨迹完全相等检查已执行；失败会中止。不将单样本与批处理的舍入差异混入重复运行的非确定性。
- fp16 尚未测量；不推断其他设备、驱动或dtype也确定。

## 噪声曲线（val，64题，非test）

![CODI 噪声曲线](figures/M0_codi_noise_v1.png)

| sigma | 准确率 % ±95%CI | 不同有效答案数 ±95%CI | 无效输出 % ±95%CI |
|---:|---:|---:|---:|
{table}

单次基线在该面板为{100*noise['aggregate']['accuracy']['mean']:.4f}%，K=16批处理sigma=0为{100*noise['noise']['0']['accuracy']['mean']:.4f}%。这是实测的batch数值/输出差异，曲线内部所有sigma均使用相同K=16协议，不能混用单次基线计算噪声收益。当前所有非零sigma的均值都低于同batch的sigma=0基线；答案多样性增加并不意味着正确率提高。

来源：`runs/{c['noise_run']}/metrics.json`；每个 rollout 原文、答案、seed、sigma、problem ID 均在 events.jsonl。未为阻塞的源绘制虚构曲线。该曲线不足以证明潜空间搜索或校准收益。

## Coconut 训练预算与负面结果

使用官方 GPT-2 架构与真实 ProsQA 前8条训练样本、随机初始化权重测**计算吞吐**，不是训练精度实验。配置 `M0_coconut_cost.yaml` 的初次运行快照 batch=8，FP32，与官方dtype一致；0/3/6 latent阶段的batch耗时均值分别约0.181/2.478/4.171秒。运行 `20261028T091345344806Z_324d80d867fe_b8cc5d`。

按最慢已测阶段恒定吞吐外推，ProsQA 17886题×50epochs 为约129.52 GPU小时/seed，三个seed约388.56 GPU小时；这是共享GPU上的保守估计，不是实际训练耗时，也未计验证、checkpoint、DDP开销。GSM8K-Aug的2620.64 GPU小时估算使用ProsQA序列长度做代理，**不是有效的GSM8K精确预算**；要得到可信预算须用实际增强训练数据重测。最初公式使用385000近似题数；官方论文实际385620，已在后续配置修正，未改旧结果。

随后按官方batch=32做探针时触发CUDA OOM（尝试额外申请3.64 GiB），完整 traceback 已记录；没有自动减batch重跑。因此Coconut两个数据集均暂缓：ProsQA成本估算超过40 GPU小时，GSM8K还存在实际recipe显存与预算未验证阻塞。不能声称两者已按官方recipe训练。

## 成本与运行命令

| 运行 | GPU小时（单卡wall，包括模型加载） | 源forward调用 |
|---|---:|---:|
| CODI smoke b16 | {smoke['gpu_hours']:.4f} | {smoke['source_forward_calls']} |
| 官方脚本test | {reference['gpu_hours']:.4f} | {reference['source_forward_calls']} |
| CODI test | {repro['gpu_hours']:.4f} | {repro['source_forward_calls']} |
| CODI noise | {noise['gpu_hours']:.4f} | {noise['source_forward_calls']} |

GPU为实测 RTX4090，详见 M0_env。forward为实际模型调用次数，batch调用算一次，不等于样本token总数；逐结果另存batch_size。共享GPUwall不可当作独占效率。失败探针的成本未完整仪表化，见BLOCKERS，不能计成零。

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_eval.py configs/M0_codi_gpt2_smoke_b16.yaml
PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_eval.py configs/M0_codi_gpt2_repro_b16.yaml smoke_run=runs/{c['smoke_run']}
PYTHONPATH=. OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_eval.py configs/M0_codi_gpt2_noise_b16.yaml smoke_run=runs/{c['smoke_run']}
```

尚需：解决LLaMA官方授权；在足够显存下验证Huginn及官方评测协议；对Coconut制定40GPU小时以内可执行的官方训练方案或保持暂缓；补齐这些源的复现、确定性和噪声曲线。M0不宣称全部完成，后续结果须另建运行与报告版本。
'''
    Path('reports/M0_repro.md').write_text(text)
    write_json(report_run.path/'sources.json',{'repro':c['repro_run'],'noise':c['noise_run'],'smoke':c['smoke_run'],'reference':c['reference_run']})
    report_run.write_metrics(status='completed',kind='derived_report',delta_pp=delta,gpu_hours=0,source_forward_calls=0)
    print(report_run.path)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
