import argparse,json,hashlib
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def read(p):return json.loads(Path(p).read_text())
def fmt(s):return f"{100*s['mean']:.3f}% [{100*s['ci95'][0]:.3f}, {100*s['ci95'][1]:.3f}]"
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 off=read(Path(c['official'])/'metrics.json');ref={r['id']:r for r in read(Path(c['official'])/'predictions.json')}
 rows=[f"| 官方 test.py batch128 | {fmt(off['accuracy'])} | 100%（自身） |"]
 all_disagreements={};artifacts=[]
 for key,bs in [('clr16',16),('clr128',128)]:
  path=Path(c[key]);metrics=read(path/'metrics.json');values=metrics['results'][str(bs)]
  if metrics['status']!='completed':raise RuntimeError('Completed results required')
  rows.append(f"| CLR batch{bs} | {fmt(values['accuracy'])} | {fmt(values['agreement_official'])} |")
  predictions=read(path/f'predictions_b{bs}.json');diff=[{**r,'official':ref[r['id']]} for r in predictions if r['pred']!=ref[r['id']]['official_answer']]
  all_disagreements[str(bs)]=diff;write_json(run.path/f'disagreements_b{bs}.json',diff)
 table='| 运行 | 原始test1319准确率（95% CI） | 对官方逐题一致率（95% CI） |\n|---|---|---|\n'+'\n'.join(rows)
 difference=100*off['accuracy']['mean']-c['paper_reference_percent']
 report=f'''# M0b CODI-LLaMA v2：复现与差异分析

作者checkpoint已取得、校验并真实加载；完成smoke、确定性检查和全1319题官方/CLR对照。**噪声曲线尚未完成，M0b不标记整体验收。** 底模是已验证同一性的ModelScope版本，训练后checkpoint是作者锁定的公开版本，未换模型。

{table}

均为按题bootstrap10,000次95%区间，不用3次greedy重复制造零CI。官方清洁test1312子集为{fmt(off['clean_subset_accuracy'])}。完整原始指标及run清单见 [复现记录](M0b_codi_llama_repro_v2.md)。8题同输入重复的latent最大差为0；真实smoke只说明机制与运行正常，不保证所有输入/批量不变。

## 逐题接口差异

batch16有{len(all_disagreements['16'])}题与官方答案不同，同batch128后剩{len(all_disagreements['128'])}题。明细见 `{run.path}/disagreements_b16.json` 和 `disagreements_b128.json`。batch128对照没有完全消除残差，因此不能把全部差异归因于batch大小，也不能声称逐题100%等价。精度/内核执行路径等残差原因尚需进一步定位；不针对错误题修改解码、正则或划分。

## 与论文数字偏差超过2pp的分析

固定参考仍为论文v3附录Table A4的β=20条目{c['paper_reference_percent']}%，本次官方脚本结果相差 **{difference:+.3f}pp**。该条目是超参数消融结果，作者发布checkpoint到该条目的精确映射尚无训练manifest可证实；不能在看到结果后另挑接近的数字当验收目标。[论文Table A4](https://arxiv.org/html/2502.21074v3#A6.T4)

已排查的解释：

- 基础模型镜像不是替代版本：权重LFS指针与配置/tokenizer的Git对象均与锁定上游一致。
- 这不是仅由CLR接口引入的偏差：官方脚本本身达到55.497%，CLR同batch为55.421%。
- 移除7个训练13-gram重叠问题后，官方准确率为55.335%，仍比51.9%高3.435pp，不能用这7题解释全部差异。比较全test是为官方benchmark；clean test仍用于后续v2边界。
- 参数遵循已锁定官方recipe：bf16、6连续输入、LoRA128/32、projection2048、greedy、256输出上限。没有根据test分数调这些参数。
- checkpoint存在官方当前模型类未定义的4个dynamic_cls张量；官方脚本原本strict=False忽略它们。v2显式以惰性buffer保留这四键，最终strict=True；其余额外/缺失键仍报错，未使用或猜测这个头的计算。这个代码/权重schema差异提示需要核对发布checkpoint的训练版本，但**不是已证明的准确率偏差成因**。

仍未解决：发布checkpoint与论文实验条目/训练seed/训练代码版本的对应关系，以及不同torch/transformers/数值执行路径对答案的影响。当前结果可复查，但不能据此声称完整还原论文训练实验。

## 版本、失败与成本

v1保持不变。第一次strict加载失败和GPU5官方入口显存不足均保留在BLOCKERS；后续显式使用GPU4，未降低模型/精度/latent/batch。8项最小测试通过（包含auxiliary键拒绝与真实微型LLaMA反馈测试），不是替代benchmark。

资源下载、源码版本、环境、GPU型号、wall GPU小时与前向次数位于每个run。官方full：{off['gpu_hours']:.6f} GPU wall小时、{off['source_forward_calls']}次forward。CLR其他阶段成本见其metrics，重试不隐去。精确执行配置在run/config.yaml；辅助对照命令为：

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0b_codi_llama_eval_v2.py configs/M0b_codi_llama_repro_v2_gpu4.yaml smoke_run=runs/20261029T013734197681Z_7bcbbe6fbd3d_77ffd6 reference={c['official']}/predictions.json 'batch_sizes=[128]'
```
'''
 output=Path(c['output'])
 if output.exists():raise RuntimeError('Report exists; use a new version')
 output.write_text(report,encoding='utf-8')
 job=read(c['job_status']);paths=set(job['runs'].values())|{c['clr128']}
 costs=[]
 for path in paths:
  m=read(Path(path)/'metrics.json');costs.append({'run':path,'gpu_hours':m.get('gpu_hours'),'source_forward_calls':m.get('source_forward_calls')});artifacts.extend(p for p in Path(path).rglob('*') if p.is_file())
 write_json(run.path/'costs.json',costs)
 write_json(run.path/'paper_reference.json',{'url':c['paper_reference_url'],'section':'Appendix F Table A4, beta=20','reference_percent':c['paper_reference_percent'],'delta_pp':difference,'checkpoint_to_paper_mapping':'unresolved'})
 run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,disagreement_counts={k:len(v) for k,v in all_disagreements.items()},paper_delta_pp=difference)
 print(run.path,flush=True)
