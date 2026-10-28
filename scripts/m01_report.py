import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.stats_v2 import bootstrap_mean

def read(path):return json.loads(Path(path).read_text())
def fmt(v):
 m=v['mean']*100;l,h=[x*100 for x in v['ci95']]
 return f'{m:.3f}% [{l:.3f}, {h:.3f}]'
def main(c):
 out=Path('reports/evidence_v2');out.mkdir(exist_ok=True)
 refs=read(Path(c['reference'])/'per_question.json');refmap={r['problem_id']:r for r in refs}
 variants={};tab=[]
 for key in ['padding_candidate','stride_candidate','bf16','fp32']:
    for p in sorted(Path(c[key]).glob('predictions_b*.json')):
        rows=read(p);name=key+'_'+p.stem;variants[name]={r['id']:r for r in rows}
        tab.append(f"| {name} | {fmt(bootstrap_mean([r['correct'] for r in rows]))} | {fmt(bootstrap_mean([r['pred']==refmap[r['id']]['official_answer'] for r in rows]))} |")
 details=[]
 for r in refs:
    preds={name:items[r['problem_id']]['pred'] for name,items in variants.items()}
    ctx=next(iter(variants.values()))[r['problem_id']]['meta']
    details.append({**r,'v2_predictions':preds,'padding_audit':{'v1_pad_count':0,'v2_pad_count':ctx['pad_count'],'v2_pad_to':ctx['pad_to']},
      'audit':{'padding_direction':'official/v2 left; v1 unpadded','prefill_mask':'zero pads, one question+BOT','cached_mask':'omitted, matching official','position_ids':'implicit absolute, includes left pads','latent_steps':6,'BOT':50258,'EOT':50259,'max_new_tokens':256,'answer_regex':r'-?\d+\.?\d*','extraction':'remove commas, last numeric substring; invalid=None vs official inf, both incorrect against finite gold'},
      'explanation':'Padding/context changed for v1; canonical-context v2 isolates additional tensor-layout and arithmetic effects. Individual causal attribution is not implied by aggregate agreement.'})
 write_json(out/'protocol_all_questions.json',details)
 write_json(out/'v1_all_454_disagreements.json',[r for r in details if not r['agree']])
 last={n:x for n,x in variants.items() if n.startswith('bf16_')}
 write_json(out/'v2_remaining_disagreements.json',[r for r in details if any(r['v2_predictions'][n]!=r['official_answer'] for n in last)])
 pairwise=[]
 for x,y in [(1,4),(1,16),(4,16)]:
    a=last.get(f'bf16_predictions_b{x}');b=last.get(f'bf16_predictions_b{y}')
    if a and b:pairwise.append({'batch_sizes':[x,y],**bootstrap_mean([a[i]['pred']==b[i]['pred'] for i in a])})
 write_json(out/'batch_pairwise.json',pairwise)
 freeze=read('reports/M0_codi_freeze_v1.json');bad=[p for p,h in freeze['sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 write_json(out/'v1_integrity.json',{'checked_files':len(freeze['sha256']),'changed':bad})
 if bad:raise RuntimeError('Frozen v1 files changed')
 d=read(Path(c['dedup'])/'metrics.json');det=read(Path(c['determinism'])/'metrics.json')
 clean=set(read('data/splits/gsm8k_M01_v2.json')['test']);clean_tab=[]
 for n,rows in variants.items():
    if n.startswith(('bf16_','fp32_')):
        x=[r for i,r in rows.items() if i in clean];clean_tab.append(f"| {n} | {len(x)} | {fmt(bootstrap_mean([r['correct'] for r in x]))} |")
 text='''# M0.1 协议与去重（v2，审查稿）

状态：2026-10-28 用户明确接受 bf16 复放验收，并授权固定 batch=1、fp32 完整噪声实验。G-σ仍须完成并审查，不能进入M1。官方bf16复放与fp32生成的差异保留，不把99.92%冒充fp32的一致率。

## 逐题对照

所有1319题、全部v1不一致题、各v2版本预测及逐项协议检查保存在 [protocol_all_questions.json](evidence_v2/protocol_all_questions.json)、[v1_all_454_disagreements.json](evidence_v2/v1_all_454_disagreements.json)、[v2_remaining_disagreements.json](evidence_v2/v2_remaining_disagreements.json)。比较答案数值，不要求文字输出完全相同。

v1与官方预测一致率仅65.580%，454题不同。不能用两者准确率接近代替接口一致性。v2保留官方cohort=128原始顺序的左padding宽度；上下文只由题目长度确定，不使用答案。prefill传mask，潜在步与答案解码不续传mask，position ids保持官方隐式绝对位置，连续输入6步，BOT/EOT=50258/50259，不额外插EOS，greedy上限256，排除词表最后一项。数值正则、去逗号及取最后数值一致。

仅修padding仍不足：逐行cat隐状态使张量连续化，改变bf16投影的数值结果。v2零噪声直接保留官方切片/unsqueeze布局，EOT使用官方expand布局。未用测试答案选择噪声或训练参数。

以下是确定性评估：按题bootstrap 10,000次的95%区间，**不是三个推理seed的零方差区间**。区间以不对称端点报告，避免虚假的对称“±”。初始候选不覆盖，所有结果均保留。

| 配置/真实batch | 准确率（95% CI） | 与原官方答案一致率（95% CI） |
|---|---|---|
'''+ '\n'.join(tab)+f'''

fp32 batch=1全量运行 `{c['fp32']}` 的官方一致率为95.299%，低于99%门槛。即使bf16官方batch复放达到99%以上，也不能据此宣布fp32生产协议通过原门槛。`configs/protocol_codi_v2.yaml` 已明确固定fp32、物理batch=1，并记录用户接受复放验收及授权完整噪声实验的决定；后续dump/rollout/搜索须引用它，不能临时更换batch或精度。原始500验证池重划分获用户明确授权。

## 剩余1题追踪

原始test索引1306，ID `b209f7b0961d62d42f0e75ff41a3a83d679a19b3828fb251c8fe4a68460449c2`，官方答案7、bf16复放答案8；位于最后39题cohort、宽度126、该题无padding。
追踪运行 `{c['residual']}`：参数逐张量相等，prefill输入、mask、hidden与logits逐元素相同；首次投影后的潜在输入最大差0.25，随后递归放大。初始条件/权重/正则不能解释此题；已将差异定位到投影数值路径，具体dispatch原因仍在核验，不能声称已完全归因。详见run内 `trace_comparison.json`。

## 数据去重与v2冻结

训练资源：`zen-E/GSM8k-Aug` revision `c4972fba71a612e2089d8fb10f78caa01c740083`，385620条，385529个规范化题目。SHA-256 `07b2e0180ec74693de60503260c51d86e71931c7569a026f3240939334213609`。
NFKC+casefold、规范弯引号、保留带符号小数、去标点后的词序列；精确匹配与任意连续13-token匹配分别报告。检查全量发布语料；官方训练还过滤长度/答案格式，因此这是保守的污染筛查，不证明每个命中都实际参加训练。13-gram模板重合也可能是假阳性，仍按预先规则移除。

| 原划分 | N | 精确重叠 | 13-gram重叠 | 并集移除 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(f"| {k} | {v['n']} | {v['exact_overlap']} | {v['13gram_overlap']} | {v['union_overlap']} |" for k,v in d['overlap'].items())+f'''

原500池剩493题，按problem ID排序冻结val=300、cal=193；test剩1312。v1不改动。split: `data/splits/gsm8k_M01_v2.json`，SHA `{d['split_sha256']}`。移除问题及训练匹配见 `{c['dedup']}/per_problem_overlap.json` 与 `removed_from_cal_test.json`；重分配见 `reassignment.json`。去重仅依据题目文本，与正确率无关。

测试集完整1319题仅用于请求的官方协议对照；清洁test1312题是v2后续评估边界。以下仅按冻结ID过滤已完成预测，没有重新调参：

| 配置 | N | 清洁test准确率（95% CI） |
|---|---:|---|
'''+ '\n'.join(clean_tab)+f'''

## 确定性与成本

在8个val问题上，bf16与fp32各自同一输入重复两次、batch=1，最大潜状态绝对差均为0，答案全部一致。此结论限于相同上下文/形状；跨batch结果不同，不能称为跨batch确定。证据 `{c['determinism']}`。
每个run保存配置、git/dirty diff、环境与GPU型号、forward次数和GPU wall小时。多个试验并发在同一GPU，wall GPU小时相加不是独占GPU利用小时；完整成本清单见 `evidence_v2/costs.json`。对失败入口无法恢复的成本记未知。

## 复现命令与版本

```bash
PYTHONPATH=. .venv/bin/python scripts/m01_compare_v1.py configs/M01_compare_v1.yaml
PYTHONPATH=. .venv/bin/python scripts/m01_fetch_aug.py configs/M01_fetch_aug.yaml
PYTHONPATH=. .venv/bin/python scripts/m01_dedup.py configs/M01_dedup.yaml
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/m01_protocol_eval.py configs/M01_protocol_smoke.yaml
# 全量的protocol/smoke_run/reference/batch_sizes覆盖项以各run/config.yaml为准
PYTHONPATH=. .venv/bin/python scripts/m01_determinism.py configs/M01_determinism.yaml
PYTHONPATH=. .venv/bin/python scripts/m01_report.py configs/M01_report.yaml
PYTHONPATH=. .venv/bin/python -m pytest tests/test_m01.py -q
```

v1冻结清单已复核{len(freeze['sha256'])}个文件，变更0。v2尚未完成整体验收，不伪造冻结完成标记；后续修改继续保留已有run和候选代码git版本。
'''
 text+='\n## 实际batch两两一致性\n\n| batch对 | 答案一致率（95% CI） |\n|---|---|\n'+'\n'.join(f"| {r['batch_sizes']} | {fmt(r)} |" for r in pairwise)+'\n\n上述均为真实物理batch，没有把batch参数包装成逐题循环来声称100%一致。fp32与bf16同为batch1时，1249/1319题答案相同（94.693%），另70题不同。\n'
 Path('reports/M0.1_protocol.md').write_text(text)
 noise=read(Path(c['noise_smoke'])/'summary.json');nm=read(Path(c['noise_smoke'])/'metrics.json')
 fig,axes=plt.subplots(1,4,figsize=(16,3.6))
 for f in [0,3]:
    rr=[r for r in noise if r['setting']['from_step']==f and r['setting']['temperature']==0]
    sig=[r['setting']['sigma'] for r in rr]
    for ax,key in zip(axes,['accuracy','distinct_answers','nondegenerate','invalid_rate']):ax.plot(sig,[r['metrics'][key]['mean'] for r in rr],marker='o',label=f'from_step={f}')
 for i,(ax,title) in enumerate(zip(axes,['Accuracy','Distinct valid answers','Nondegenerate fraction','Invalid answer rate'])):
    ax.set(xlabel='sigma',ylabel=title)
    if i!=1:ax.set_ylim(-.03,1.03)
    ax.legend()
 fig.suptitle('SMOKE ONLY: n=1, K=16, seed=11; NOT a validation estimate');fig.tight_layout();fig.savefig(out/'noise_smoke.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(3,6,figsize=(16,8))
 for ax,r in zip(axes.flat,noise):
    h=r['metrics']['V_histogram'];ax.bar([i/16 for i in h['correct_counts']],h['counts'],width=.05);s=r['setting'];ax.set_title(f"s={s['sigma']} f={s['from_step']} temp={s['temperature']}");ax.set(ylim=(0,1.2),xlabel='V',ylabel='Problems')
 fig.suptitle('SMOKE ONLY: one problem per histogram, K=16');fig.tight_layout();fig.savefig(out/'V_histogram_smoke.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {r['setting']['sigma']} | {r['setting']['from_step']} | {r['setting']['temperature']} | {r['metrics']['accuracy']['mean']:.3f} | {r['metrics']['distinct_answers']['mean']:.1f} | {r['metrics']['nondegenerate']['mean']:.3f} | {r['metrics']['invalid_rate']['mean']:.3f} |" for r in noise)
 Path('reports/M0.1_noise.md').write_text(f'''# M0.1 噪声曲线（v2，未完成）

**G-σ尚未验收，完整结果尚未完成，不推荐sigma。** 用户已于2026-10-28明确接受bf16复放验收，授权固定fp32协议完整噪声实验；原95.299%生成一致率仍如实保留。清洁val面板固定300题。

## 已完成smoke

运行 `{c['noise_smoke']}`：1/300题（0.333%），K=16，seed11，18设置，{nm['wall_seconds']:.2f}秒（低于15分钟），forward={nm['source_forward_calls']}，GPU wall={nm['gpu_hours']:.5f}小时。完整设计seed=11/22/33，不把仅1seed smoke伪装为三个seed实验。烟测只检查代码执行和指标完整性。

| sigma | from_step | temperature | accuracy | distinct valid answers | nondegenerate | invalid rate |
|---|---|---|---|---|---|---|
{table}

![Smoke curve](evidence_v2/noise_smoke.png)

![V histogram smoke](evidence_v2/V_histogram_smoke.png)

## 固定完整方案与指标

`configs/M01_noise_full.yaml`：val300题，sigma=0/.1/.3/.5/.75/1/1.5/2，from_step=0/3（T=6），K=16，3个随机seed。对照为sigma0、temperature=.5/1；全词表采样，不启top-k/top-p。源协议只从 `configs/protocol_codi_v2.yaml` 读取，物理batch=1、fp32。随机数由实验seed、problem ID、rollout编号稳定哈希生成，与问题遍历顺序无关；不修改全局RNG。

V=每题16次正确次数/16。非退化比例按每个seed的0<V<1题目比例计算，再汇总seed；不把3seed池化成K48。无效答案为提取不到数值；distinct valid排除无效，同时另存including-invalid指标。所有统计先归约到问题，再做按题10k bootstrap，不能把rollout当独立题目。V直方图有17个离散bin。完整报告应另外给出3seed均值与95%区间，当前没有这样的完整结果。

推荐规则固定为：相对同协议sigma0下降严格<5pp、非退化比例≥30%的高斯设置中，非退化比例最大；并列按更小下降、更小sigma、更早from_step。无设置通过时返回selected=null、等待审查，不能自行挑一个。smoke无论多好都不能推荐。

按smoke实测线性外推约 **{nm['gpu_hours']*300*3:.1f} GPU wall小时**，不是承诺成本；单题长度、共享GPU竞争与高噪声长输出会改变吞吐。未偷偷减题、减K、挑seed或改batch。完整实验命令（用户授权已记录；须使用授权后协议快照匹配的smoke_run，下面历史smoke仅作旧运行记录）：

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/m01_noise.py configs/M01_noise_full.yaml smoke_run={c['noise_smoke']}
```

同一输入两次潜状态最大差0（smoke，fp32）。更全面的8题双精度确定性检查见协议报告。每条rollout文本、提取值、正确性、meta写events.jsonl，每个setting/seed的逐题向量写panel_*.json，V直方图与bootstrap写summary.json。
''')
 costs=[]
 for p in sorted(Path('runs').glob('*/config.yaml')):
    if 'M0.1' in p.read_text() or 'M0b' in p.read_text():
        m=read(p.parent/'metrics.json') if (p.parent/'metrics.json').exists() else {}
        costs.append({'run':p.parent.name,**{k:m.get(k) for k in ['status','wall_seconds','gpu_hours','source_forward_calls']}})
 write_json(out/'costs.json',costs)
 print('Wrote protocol/noise reports and traceable evidence')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
