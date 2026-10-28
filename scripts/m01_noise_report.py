"""Generate final noise artifacts only from a completed full run."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config

def read(p):return json.loads(Path(p).read_text())
def fmt(x):
    lo,hi=x.get('ci95',[x['mean']-x.get('ci95_halfwidth',0),x['mean']+x.get('ci95_halfwidth',0)])
    return f"{x['mean']:.4f} [{lo:.4f}, {hi:.4f}]"
def main(c):
    report=Path('reports/M0.1_noise_full_v2.md')
    if report.exists():raise RuntimeError('Completed full report exists; create a new version')
    run=Path(c['run']);m=read(run/'metrics.json')
    if m['status']!='completed' or m['questions']<300 or len(m['seeds'])<3 or m['K']!=16:raise RuntimeError('Completed full panel required')
    rows=read(run/'seed_summary.json');raw=read(run/'summary.json');rec=read(run/'recommendation.json')
    if len(rows)!=18 or len(raw)!=54:raise RuntimeError('Incomplete setting/seed grid')
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    keys=['accuracy','distinct_answers','nondegenerate','invalid_rate']
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for f in [0,3]:
        group=[r for r in rows if r['setting']['temperature']==0 and r['setting']['from_step']==f]
        for ax,key in zip(axes,keys):
            ax.plot([r['setting']['sigma'] for r in group],[r['metrics'][key]['mean'] for r in group],marker='o',label=f'from_step={f}')
    for ax,key in zip(axes,keys):ax.set(xlabel='sigma',ylabel=key);ax.legend()
    fig.suptitle('CODI fp32 batch=1; clean val n=300; K=16; seeds=11,22,33');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,6,figsize=(18,9))
    for ax,row in zip(axes.flat,rows):
        for s in m['seeds']:
            r=next(r for r in raw if r['setting']==row['setting'] and r['seed']==s);h=r['metrics']['V_histogram']
            ax.plot([k/16 for k in range(17)],h['counts'],marker='.',label=f'seed={s}')
        setting=row['setting'];ax.set_title(f"sigma={setting['sigma']} f={setting['from_step']} temp={setting['temperature']}");ax.set(xlabel='V',ylabel='Problems');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'V_histograms.png',dpi=160);plt.close(fig)
    table='| sigma | from_step | temp | accuracy | distinct valid | nondegenerate | invalid |\n|---|---|---|---|---|---|---|\n'
    for r in rows:
        s=r['setting'];table+='| '+ ' | '.join([str(s[k]) for k in ['sigma','from_step','temperature']]+[fmt(r['metrics'][k]) for k in keys])+' |\n'
    text=f'''# M0.1 完整噪声实验 v2

运行：`{run}`。固定清洁val={m['questions']}，K={m['K']}，随机seed={m['seeds']}，全部18设置完成。模型协议见run/protocol.json；原官方bf16一致性差异和用户接受决定保留，不改变测试边界。

{table}

值为均值[95% CI]。sigma0且greedy的确定性项按题bootstrap 10k；随机项为3seed均值的Student-t区间，各seed按题bootstrap另存summary.json。非退化比例先在每seed的K16上计算，未池化为K48。无效提取值不计入distinct valid，含无效的distinct另存JSON。温度对照为完整词表采样，无top-k/top-p。

![Curves]({out.name}/curves.png)

![V histograms]({out.name}/V_histograms.png)

推荐规则与输出（不得因结果不好修改阈值）：

```json
{json.dumps(rec,ensure_ascii=False,indent=2)}
```

未通过双门槛则不推荐sigma、等待审查。通过时只报告候选，M1仍须用户审查M0.1第1–4项后进入。

计算成本：{m['gpu_hours']:.4f} GPU wall小时，{m['source_forward_calls']}次源forward；GPU型号/环境详见run/environment.json，CUDA设备及权重revision见run/protocol.json。完整命令见run/config.yaml。所有rollout保存在events.jsonl，逐题K16向量在panel_*.json。
'''
    report.write_text(text,encoding='utf-8')
    files=[p for p in run.rglob('*') if p.is_file()]+list(out.glob('*.png'))+[report]
    manifest={'status':'completed_noise_artifacts_only; M0.1 review pending','run':str(run),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('reports/M0.1_noise_full_v2_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with Path('reports/M0.1_noise.md').open('a',encoding='utf-8') as f:f.write('\n\n完整实验已完成，见 [完整噪声报告](M0.1_noise_full_v2.md)。上文smoke/运行中记录为历史记录。\n')
    with Path('docs/EXPERIMENTS.md').open('a',encoding='utf-8') as f:f.write('\n\n## M0.1 完整噪声 v2 完成（待审查）\n\n'+f'运行 `{run}`；完整配置、成本、图与负面结果见 `reports/M0.1_noise_full_v2.md`。\n\n'+table+'\n推荐输出：`'+json.dumps(rec,ensure_ascii=False)+'`\n')
    with Path('docs/BLOCKERS.md').open('a',encoding='utf-8') as f:f.write('\n\n## M0.1 完整噪声运行结束\n'+f'运行 `{run}` 完成；G-σ结果见完整报告，M0.1审查及M1仍未授权。推荐输出：'+json.dumps(rec,ensure_ascii=False)+'\n')
    print(report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
