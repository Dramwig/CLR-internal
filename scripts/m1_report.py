"""Traceable M1 gate/final metrics report; never fabricates missing full outputs."""
import argparse,json,hashlib
from pathlib import Path
import torch
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.stats_v2 import bootstrap_mean

def main(c):
 with recorded(c) as run:
    lines=['# M1 实测结果与验收状态','']
    records={}
    for key in ['smoke_run','gate_run','budget_run','generation_run']:
        if c.get(key):
            records[key]=json.loads(Path(c[key],'metrics.json').read_text())
            lines.extend([f'## {key}',f'运行：`{c[key]}`。','```json',json.dumps(records[key],ensure_ascii=False,indent=2),'```',''])
    full=records.get('generation_run',{})
    figures=Path('reports/M1_figures');figures.mkdir(exist_ok=True)
    if full.get('status')=='completed':
        rows=[];variances=[];early=[];root=Path(c['output'])
        for split in ['reader_train','reader_val','val','cal','test']:
            early.extend(pq.read_table(root/split/'early_exit.parquet').to_pylist())
            if (root/split/'mc.parquet').exists():rows.extend(pq.read_table(root/split/'mc.parquet').to_pylist())
            if (root/split/'variance64.parquet').exists():variances.extend(pq.read_table(root/split/'variance64.parquet').to_pylist())
        stats=[]
        for split in ['val','cal','test']:
            for policy in ['pi','pi_T']:
                fig,axes=plt.subplots(2,3,figsize=(12,6))
                for t,ax in enumerate(axes.flat):
                    subset=[r for r in rows if r['split']==split and r['policy']==policy and r['t']==t];v=[r['value'] for r in subset]
                    stats.append({'split':split,'policy':policy,'t':t,'mean':bootstrap_mean(v),'nondegenerate':bootstrap_mean([0<x<1 for x in v]),'invalid':bootstrap_mean([r['invalid_rate'] for r in subset])})
                    ax.hist(v,bins=np.arange(-.5,17)/16);ax.set_title(f't={t}, n={len(v)}');ax.set_xlabel('V_hat (K=16)')
                fig.tight_layout();fig.savefig(figures/f'{split}_{policy}_V.png');plt.close(fig)
                lines.append(f'![{split} {policy}](M1_figures/{split}_{policy}_V.png)')
        vstats=[]
        for policy in ['pi','pi_T']:
            subset=[r for r in variances if r['policy']==policy]
            estimates=[float(np.var(r['blocks16'],ddof=1)) for r in subset]
            vstats.append({'policy':policy,'pairs':len(subset),'K16_sampling_variance_from_four_K16_blocks':bootstrap_mean(estimates),'limitations':'Only4 blocks per pair; noisy variance estimates; distinct seeds from main K16 labels'})
        write_json(run.path/'step_statistics.json',stats);write_json(run.path/'variance_statistics.json',vstats)
        lines.extend(['## 按题bootstrap10k标签统计','|split|policy|t|V mean [95% CI]|nondegenerate [95% CI]|','|---|---|---|---|---|'])
        for s in stats:lines.append(f"|{s['split']}|{s['policy']}|{s['t']}|{s['mean']['mean']:.4f} {s['mean']['ci95']}|{s['nondegenerate']['mean']:.4f} {s['nondegenerate']['ci95']}|")
        lines.extend(['','K64方差结果：','```json',json.dumps(vstats,ensure_ascii=False,indent=2),'```'])
    else:lines.append('**M1全量尚未完成；不存在完整MC直方图或K64方差结论。**')
    path=Path('reports/M1_results.md');path.write_text('\n'.join(lines)+'\n')
    run.write_metrics(status='completed',generation_completed=full.get('status')=='completed',report=str(path),report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_forward_calls=0,gpu_hours=0)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
