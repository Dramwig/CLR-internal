"""Aggregate M10e final predictions with the frozen Table 1 estimand."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import softmax
from clr.paper.statistics import metric_summary, auc_summary
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from scripts.m10e import stats as per_step_stats


def main(c):
    root=Path(c['output'])
    assert not (root/'freeze.json').exists(), 'Do not overwrite frozen statistics'
    with recorded(c) as run:
        summaries=[];checks=[];rows=[]
        for key in ['gpt2_HSRM','llama_HSRM','gpt2_CLUE','llama_CLUE','llama_B2']:
            final=json.loads((root/f'final_{key}.json').read_text())
            rawstats=json.loads(Path(final['statistics']).read_text());points=[];aucs=[];native=[];canonical=[]
            for p in sorted(Path(final['run']).glob('*.parquet')):
                method,q,rs,split=p.stem.split('_',3)
                random_seed=0 if rs=='None' else int(rs)
                data=pq.read_table(p).to_pylist()
                for kind in (['clean','traj'] if q=='Q1' and split=='reader_val' else ['clean']):
                    ds=[r for r in data if r['state_ref']['kind']==kind]
                    if not ds:continue
                    x=np.array([r['calibrated_logits'] for r in ds],dtype=np.float64)
                    prob=softmax(x,axis=1)
                    y=np.eye(4)[[r['label'] for r in ds]] if q=='Q3' else np.array([[1-r['label_soft'],r['label_soft']] for r in ds])
                    frame=pd.DataFrame(dict(problem_id=[r['problem_id'] for r in ds],t=[r['state_ref']['t'] for r in ds],
                        nll=-(y*np.log(prob.clip(1e-12))).sum(1),accuracy=y[np.arange(len(prob)),prob.argmax(1)],
                        brier=(1-2*(prob*y).sum(1)+(prob*prob).sum(1))/(1 if q=='Q3' else 2),
                        label=[r['label'] for r in ds],score=prob[:,1]))
                    common=dict(source=final['source'],model=method,seed=random_seed,split=split,qtype=q,kind=kind)
                    # Preserve the original online statistics and reproduce
                    # them at their actual dtype. Table 1's frozen exporter
                    # scripts/m9_export.py uses float64 from serialized logits.
                    dtype=np.float64 if method=='CLUE' else np.float32
                    native.extend(dict(seed=random_seed,split=split,qtype=q,**r) for r in per_step_stats(ds,x.astype(dtype),q))
                    canonical.extend(dict(seed=random_seed,split=split,qtype=q,**r) for r in per_step_stats(ds,x,q))
                    g=frame.groupby(['problem_id','t'])[['nll','accuracy','brier']].mean()
                    g['instances']=frame.groupby(['problem_id','t']).size()
                    points.append(g.reset_index().assign(**common))
                    if q=='Q2':aucs.append(frame.assign(**common))
                    rows.append(dict(**common,rows=len(ds),problems=frame.problem_id.nunique()))
            frame=pd.concat(points,ignore_index=True)
            frame.to_csv(root/f'problem_metrics_{key}.csv',index=False,float_format='%.15g')
            this=[]
            for sk,g in frame.groupby(['source','model','split','qtype','kind']):
                common=dict(zip(['source','model','split','qtype','kind'],sk))
                this.extend(dict(**common,**r) for r in metric_summary(g,['nll','accuracy','brier']))
            for split,g in pd.concat(aucs,ignore_index=True).groupby('split'):
                this.append(dict(source=final['source'],model=final['method'],split=split,qtype='Q2',kind='clean',**auc_summary(g)))
            # Agree independently with final per-seed/per-step micro metrics.
            for r in this:
                vals=[s[r['metric']] for s in rawstats if s['split']==r['split'] and s['qtype']==r['qtype'] and s['kind']==r['kind']]
                panel=lambda rs:[s[r['metric']] for s in rs if s['split']==r['split'] and s['qtype']==r['qtype'] and s['kind']==r['kind']]
                native_error=abs(np.mean(vals)-np.mean(panel(native)))
                assert native_error<1e-7,(r,'native replay differs',native_error)
                canonical_error=abs(np.mean(panel(canonical))-r['mean'])
                assert canonical_error<1e-7,(r,'float64 Table 1 replay differs',canonical_error)
                checks.append(dict(key=key,split=r['split'],qtype=r['qtype'],kind=r['kind'],metric=r['metric'],
                    native_replay_absolute_difference=native_error,canonical_replay_absolute_difference=canonical_error,
                    online_vs_table1_precision_difference=float(r['mean']-np.mean(vals))))
            summaries.extend(this)
            write_json(run.path/'progress.json',dict(completed=key,summary_rows=len(summaries)))
            print('summarized',key,flush=True)
        summary=pd.DataFrame(summaries);summary.to_csv(root/'summary.csv',index=False,float_format='%.15g')
        write_json(root/'aggregation_checks.json',dict(passed=True,checks=checks,row_counts=rows))
        main=[]
        for r in summaries:
            if r['split']=='test' and r['kind']=='clean' and ((r['qtype']=='Q1' and r['metric']=='nll') or (r['qtype']=='Q2' and r['metric']=='auroc') or (r['qtype'] in ['Q3','Q4'] and r['metric']=='accuracy')):
                main.append(dict(**r,display=f"{r['mean']*100:.1f}" if r['qtype'] in ['Q3','Q4'] else f"{r['mean']:.3f}"))
        write_json(root/'table_cells.json',dict(cells=main,protocol='test, clean, seen templates; micro within t; equal across t and seeds 11/22/33; CLUE deterministic once',
            uncertainty='95% problem-cluster bootstrap 10000, same problem draw across steps and seeds; conditional on fixed fitted models',run=str(run.path)))
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
