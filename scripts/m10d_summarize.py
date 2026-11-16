"""Derive the four manuscript cells from final predictions and frozen M10 CSV."""
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


def main(c):
    root=Path(c['output'])
    with recorded(c) as run:
        final=json.loads((root/'final.json').read_text());points=[];aucs=[]
        for rs in c['seeds']:
            for split in ['reader_val','test']:
                for q in ['Q1','Q2']:
                    raw=pq.read_table(Path(final['run'])/f'B4_{q}_{rs}_{split}.parquet').to_pylist()
                    for kind in (['clean','traj'] if q=='Q1' and split=='reader_val' else ['clean']):
                        data=[r for r in raw if r['state_ref']['kind']==kind]
                        p=softmax(np.array([r['logits'] for r in data])/data[0]['temperature'],axis=1)
                        y=np.array([[1-r['label_soft'],r['label_soft']] for r in data])
                        f=pd.DataFrame(dict(problem_id=[r['problem_id'] for r in data],t=[r['state_ref']['t'] for r in data],
                            nll=-(y*np.log(p.clip(1e-12))).sum(1),accuracy=y[np.arange(len(p)),p.argmax(1)],
                            label=[r['label'] for r in data],score=p[:,1]))
                        common=dict(source='llama',model='B4',seed=rs,split=split,qtype=q,kind=kind)
                        g=f.groupby(['problem_id','t'])[['nll','accuracy']].mean();g['instances']=f.groupby(['problem_id','t']).size()
                        points.append(g.reset_index().assign(**common))
                        if q=='Q2':aucs.append(f.assign(**common))
        frame=pd.concat(points,ignore_index=True);frame.to_csv(root/'problem_metrics.csv',index=False,float_format='%.15g')
        summary=[]
        for key,g in frame.groupby(['source','model','split','qtype','kind']):
            common=dict(zip(['source','model','split','qtype','kind'],key))
            summary.extend(dict(**common,**v) for v in metric_summary(g,['nll','accuracy']))
        af=pd.concat(aucs,ignore_index=True)
        for split,g in af.groupby('split'):
            summary.append(dict(source='llama',model='B4',split=split,qtype='Q2',kind='clean',**auc_summary(g)))
        pd.DataFrame(summary).to_csv(root/'B4_summary.csv',index=False,float_format='%.15g')
        old=pd.read_csv('data/M10_v1/paper_inputs/main.csv');cells=[]
        for source,model,values in [('gpt2','B6',old.to_dict('records')),('llama','B4',summary)]:
            for q,metric in [('Q1','nll'),('Q2','auroc')]:
                v=next(r for r in values if r['source']==source and r['model']==model and r['split']=='test' and r['qtype']==q and r['metric']==metric)
                cells.append(dict(source=source,model=model,qtype=q,metric=metric,mean=float(v['mean']),lo=float(v['lo']),hi=float(v['hi']),
                    table=f"{v['mean']:.3f}",origin='found_frozen' if model=='B6' else 'new_training_final',
                    evidence='data/M10_v1/paper_inputs/main.csv' if model=='B6' else str(root/'B4_summary.csv')))
        write_json(root/'four_cells.json',dict(cells=cells,protocol='test; clean; seen templates; micro within t, equal across t and fixed seeds 11/22/33',
            uncertainty='95% problem-cluster bootstrap 10000, preserving steps/seeds; conditional on three trained seeds; B6 CIs reused',summary_run=str(run.path)))
        # Independently agree with the frozen per-step final evaluation metrics.
        stats=json.loads(Path(final['statistics']).read_text())
        for cell in cells:
            if cell['model']!='B4':continue
            vals=[r[cell['metric']] for r in stats if r['split']=='test' and r['kind']=='clean' and r['qtype']==cell['qtype']]
            assert abs(np.mean(vals)-cell['mean'])<1e-7,(vals,cell)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)
        print(json.dumps(cells,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
