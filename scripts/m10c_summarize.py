"""YAML-only CPU aggregation; no fitting or new scientific predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.paper.m10c_cost import sha, summarize_conformal


def main(c):
    root = Path(c['output']); dest = root / 'paper_inputs'; dest.mkdir(exist_ok=True)
    used = {}
    def readj(p):
        used[str(p)] = sha(p)
        return json.loads(Path(p).read_text())
    measure = readj(c['measurement'])
    if not measure['passed']: raise ValueError('Measurement gate failed')
    panel = readj(Path(measure['run']) / 'panel.json')
    # Exact checkpoint matrix shapes, no source forward or altered model.
    ckpt = load_config(c['source_protocol'])['source']['checkpoint']
    used[ckpt] = sha(ckpt)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if 'state_dict' in state: state = state['state_dict']
    shapes = {k: list(v.shape) for k,v in state.items()
              if v.ndim == 2 and not any(s in k for s in ['wte.weight','wpe.weight'])}
    linear = sum(2 * int(np.prod(v)) for v in shapes.values())
    profile = dict(method='Cached one-token step; 2 FLOP/MAC; unmerged LoRA, projection and full vocabulary head; excludes prefill',
                   linear_flops=linear, attention_flops_per_context_token=4*12*768,
                   linear_weight_shapes=shapes, checkpoint=ckpt, checkpoint_sha256=used[ckpt])
    write_json(root / 'source_profile_reproduced.json', profile)
    if linear != readj(root/'source_profile.json')['linear_flops']: raise ValueError('Source profile differs')
    manifest = c['source_manifest']; used[manifest] = sha(manifest)
    mf = pq.read_table(manifest, columns=['problem_id','pad_to']).to_pandas().drop_duplicates()
    widths = mf.set_index('problem_id').pad_to.to_dict()
    den = np.mean([linear+4*12*768*(widths[r['problem_id']]+2+r['state_ref']['t']) for r in panel['rows']['Q3']])
    # Historical completed resident costs. No allocation by sample or parameter count.
    p = c['clr_costs']; used[p] = sha(p); f = pd.read_csv(p)
    f = f[(f.stage=='train') & (f['case']=='main') & (f.variant=='lite') & (f.status=='completed')]
    if set(f.seed) != {11,22,33} or len(f)!=3: raise ValueError('CLR cost seeds')
    costs = [dict(model='CLR-lite',seed=int(r.seed),run=r.run,gpu_hours=r.gpu_hours,scope='Q1-Q4 joint completed training; loading, val and early stopping included') for r in f.itertuples()]
    for model, pattern, ledger_path in [('B6',c['b6_refs'],c['b6_ledger']),('B5c',c['b5c_refs'],c['b5c_ledger'])]:
        ledger = readj(ledger_path)
        for seed in [11,22,33]:
            ref = readj(pattern.format(seed=seed)); run = str(Path(ref['run']).resolve())
            matches = [r for r in ledger if str(Path(r['run']).resolve())==run and r['status']=='completed']
            if len(matches)!=1: raise ValueError('Missing historical cost '+run)
            hist = readj(Path(run)/'history.json')
            costs.append(dict(model=model,seed=seed,run=run,gpu_hours=matches[0]['gpu_hours'],
                              scope='Q1-Q4 per-type hyperparameter sweeps (B6)' if model=='B6' else 'Q3-Q4 joint single full pass (B5c)',
                              pure_Q3_train_gpu_hours=sum(h.get('task_seconds',{}).get('Q3',h.get('training_seconds',0) if h.get('task')=='Q3' else 0) for h in hist)/3600,
                              pure_Q4_train_gpu_hours=sum(h.get('task_seconds',{}).get('Q4',h.get('training_seconds',0) if h.get('task')=='Q4' else 0) for h in hist)/3600))
    cost = pd.DataFrame(costs); cost.to_csv(dest/'historical_training.csv',index=False,float_format='%.15g')
    training = cost.groupby('model').gpu_hours.agg(['sum','median']).to_dict('index')
    shared = readj(c['b3_metrics'])
    training['B3'] = dict(sum=None,median=None,shared_run_total_gpu_hours=shared['gpu_hours'],scope='B0-B4 all qtypes/seeds including fitting/cal/test; isolated B3 unavailable')
    write_json(root/'training_cost_summary.json',training)
    for p in [c['main_csv'],c['clr_conformal'],c['b5c_conformal'],c['source_protocol']]: used[p]=sha(p)
    main = pd.read_csv(c['main_csv'])
    rows=[]
    for r in measure['results']:
        a = main[(main.source=='gpt2') & (main.model==r['model']) & (main.qtype==r['qtype']) & (main.split=='test') & (main.metric=='accuracy')]
        if len(a)!=1: raise ValueError('Accuracy cell')
        rows.append({**{k:v for k,v in r.items() if not k.startswith('gpu_') and k!='timed_rounds'},
                     'test_accuracy':float(a.iloc[0]['mean']),'accuracy_lo':float(a.iloc[0].lo),'accuracy_hi':float(a.iloc[0].hi),
                     'source_step_flops':den,'source_step_ratio':r['flops_per_query']/den,
                     'historical_training_gpu_hours_3seeds':training[r['model']]['sum'],
                     'historical_training_gpu_hours_median_seed':training[r['model']]['median']})
    pd.DataFrame(rows).to_csv(dest/'cost_precision.csv',index=False,float_format='%.15g')
    cp, pairs = summarize_conformal(c['clr_conformal'], c['b5c_conformal'])
    cp.to_csv(dest/'conformal_paired.csv',index=False,float_format='%.15g')
    pairs.to_csv(root/'conformal_pairs.csv',index=False,float_format='%.15g')
    write_json(root/'summary_inputs.json',{'sha256':used,'new_calibration':False,'new_test_inference':False,'bootstrap_unit':'problem; preserves all steps and fixed 3 seeds; 10000 draws'})
    write_json(dest/'freeze.json',{'version':'M10c_paper_inputs_v1','sha256':{str(p):sha(p) for p in dest.glob('*.csv')}})
    print(cp.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');main(load_config(p.parse_args().config))
