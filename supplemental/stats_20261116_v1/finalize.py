"""Verify fixed inputs, exposure, paired estimates and all budget receipts; freeze."""
import sys
sys.dont_write_bytecode=True
import json,re
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from supplemental.stats_20261116_v1.runtime import digest,cost

def main(c):
    root=Path(c['output']);assert not (root/'freeze.json').exists()
    for n in ['reproduction_passed.json','a_validation.json','b_reused_per_type_audit.json']:
        assert json.loads((root/n).read_text())['passed']
    for n in ['a_completed.json','b_completed.json','b_analysis_completed.json']:assert json.loads((root/n).read_text())['status']=='completed'
    expected=json.loads((root/'frozen_scientific_input_sha256.json').read_text())
    for manifest in ['input_sha256_a.json','input_sha256_b.json']:
        for p,h in json.loads((root/manifest).read_text()).items():
            if p in expected:assert expected[p]==h
            expected[p]=h
    # Verify the actual frozen training/evaluation dump and QA inputs as well.
    m6=json.loads(Path('data/M6_v1/freeze.json').read_text());frozen=m6.get('sha256',m6.get('files',{}))
    new_inputs=0
    for p,h in frozen.items():
        try:relative=str(Path(p).absolute().relative_to(Path.cwd()))
        except ValueError:continue
        if relative.startswith(('data/M6_v1/qa/','data/dumps/codi_llama_M6_v1/')) or relative=='data/M6_v1/llama_normalization.npz':
            expected[p]=h;new_inputs+=1
    if new_inputs==0:raise ValueError('No M6 frozen training data hashes found')
    for p,h in expected.items():
        if digest(p)!=h:raise ValueError('Input changed: '+p)
    split_ids={s:set(pq.read_table(f'data/M6_v1/qa/{s}.parquet',columns=['problem_id']).column(0).to_pylist()) for s in ['reader_train','reader_val','cal','test']}
    for a,ai in split_ids.items():
        for b,bi in split_ids.items():
            if a!=b:assert not ai&bi,(a,b,'problem leakage')
    ca=pq.read_table('data/M6_v1/qa/cal.parquet',columns=['problem_id','cal_part']).to_pandas().drop_duplicates()
    cal_ids={part:set(g.problem_id) for part,g in ca.groupby('cal_part')};assert not cal_ids['cal_A']&cal_ids['cal_B'];assert cal_ids['cal_A']|cal_ids['cal_B']==split_ids['cal']
    target=json.loads(Path('data/M10_v1/exposure_targets.json').read_text())['llama'];training=[];q1=[]
    lock=json.loads((root/'b_final_lock.json').read_text());access=json.loads((root/'test_access.json').read_text());assert lock['all_training_locked'] and access['timestamp']>=lock['timestamp']
    for ab in ['A1','A2','A4']:
        for seed in [11,22,33]:
            ref=json.loads((root/f'trained_{ab}_lite_{seed}.json').read_text());run=Path(ref['run']);history=json.loads((run/'history.json').read_text());best=json.loads((run/'best.json').read_text());manifest=json.loads((run/'training_manifest.json').read_text())
            counts=target[str(seed)]['counts'];assert counts==manifest['train_counts'];assert 8<=len(history)<=10
            eligible=[h for h in history if h['epoch']>=8];selected=min(eligible,key=lambda x:x['selection_nll']);assert selected==best
            for i,h in enumerate(history,1):assert h['examples']=={q:i*n for q,n in counts.items()}
            assert digest(run/'best.pt')==lock['checkpoint_sha256'][f'{ab}_{seed}']
            training.append(dict(variant=ab,seed=seed,epochs=len(history),selected_epoch=best['epoch'],selected_exposure=best['examples'],checkpoint_sha256=digest(run/'best.pt')))
    jobs=[]
    for p in (root/'jobs').glob('*.json'):
        r=json.loads(p.read_text());run=Path(r['run']);m=json.loads((run/'metrics.json').read_text());cfg=json.loads((run/'config.json').read_text());assert m['status']=='completed';assert m['source_forward_calls']==0
        jobs.append(dict(run=str(run),stage=cfg['stage'] if 'stage' in cfg else 'train',case=cfg['case'],mode=cfg['mode'],seed=cfg['seed'],gpu_hours=m['gpu_hours']))
    assert len(jobs)==24 and cost()<25
    b=json.loads((root/'b_results.json').read_text());assert len(b)==60
    for r in b:
        for field in ['value','baseline','delta']:
            x=r[field];assert all(np.isfinite(v) for v in x.values());assert x['lo']<=x['hi'] and x['lo90']<=x['hi90'];assert x['lo']<=x['lo90']<=x['hi90']<=x['hi']
    # Full precision new predictions agree independently with saved online step metrics.
    pm=pd.read_parquet(root/'b_problem_metrics.parquet');af=pd.read_parquet(root/'b_auroc_observations.parquet')
    from sklearn.metrics import roc_auc_score
    errors=[]
    for ab in ['A1','A2','A4']:
        for seed in [11,22,33]:
            ref=json.loads((root/f'evaluated_{ab}_{seed}.json').read_text());stats=json.loads(Path(ref['run'],'statistics.json').read_text())
            for row in stats:
                if row['calibration']!='temperature' or row['kind']!='clean' or row['heldout']:continue
                q=row['qtype'];f=pm[(pm.model==ab)&(pm.seed==seed)&(pm.split==row['split'])&(pm.qtype==q)&(pm.t==row['t'])]
                m='nll' if q=='Q1' else 'auroc' if q=='Q2' else 'accuracy'
                if m=='auroc':
                    g=af[(af.model==ab)&(af.seed==seed)&(af.split==row['split'])&(af.t==row['t'])];value=roc_auc_score(g.label,g.score)
                else:value=float((f[m]*f.instances).sum()/f.instances.sum())
                error=abs(value-row[m]);assert error<1e-7,(ab,seed,row,error);errors.append(error)
    text=(root/'REPORT.md').read_text();assert len(re.findall(r'^## [1-6]\. ',text,re.M))==6
    assert '实验运行中' not in text and '尚在执行' not in text
    write_json(root/'final_audit.json',{'passed':True,'timestamp':datetime.now().astimezone().isoformat(),'verified_old_input_files':len(expected),'M6_dump_QA_files':new_inputs,'split_problem_counts':{s:len(ids) for s,ids in split_ids.items()},'cal_problem_counts':{s:len(ids) for s,ids in cal_ids.items()},'problem_split_disjoint':True,'training':training,'jobs':jobs,'gpu_hours':cost(),'source_forward_calls':0,'online_prediction_metric_checks':len(errors),'max_online_replay_error':max(errors),'report_bytes':len(text.encode())})
    live=json.loads((root/'LIVE.json').read_text())
    live.update(status='completed',timestamp=datetime.now().astimezone().isoformat(),gpu_hours=cost(),new_training_completed=9,new_evaluation_completed=9,running=[],remaining=[],report=str((root/'REPORT.md').absolute()),final_audit=str((root/'final_audit.json').absolute()),freeze=str((root/'freeze.json').absolute()))
    write_json(root/'LIVE.json',live)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='freeze.json')
    write_json(root/'freeze.json',{'version':'stats_20261116_v1','created_at':datetime.now().astimezone().isoformat(),'sha256':{str(p):digest(p) for p in files}})
    print('Completed and frozen:',root/'REPORT.md','GPUh',cost(),'files',len(files),flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))
