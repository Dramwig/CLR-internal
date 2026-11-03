"""Resumable sequential training controller; final-data access is a separate stage."""
import argparse,json,subprocess,time,hashlib,os
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import torch
from clr.data.m4.reader import rows_for
from clr.data.questions.core import seed
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def runs_matching(case,s):
    result=[]
    for p in Path('runs').glob('*/config.json'):
        c=json.loads(p.read_text())
        if c.get('milestone')=='M4' and c.get('stage')=='train' and c.get('case')==case and c.get('seed')==s:result.append(p.parent)
    return sorted(result)
def used_hours():
    total=0.
    for p in Path('runs').glob('*/metrics.json'):
        cpath=p.parent/'config.json'
        if cpath.exists() and json.loads(cpath.read_text()).get('milestone')=='M4':total+=json.loads(p.read_text()).get('gpu_hours',0.)
    return total

def main(base):
    out=Path(base['output']);directory=out/'train_configs';directory.mkdir(exist_ok=True);manifest=out/'training_suite.json';results=json.loads(manifest.read_text()) if manifest.exists() else {};manifest_lock=Lock()
    # Nested, paired Q4 labels: no test/cal accesses and no choice of panels by outcome.
    panel=out/'fewshot_panels.json'
    if not panel.exists():
        rows=rows_for(base,'reader_train','train',qtypes=['Q4']);by={}
        for r in rows:by.setdefault(r['problem_id'],{}).setdefault(r['state_ref']['t'],set()).add(r['label'])
        eligible=sorted([p for p,ts in by.items() if any(ys=={0,1} for ys in ts.values())],key=lambda p:seed(11,p,'M4fewshot'))
        if len(eligible)<5000:raise ValueError('Insufficient independent Q4 problems for N10000 panel')
        pairs=[]
        for pid in eligible[:5000]:
            ts=sorted(t for t,ys in by[pid].items() if ys=={0,1});t=ts[seed(11,pid,'step')%len(ts)];pairs.extend([[pid,t,0],[pid,t,1]])
        write_json(panel,{'seed':11,'panels':{str(n):pairs[:n] for n in [100,1000,10000]},'independent_problems':{'100':50,'1000':500,'10000':5000}})
    def train(case,s,**overrides):
        key=f'{case}_{s}'
        if key in results:return Path(results[key])
        existing=runs_matching(case,s)
        if existing:
            p=existing[-1];m=p/'metrics.json'
            while not m.exists() or json.loads(m.read_text()).get('status')=='running':
                print(f'Waiting for active {p}',flush=True);time.sleep(30)
            data=json.loads(m.read_text())
            if data.get('status')!='completed':raise RuntimeError(f'Prior run failed; explicit repair required: {p}')
        else:
            remaining=base['total_budget_hours']-used_hours()
            if remaining<1:raise RuntimeError('M4 total GPU budget exhausted before next job')
            c={**base,'case':case,'seed':s,**overrides};c['run_budget_hours']=min(c['run_budget_hours'],remaining)
            path=directory/f'{key}.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False))
            with (Path('reports')/f'M4_{key}.log').open('w') as log:
                subprocess.run(['.venv/bin/python','scripts/m4_train.py',str(path)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'.','OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1'})
            p=runs_matching(case,s)[-1]
        with manifest_lock:
            results[key]=str(p);write_json(manifest,results)
        print(key,str(p),flush=True);return p
    pilots={k:train(base.get('pilot_cases',{}).get(str(k),f'pilot_k{k}'),11,trial=True,cross_every=k) for k in base['cross_grid']}
    selected=min(pilots,key=lambda k:json.loads((pilots[k]/'metrics.json').read_text())['best_val_nll'])
    write_json(out/'hyperparameters_lock.json',{'cross_every':selected,'criterion':'reader_val macro-qtype NLL','pilot_runs':{str(k):str(v) for k,v in pilots.items()},'test_opened':False})
    workers=list(zip(base['seeds'],base['reader_devices']))
    if len(workers)!=len(base['seeds']) or len(set(base['reader_devices']))!=len(workers):raise ValueError('One explicit distinct device per seed required')
    def parallel(fn):
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures=[pool.submit(fn,s,device) for s,device in workers]
            for future in futures:future.result()
    parallel(lambda s,device:train('main',s,cross_every=selected,reader_device=device))
    parallel(lambda s,device:train('b3_multi',s,model_kind='b3_multi',micro_batch=32,gradient_accumulation=1,eval_batch=1024,reader_device=device))
    def transfer(s,device):
        pre=train('leave_q4_out',s,cross_every=selected,qtypes=['Q1','Q2','Q3'],reader_device=device)
        for n in [100,1000,10000]:
            settings={'qtypes':['Q4'],'fewshot_n':n,'fewshot_manifest':str(panel),'epochs':20,'validation_epochs':[2,5,10,20],'reader_device':device}
            train(f'transfer_{n}',s,cross_every=selected,initial_checkpoint=str(pre/'best.pt'),**settings)
            train(f'scratch_b3_{n}',s,model_kind='b3',micro_batch=32,gradient_accumulation=1,eval_batch=1024,**settings)
    parallel(transfer)
    if not (out/'qa/unseen_train.parquet').exists():raise RuntimeError('Unseen QA not complete; rerun controller after source/QA completion')
    parallel(lambda s,device:train('augmented',s,cross_every=selected,augmentation=True,reader_device=device))
    hashes={}
    for name,p in results.items():
        if name.startswith('pilot'):continue
        hashes[name]=hashlib.sha256((Path(p)/'best.pt').read_bytes()).hexdigest()
    write_json(out/'training_complete.json',{'runs':results,'checkpoint_sha256':hashes,'gpu_hours_to_date':used_hours(),'all_training_locked':True,'test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
