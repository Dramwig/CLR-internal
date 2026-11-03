"""Resume final evaluation only after mandatory training lock and val smoke."""
import argparse,json,subprocess,os,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    root=Path(c['output']);lock=json.loads(Path(c['eval_lock']).read_text())
    if not lock['all_training_locked']:raise ValueError('Training not locked')
    smoke=json.loads(Path(c['evaluation_smoke_run'],'metrics.json').read_text())
    if smoke.get('status')!='completed' or smoke.get('test_opened'):raise ValueError('Val-only evaluation smoke required')
    path=root/'evaluation_suite.json';results=json.loads(path.read_text()) if path.exists() else {};configs=root/'eval_configs';configs.mkdir(exist_ok=True)
    jobs=[]
    for key,directory in lock['runs'].items():
        tr=json.loads(Path(directory,'config.json').read_text());case=tr['case']
        if case.startswith('pilot') or case=='leave_q4_out':continue
        jobs.append({**c,'case':case,'seed':tr['seed'],'model_kind':tr['model_kind'],'qtypes':tr['qtypes'],'checkpoint':str(Path(directory)/'best.pt'),'cross_every':tr['cross_every'],'eval_batch':16 if tr['model_kind']=='clr' else 1024,'eval_splits':['reader_val','test','unseen_eval'] if case in ['main','augmented','b3_multi'] else ['reader_val','test']})
    for name in ['B0','B3','B4']:
        for s in c['seeds']:jobs.append({**c,'case':name,'seed':s,'model_kind':'reference','reference_name':name,'qtypes':['Q1','Q2'] if name=='B4' else ['Q1','Q2','Q3','Q4'],'eval_batch':1024,'eval_splits':['reader_val','test','unseen_eval'] if name=='B3' else ['reader_val','test']})
    for config in jobs:
        key=f"{config['case']}_{config['seed']}"
        if key in results:continue
        # A failed final run is not retried silently; preserve its predictions for audit.
        existing=[]
        for p in Path('runs').glob('*/config.json'):
            r=json.loads(p.read_text())
            if r.get('milestone')=='M4' and r.get('stage')=='evaluate' and r.get('mode')=='full' and r.get('case')==config['case'] and r.get('seed')==config['seed']:existing.append(p.parent)
        if existing:
            run=sorted(existing)[-1];m=json.loads((run/'metrics.json').read_text())
            if m.get('status')!='completed':raise RuntimeError(f'Incomplete final evaluation requires explicit investigation: {run}')
        else:
            cfg=configs/f'{key}.yaml';cfg.write_text(yaml.safe_dump(config,sort_keys=False))
            with (Path('reports')/f'M4_eval_{key}.log').open('w') as log:
                subprocess.run(['.venv/bin/python','scripts/m4_evaluate.py',str(cfg)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'.','OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1'})
            candidates=[]
            for p in Path('runs').glob('*/config.json'):
                r=json.loads(p.read_text())
                if r.get('stage')=='evaluate' and r.get('case')==config['case'] and r.get('seed')==config['seed'] and r.get('mode')=='full':candidates.append(p.parent)
            run=sorted(candidates)[-1]
        results[key]=str(run);write_json(path,results);print(key,run,flush=True)
    write_json(root/'evaluation_complete.json',{'runs':results,'test_opened':True,'post_test_tuning':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
