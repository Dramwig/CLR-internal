"""Final M4.1 evaluation after locked training and a val-only evaluator smoke."""
import argparse,json,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 root=Path(c['output']);lock=json.loads(Path(c['eval_lock']).read_text())
 if not lock.get('all_training_locked'):raise ValueError('Training lock absent')
 for v in ['v1','lite']:
    smoke=json.loads(Path(c['evaluation_smokes'][v],'metrics.json').read_text())
    if smoke.get('status')!='completed' or smoke.get('test_opened'):raise ValueError('Val-only evaluation smoke missing')
 path=root/'evaluation_suite.json';results=json.loads(path.read_text()) if path.exists() else {};mutex=Lock();cfgdir=root/'eval_configs';cfgdir.mkdir(exist_ok=True)
 jobs=[]
 for key,run in lock['runs'].items():
    tr=json.loads(Path(run,'config.json').read_text())
    if tr['case']=='leave_q4_out':continue
    case=f"{tr['case']}_{tr['variant']}";jobs.append({**c,'stage':'evaluate','mode':'full','case':case,'variant':tr['variant'],'seed':tr['seed'],'qtypes':tr['qtypes'],'model_kind':'b3_multi' if tr['variant']=='b3_multi' else 'clr','checkpoint':str(Path(run)/'best.pt'),'eval_batch':256 if tr['variant']=='v1' else 1024})
 def worker(seed,device):
    for cfg in jobs:
        if cfg['seed']!=seed:continue
        key=f"{cfg['case']}_{seed}"
        if key in results:continue
        existing=[]
        for p in Path('runs').glob('*/config.json'):
            old=json.loads(p.read_text())
            if old.get('milestone')=='M4.1' and old.get('stage')=='evaluate' and old.get('mode')=='full' and old.get('case')==cfg['case'] and old.get('seed')==seed:existing.append(p.parent)
        if existing:raise RuntimeError('Explicit investigation required before resuming final inference: '+str(existing))
        cfg['reader_device']=device;p=cfgdir/f'{key}.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False))
        with Path(f'reports/M41_eval_{key}.log').open('w') as log:subprocess.run(['.venv/bin/python','scripts/m41_evaluate.py',str(p)],check=True,stdout=log,stderr=subprocess.STDOUT)
        found=[]
        for p in Path('runs').glob('*/config.json'):
            old=json.loads(p.read_text())
            if old.get('milestone')=='M4.1' and old.get('stage')=='evaluate' and old.get('mode')=='full' and old.get('case')==cfg['case'] and old.get('seed')==seed:found.append(p.parent)
        if len(found)!=1:raise ValueError('Ambiguous evaluation run')
        with mutex:
            results[key]=str(found[0]);write_json(path,results);write_json(Path('docs/M41_LIVE.json'),{'stage':'evaluation','completed_jobs':len(results),'planned_jobs':len(jobs),'test_opened':True})
 with ThreadPoolExecutor(max_workers=3) as pool:
    fs=[pool.submit(worker,s,d) for s,d in zip(c['seeds'],c['reader_devices'])]
    for f in fs:f.result()
 # Frozen controls are reused only after final training lock; no repeated tuning.
 old=json.loads(Path('data/M4_v0/evaluation_suite.json').read_text())
 for case in ['B0','B3']+[f'scratch_b3_{n}' for n in [100,1000,10000]]:
    for seed in c['seeds']:results[f'{case}_{seed}']=old[f'{case}_{seed}']
 write_json(root/'evaluation_complete.json',{'runs':results,'test_opened':True,'post_test_tuning':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
