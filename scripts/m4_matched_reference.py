"""Pre-final extra control: independent B3 probes matched to CLR best-epoch exposure."""
import argparse,json,hashlib,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m4_status import update

def main(c):
    root=Path(c['output']);original=json.loads((root/'training_complete.json').read_text());new=root/'training_complete_with_matched.json'
    if new.exists():return
    def worker(s,device):
        mainrun=Path(original['runs'][f'main_{s}']);epochs=json.loads((mainrun/'best.json').read_text())['epoch'];case='B3_matched';existing=[]
        for p in Path('runs').glob('*/config.json'):
            cfg=json.loads(p.read_text())
            if cfg.get('milestone')=='M4' and cfg.get('stage')=='train' and cfg.get('case')==case and cfg.get('seed')==s:existing.append(p.parent)
        if existing:
            run=sorted(existing)[-1]
            if json.loads((run/'metrics.json').read_text()).get('status')!='completed':raise RuntimeError(f'Matched reference failed or active: {run}')
            return s,run
        cfg={**c,'seed':s,'case':case,'model_kind':'b3_matched','reader_device':device,'epochs':epochs,'validation_epochs':[epochs],'patience':999,'micro_batch':8,'gradient_accumulation':4,'eval_batch':1024}
        path=root/'train_configs'/f'B3_matched_{s}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        with Path(f'reports/M4_B3_matched_{s}.log').open('w') as log:subprocess.run(['.venv/bin/python','scripts/m4_train.py',str(path)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'.','OMP_NUM_THREADS':'2'})
        for p in sorted(Path('runs').glob('*/config.json'),reverse=True):
            cfg=json.loads(p.read_text())
            if cfg.get('milestone')=='M4' and cfg.get('stage')=='train' and cfg.get('case')==case and cfg.get('seed')==s:return s,p.parent
        raise RuntimeError('Matched reference run missing')
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,s,device) for s,device in zip(c['seeds'],c['reader_devices'])]
        for i,future in enumerate(futures,1):
            s,run=future.result();key=f'B3_matched_{s}';original['runs'][key]=str(run);original['checkpoint_sha256'][key]=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest();update('matched_reference',i,3,run)
    original['matched_reference']='Four separate B3 architectures; same seed/main best-epoch exposure; final epoch checkpoint; no test access';write_json(new,original)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
