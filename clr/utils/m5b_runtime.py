"""M5b startup snapshots and conservative aggregate budget accounting."""
import hashlib,json,shutil,time
from pathlib import Path
from clr.utils.idea_a import write_json


def snapshot(run, config, files=()):
    names=set(files)|{'docs/M5b_TASK.md','clr/utils/m5b_runtime.py'}
    hashes={}
    for name in sorted(names):
        p=Path(name)
        target=run.path/'source_snapshot'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(run.path/'source_sha256.json',hashes)


def hours_used(exclude=None):
    total=0.; ledger=[]
    for p in Path('runs').glob('*/config.json'):
        c=json.loads(p.read_text())
        if c.get('milestone')!='M5b' or (exclude and p.parent.resolve()==Path(exclude).resolve()):continue
        values=[]
        for name,key in [('metrics.json','gpu_hours'),('budget.json','own_gpu_hours')]:
            f=p.parent/name
            if f.exists():values.append(json.loads(f.read_text()).get(key,0.))
        h=max(values,default=0.);total+=h;ledger.append({'run':str(p.parent),'gpu_hours':h,'stage':c.get('stage'),'case':c.get('case')})
    return total,ledger


class Budget:
    def __init__(self,run,c,start):
        self.path=run.path;self.start=start;self.limit=c.get('max_gpu_hours',25);self.job_limit=(900/3600 if c['stage']=='smoke' else c.get('run_budget_hours',20))
    def check(self):
        own=(time.monotonic()-self.start)/3600;prior,_=hours_used(self.path);total=prior+own
        write_json(self.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total,'limit':self.limit})
        if total>=self.limit or own>=self.job_limit:raise TimeoutError('M5b explicit aggregate or per-run time budget reached')
