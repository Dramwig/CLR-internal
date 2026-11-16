"""M6 startup snapshots and GPU-wall-time budget, including failed runs."""
import hashlib, json, shutil, time, fcntl, os
from pathlib import Path
from clr.utils.idea_a import write_json

ROOT = Path('data/M6_v1')

def hours(exclude=None):
    rows=[]
    for p in (ROOT/'runs').glob('*.json'):
        item=json.loads(p.read_text()); path=Path(item['run'])
        if exclude and path.resolve()==Path(exclude).resolve(): continue
        m=json.loads((path/'metrics.json').read_text()) if (path/'metrics.json').exists() else {}
        b=json.loads((path/'budget.json').read_text()) if (path/'budget.json').exists() else {}
        rows.append({**item,'gpu_hours':max(m.get('gpu_hours',0),b.get('own_gpu_hours',0))})
    return rows

def snapshot(run,c):
    (ROOT/'runs').mkdir(parents=True,exist_ok=True)
    files=set(map(str,Path('clr').rglob('*.py')))|set(map(str,Path('scripts').glob('m6*.py')))|{'docs/M6_TASK.md'}
    for key in ['protocol','templates']:
        if c.get(key): files.add(c[key])
    hashes={}
    for name in sorted(files):
        p=Path(name); dst=run.path/'source_snapshot'/p; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst); hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(run.path/'source_sha256.json',hashes)
    write_json(run.path/'runtime_library_environment.json',{k:os.environ.get(k) for k in ['LD_LIBRARY_PATH','LD_PRELOAD','HF_HUB_OFFLINE','PYTHONPATH']})
    write_json(ROOT/'runs'/f'{run.path.name}.json',{'run':str(run.path),'section':c['section'],'stage':c['stage'],'budget_group':c.get('budget_group')})

class Budget:
    def __init__(self,run,c,start):
        self.run=run; self.c=c; self.start=start; self.active=False
        snapshot(run,c)
    def check(self,active=True):
        self.active |= active
        own=(time.monotonic()-self.start)/3600 if self.active else 0
        ledger=hours(self.run.path); total=sum(x['gpu_hours'] for x in ledger)+own
        section=sum(x['gpu_hours'] for x in ledger if x['section']==self.c['section'])+own
        write_json(self.run.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total,'section_gpu_hours':section})
        cap={'A':3,'B':30,'CD':27}[self.c['section']]
        if total>=60 or section>=cap: raise TimeoutError('M6 aggregate/section GPU-hour cap reached')
        if self.c.get('budget_group'):
            group=sum(x['gpu_hours'] for x in ledger if x.get('budget_group')==self.c['budget_group'])+own
            if group>=self.c['budget_group_hours']-self.c.get('budget_guard_hours',0):raise TimeoutError('M6 budget group cap/guard reached: '+self.c['budget_group'])
        if self.c.get('mode')=='smoke' and own>0.25: raise TimeoutError('M6 smoke >15 minutes')
        return own

def live(**values):
    p=Path('docs/M6_LIVE.json')
    with Path('docs/M6_LIVE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);old=json.loads(p.read_text()) if p.exists() else {}
        old.update(values);old['updated_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();write_json(p,old)
