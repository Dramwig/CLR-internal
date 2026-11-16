"""Independent M9 ledger; frozen M6 implementations are imported read-only."""
import json,time,shutil
from pathlib import Path
from clr.utils.idea_a import write_json

def hours(exclude=None):
    result=[]
    for p in Path('runs').glob('*/config.json'):
        c=json.loads(p.read_text())
        if c.get('milestone')!='M9' or (exclude and p.parent.resolve()==Path(exclude).resolve()):continue
        m=json.loads((p.parent/'metrics.json').read_text()) if (p.parent/'metrics.json').exists() else {}
        b=json.loads((p.parent/'budget.json').read_text()) if (p.parent/'budget.json').exists() else {}
        result.append({'run':str(p.parent),'mode':c.get('mode'),'stage':c.get('stage'),'gpu_hours':max(m.get('gpu_hours',0),b.get('own_gpu_hours',0)),'source_forward_calls':m.get('source_forward_calls'),'status':m.get('status')})
    return result

def live(**kw):
    p=Path('docs/M9_LIVE.json');old=json.loads(p.read_text()) if p.exists() else {}
    write_json(p,{**old,**kw,'updated_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})

def snapshot(run,c):
    paths=set(Path('clr').rglob('*.py'))|set(Path('scripts').glob('m9*.py'))|set(Path('scripts').glob('m6*.py'))|set(Path('scripts/paper').glob('*.py'))|set(Path('configs/M9').rglob('*.yaml'))|{Path('docs/M9_TASK.md'),Path('tests/test_m9.py')}
    import hashlib
    hashes={}
    for p in sorted(paths):
        dst=run.path/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(run.path/'source_sha256.json',hashes)

class Budget:
    def __init__(self,run,c,start):
        self.run,self.c,self.start=run,c,start;self.active=False;snapshot(run,c)
    def check(self,active=True):
        self.active|=active;own=(time.monotonic()-self.start)/3600 if self.active else 0
        total=sum(r['gpu_hours'] for r in hours(self.run.path))+own
        write_json(self.run.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total})
        self.run.write_metrics(gpu_hours=own,source_forward_calls=0)
        if total>=self.c.get('budget_gpu_hours',10)-.03:raise TimeoutError('M9 cumulative GPU budget guard reached')
        if self.c.get('mode')=='smoke' and own>.25:raise TimeoutError('M9 smoke >15min')
        return own
