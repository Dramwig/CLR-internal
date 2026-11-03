"""Conservative sum of M4 GPU job wall times, including failures and active jobs."""
import json,time
from pathlib import Path
from clr.utils.idea_a import write_json

def hours_used(exclude=None):
    total=0.
    for p in Path('runs').glob('*/config.json'):
        if exclude is not None and p.parent.resolve()==Path(exclude).resolve():continue
        config=json.loads(p.read_text())
        if config.get('milestone')!='M4':continue
        values=[]
        for name in ['metrics.json','progress.json','budget.json']:
            path=p.parent/name
            if path.exists():values.append(json.loads(path.read_text()).get('own_gpu_hours',0.) if name=='budget.json' else json.loads(path.read_text()).get('gpu_hours',0.))
        total+=max(values,default=0.)
    return total

class Budget:
    def __init__(self,run,c,start):
        self.path=run.path;self.limit=c['total_budget_hours'];self.start=start;self.job_limit=c.get('run_budget_hours',80)
    def check(self):
        elapsed=(time.monotonic()-self.start)/3600;used=hours_used(self.path)
        write_json(self.path/'budget.json',{'own_gpu_hours':elapsed,'other_gpu_hours':used,'total_gpu_hours':used+elapsed,'limit':self.limit})
        if used+elapsed>=self.limit or elapsed>=self.job_limit:raise TimeoutError('M4 GPU budget exhausted; no silent experiment reduction')
