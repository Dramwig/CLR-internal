"""M4.1 standalone aggregate budget; immutable M4 accounting is not modified."""
import json,time
from pathlib import Path
from clr.utils.idea_a import write_json

def hours_used(exclude=None):
    total=0.
    for p in Path('runs').glob('*/config.json'):
        if exclude and p.parent.resolve()==Path(exclude).resolve():continue
        if json.loads(p.read_text()).get('milestone')!='M4.1':continue
        values=[]
        for name,key in [('metrics.json','gpu_hours'),('budget.json','own_gpu_hours')]:
            f=p.parent/name
            if f.exists():values.append(json.loads(f.read_text()).get(key,0.))
        total+=max(values,default=0.)
    return total

class Budget:
    def __init__(self,run,c,start):self.path=run.path;self.start=start;self.limit=c['total_budget_hours'];self.job_limit=c['run_budget_hours']
    def check(self):
        own=(time.monotonic()-self.start)/3600;total=hours_used(self.path)+own
        write_json(self.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total,'limit':self.limit})
        if total>=self.limit or own>=self.job_limit:raise TimeoutError('M4.1 budget reached; no silent reduction')
