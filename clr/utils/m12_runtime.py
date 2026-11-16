"""Recorded M12 budget: sum job wall time, including failed jobs, no source calls."""
import json, time, os, hashlib, tarfile
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from clr.utils.records import recorded as base_recorded
from clr.utils.idea_a import write_json
ROOT=Path('data/M12_ablation')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def cost(exclude=None):
    total=0.
    for p in (ROOT/'jobs').glob('*.json'):
        row=json.loads(p.read_text());path=Path(row['run'])
        if exclude and path.resolve()==Path(exclude).resolve():continue
        values=[0.]
        for name,key in [('metrics.json','gpu_hours'),('budget.json','own_gpu_hours')]:
            f=path/name
            if f.exists():values.append(json.loads(f.read_text()).get(key,0.))
        if not (path/'metrics.json').exists() or json.loads((path/'metrics.json').read_text()).get('status') not in ['completed','failed','abandoned']:
            try:os.kill(row['pid'],0);values.append((time.time()-row['started'])/3600)
            except ProcessLookupError:pass
        total+=max(values)
    return total

@contextmanager
def recorded(c):
    with base_recorded(c) as run:
        (ROOT/'jobs').mkdir(exist_ok=True)
        write_json(ROOT/'jobs'/f'{run.path.name}.json',{'run':str(run.path),'pid':os.getpid(),'started':time.time()})
        paths=sorted(set([*Path('clr').rglob('*.py'),*Path('scripts').glob('m12*.py'),Path('scripts/m41_train.py'),Path('scripts/m4_train.py'),*Path('configs/M12').glob('*.yaml')]))
        write_json(run.path/'code_sha256.json',{str(p):digest(p) for p in paths})
        with tarfile.open(run.path/'startup_source.tar.gz','w:gz') as tar:
            for p in paths:tar.add(p)
        yield run

class Budget:
    def __init__(self,run,c,start):self.path=run.path;self.start=start;self.c=c
    def check(self):
        own=(time.monotonic()-self.start)/3600;total=cost(self.path)+own
        write_json(self.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total,'limit':self.c['total_budget_hours']})
        if total>=self.c['total_budget_hours']-.05:raise TimeoutError('M12 aggregate GPU budget reached')
        if datetime.now().astimezone()>=datetime.fromisoformat(self.c['deadline']):raise TimeoutError('M12 deadline reached')
        if self.c['mode']=='smoke' and own*3600>=900:raise TimeoutError('M12 smoke 900 second limit')
