"""M8b-specific provenance and 25 GPU-hour ledger."""
import json,time,fcntl
from pathlib import Path
from datetime import datetime,timezone
from clr.audit.features import sha256
from clr.sources.kv_relay import verify_idea_a
from clr.utils.idea_a import write_json

def live(**kw):
    p=Path('docs/M8b_LIVE.json')
    with Path('docs/M8b_LIVE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=json.loads(p.read_text()) if p.exists() else {}
        write_json(p,{**old,**kw,'updated_utc':datetime.now(timezone.utc).isoformat()})
        fcntl.flock(lock,fcntl.LOCK_UN)

def snapshot(run,c):
    verify_idea_a(c['idea_a'])
    paths=[*Path('clr').rglob('*m8b*.py'),*Path('scripts').glob('m8b*.py'),*Path('configs/M8b').glob('*.yaml'),Path('docs/M8b_TASK.md'),Path('tests/test_m8b.py')]
    hashes={}
    for p in paths:
        if p.is_file():
            d=run.path/'code'/p;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(p.read_bytes());hashes[str(p)]=sha256(p)
    write_json(run.path/'code_sha256.json',hashes)
    write_json(run.path/'anchors.json',{p:sha256(p) for p in ['data/M71_v1/freeze.json','data/M8a_v1/freeze.json']})

def total_cost(exclude=None):
    total=0.
    for p in Path('runs').glob('*/config.json'):
        if exclude and p.parent.resolve()==Path(exclude).resolve():continue
        c=json.loads(p.read_text())
        if c.get('milestone')!='M8b':continue
        q=p.parent/'metrics.json'
        if q.exists():total+=json.loads(q.read_text()).get('gpu_hours',0)
    upper=Path("data/M8b_v1/orphan_cost_upper.json")
    if upper.exists():total+=json.loads(upper.read_text())["gpu_hours_upper_bound"]
    return total

class Cost:
    def __init__(self,run,c,gpu=True):
        self.run,self.c,self.gpu=run,c,gpu;self.start=time.monotonic();self.forward_calls=0;self.test_opened=False
    def update(self,**kw):
        self.run.write_metrics(gpu_hours=(time.monotonic()-self.start)/3600 if self.gpu else 0,source_forward_calls=self.forward_calls,test_opened=self.test_opened,cost_basis='GPU residency including smoke/load/CPU work/failures; CPU-only=0',**kw)
    def check(self):
        elapsed=time.monotonic()-self.start
        if elapsed>self.c.get('max_seconds',86400):raise RuntimeError('M8b wall-time limit')
        if self.gpu and total_cost(self.run.path)+elapsed/3600>self.c['budget_gpu_hours']:raise RuntimeError('M8b total GPU budget exceeded')
