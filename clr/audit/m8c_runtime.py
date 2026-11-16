"""M8c-specific provenance and cumulative GPU-hour ledger."""
import json,time,fcntl
from pathlib import Path
from datetime import datetime,timezone
from clr.audit.features import sha256
from clr.sources.kv_relay import verify_idea_a
from clr.utils.idea_a import write_json

def live(**kw):
    p=Path('docs/M8c_LIVE.json')
    with Path('docs/M8c_LIVE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=json.loads(p.read_text()) if p.exists() else {}
        write_json(p,{**old,**kw,'updated_utc':datetime.now(timezone.utc).isoformat()})
        fcntl.flock(lock,fcntl.LOCK_UN)

def snapshot(run,c):
    verify_idea_a(c['idea_a'])
    paths=[*Path('clr').rglob('*m8c*.py'),*Path('scripts').glob('m8c*.py'),*Path('configs/M8c').glob('*.yaml'),Path('docs/M8c_TASK.md'),Path('tests/test_m8c.py')]
    hashes={}
    for p in paths:
        if p.is_file():
            d=run.path/'code'/p;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(p.read_bytes());hashes[str(p)]=sha256(p)
    write_json(run.path/'code_sha256.json',hashes)
    write_json(run.path/'anchors.json',{p:sha256(p) for p in ['data/M71_v1/freeze.json','data/M8a_v1/freeze.json','data/M8b_v1/freeze.json']})

def total_cost(exclude=None):
    total=0.
    for p in Path('runs').glob('*/config.json'):
        if exclude and p.parent.resolve()==Path(exclude).resolve():continue
        c=json.loads(p.read_text())
        if c.get('milestone')!='M8c':continue
        q=p.parent/'metrics.json'
        if q.exists():total+=json.loads(q.read_text()).get('gpu_hours',0)
    extra=Path('data/M8c_v1/cost_upper_additions.json')
    if extra.exists():total+=sum(r['gpu_hours_upper_addition'] for r in json.loads(extra.read_text())['rows'])
    return total

class Cost:
    def __init__(self,run,c,gpu=True):
        self.run,self.c,self.gpu=run,c,gpu;self.start=time.monotonic();self.forward_calls=0;self.test_opened=False;self.last_check=-100.;self.other_cost=0.
    def update(self,**kw):
        self.run.write_metrics(gpu_hours=(time.monotonic()-self.start)/3600 if self.gpu else 0,source_forward_calls=self.forward_calls,test_opened=self.test_opened,cost_basis='GPU residency including smoke/load/CPU work/failures; CPU-only=0',**kw)
    def check(self):
        elapsed=time.monotonic()-self.start
        if elapsed>self.c.get('max_seconds',86400):raise RuntimeError('M8c wall-time limit')
        if elapsed-self.last_check>=10:
            self.other_cost=total_cost(self.run.path);self.last_check=elapsed
            self.update(status='running')
        if self.gpu and self.other_cost+elapsed/3600>self.c['budget_gpu_hours']-self.c.get('shutdown_reserve_gpu_hours',.02):raise RuntimeError('M8c GPU budget shutdown reserve reached')
