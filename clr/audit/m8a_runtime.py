"""M8a provenance, immutable dependencies and cumulative budget."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from clr.audit.features import sha256
from clr.sources.kv_relay import verify_idea_a
from clr.utils.idea_a import write_json


def live(**fields):
    path=Path('docs/M8a_LIVE.json')
    old=json.loads(path.read_text()) if path.exists() else {}
    write_json(path,{**old,**fields,'updated_utc':datetime.now(timezone.utc).isoformat()})


def snapshot(run,c):
    verify_idea_a(c['idea_a'])
    paths=[*Path('clr').rglob('*m8*.py'),*Path('scripts').glob('m8*.py'),
           *Path('configs/M8a').glob('*.yaml'),Path('docs/M8a_TASK.md'),
           Path('tests/test_m8a.py')]
    paths += [Path(p) for p in ['clr/audit/features.py','clr/sources/kv_relay.py',
        'clr/audit/m71_reader.py','clr/audit/m71_evaluation.py','clr/audit/m7_evaluation.py',
        'clr/models/m71.py','clr/data/m7.py','clr/data/m71.py']]
    hashes={}
    for p in paths:
        if p.is_file():
            dst=run.path/'code'/p;dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(p.read_bytes());hashes[str(p)]=sha256(p)
    write_json(run.path/'code_sha256.json',hashes)
    write_json(run.path/'upstream_anchors.json',{'data/M71_v1/freeze.json':sha256('data/M71_v1/freeze.json')})


def total_cost(exclude=None):
    total=0.
    for p in Path('runs').glob('*/config.json'):
        if exclude and p.parent.resolve()==Path(exclude).resolve():continue
        try:
            c=json.loads(p.read_text())
            if c.get('milestone') not in ['M8','M8a']:continue
            m=json.loads((p.parent/'metrics.json').read_text())
            total+=m.get('gpu_hours',0)
        except (ValueError,FileNotFoundError):continue
    return total


class Cost:
    def __init__(self,run,c,gpu=True):
        self.run,self.c,self.gpu=run,c,gpu;self.start=time.monotonic()
        self.forward_calls=0;self.test_opened=False
    def update(self,**extra):
        self.run.write_metrics(gpu_hours=(time.monotonic()-self.start)/3600 if self.gpu else 0,
            source_forward_calls=self.forward_calls,test_opened=self.test_opened,
            cost_basis='shared GPU residency including load, CPU work, smoke, failures; CPU-only=0',**extra)
    def check(self):
        if time.monotonic()-self.start>self.c.get('max_seconds',900):
            raise RuntimeError('M8a configured wall time exceeded')
        if self.gpu and total_cost(self.run.path)+(time.monotonic()-self.start)/3600>self.c['budget_gpu_hours']:
            raise RuntimeError('M8a cumulative GPU budget exceeded')
