"""M7-only persistence and cost accounting; independent from the M6 controller."""
import hashlib
import json
import os
import fcntl
from pathlib import Path
import time
from datetime import datetime, timezone
import torch
from clr.utils.idea_a import write_json


def live(**values):
    p=Path('docs/M7_LIVE.json')
    with p.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        old=json.loads(p.read_text()) if p.exists() else {}
        write_json(p, {**old, **values, 'updated_utc': datetime.now(timezone.utc).isoformat()})


def snapshot(run):
    from clr.sources.kv_relay import verify_idea_a
    write_json(run.path/'generator_startup_sha256.json',verify_idea_a('/home/me/PaperField/LatDefer'))
    paths = [*Path('clr').rglob('*m7*.py'), Path('clr/sources/kv_relay.py'),
             Path('clr/models/clr_v0.py'),Path('clr/models/clr_v1.py'),Path('scripts/m7_env.sh'),
             *Path('scripts').glob('m7*.py'), *Path('configs/M7').glob('*.yaml')]
    target=run.path/'code'; target.mkdir()
    hashes={}
    for p in paths:
        if p.is_file():
            dest=target/p; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(p.read_bytes())
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(run.path/'code_sha256.json',hashes)


class Cost:
    def __init__(self, run, config):
        self.run=run; self.config=config; self.start=time.monotonic()
        self._scan_time=-1.;self._other_cost=0.
    def elapsed(self):return time.monotonic()-self.start
    def update(self, relay=None, **extra):
        out={'gpu_hours': self.elapsed()/3600,
             'cost_basis':'elapsed CUDA process residency on one shared GPU, includes load/CPU work',
             'test_opened':False, **extra}
        if relay:
            out.update(source_forward_calls=relay.forward_calls,source_forward_examples=relay.forward_examples,
                       source_prefill_tokens=relay.prefill_tokens)
        self.run.write_metrics(**out)
        return out
    def check(self):
        if self.elapsed()>self.config.get('max_seconds',40*3600):raise RuntimeError('M7 configured runtime limit exceeded')
        # Every M7 run, including smoke, failed attempts and this active run.
        if self._scan_time<0 or self.elapsed()-self._scan_time>60:
            self._other_cost=0.
            for p in Path('runs').glob('*/metrics.json'):
                if p.parent==self.run.path:continue
                try:
                    if json.loads((p.parent/'config.json').read_text()).get('milestone')=='M7':
                        self._other_cost+=json.loads(p.read_text()).get('gpu_hours',0)
                except FileNotFoundError:continue
            self._scan_time=self.elapsed()
        total=self.elapsed()/3600+self._other_cost
        if total>=self.config.get('budget_gpu_hours',40):raise RuntimeError('M7 cumulative GPU-hour budget exhausted')
