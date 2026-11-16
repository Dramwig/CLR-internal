"""M7.1-only provenance, budget and persistent status (M7 remains immutable)."""
import fcntl
import json
import time
from datetime import datetime,timezone
from pathlib import Path

from clr.audit.features import sha256
from clr.sources.kv_relay import verify_idea_a
from clr.utils.idea_a import write_json


def live(**updates):
    path=Path('docs/M71_LIVE.json')
    with path.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        old=json.loads(path.read_text()) if path.exists() else {}
        if old.get('test_opened',False):updates['test_opened']=True
        write_json(path,{**old,**updates,'updated_utc':datetime.now(timezone.utc).isoformat()})


def snapshot(run):
    write_json(run.path/'generator_startup_sha256.json',verify_idea_a('/home/me/PaperField/LatDefer'))
    paths=[Path('clr/audit/features.py'),*Path('clr').rglob('*m71*.py'),
           *Path('scripts').glob('m71*.py'),*Path('configs/M71').glob('*.yaml'),
           Path('scripts/m7_env.sh'),Path('clr/data/m7.py'),Path('clr/sources/kv_relay.py'),
           Path('clr/models/clr_v1.py'),Path('clr/models/clr_v0.py'),Path('docs/M7.1_TASK.md'),
           Path('tests/test_m71.py')]
    hashes={};target=run.path/'code';target.mkdir()
    for path in paths:
        if not path.is_file():continue
        dest=target/path;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(path.read_bytes())
        hashes[str(path)]=sha256(path)
    write_json(run.path/'code_sha256.json',hashes)
    write_json(run.path/'upstream_freeze_anchors.json',{
        'data/M7_v1/freeze.json':sha256('data/M7_v1/freeze.json'),
        'data/M6_v1/freeze.json':sha256('data/M6_v1/freeze.json')})


class Cost:
    def __init__(self,run,config):
        self.run=run;self.config=config;self.start=time.monotonic();self.last_scan=-100.;self.other=0.
        self.test_opened=False
    def elapsed(self):return time.monotonic()-self.start
    def update(self,relay=None,**extra):
        self.test_opened|=bool(extra.pop('test_opened',False))
        values={'gpu_hours':self.elapsed()/3600,'test_opened':self.test_opened,
                'cost_basis':'one shared-GPU process residency, including CPU/load/failed work',**extra}
        if relay:
            values.update(source_forward_calls=relay.forward_calls,source_forward_examples=relay.forward_examples,
                          source_prefill_tokens=relay.prefill_tokens)
        self.run.write_metrics(**values)
    def check(self):
        if self.elapsed()>self.config['max_seconds']:raise RuntimeError('M7.1 runtime limit exceeded')
        if self.elapsed()-self.last_scan>60:
            self.other=0.
            for path in Path('runs').glob('*/config.json'):
                if path.parent==self.run.path:continue
                cfg=json.loads(path.read_text())
                if cfg.get('milestone')=='M7.1':
                    self.other+=json.loads((path.parent/'metrics.json').read_text()).get('gpu_hours',0.)
            self.last_scan=self.elapsed()
        if self.other+self.elapsed()/3600>=self.config['budget_gpu_hours']:
            raise RuntimeError('M7.1 cumulative GPU budget exhausted')


def assert_final_lock(config):
    lock=Path(config['output'])/'final_lock.json'
    for path,digest in json.loads(lock.read_text())['sha256'].items():
        if sha256(path)!=digest:raise ValueError('M7.1 final-lock hash changed: '+path)
