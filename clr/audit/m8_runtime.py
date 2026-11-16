"""M8 provenance and cumulative GPU residency accounting; upstream is immutable."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from clr.audit.features import sha256
from clr.utils.idea_a import write_json


def live(**updates):
    path = Path('docs/M8_LIVE.json')
    old = json.loads(path.read_text()) if path.exists() else {}
    write_json(path, {**old, **updates, 'updated_utc': datetime.now(timezone.utc).isoformat()})


def snapshot(run):
    paths = [*Path('clr').rglob('*m8*.py'), *Path('scripts').glob('m8*.py'),
             *Path('configs/M8').glob('*.yaml'), Path('tests/test_m8.py'), Path('docs/M8_TASK.md')]
    # Snapshot imported implementations as well as new M8 code.
    paths += [Path(p) for p in ['clr/audit/m71_reader.py', 'clr/audit/m71_evaluation.py',
        'clr/audit/m7_evaluation.py', 'clr/audit/features.py', 'clr/models/m71.py',
        'clr/models/clr_v1.py', 'clr/data/m7.py', 'clr/data/m71.py', 'clr/sources/kv_relay.py']]
    hashes = {}
    for path in paths:
        if not path.is_file(): continue
        target = run.path/'code'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[str(path)] = sha256(path)
    write_json(run.path/'code_sha256.json', hashes)
    write_json(run.path/'upstream_anchors.json', {'data/M71_v1/freeze.json': sha256('data/M71_v1/freeze.json')})


class Cost:
    def __init__(self, run, config):
        self.run, self.config = run, config
        self.start = time.monotonic()
        self.test_opened = False

    def update(self, **extra):
        self.test_opened |= extra.pop('test_opened', False)
        self.run.write_metrics(gpu_hours=(time.monotonic()-self.start)/3600,
            cost_basis='shared GPU process residency including load, CPU work, failures',
            source_forward_calls=0, test_opened=self.test_opened, **extra)

    def check(self):
        if time.monotonic()-self.start > self.config['max_seconds']:
            raise RuntimeError('M8 configured runtime limit exceeded')
