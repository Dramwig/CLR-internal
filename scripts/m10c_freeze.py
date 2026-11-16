"""Freeze M10c after reports; never modify previous milestone artifacts."""
import argparse,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.paper.m10c_cost import sha


def main(c):
    root=Path(c['output'])
    if (root/'freeze.json').exists():raise ValueError('Already frozen')
    for p in ['paper_completed.json','reproducibility.json']:
        d=json.loads((root/p).read_text())
        if not d.get('passed',d.get('completed')):raise ValueError('Incomplete '+p)
    if not json.loads((root/'upstream_audit.json').read_text())['scientific_inputs_intact']:raise ValueError('Scientific integrity failed')
    if len(Path(c['review']).read_bytes())>8000 or len(Path(c['summary']).read_bytes())>10000:raise ValueError('Review/summary size')
    # No mutable controller stdout is frozen while a writer is still active.
    ledger=json.loads((root/'cost_ledger.json').read_text())
    paths=set(root.rglob('*'))|set(Path('configs/M10c').rglob('*'))|set(Path('scripts').glob('m10c*.py'))|set(Path('scripts/paper').glob('make_t6*.py'))|set(Path('paper').rglob('*'))
    paths|={Path(p) for p in ['clr/paper/m10c_cost.py','clr/paper/render_m10c.py','tests/test_m10c.py','docs/M10c_TASK.md','docs/M10c_LIVE.json','reports/M10c_REVIEW.md','reports/M10c_details.md','reports/FINAL_SUMMARY.md']}
    for r in ledger:paths.update(Path(r['run']).rglob('*'))
    paths={p for p in paths if p.is_file() and p.suffix!='.pyc' and p!=root/'freeze.json' and p.name not in ['finish.log','completion_audit.json']}
    write_json(root/'freeze.json',dict(version='M10c_v1',status='complete_pending_review_read_only',no_training=True,no_recalibration=True,no_new_test_inference=True,
        gpu_hours=sum(r['gpu_hours'] for r in ledger),sha256={str(p):sha(p) for p in sorted(paths)},
        upstream_integrity='Scientific files intact; three disclosed pre-existing administrative/log exceptions, archived paper supersessions'))
    frozen=json.loads((root/'freeze.json').read_text());bad=[p for p,h in frozen['sha256'].items() if sha(p)!=h]
    write_json(root/'completion_audit.json',dict(passed=not bad,errors=bad,frozen_files=len(frozen['sha256']),freeze_sha256=sha(root/'freeze.json')))
    if bad:raise ValueError('Completion byte audit failed')
    print('M10c frozen:',len(frozen['sha256']),'files',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');main(load_config(p.parse_args().config))
