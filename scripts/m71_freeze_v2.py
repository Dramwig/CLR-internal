"""Freeze only closed M7.1 runs; the active freezer run is referenced separately."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from clr.audit.features import sha256
from clr.audit.m71_runtime import snapshot,live
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);completion=json.loads((root/'completion.json').read_text())
        if (root/'freeze.json').exists():raise FileExistsError('M71 already frozen')
        review=Path('reports/M7.1_REVIEW.md')
        if review.stat().st_size>8000:raise ValueError('Review length exceeded')
        files=[p for p in root.rglob('*') if p.is_file() and p.name not in ['freeze.json','verification.json']]
        files += [review,Path('reports/M7.1_details.md'),Path('docs/M7.1_TASK.md'),Path('tests/test_m71.py'),Path('clr/audit/features.py')]
        files += [*Path('scripts').glob('m71*.py'),*Path('clr').rglob('*m71*.py'),*Path('configs/M71').glob('*.yaml'),*Path('reports/M71_evidence').glob('*.log')]
        files += list(Path('data/splits').glob('M71_v1_*.json'))
        costs=[]
        for p in Path('runs').glob('*/config.json'):
            if p.parent.resolve()==run.path.resolve() or json.loads(p.read_text()).get('milestone')!='M7.1':continue
            metrics=json.loads((p.parent/'metrics.json').read_text())
            if metrics['status'] not in ['completed','failed']:raise ValueError('M71 run still open '+str(p.parent))
            costs.append(metrics.get('gpu_hours',0))
            files.extend(x for x in p.parent.rglob('*') if x.is_file())
        if sum(costs)>30:raise ValueError('M71 GPU budget exceeded')
        total_bytes=sum(p.stat().st_size for p in set(files))
        if total_bytes>15*2**30:raise ValueError('M71 total frozen artifacts exceed 15GiB')
        hashes={str(p.resolve()):sha256(p) for p in sorted(set(files))}
        write_json(root/'freeze.json',{'version':'M71_v1','status':completion['status'],'freezer_run':str(run.path),
            'test_opened':completion['test_opened'],'artifact_bytes':total_bytes,'gpu_hours':sum(costs),'sha256':hashes})
        mismatches=[path for path,digest in hashes.items() if sha256(path)!=digest]
        write_json(root/'verification.json',{'run':str(run.path),'files':len(hashes),'mismatches':mismatches,'passed':not mismatches})
        if mismatches:raise ValueError('M71 artifacts changed during freeze')
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=completion['test_opened'],files=len(hashes))
        live(stage='frozen_complete_pending_review',milestone_complete=True,test_opened=completion['test_opened'],
             review=str(review),freeze=str(root/'freeze.json'),gpu_hours=sum(costs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))
