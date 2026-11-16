"""Read-only byte verification of relevant frozen scientific inputs."""
import argparse
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from scripts.m10d import digest


def main(c):
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'freeze.json').exists(), 'Verify frozen hashes read-only; do not replace audit files'
    refs=[Path(s[k]).resolve() for s in c['sources'].values() for k in ['root','qa']]
    selected={};manifests={}
    for name in ['M23_v1a','M6_v1','M9_v1','M10_v1','M10b_v1','M10c_v1','M10d_v1']:
        p=Path('data')/name/'freeze.json';manifests[str(p)]=digest(p)
        d=json.loads(p.read_text());files=d.get('sha256',d.get('files',{}))
        for path,h in files.items():
            absolute=Path(path).resolve()
            if any(absolute==base or base in absolute.parents for base in refs):
                if str(absolute) in selected: assert selected[str(absolute)]==h
                selected[str(absolute)]=h
    def verify(item):
        p,h=item;actual=digest(p)
        return dict(path=p,expected=h,actual=actual) if actual!=h else None
    with ThreadPoolExecutor(max_workers=2) as pool:bad=[v for v in pool.map(verify,selected.items()) if v]
    result=dict(checked=len(selected),manifest_sha256=manifests,selected_sha256=selected,mismatches=bad,passed=not bad,
        note='Test bytes hashed for integrity only; no test samples/metrics decoded.')
    previous=root/'upstream_before.json'
    if previous.exists():
        old=json.loads(previous.read_text());assert old['manifest_sha256']==manifests
        assert old['selected_sha256']==selected
        write_json(root/'upstream_after.json',result)
    else:
        write_json(previous,result)
        papers=root/'literature';papers.mkdir(exist_ok=True)
        for name in ['hsrm','clue']:
            for ext in ['html','txt']:shutil.copyfile(f'/tmp/M10e_{name}.{ext}',papers/f'{name}.{ext}')
        write_json(papers/'sources.json',dict(hsrm=dict(url='https://arxiv.org/html/2608.30841',sha256=digest(papers/'hsrm.html'),read='Section 3 and Appendix C'),
            clue=dict(url='https://arxiv.org/html/2510.01591',sha256=digest(papers/'clue.html'),read='Section 3 and Appendix A')))
        snapshot=root/'code_before';snapshot.mkdir(exist_ok=True)
        for p in ['scripts/m10e.py','scripts/m10e_audit.py','clr/baselines/m10e.py','tests/test_m10e.py','configs/M10e/run.yaml']:
            dst=snapshot/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    assert not bad,bad
    print('verified',len(selected),'frozen scientific inputs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
