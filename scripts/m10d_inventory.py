"""Read-only provenance search and selected frozen-input byte verification."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from scripts.m10d import digest


def main(c):
    manifests={};selected={};search={}
    for m in ['M6','M9','M10']:
        p=Path(f'data/{m}_v1/freeze.json');d=json.loads(p.read_text())
        files=d.get('sha256',d.get('files',{}))
        search[m]=dict(checkpoint_paths=[k for k in files if 'B4_' in Path(k).name],
                       trained=list(map(str,Path(f'data/{m}_v1/trained').glob('*[bB]4*'))))
        manifests[str(p)]=digest(p)
        for k,h in files.items():
            path=Path(k).resolve()
            if (str(path).startswith(str(Path(c['root']).resolve())+'/')
                or str(path).startswith(str(Path(c['qa']).resolve())+'/')
                or str(path)==str(Path('data/M10_v1/paper_inputs/main.csv').resolve())):
                selected[str(path)]=h
    # B6 cached derivation includes every prediction/calibration source hash.
    import pickle
    for seed in c['seeds']:
        p=Path(f'data/M10_v1/paper_cache/gpt2_B6_{seed}.pkl')
        selected[str(p.resolve())]=json.loads(Path('data/M10_v1/freeze.json').read_text())['sha256'][str(p)]
        with p.open('rb') as f:cache=pickle.load(f)
        for k,h in cache['sources'].items():selected[str(Path(k).resolve())]=h
    def verify(item):
        p,h=item;actual=digest(p);return dict(path=p,expected=h,actual=actual) if actual!=h else None
    with ThreadPoolExecutor(max_workers=2) as pool:bad=[x for x in pool.map(verify,selected.items()) if x]
    out=dict(manifest_sha256=manifests,b4_search=search,selected_sha256=selected,
             checked=len(selected),mismatches=bad,passed=not bad,
             note='Only file hashes read for test; no test samples or metrics used for selection.')
    write_json(Path(c['output'])/'upstream_inputs.json',out)
    assert not bad,bad
    print('verified',len(selected),'frozen inputs; LLaMA B4 checkpoint search empty',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
