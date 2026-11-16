"""Freeze the validation-selected evaluation protocol and outcome-free test IDs."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.audit.features import sha256
from clr.audit.m8a_runtime import snapshot,Cost,live,total_cost
from clr.data.m71 import fixed_hash_subset
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        assert json.loads((root/'P0.json').read_text())['passed']
        for name in c['required_smokes']:
            s=json.loads((root/name).read_text());assert not s['test_opened'] and s['groups']==20 and s['seconds']<=900
            assert json.loads((Path(s['run'])/'metrics.json').read_text())['status']=='completed'
        verification={}
        for path in c['freezes']:
            f=json.loads(Path(path).read_text());mapping=f.get('sha256',f.get('files'))
            def check(item):
                p,h=item;return p if sha256(p)!=h else None
            with ThreadPoolExecutor(max_workers=2) as executor:bad=[x for x in executor.map(check,mapping.items()) if x]
            verification[path]={'files':len(mapping),'mismatches':bad,'freeze_sha256':sha256(path)}
            assert not bad,bad
        rows=[json.loads(x) for x in Path('data/M71_v1/pool/source/test/manifest.jsonl').read_text().splitlines()]
        ids=fixed_hash_subset([r['group_id'] for r in rows],500)
        write_json(root/'test500_ids.json',{'group_ids':ids,'selection':'SHA256(group_id) ascending, first500 of original test2000; no outcome selection','run':str(run.path)})
        paths=[*Path('clr').rglob('*m8*.py'),*Path('scripts').glob('m8a*.py'),*Path('configs/M8a').glob('*.yaml'),
            Path('tests/test_m8a.py'),Path('docs/M8a_TASK.md'),root/'test500_ids.json',root/'P0.json',root/'diffmean.pt',root/'diffmean.json']
        for seed in [11,22,33]:
            for kind in ['CLR']:
                t=json.loads(Path(f'data/M71_v1/pool/training/{kind}_{seed}.json').read_text())
                paths.extend([Path(t['checkpoint']),Path(f'data/M71_v1/pool/calibration/{kind}_{seed}.json')])
        paths.extend(Path(p) for p in ['clr/audit/features.py','clr/sources/kv_relay.py','clr/audit/m71_reader.py','clr/audit/m71_evaluation.py','clr/audit/m7_evaluation.py','clr/models/m71.py','clr/data/m7.py','clr/data/m71.py'])
        lock={str(p.resolve()):sha256(p) for p in paths if p.is_file()}
        assert not (root/'final_lock.json').exists()
        write_json(root/'final_lock.json',{'run':str(run.path),'sha256':lock,'test_opened':False,'verification':verification,
            'cost_before_test_GPUh':total_cost(),'budget':c['budget_gpu_hours']})
        cost.update(status='completed');live(stage='final_protocol_locked',final_lock=str(root/'final_lock.json'),test_groups=500)
        print('locked',len(lock),'files; upstream verified',sum(x['files'] for x in verification.values()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
