"""Close validation development and pin models/settings before opening test."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import yaml
from clr.audit.features import sha256
from clr.audit.m8b_runtime import snapshot,Cost,total_cost,live
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from clr.utils.config import load_config

def main(c):
    root=Path(c['output'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False)
        smoke=json.loads((root/c['smoke_marker']).read_text());assert smoke['groups']==20 and not smoke['test_opened'] and smoke['seconds']<=900
        assert json.loads((root/'upstream_preflight.json').read_text())['passed']
        # Budget bound uses measured all-design smoke and a declared safety factor.
        projected=c['projected_gpu_hours']
        if total_cost()+projected>c['budget_gpu_hours']:raise RuntimeError('Prospective matrix exceeds remaining GPU budget')
        paths=[*Path('clr').rglob('*m8b*.py'),*Path('scripts').glob('m8b_evaluate.py'),*Path('scripts').glob('m8b_overhead.py'),*Path('configs/M8b').glob('final_*.yaml'),Path('docs/M8b_TASK.md')]
        paths += [Path(p) for p in ['clr/audit/features.py','clr/audit/m71_reader.py','clr/audit/m71_storage.py','clr/audit/m8a_perturb.py','clr/models/m71.py','clr/models/clr_v1.py','clr/models/clr_v0.py','clr/sources/kv_relay.py','clr/data/m71.py','clr/data/m7.py','scripts/m8a_evaluate.py']]
        for d in ['R1','R2','R3','R4']:
            for s in c['reader_seeds']:
                cal=root/c.get('calibration_folder','calibration')/f'{d}_{s}.json';assert cal.exists();paths.append(cal)
                if d!='R1':
                    ref=root/c.get('training_folder','training')/f'{d}_{s}.json';paths.extend([ref,Path(json.loads(ref.read_text())['checkpoint'])])
                else:
                    ref=Path(f'data/M71_v1/pool/training/CLR_{s}.json');paths.extend([ref,Path(json.loads(ref.read_text())['checkpoint'])])
        paths.extend([root/'variant_selection.json',root/'receiver_projection.pt',Path('data/M71_v1/projection.pt'),Path('data/M8a_v1/test500_ids.json')])
        if (root/'sensitivity_ids.json').exists():paths.append(root/'sensitivity_ids.json')
        paths.append(root/'budget_lock.json')
        lock={'run':str(run.path),'sha256':{str(p):sha256(p) for p in paths},'test_id_source':'data/M8a_v1/test500_ids.json','budget_before_test':total_cost(),'projected_gpu_hours':projected,'configuration':'final configs locked before test; runtime report seeds independently drawn and saved after each run','smoke':smoke}
        if (root/'final_lock.json').exists():raise ValueError('Final lock already exists')
        write_json(root/'final_lock.json',lock);cost.update(status='completed');live(stage='final_locked',test_opened=False,final_lock=str(root/'final_lock.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
