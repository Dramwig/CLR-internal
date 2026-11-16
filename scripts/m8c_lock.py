"""Freeze protocol/weights/IDs before the first final test evaluation."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from clr.audit.features import sha256
from clr.audit.m8c_runtime import snapshot,Cost,total_cost,live
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        assert not (root/'final_lock.json').exists()
        # The 2026-11-09 user revision forbids falling below 200 test groups.
        audit_path=root/c.get('feasibility_name','budget_feasibility.json')
        audit=json.loads(audit_path.read_text())
        if not audit['passed']:
            raise ValueError('M8c test remains closed: budget feasibility gate did not reach 200 groups')
        n=int(c['test_groups'])
        if n!=200 or n>int(audit['maximum_feasible_groups']):
            raise ValueError('M8c requires exactly 200 groups within the passing budget audit')
        assert json.loads((root/'upstream_prelock.json').read_text())['passed']
        assert json.loads((root/'gate_short_val200_summary.json').read_text())['passed']
        for s in c['reader_seeds']:
            ref=json.loads((root/c['training_folder']/f'R4_{s}.json').read_text())
            assert ref['body_frozen'] and ref['selected_exposure']>=4224000 and ref['train_groups']==4000
        # Frozen IDs can be reused without opening test examples or predictions.
        gids=json.loads(Path('data/M8a_v1/test500_ids.json').read_text())['group_ids']
        if not 1<=n<=len(gids):raise ValueError('test_groups must be within frozen M8a test500 IDs')
        write_json(root/'test_ids.json',{'group_ids':gids[:n], 'source':'data/M8a_v1/test500_ids.json',
                                        'selection':'fixed hash-ranked prefix; no outcome selection'})
        secondary_ids=gids[:100]
        write_json(root/'secondary_ids.json',{'group_ids':secondary_ids,'selection':'first 100 locked main test IDs; no outcome selection'})
        write_json(root/'minimum_ids.json',{'group_ids':secondary_ids,'selection':'first 100 locked main test IDs; no outcome selection'})
        paths=set(Path('clr').rglob('*m8c*.py'))|set(Path('scripts').glob('m8c*.py'))|set(Path('configs/M8c').glob('*.yaml'))
        paths|={Path('docs/M8c_TASK.md'),Path('tests/test_m8c.py'),root/'minimum_ids.json',root/'secondary_ids.json',root/'test_ids.json',audit_path,Path('data/M8a_v1/test500_ids.json'),Path('data/M71_v1/projection.pt'),Path('data/M8b_v1/receiver_projection.pt')}
        model=Path(c['model_path'])
        paths.update(p for p in model.glob('*') if p.is_file())
        paths.update(p for p in Path(c['tokenizer_path']).glob('*') if p.is_file())
        idea=Path(c['idea_a'])
        paths.update(idea/p for p in ['latdef/data/synthetic_v2.py','latdef/data/synthetic.py','latdef/data/schema.py','docs/v2-design-freeze.json','configs/synthetic/v2.yaml'])
        for split in ['reader_train','reader_val','cal_A','cal_B']:
            folder=root/'source'/split;manifest=folder/'manifest.jsonl'
            paths.update([manifest,folder/'completed.json'])
            paths.update(Path(entry['file']) for entry in map(json.loads,manifest.read_text().splitlines()))
        for d in ['D0','R4']:
            for s in c['reader_seeds']:
                paths.add(root/c['calibration_folder']/f'{d}_{s}.json')
                paths.add(root/c['calibration_folder']/f'{d}_{s}_valQA.json')
                p=root/c['training_folder']/f'R4_{s}.json' if d=='R4' else Path(f'data/M71_v1/pool/training/CLR_{s}.json')
                paths.add(p);paths.add(Path(json.loads(p.read_text())['checkpoint']))
        spent=total_cost();budget=float(c['budget_gpu_hours']);forecast=float(c['forecast_total_gpu_hours'])
        if spent>forecast or forecast>budget:raise ValueError(f'Budget forecast invalid: spent={spent:.3f}, forecast={forecast:.3f}, cap={budget:.3f}')
        write_json(root/'budget_lock.json',{'run':str(run.path),'spent_gpu_hours':spent,'estimated_total_gpu_hours':forecast,'budget':budget,'test_groups':n,
                    'test_pool_groups':len(gids),'plan':c['budget_plan'],
                    'selection_rule':'target hit, then minimum attack-side selection NLL, then maximum target logp; report panel excluded','test_opened':False})
        paths.add(root/'budget_lock.json')
        write_json(root/'final_lock.json',{'run':str(run.path),'test_opened':False,'n':n,
                    'main':{'kind':'E3','lambdas':[1.0],'epsilons':[.01,.1,1.0],'steps':200,'designs':['D0','R4'],'reader_seeds':c['reader_seeds']},
                    'budget_gpu_hours':budget,'spent_at_lock_gpu_hours':spent,
                    'secondary_group_ids':secondary_ids,'secondary_execution_order':['minimum_search','E1_E2','lambda_sensitivity'],
                    'secondary':{'lambdas':[.1,10.0],'epsilon':.1,'binary_steps_max':6,'kinds':['E1','E2']},
                    'removed_settings':{'epsilons':[.03,.3],'reason':'user revision: fixed primary200 within 21 GPUh'},
                    'sha256':{str(p):sha256(p) for p in sorted(paths)}})
        cost.update(status='completed');live(stage='final_protocol_locked',test_opened=False,final_lock=str(root/'final_lock.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
