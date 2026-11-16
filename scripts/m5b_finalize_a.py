"""Wait for all six pre-specified A trainings, lock, smoke, then final evaluation."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def main(c):
    root=Path(c['output']);state={'stage':'waiting_training'};write_json(root/'A_pipeline.json',state)
    train={}
    while len(train)<6:
        for case in ['B3-prefix','CLR-lite-single']:
            for seed in c['seeds']:
                key=f'{case}_{seed}';p=root/f'trained_{key}_full.json'
                if p.exists():
                    r=Path(json.loads(p.read_text())['run']);m=json.loads((r/'metrics.json').read_text())
                    if m.get('status')=='completed':train[key]=str(r)
        state.update(completed_training=list(train));write_json(root/'A_pipeline.json',state)
        for name in ['queue_single.json','queue_prefix.json']:
            p=root/name
            if p.exists() and json.loads(p.read_text()).get('stage')=='failed':raise RuntimeError(f'A queue failed: {name}')
        if len(train)<6:time.sleep(20)
    hashes={}
    for r in train.values():
        for p in Path(r).glob('*best.pt'):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    lock={'all_training_locked':True,'training_runs':train,'checkpoint_sha256':hashes,'protocol_hash':hashlib.sha256(Path('docs/M5b_TASK.md').read_bytes()).hexdigest(),'historical_invalid_A_test_exposure':'data/M5b_v1/invalid_development_runs.json','post_test_hyperparameter_selection':False}
    write_json(root/'A_training_lock.json',lock)
    cfgroot=root/'pipeline_configs';cfgroot.mkdir(exist_ok=True);evals={}
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(c['gpu'])}
    for case in ['B3-prefix','CLR-lite-single']:
        base={**c,'case':case,'seed':11,'train_run':train[f'{case}_11'],'stage':'smoke','training_lock':str(root/'A_training_lock.json'),'eval_smoke_run':None}
        path=cfgroot/f'eval_smoke_{case}.yaml';path.write_text(yaml.safe_dump(base,sort_keys=False));state.update(stage='eval_smoke',case=case);write_json(root/'A_pipeline.json',state)
        with Path(f'reports/M5b_eval_smoke_{case}.log').open('w') as log:r=subprocess.run([sys.executable,'scripts/m5b_evaluate_a.py',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'A evaluation smoke failed {case}')
        smoke=json.loads((root/f'evaluated_{case}_11_smoke.json').read_text())['run']
        for seed in c['seeds']:
            full={**base,'stage':'full','seed':seed,'train_run':train[f'{case}_{seed}'],'eval_smoke_run':smoke};path=cfgroot/f'eval_full_{case}_{seed}.yaml';path.write_text(yaml.safe_dump(full,sort_keys=False));state.update(stage='final_evaluation',case=case,seed=seed);write_json(root/'A_pipeline.json',state)
            with Path(f'reports/M5b_eval_{case}_{seed}.log').open('w') as log:r=subprocess.run([sys.executable,'scripts/m5b_evaluate_a.py',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'A final evaluation failed {case}/{seed}')
            evals[f'{case}_{seed}']=json.loads((root/f'evaluated_{case}_{seed}_full.json').read_text())['run'];write_json(root/'A_evaluation_complete.json',{'runs':evals,'test_scope':'pre-specified A only'})
    state.update(stage='completed',evaluations=evals);write_json(root/'A_pipeline.json',state)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
