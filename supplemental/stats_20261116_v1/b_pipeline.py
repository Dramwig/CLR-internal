"""Smoke -> three-seed complete training -> common lock -> final eval."""
import sys
sys.dont_write_bytecode=True
import json,subprocess,time,concurrent.futures
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from supplemental.stats_20261116_v1.runtime import cost,digest

def main(c):
    root=Path(c['output']);pkg='supplemental.stats_20261116_v1';start=time.time()
    def execute(module,config,name):
        log=root/(name+'.log')
        if log.exists():raise FileExistsError(log)
        with log.open('w') as stream:subprocess.run([sys.executable,'-m',pkg+'.'+module,str(config)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    execute('protocol_smoke',root/'config.yaml','protocol_smoke')
    assert json.loads((root/'reproduction_passed.json').read_text())['passed']
    assert json.loads((root/'b_budget_preestimate.json').read_text())['estimated_gpu_hours']<=25
    for ab in ['A1','A2','A4']:
        cfg=root/f'b_smoke_{ab}.yaml';execute('b_train',cfg,'b_smoke_train_'+ab)
        r=json.loads((root/f'trained_smoke_{ab}_lite_11.json').read_text());d=load_config(cfg);d['checkpoint']=str(Path(r['run'])/'best.pt')
        ep=root/f'b_smoke_evaluate_{ab}.yaml';ep.write_text(yaml.safe_dump(d,sort_keys=False));execute('b_evaluate',ep,'b_smoke_evaluate_'+ab)
    write_json(root/'b_smoke_passed.json',{'passed':True,'test_opened':False,'gpu_hours':cost(),'elapsed_seconds':time.time()-start})
    def train_seed(seed):
        for ab in ['A1','A2','A4']:
            execute('b_train',root/f'b_{ab}_{seed}.yaml',f'b_train_{ab}_{seed}')
            write_json(root/f'b_training_progress_{seed}.json',{'last_completed':ab,'seed':seed,'total_gpu_hours':cost()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(train_seed,[11,22,33]))
    checkpoints={};eval_configs={}
    for seed in [11,22,33]:
        for ab in ['A1','A2','A4']:
            r=json.loads((root/f'trained_{ab}_lite_{seed}.json').read_text());path=Path(r['run']);cfg=load_config(root/f'b_{ab}_{seed}.yaml')
            manifest=json.loads((path/'training_manifest.json').read_text());history=json.loads((path/'history.json').read_text());best=json.loads((path/'best.json').read_text())
            assert 8<=best['epoch']<=10 and 8<=len(history)<=10
            counts=manifest['train_counts'];target=json.loads(Path('data/M10_v1/exposure_targets.json').read_text())['llama'][str(seed)]['counts'];assert counts==target
            assert all(history[-1]['examples'][q]==counts[q]*len(history) for q in counts)
            cfg['checkpoint']=str(path/'best.pt');ep=root/f'b_eval_{ab}_{seed}.yaml';ep.write_text(yaml.safe_dump(cfg,sort_keys=False));eval_configs[ab,seed]=ep
            checkpoints[f'{ab}_{seed}']=digest(cfg['checkpoint'])
    files=[*[root/p for p in ['b_train.py','b_evaluate.py','model_data.py','runtime.py','b_pipeline.py']],*root.glob('b_*.yaml'),*Path('clr').rglob('*.py'),Path('scripts/m4_train.py'),Path('scripts/m41_evaluate.py')]
    write_json(root/'b_final_lock.json',{'all_training_locked':True,'checkpoint_sha256':checkpoints,'code_sha256':{str(p):digest(p) for p in files},'timestamp':time.time(),'test_opened':False})
    def eval_seed(seed):
        for ab in ['A1','A2','A4']:execute('b_evaluate',eval_configs[ab,seed],f'b_eval_{ab}_{seed}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(eval_seed,[11,22,33]))
    write_json(root/'b_completed.json',{'status':'completed','gpu_hours':cost(),'elapsed_seconds':time.time()-start})

if __name__=='__main__':main(load_config(sys.argv[1]))
