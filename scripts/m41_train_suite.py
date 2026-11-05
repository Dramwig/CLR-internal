"""Full main comparisons then validation-selected leave-one-type transfer."""
import argparse,json,subprocess,time,hashlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m41_status import update

def main(c):
 root=Path(c['output']);manifest=root/'training_suite.json';results=json.loads(manifest.read_text()) if manifest.exists() else {};mutex=Lock();cfgdir=root/'train_configs';cfgdir.mkdir(exist_ok=True)
 def run(cfg):
    key=f"{cfg['case']}_{cfg['variant']}_{cfg['seed']}"
    if key in results:return results[key]
    receipt=root/f'trained_{key}.json'
    if receipt.exists():directory=json.loads(receipt.read_text())['run']
    else:
        p=cfgdir/f'{key}.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False))
        with Path(f'reports/M41_train_{key}.log').open('w') as log:subprocess.run(['.venv/bin/python','scripts/m41_train.py',str(p)],check=True,stdout=log,stderr=subprocess.STDOUT)
        directory=json.loads(receipt.read_text())['run']
    with mutex:
        results[key]=directory;write_json(manifest,results);update('training',f'C训练完成{len(results)}项，最终test尚未打开',completed_jobs=len(results),last_run=directory,test_opened=False)
    return directory
 def mainworker(seed,device):
    for v in ['v1','lite','b3_multi']:run({**c,'seed':seed,'variant':v,'reader_device':device,'batch_size':256 if v=='v1' else 1024,'eval_batch':256 if v=='v1' else 1024,'head_lr':.001 if v=='b3_multi' else c['head_lr'],**c.get('per_seed_overrides',{}).get(str(seed),{})})
 with ThreadPoolExecutor(max_workers=3) as pool:
    fs=[pool.submit(mainworker,s,d) for s,d in zip(c['seeds'],c['reader_devices'])]
    for f in fs:f.result()
 scores={v:sum(json.loads(Path(results[f'main_{v}_{s}'],'best.json').read_text())['selection_nll'] for s in c['seeds'])/3 for v in ['v1','lite']};chosen=min(scores,key=scores.get)
 write_json(root/'variant_selection.json',{'selected':chosen,'reader_val_macro_nll':scores,'test_opened':False})
 def transferworker(seed,device):
    base={**c,'seed':seed,'variant':chosen,'reader_device':device,'batch_size':256 if chosen=='v1' else 1024,'eval_batch':256 if chosen=='v1' else 1024,**c.get('per_seed_overrides',{}).get(str(seed),{})}
    pre=run({**base,'case':'leave_q4_out','qtypes':['Q1','Q2','Q3'],'epochs':2,'min_epochs':1})
    for n in [100,1000,10000]:run({**base,'case':f'transfer_{n}','qtypes':['Q4'],'epochs':20,'min_epochs':1,'patience':4,'fewshot_n':n,'batch_size':32,'initial_checkpoint':str(Path(pre)/'best.pt')})
 with ThreadPoolExecutor(max_workers=3) as pool:
    fs=[pool.submit(transferworker,s,d) for s,d in zip(c['seeds'],c['reader_devices'])]
    for f in fs:f.result()
 hashes={k:hashlib.sha256(Path(p,'best.pt').read_bytes()).hexdigest() for k,p in results.items()}
 write_json(root/'training_complete.json',{'all_training_locked':True,'runs':results,'checkpoint_sha256':hashes,'variant_selection':chosen,'test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
