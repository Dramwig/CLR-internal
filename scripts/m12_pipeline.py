"""Priority-controlled M12; lock every retained training before any test read."""
import argparse,json,time,subprocess,os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m12_runtime import cost,digest

ROOT=Path('data/M12_ablation')
def state(stage,**kwargs):
    old=json.loads(Path('docs/M12_LIVE.json').read_text()) if Path('docs/M12_LIVE.json').exists() else {}
    write_json('docs/M12_LIVE.json',{**old,'stage':stage,'updated':datetime.now().astimezone().isoformat(),'gpu_hours':cost(),**kwargs})

def launch(module,c,label):
    p=Path('configs/M12')/(label+'.yaml');p.write_text(yaml.safe_dump(c,sort_keys=False))
    with Path('reports/M12_evidence',label+'.log').open('w') as f:
        subprocess.run(['.venv/bin/python','-m',module,str(p)],stdout=f,stderr=subprocess.STDOUT,check=True)

def main(c):
    for p in [ROOT/'training_complete.json',ROOT/'test_access.json']:
        if p.exists():raise ValueError('Existing lock/test receipt; resume explicitly, never retrain')
    if not json.loads((ROOT/'smoke.json').read_text())['passed']:raise ValueError('Smoke gate')
    results={};retained=[];dropped=[];decisions=[]
    seeds=[11,22,33];expected_per_run=2.16
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    resource=[tuple(map(int,line.split(','))) for line in snapshot.strip().splitlines()]
    candidates=sorted(resource,key=lambda x:(x[1],-x[2]),reverse=True)
    if candidates[2][1]<4096:raise RuntimeError('Three GPUs with 4GiB free required; no batch/model fallback')
    devices=c.get('m12_seed_devices',[f'cuda:{r[0]}' for r in candidates[:3]])
    if len(devices)!=3 or len(set(devices))!=3:raise ValueError('Three distinct seed devices required')
    free_by_device={f'cuda:{r[0]}':r[1] for r in resource}
    if any(free_by_device.get(d,0)<4096 for d in devices):raise RuntimeError('Selected GPUs require 4GiB free; no batch/model fallback')
    write_json(ROOT/'resource_assignment.json',{'snapshot':snapshot,'seed_devices':dict(zip(seeds,devices)),'timestamp':datetime.now().astimezone().isoformat()})
    for a in ['A1','A2','A3','A4','A5']:
        hours=cost();need=3*expected_per_run*1.2;seconds=(datetime.fromisoformat(c['deadline'])-datetime.now().astimezone()).total_seconds()
        decision={'ablation':a,'used_gpu_hours':hours,'training_estimate_gpu_hours':need,'final_reserved_gpu_hours':2.,'seconds_to_deadline':seconds}
        already_complete=all((ROOT/f'trained_{a}_lite_{s}.json').exists() for s in seeds)
        decision['reused_completed_training']=already_complete
        if not already_complete and (hours+need+2>30 or seconds<expected_per_run*1.2*3600+3600):
            dropped.extend(x for x in ['A1','A2','A3','A4','A5'] if x not in retained)
            decision['admitted']=False;decisions.append(decision);break
        decision['admitted']=True;decisions.append(decision);write_json(ROOT/'admission.json',decisions)
        state('training',active_ablation=a,retained=retained,dropped=dropped,test_opened=False)
        def worker(pair):
            seed,device=pair;cfg={**c,'case':a,'ablation':a,'seed':seed,'reader_device':device}
            receipt=ROOT/f'trained_{a}_lite_{seed}.json'
            if not receipt.exists():launch('scripts.m12_train',cfg,f'train_{a}_{seed}')
            r=json.loads(receipt.read_text())['run'];metrics=json.loads(Path(r,'metrics.json').read_text())
            if metrics['status']!='completed':raise ValueError('Non-completed training')
            return f'{a}_{seed}',r
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                for key,r in pool.map(worker,zip(seeds,devices)):
                    results[key]=r;write_json(ROOT/'training_runs.json',results)
        except Exception as e:
            write_json(ROOT/'training_exception.json',{'ablation':a,'error':str(e),'results':results,'retained':retained})
            # A recorded failure never silently shrinks data/settings. Stop queue;
            # retain only whole completed three-seed variants for final delivery.
            dropped.extend(x for x in ['A1','A2','A3','A4','A5'] if x not in retained)
            state('training_failed_attention',error=str(e),retained=retained,dropped=dropped);raise
        retained.append(a)
        values=[json.loads(Path(results[f'{a}_{s}'],'metrics.json').read_text())['gpu_hours'] for s in seeds]
        expected_per_run=max(values)
        write_json(ROOT/'training_status.json',{'retained':retained,'dropped':dropped,'runs':results})
        state('training_variant_completed',retained=retained,dropped=dropped,test_opened=False)
    write_json(ROOT/'admission.json',decisions)
    codepaths=['clr/models/m12.py','clr/data/m12.py','scripts/m12_train.py','scripts/m12_evaluate.py','scripts/m12_permutation.py','clr/models/clr_v1.py','clr/models/clr_v0.py','clr/data/m4/reader.py','clr/calib/m4.py','clr/calib/m4_evaluation.py']
    lock={'all_training_locked':True,'created':datetime.now().astimezone().isoformat(),'retained':retained,'dropped':dropped,'runs':results,'checkpoint_sha256':{k:digest(Path(v)/'best.pt') for k,v in results.items()},'code_sha256':{p:digest(p) for p in codepaths},'train_config_sha256':{k:digest(Path(v)/'config.json') for k,v in results.items()},'test_opened':False,'priorities':'A1>A2>A3>A4>A5','protocol':'M41 complete exposure and earliest8/latest10/patience1'}
    write_json(ROOT/'training_complete.json',lock);state('locked_before_test',retained=retained,dropped=dropped,test_opened=False)
    evaluations={}
    def evaluate_worker(pair):
        seed,device=pair
        for a in retained:
            cfg={**c,'mode':'full','case':a,'ablation':a,'seed':seed,'reader_device':device,'checkpoint':str(Path(results[f'{a}_{seed}'])/'best.pt'),'eval_lock':str(ROOT/'training_complete.json')}
            receipt=ROOT/f'evaluated_{a}_{seed}.json'
            if not receipt.exists():launch('scripts.m12_evaluate',cfg,f'eval_{a}_{seed}')
            evaluations[f'{a}_{seed}']=json.loads(receipt.read_text())['run'];write_json(ROOT/f'evaluation_seed_{seed}.json',{k:v for k,v in evaluations.items() if k.endswith('_'+str(seed))})
    state('final_evaluation',test_opened=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(evaluate_worker,zip(seeds,devices)))
    write_json(ROOT/'evaluation_complete.json',evaluations);state('evaluations_completed',test_opened=True,retained=retained,dropped=dropped)
    launch('scripts.m12_permutation',{**c,'mode':'full','reader_device':devices[0]},'permutation')
    state('statistics',test_opened=True)
    launch('scripts.m12_summarize',c,'summarize')
    launch('scripts.m12_finish',c,'finish')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
