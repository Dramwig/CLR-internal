"""All-training lock precedes M9-A final inference; sequential 10 GPUh ledger."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time,hashlib
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m9_runtime import hours,snapshot,live

def main(c):
 with recorded(c) as run:
    snapshot(run,c);root=Path(c['output']);models={}
    smoke=json.loads(Path(c['smoke_ref']).read_text())
    assert json.loads((Path(smoke['run'])/'metrics.json').read_text())['status']=='completed'
    for cfg in c['training_configs']:
        t=load_config(cfg);log=Path('reports/M9_evidence')/f'train_{t["seed"]}.log'
        with log.open('x') as f:
            subprocess.run([sys.executable,'-u','-m','scripts.m9_train',cfg],stdout=f,stderr=subprocess.STDOUT,check=True)
        ref=json.loads((root/'trained'/f'multi_{t["seed"]}.json').read_text());key=f'multi_lite_{t["seed"]}'
        models[key]={'config':t,'checkpoint':ref['checkpoint'],'sha256':ref['sha256']}
        live(stage='training',trained_seeds=list(models),test_opened=False)
        assert sum(r['gpu_hours'] for r in hours())<c['budget_gpu_hours']-.3,'Inference reserve exhausted'
    lock={'all_training_locked':True,'models':models,'checkpoint_sha256':{k:v['sha256'] for k,v in models.items()},'test_opened':False}
    write_json(root/'training_lock.json',lock);index={}
    for key,m in models.items():
        index[key]={}
        for s in ['gpt2','llama']:
            e={**m['config'],'stage':'evaluate','mode':'full','model_key':key,'source':s,'checkpoint':m['checkpoint'],'eval_lock':str(root/'training_lock.json'),'eval_splits':['reader_val','test']}
            cfg=Path('configs/M9')/f'evaluate_{key}_{s}.yaml';cfg.write_text(yaml.safe_dump(e,sort_keys=False))
            with (Path('reports/M9_evidence')/f'evaluate_{key}_{s}.log').open('x') as f:
                subprocess.run([sys.executable,'-u','-m','scripts.m9_evaluate',str(cfg)],stdout=f,stderr=subprocess.STDOUT,check=True)
            ref=json.loads((root/'evaluated'/f'{key}_{s}.json').read_text());index[key][s]=ref['run'];write_json(root/'evaluation_index.json',index)
    live(stage='A_evaluation_completed',test_opened=True,primary_experiment_complete=True)
    write_json(root/'experiment_complete.json',{'run':str(run.path),'models':list(models),'gpu_hours':sum(r['gpu_hours'] for r in hours())})
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
