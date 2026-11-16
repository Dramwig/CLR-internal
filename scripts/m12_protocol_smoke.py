"""Real-token serializer equality and all calibration branches on val proxies."""
import argparse,json,time,subprocess
from pathlib import Path
import yaml,numpy as np,torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m12_runtime import recorded,Budget
from clr.data.m4.reader import rows_for,PrefixBank
from clr.data.m12 import TensorSerializer
from clr.models.clr_v0 import Serializer
from clr.models.m12 import make_model

def main(c):
    controller_start=time.monotonic();torch.set_num_threads(2)
    training=json.loads(Path(c['output'],'smoke_train.json').read_text());checks=[]
    with recorded({**c,'case':'tensor_equivalence','mode':'smoke'}) as run:
        start=time.monotonic();budget=Budget(run,c,start)
        tokenizer=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True);templates=load_config(c['templates'])
        reference=Serializer(tokenizer,templates);cached=TensorSerializer(tokenizer,templates)
        rows=rows_for(c,'reader_val','validate',ids=set(sorted({r['problem_id'] for r in rows_for(c,'reader_val','validate',qtypes=['Q2'])})[:50]))
        for q in c['qtypes']:
            rs=[r for r in rows if r['qtype']==q][:1024]
            a=reference.batch(rs);b=cached.batch(rs)
            if any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Tensor cache changes serialization')
            timings={}
            for name,s in [('original',reference),('tensor_cache',cached)]:
                t=time.monotonic()
                for _ in range(5):s.batch(rs)
                timings[name]=(time.monotonic()-t)/5
            checks.append({'qtype':q,'n':len(rs),'equal':True,'seconds':timings})
        write_json(run.path/'checks.json',checks);run.write_metrics(status='completed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False)
    for row in training['jobs']:
        cfg={**c,'mode':'smoke','ablation':row['ablation'],'case':'smoke_eval_'+row['ablation'],'checkpoint':str(Path(row['run'])/'best.pt')}
        p=Path('configs/M12')/f"smoke_eval_{row['ablation']}.yaml";p.write_text(yaml.safe_dump(cfg,sort_keys=False))
        with Path(f"reports/M12_evidence/smoke_eval_{row['ablation']}.log").open('w') as log:
            subprocess.run(['.venv/bin/python','-m','scripts.m12_evaluate',str(p)],check=True,stdout=log,stderr=subprocess.STDOUT)
        if time.monotonic()-controller_start>=850:raise TimeoutError('Protocol smoke exceeds gate')
    write_json(Path(c['output'])/'smoke.json',{'passed':True,'training':training,'serializer_checks':checks,'protocol_elapsed_seconds':time.monotonic()-controller_start,'test_opened':False,'train_problems':20,'proxy_reader_val_problems':50,'proxy_cal_A_B_eval':[15,15,20]})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
