"""Locked M6 inference and source-wise calibration; smoke uses disjoint val IDs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from scripts.m6_train import build,forward,restore
from clr.data.m6 import Bank,rows
from clr.models.clr_v0 import Serializer,QTYPES
from clr.calib.m4 import fit,apply
from clr.calib.m4_evaluation import target_array,summarize,thresholds
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import Budget

@torch.no_grad()
def predict(model,c,rs,bank,serializer,templates,norm,source,budget):
    result=[];model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['kind']!='b3'):
        for i in range(0,len(rs),c['eval_batch']):
            x=forward(model,c,rs[i:i+c['eval_batch']],bank,serializer,templates,norm,source);result.append(x.cpu().numpy())
            if i%(100*c['eval_batch'])==0:budget.check()
    return np.concatenate(result)

def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);torch.set_num_threads(2);torch.manual_seed(c['seed'])
    try:
        if c['mode']!='smoke':
            lock=json.loads(Path(c['eval_lock']).read_text());key=c['model_key'];h=hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest()
            if not lock.get('all_training_locked') or lock['checkpoint_sha256'].get(key)!=h:raise ValueError('Final inference requires exact all-training lock')
        model=build(c);restore(model,c,torch.load(c['checkpoint'],map_location='cpu',weights_only=True));model.to(c['device']).eval()
        templates=load_config(c['templates']);serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True),templates) if c['kind'] not in ['b3','b5b'] else None
        s=c['source'];spec=c['sources'][s];no=np.load(spec['normalization']);norm={s:[torch.from_numpy(no[k]).to(c['device']) for k in ['mean','scale']]}
        if c['mode']=='smoke':
            proxy=rows(spec['qa'],'val','validate');ids=sorted({r['problem_id'] for r in proxy})[:15];caids=set(ids[:5]);cbids=set(ids[5:10]);evalids=set(ids[10:])
            cr=[r for r in proxy if r['problem_id'] in caids|cbids];cbank=Bank(spec['root'],'val','validate',caids|cbids)
        else:cr=rows(spec['qa'],'cal','calibrate');cbank=Bank(spec['root'],'cal','calibrate')
        calibrations={};threshold_map={};stats=[];cps=[]
        for q in QTYPES:
            ca=[r for r in cr if r['qtype']==q and (r['problem_id'] in caids if c['mode']=='smoke' else r['cal_part']=='cal_A')]
            cb=[r for r in cr if r['qtype']==q and (r['problem_id'] in cbids if c['mode']=='smoke' else r['cal_part']=='cal_B')]
            a=predict(model,c,ca,cbank,serializer,templates,norm,s,budget);b=predict(model,c,cb,cbank,serializer,templates,norm,s,budget)
            calibrations[q]={'temperature':fit(a,target_array(ca,q),'temperature')};threshold_map[q]={'temperature':thresholds(cb,b,calibrations[q]['temperature'],q,c['seed'])}
            # Keep calibration predictions for RQ4; no evaluation labels enter fit.
            allr=ca+cb;allx=np.concatenate([a,b]);pred=[{**r,'logits':x.tolist(),'calibrated_logits':apply(x[None],calibrations[q]['temperature'])[0].tolist()} for r,x in zip(allr,allx)]
            pq.write_table(pa.Table.from_pylist(pred),run.path/f'{s}_{q}_cal.parquet',compression='zstd')
        write_json(run.path/'calibration.json',calibrations);del cbank,cr
        for split in (['val'] if c['mode']=='smoke' else c['eval_splits']):
            phase='validate' if c['mode']=='smoke' else ('train' if split=='unseen_train' else 'final');ids=evalids if c['mode']=='smoke' else None
            location=c.get('extra_eval',{}).get(split,spec)
            bank=Bank(location['root'],split,phase,ids);rs=rows(location['qa'],split,phase,ids)
            for q in QTYPES:
                qr=[r for r in rs if r['qtype']==q];x=predict(model,c,qr,bank,serializer,templates,norm,s,budget)
                pr=[{**r,'logits':v.tolist(),'calibrated_logits':apply(v[None],calibrations[q]['temperature'])[0].tolist()} for r,v in zip(qr,x)]
                pq.write_table(pa.Table.from_pylist(pr),run.path/f'{s}_{q}_{split}.parquet',compression='zstd')
                st,cp=summarize(qr,x,q,calibrations[q],threshold_map[q],c['seed'])
                stats += [dict(source=s,split=split,seed=c['seed'],model_key=c['model_key'],**v) for v in st]
                cps += [dict(source=s,split=split,seed=c['seed'],model_key=c['model_key'],**v) for v in cp]
                write_json(run.path/'statistics.json',stats);write_json(run.path/'conformal.json',cps)
                print(split,q,len(qr),flush=True)
            del bank,rs
        (Path(c['output'])/'evaluated').mkdir(parents=True,exist_ok=True)
        write_json(Path(c['output'])/'evaluated'/f"{c['model_key']}_{s}.json",{'run':str(run.path)})
        run.write_metrics(status='completed',test_opened=c['mode']!='smoke' and 'test' in c['eval_splits'],gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)
    finally:
        run.write_metrics(gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
