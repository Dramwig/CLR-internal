"""Final locked-model inference/calibration; smoke evaluates reader_val proxies only."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.m4_budget import Budget
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.models.clr_v0 import CLR,Serializer
from clr.baselines.b3_multi_v0 import B3Multi,B3PerType
from clr.baselines.probes import Probe
from clr.data.m4.reader import PrefixBank,rows_for,probe_inputs
from clr.calib.m4 import fit
from clr.calib.m4_evaluation import target_array,thresholds,summarize

@torch.no_grad()
def predict(model,kind,rows,bank,serializer,templates,norm,device,batch,budget=None):
    result=[];model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=kind=='clr'):
        for i in range(0,len(rows),batch):
            if budget is not None and (i//batch)%100==0:budget.check()
            rs=rows[i:i+batch];prefix,mask=bank.batch(rs,device);q=rs[0]['qtype']
            if kind=='clr':logits=model(serializer.batch(rs,device),prefix,mask,q)
            else:
                inputs=probe_inputs(rs,prefix,mask,templates,*norm)
                logits=model(*inputs,q) if kind in ['b3_multi','b3_matched'] else model(*inputs)
            result.append(logits.float().cpu().numpy())
    return np.concatenate(result)

def main(c):
 with recorded(c) as run:
    start=time.monotonic();status='failed';torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device'];kind=c['model_kind'];smoke=c['mode']=='smoke'
    budget=Budget(run,c,start)
    try:
        budget.check()
        if not smoke:
            lock=json.loads(Path(c['eval_lock']).read_text())
            if not lock.get('all_training_locked'):raise ValueError('All mandatory training must be locked before final evaluation')
            if c['checkpoint']:
                key=f"{c['case']}_{c['seed']}";digest=hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest()
                if lock['checkpoint_sha256'].get(key)!=digest:raise ValueError('Final checkpoint differs from training lock')
        templates=load_config(c['templates']);normfile=np.load(Path(c['m3_baseline_run'])/'normalization.npz');norm=[torch.from_numpy(normfile[k]).to(device) for k in ['mean','scale']]
        serializer=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates,c['max_length']) if kind=='clr' else None
        model=None
        if kind=='clr':
            model=CLR.pretrained(c['model_path'],k=c['cross_every'],gradient_checkpointing=False).to(device)
            model.restore_trainable(torch.load(c['checkpoint'],map_location='cpu',weights_only=True))
        elif kind in ['b3_multi','b3_matched']:
            model=(B3Multi() if kind=='b3_multi' else B3PerType()).to(device);model.load_state_dict(torch.load(c['checkpoint'],map_location=device,weights_only=True))
        elif kind=='b3':
            model=Probe('B3','Q4').to(device);model.load_state_dict(torch.load(c['checkpoint'],map_location=device,weights_only=True))
        # Smoke uses disjoint reader_val IDs for proxy cal_A/cal_B/evaluation, never cal/test.
        proxyids=None
        if smoke:
            allval=rows_for(c,'reader_val','validate',qtypes=c['qtypes']);pids=sorted({r['problem_id'] for r in allval})[:50];proxyids=set(pids)
            caids=set(pids[:15]);cbids=set(pids[15:30]);evalids=set(pids[30:]);calrows=[r for r in allval if r['problem_id'] in caids|cbids];del allval
            calbank=PrefixBank(c,'reader_val','validate',proxyids)
        else:calrows=rows_for(c,'cal','calibrate',qtypes=c['qtypes']);calbank=PrefixBank(c,'cal','calibrate')
        calibrations={};threshold_map={};allstats=[];allcp=[]
        for q in c['qtypes']:
            if kind=='reference':
                name=c['reference_name'];trainedq='Q1' if name=='B4' else q;model=Probe(name,trainedq).to(device)
                path=Path(c['m3_baseline_run'])/f'{name}_{trainedq}_{c["seed"]}.pt';model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            ca=[r for r in calrows if (r['problem_id'] in caids if smoke else r['cal_part']=='cal_A') and r['qtype']==q]
            cb=[r for r in calrows if (r['problem_id'] in cbids if smoke else r['cal_part']=='cal_B') and r['qtype']==q]
            a=predict(model,kind,ca,calbank,serializer,templates,norm,device,c['eval_batch'],budget);b=predict(model,kind,cb,calbank,serializer,templates,norm,device,c['eval_batch'],budget)
            if q=='Q3' and kind=='clr':
                sample=ca[:c['eval_batch']];perm=[2,0,3,1];permuted=[{**r,'options':[r['options'][i] for i in perm]} for r in sample]
                original=predict(model,kind,sample,calbank,serializer,templates,norm,device,c['eval_batch'],budget)
                changed=predict(model,kind,permuted,calbank,serializer,templates,norm,device,c['eval_batch'],budget)
                delta=float(np.max(np.abs(original[:,perm]-changed)))
                write_json(run.path/'option_invariance.json',{'max_abs_logit_difference':delta,'panel':'cal_A' if not smoke else 'reader_val_proxy','problems':len(sample)})
                if delta!=0:raise ValueError('Trained CLR option permutation not exactly invariant')
            calibrations[q]={m:fit(a,target_array(ca,q),m) for m in ['temperature','affine']}
            threshold_map[q]={m:thresholds(cb,b,calibrations[q][m],q,c['seed']) for m in calibrations[q]}
        write_json(run.path/'calibration.json',calibrations);write_json(run.path/'thresholds.json',{q:{m:{k:(v if np.isfinite(v) else None) for k,v in ts.items()} for m,ts in methods.items()} for q,methods in threshold_map.items()})
        del calbank,calrows
        for split in (['reader_val'] if smoke else c['eval_splits']):
            phase='validate' if smoke else 'final';bank=PrefixBank(c,split,phase,proxyids if smoke else None)
            for q in c['qtypes']:
                if kind=='reference':
                    name=c['reference_name'];trainedq='Q1' if name=='B4' else q;model=Probe(name,trainedq).to(device)
                    model.load_state_dict(torch.load(Path(c['m3_baseline_run'])/f'{name}_{trainedq}_{c["seed"]}.pt',map_location=device,weights_only=True))
                for held in [False,True]:
                    rows=rows_for(c,split,phase,heldout=held,ids=evalids if smoke else None,qtypes=[q]);logits=predict(model,kind,rows,bank,serializer,templates,norm,device,c['eval_batch'],budget)
                    pred=[]
                    for r,x in zip(rows,logits):
                        pred.append({'problem_id':r['problem_id'],'t':r['state_ref']['t'],'traj_id':r['state_ref']['traj_id'],'kind':r['state_ref']['kind'],'template_id':r['template_id'],'heldout':held,'replicate':r['replicate'],'label':r['label'],'label_soft':r['label_soft'],'logits':x.tolist()})
                    pq.write_table(pa.Table.from_pylist(pred),run.path/f'{q}_{split}_{held}.parquet',compression='zstd')
                    subsets={'all':list(range(len(rows)))}
                    if split=='unseen_eval':
                        overlap=json.loads((Path(c['output'])/'unseen_train_eval_overlap.json').read_text())
                        # Schema is explicitly supplied by the pre-final overlap audit.
                        contaminated={pid for pid,detail in overlap.items() if detail['overlap']}
                        subsets['no_train_13gram_overlap']=[i for i,r in enumerate(rows) if r['problem_id'] not in contaminated]
                        for pool in ['svamp','asdiv']:subsets[pool]=[i for i,r in enumerate(rows) if r['problem_id'].lower().startswith(pool+':')]
                    for subset,ii in subsets.items():
                        if not ii:continue
                        stats,cp=summarize([rows[i] for i in ii],logits[ii],q,calibrations[q],threshold_map[q],c['seed'])
                        common={'case':c['case'],'seed':c['seed'],'split':split,'subset':subset}
                        allstats += [{**common,**x} for x in stats];allcp += [{**common,**x} for x in cp]
                    write_json(run.path/'statistics.json',allstats);write_json(run.path/'conformal.json',allcp)
                    print(split,q,held,len(rows),flush=True)
            del bank
        status='completed';run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,statistics_rows=len(allstats),conformal_rows=len(allcp),test_opened=not smoke and 'test' in c['eval_splits'])
        print(run.path,flush=True)
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
