"""Locked final evaluation of trained M5b-A controls; smoke uses val proxies only."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.m5b_runtime import Budget,snapshot
from clr.models.m5b_controls import PrefixProbe,current_only,prefix_features
from clr.models.clr_v1 import make_model
from clr.models.clr_v0 import Serializer
from clr.data.m4.reader import rows_for,PrefixBank,probe_inputs
from clr.calib.m4 import fit
from clr.calib.m4_evaluation import target_array,thresholds,summarize


@torch.inference_mode()
def predict(model,case,rows,bank,serializer,templates,norm,device,batch,budget):
    out=[]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=case=='CLR-lite-single'):
        for i in range(0,len(rows),batch):
            rs=rows[i:i+batch]
            if case=='CLR-lite-single':
                z,m=current_only(*bank.batch(rs,device));logits=model(serializer.batch(rs,device),z,m,rs[0]['qtype'])
            else:
                ii=torch.tensor([bank.index[r['problem_id'],r['state_ref']['traj_id'],r['state_ref']['t']] for r in rs],device=device)
                # probe_inputs supplies only q/option features; its state output
                # is ignored because the bank already stores 1536-D concat input.
                from clr.baselines.data_m23 import template_features
                from clr.data.questions.core import numeric_features
                q=torch.tensor(np.stack([template_features(templates[r['qtype']][r['template_id']]) for r in rs]),device=device)
                op=torch.tensor(np.stack([numeric_features(r['options'] or ['0']) for r in rs]),device=device)
                logits=model(bank.z[ii],q,op)
            out.append(logits.float().cpu().numpy())
            if i%(batch*100)==0:budget.check()
    return np.concatenate(out)


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);status='failed';test_opened=False
    Path(c['output']).mkdir(parents=True,exist_ok=True)
    try:
        torch.set_num_threads(2);seed_everything(c['seed']);smoke=c['stage']=='smoke';case=c['case'];device=c['device'];budget.check()
        snapshot(run,c,['scripts/m5b_evaluate_a.py','clr/models/m5b_controls.py','clr/data/m4/reader.py','clr/calib/m4_evaluation.py','clr/calib/m4.py'])
        if not smoke:
            lock=json.loads(Path(c['training_lock']).read_text());gate=json.loads(Path(c['eval_smoke_run'],'metrics.json').read_text())
            if not lock['all_training_locked'] or gate.get('status')!='completed' or not gate.get('gate_passed'):raise ValueError('Locked trained controls and eval smoke required')
            for name,digest in lock['checkpoint_sha256'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed locked A checkpoint')
        templates=load_config(c['templates']);normfile=np.load(Path(c['m3_baseline_run'])/'normalization.npz');norm=[torch.tensor(normfile[k],device=device) for k in ['mean','scale']]
        model=None;serializer=Serializer(AutoTokenizer.from_pretrained(c['m41_model_path'],local_files_only=True),templates)
        if case=='CLR-lite-single':
            model=make_model({**c,'variant':'lite'}).to(device).eval();model.restore_trainable(torch.load(Path(c['train_run'])/'best.pt',map_location='cpu',weights_only=True))
        proxy=None
        if smoke:
            rs=rows_for(c,'reader_val','validate');ids=sorted({r['problem_id'] for r in rs})[:50];proxy=set(ids);caids=set(ids[:15]);cbids=set(ids[15:30]);evalids=set(ids[30:]);calrows=[r for r in rs if r['problem_id'] in caids|cbids];calbank=PrefixBank(c,'reader_val','validate',proxy)
        else:calrows=rows_for(c,'cal','calibrate');calbank=PrefixBank(c,'cal','calibrate')
        if case=='B3-prefix':calbank.z=prefix_features(calbank,*norm,device)
        calibrations={};ts={};stats=[];cp=[]
        for q in ['Q1','Q2','Q3','Q4']:
            if case=='B3-prefix':
                model=PrefixProbe(q).to(device).eval();model.load_state_dict(torch.load(Path(c['train_run'])/f'{q}_best.pt',map_location=device,weights_only=True))
            ca=[r for r in calrows if r['qtype']==q and (r['problem_id'] in caids if smoke else r['cal_part']=='cal_A')]
            cb=[r for r in calrows if r['qtype']==q and (r['problem_id'] in cbids if smoke else r['cal_part']=='cal_B')]
            a=predict(model,case,ca,calbank,serializer,templates,norm,device,c['batch'],budget);b=predict(model,case,cb,calbank,serializer,templates,norm,device,c['batch'],budget)
            calibrations[q]={'temperature':fit(a,target_array(ca,q),'temperature')};ts[q]={'temperature':thresholds(cb,b,calibrations[q]['temperature'],q,c['seed'])}
        write_json(run.path/'calibration.json',calibrations);write_json(run.path/'thresholds.json',{q:{m:{k:(v if np.isfinite(v) else None) for k,v in x.items()} for m,x in y.items()} for q,y in ts.items()})
        del calbank,calrows
        for split in (['reader_val'] if smoke else ['reader_val','test']):
            phase='validate' if split=='reader_val' else 'final'
            if split=='test':test_opened=True
            bank=PrefixBank(c,split,phase,proxy if smoke else None)
            if case=='B3-prefix':bank.z=prefix_features(bank,*norm,device)
            for q in ['Q1','Q2','Q3','Q4']:
                if case=='B3-prefix':
                    model=PrefixProbe(q).to(device).eval();model.load_state_dict(torch.load(Path(c['train_run'])/f'{q}_best.pt',map_location=device,weights_only=True))
                rs=rows_for(c,split,phase,ids=evalids if smoke else None,qtypes=[q]);logits=predict(model,case,rs,bank,serializer,templates,norm,device,c['batch'],budget)
                records=[{'problem_id':r['problem_id'],'t':r['state_ref']['t'],'traj_id':r['state_ref']['traj_id'],'kind':r['state_ref']['kind'],'template_id':r['template_id'],'heldout':False,'replicate':r['replicate'],'label':r['label'],'label_soft':r['label_soft'],'logits':x.tolist()} for r,x in zip(rs,logits)]
                pq.write_table(pa.Table.from_pylist(records),run.path/f'{q}_{split}_False.parquet',compression='zstd')
                a,b=summarize(rs,logits,q,calibrations[q],ts[q],c['seed']);common={'case':case,'seed':c['seed'],'split':split,'subset':'all'};stats.extend({**common,**x} for x in a);cp.extend({**common,**x} for x in b)
                write_json(run.path/'statistics.json',stats);write_json(run.path/'conformal.json',cp);print(split,q,len(rs),flush=True)
            del bank
        status='completed';run.write_metrics(status=status,gate_passed=True,gpu_hours=(time.monotonic()-start)/3600,test_opened=test_opened,source_forward_calls=0)
        write_json(Path(c['output'])/f"evaluated_{case}_{c['seed']}_{c['stage']}.json",{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,test_opened=test_opened)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
