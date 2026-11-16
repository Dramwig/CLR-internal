"""M5b-A frozen input-confound controls.

The smoke path is val-only.  The full path is the only M5b entry point that
may open test, and it does so only after the val smoke gate has passed.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything, write_json
from clr.models.clr_v0 import Serializer
from clr.models.clr_v1 import make_model
from clr.baselines.probes import Probe
from clr.data.m4.reader import PrefixBank, rows_for, probe_inputs
from clr.calib.m4 import fit
from clr.calib.m4_evaluation import target_array, summarize


def _norm(c, device):
    x = np.load(Path(c['m3_baseline_run'])/'normalization.npz')
    return [torch.from_numpy(x[k]).to(device) for k in ['mean', 'scale']]


@torch.no_grad()
def predict_b3(model, rows, bank, templates, norm, device, batch, prefix_mode):
    out=[]
    for i in range(0, len(rows), batch):
        rs=rows[i:i+batch]; prefix,mask=bank.batch(rs,device)
        z,q,opt=probe_inputs(rs,prefix,mask,templates,*norm)
        if prefix_mode == 'mean_current':
            # Explicitly retain the current state while exposing the prefix.
            valid=mask.float().unsqueeze(-1)
            mean=(prefix*valid).sum(1)/valid.sum(1).clamp_min(1)
            z=(mean+z)/2
        out.append(model(z,q,opt).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_lite(model, serializer, rows, bank, device, batch, single):
    out=[]
    for i in range(0, len(rows), batch):
        rs=rows[i:i+batch]; prefix,mask=bank.batch(rs,device)
        if single:
            ii=mask.long().sum(1)-1
            prefix=prefix[torch.arange(len(rs),device=device),ii,None]
            mask=torch.ones((len(rs),1),dtype=torch.bool,device=device)
        text=serializer.batch(rs,device)
        out.append(model(text,prefix,mask,rs[0]['qtype']).float().cpu().numpy())
    return np.concatenate(out)


def checkpoint(c, run):
    names=['scripts/m5b_a.py','docs/M5b_TASK.md','clr/models/clr_v0.py','clr/models/clr_v1.py',
           'clr/data/m4/reader.py','clr/calib/m4.py','clr/calib/m4_evaluation.py',c['templates']]
    h={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names}
    for s in c['seeds']:
        for q in ['Q1','Q2','Q3','Q4']:
            p=Path(c['m3_baseline_run'])/f'B3_{q}_{s}.pt';h[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        p=Path(c['m41_runs']['lite'][str(s)])/'best.pt';h[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(run.path/'input_sha256.json',h)


def main(c):
  with recorded(c) as run:
    start=time.monotonic(); status='failed'; seed_everything(c['seed']); torch.set_num_threads(c.get('threads',2)); device=c['reader_device']
    try:
      checkpoint(c,run)
      smoke=c['stage']=='smoke'
      if not smoke:
        gate=json.loads((Path(c['smoke_run'])/'metrics.json').read_text())
        if gate.get('status')!='completed' or not gate.get('gate_passed'):
          raise RuntimeError('M5b-A full run requires a completed val-only smoke')
      templates=load_config(c['templates']); norm=_norm(c,device)
      tokenizer=AutoTokenizer.from_pretrained(c['m41_model_path'],local_files_only=True)
      serializer=Serializer(tokenizer,templates)
      all_rows=[]; all_cp=[]
      splits=['reader_val'] if smoke else ['reader_val','test']
      for seed in c['seeds']:
        for variant in ['B3-prefix','CLR-lite-single']:
          for split in splits:
            phase='validate' if smoke or split=='reader_val' else 'final'
            rows=rows_for(c,split,phase,heldout=False)
            if smoke:
              ids=sorted({r['problem_id'] for r in rows})[:c.get('smoke_problems',50)]
              rows=[r for r in rows if r['problem_id'] in ids]
            bank=PrefixBank(c,split,phase,set(r['problem_id'] for r in rows) if smoke else None)
            if variant=='B3-prefix':
              models={q:Probe('B3',q,128).to(device).eval() for q in ['Q1','Q2','Q3','Q4']}
              for q,m in models.items():m.load_state_dict(torch.load(Path(c['m3_baseline_run'])/f'B3_{q}_{seed}.pt',map_location=device,weights_only=True))
            else:
              mc={**c,'variant':'lite','model_path':c['m41_model_path'],'gradient_checkpointing':False}
              model=make_model(mc).to(device).eval(); ck=Path(c['m41_runs']['lite'][str(seed)])/'best.pt'
              model.restore_trainable(torch.load(ck,map_location='cpu',weights_only=True));model.state_mean.copy_(norm[0]);model.state_scale.copy_(norm[1])
            for q in ['Q1','Q2','Q3','Q4']:
              rs=[r for r in rows if r['qtype']==q]
              if variant=='B3-prefix': logits=predict_b3(models[q],rs,bank,templates,norm,device,c['batch'],'mean_current')
              else: logits=predict_lite(model,serializer,rs,bank,device,c['batch'],True)
              # A is a pre-specified diagnostic; calibrate on frozen cal_A/B for full,
              # and on disjoint reader_val proxies for smoke.
              if smoke:
                ids=sorted({r['problem_id'] for r in rs}); ca=set(ids[:max(1,len(ids)//3)]); cb=set(ids[max(1,len(ids)//3):max(2,2*len(ids)//3)])
                cal=[r for r in rs if r['problem_id'] in ca]; val=[r for r in rs if r['problem_id'] in cb]
                if len(cal)<2 or len(val)<2: continue
                cal_idx=[i for i,r in enumerate(rs) if r['problem_id'] in ca]; val_idx=[i for i,r in enumerate(rs) if r['problem_id'] in cb]
                calib=fit(logits[cal_idx],target_array(cal,q),'temperature'); thresh=__import__('clr.calib.m4_evaluation',fromlist=['thresholds']).thresholds(val,logits[val_idx],calib,q,seed)
                stats,cp=summarize(val,logits[val_idx],q,{'temperature':calib},{'temperature':thresh},seed)
              else:
                cal=rows_for(c,'cal','calibrate',qtypes=[q]); cbank=PrefixBank(c,'cal','calibrate')
                if variant=='B3-prefix': clog=predict_b3(models[q],cal,cbank,templates,norm,device,c['batch'],'mean_current')
                else: clog=predict_lite(model,serializer,cal,cbank,device,c['batch'],True)
                ca=[r for r in cal if r['cal_part']=='cal_A']; cb=[r for r in cal if r['cal_part']=='cal_B']; ia=[i for i,r in enumerate(cal) if r['cal_part']=='cal_A']; ib=[i for i,r in enumerate(cal) if r['cal_part']=='cal_B']
                calib=fit(clog[ia],target_array(ca,q),'temperature'); thresh=__import__('clr.calib.m4_evaluation',fromlist=['thresholds']).thresholds(cb,clog[ib],calib,q,seed)
                stats,cp=summarize(rs,logits,q,{'temperature':calib},{'temperature':thresh},seed)
              for x in stats: all_rows.append({'variant':variant,'seed':seed,'split':split,'qtype':q,**x})
              for x in cp: all_cp.append({'variant':variant,'seed':seed,'split':split,'qtype':q,**x})
            del bank
      pq.write_table(pa.Table.from_pylist(all_rows),run.path/'statistics.parquet',compression='zstd')
      pq.write_table(pa.Table.from_pylist(all_cp),run.path/'conformal.parquet',compression='zstd')
      elapsed=(time.monotonic()-start)/3600
      run.write_metrics(status='completed',gate_passed=True,gpu_hours=elapsed,test_opened=not smoke,statistics_rows=len(all_rows),conformal_rows=len(all_cp))
      status='completed'; print(run.path,flush=True)
    finally:
      if status!='completed': run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,test_opened=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
