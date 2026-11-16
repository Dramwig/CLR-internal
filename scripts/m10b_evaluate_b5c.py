"""M10 GSM calibration and locked final evaluation; smoke uses val only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.data.m10 import GSMDataset,ProbeCache,SerializedCache
from clr.models.m10 import OptionFeatures
from clr.models.clr_v0 import Serializer
from clr.models.m5b_controls import current_only
from clr.calib.m4 import fit,apply
from clr.calib.m4_evaluation import target_array,summarize,thresholds
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m10b_runtime import Budget,live
from clr.models.m10_b5c import B5c

@torch.no_grad()
def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device'])
 ref=json.loads(Path(c['training_ref']).read_text());tc=ref['config'];kind=tc['kind'];qs=['Q3','Q4']
 if c['mode']!='smoke':
  lock=json.loads(Path(c['eval_lock']).read_text())
  if not lock['all_training_locked']:raise ValueError('Training not locked')
  h=hashlib.sha256(Path(ref['checkpoint']).read_bytes()).hexdigest()
  if h!=ref['sha256'] or lock['checkpoint_sha256'].get(ref['checkpoint'])!=h:raise ValueError('Checkpoint lock mismatch')
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   template=load_config(tc['templates']);serializer=Serializer(AutoTokenizer.from_pretrained(tc['tokenizer_path'],local_files_only=True),template) if kind.startswith('CLR') else None
   option=OptionFeatures(tc['tokenizer_path']) if kind=='B6' else None
   smoke=c['mode']=='smoke';ds=GSMDataset(tc,'reader_val' if smoke else 'cal','validate' if smoke else 'calibrate',15 if smoke else None,qs)
   if smoke:
    ids=sorted({r['problem_id'] for r in ds.rows});caids=set(ids[:5]);cbids=set(ids[5:10]);evalids=set(ids[10:]);
   calibration={};threshold_map={};models={}
   model=B5c(tc['qwen_path'],tc['sources'][tc['source']]['dim'],rank=16)
   model.restore_trainable(torch.load(ref['checkpoint'],map_location='cpu',weights_only=True));model=model.to(c['device']).eval()
   def predict(dataset,rr,q):
    out=[]
    for i in range(0,len(rr),c['eval_batch']):
     batch=rr[i:i+c['eval_batch']];z,mask=dataset.batch(batch,c['device'])
     with torch.autocast('cuda',dtype=torch.bfloat16):x=model(batch,z,mask,template)
     out.append(x.float().cpu().numpy());budget.check()
    return np.concatenate(out)
   for q in qs:
    ca=[r for r in ds.groups[q] if (r['problem_id'] in caids if smoke else r['cal_part']=='cal_A')];cb=[r for r in ds.groups[q] if (r['problem_id'] in cbids if smoke else r['cal_part']=='cal_B')]
    a=predict(ds,ca,q);b=predict(ds,cb,q);calibration[q]={'temperature':fit(a,target_array(ca,q),'temperature')};threshold_map[q]={'temperature':thresholds(cb,b,calibration[q]['temperature'],q,c['seed'])}
    pq.write_table(pa.Table.from_pylist([{**r,'logits':v.tolist(),'calibrated_logits':apply(v[None],calibration[q]['temperature'])[0].tolist()} for r,v in zip(ca+cb,np.concatenate([a,b]))]),run.path/f'{tc["source"]}_{q}_cal.parquet',compression='zstd')
   write_json(run.path/'calibration.json',calibration);stats=[];cps=[]
   for split in (['reader_val'] if smoke else ['reader_val','test']):
    if split=='test':budget.check(test_opened=True)
    dataset=ds if smoke else GSMDataset(tc,split,'final',qtypes=qs)
    for q in qs:
     rr=[r for r in dataset.groups[q] if not smoke or r['problem_id'] in evalids];x=predict(dataset,rr,q)
     pq.write_table(pa.Table.from_pylist([{**r,'logits':v.tolist(),'calibrated_logits':apply(v[None],calibration[q]['temperature'])[0].tolist()} for r,v in zip(rr,x)]),run.path/f'{tc["source"]}_{q}_{split}.parquet',compression='zstd')
     st,cp=summarize(rr,x,q,calibration[q],threshold_map[q],c['seed'])
     stats.extend(dict(source=tc['source'],split=split,seed=c['seed'],model_key=tc['kind'],**r) for r in st);cps.extend(dict(source=tc['source'],split=split,seed=c['seed'],model_key=tc['kind'],**r) for r in cp)
     write_json(run.path/'statistics.json',stats);write_json(run.path/'conformal.json',cps);print(split,q,len(rr),flush=True)
   folder=Path(c['output'])/'evaluated';folder.mkdir(exist_ok=True);write_json(folder/f'{tc["case"]}_{c["seed"]}_{c["mode"]}.json',{'run':str(run.path),'training':c['training_ref']})
   budget.check(status='completed',test_opened=not smoke,gate_passed=True)
  except BaseException:
   budget.check(status='failed');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
