"""Full-data trained language baseline, with independent sub-budget and smoke."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from clr.models.m10_b5c import B5c
from clr.data.m10 import GSMDataset
from clr.audit.m71_reader import Bank
from clr.data.m4.reader import targets
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m10_runtime import Budget,live


def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed']);torch.cuda.reset_peak_memory_stats(c['device'])
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   smoke=c['mode']=='smoke';audit=c['source']=='audit';qs=['A1','A4'] if audit else ['Q3','Q4'];limit=c['smoke_problems'] if smoke else None
   if not smoke:
    r=json.loads(Path(c['smoke_ref']).read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
    if m['status']!='completed':raise ValueError('B5c smoke not passed')
   if c.get('min_free_GiB') and torch.cuda.mem_get_info(c['device'])[0]/2**30<c['min_free_GiB']:raise RuntimeError('B5c independent-device memory guard failed')
   if audit:
    tr=Bank(c['audit_root'],'reader_train',limit=limit);va=Bank(c['audit_root'],'reader_val',limit=limit);mean,scale=tr.sequence_normalization();dim=512
    groups=lambda ds:{q:[r for r in ds.rows if r['task']==q] for q in qs}
    train,valid=groups(tr),groups(va)
   else:
    tr=GSMDataset(c,'reader_train','train',limit,qs);va=GSMDataset(c,'reader_val','validate',limit,qs);train,valid=tr.groups,va.groups;mean,scale=tr.mean,tr.scale;dim=c['sources'][c['source']]['dim']
   model=B5c(c['qwen_path'],dim,audit=audit,rank=16).to(c['device']);model.mean.copy_(mean);model.scale.copy_(scale);templates=load_config(c['templates']);params=[p for p in model.parameters() if p.requires_grad]
   if c.get('gradient_checkpointing'):model.body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
   optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=.01);rng=np.random.default_rng(c['seed']);counts={q:len(train[q]) for q in qs};total=sum(counts.values());weights={q:total/(len(qs)*counts[q]) for q in qs};history=[];best=float('inf');exposure={q:0 for q in qs}
   write_json(run.path/'training_manifest.json',{'counts':counts,'validation_counts':{q:len(valid[q]) for q in qs},'source_text_access':False,'test_opened':False,'LoRA_rank':16,'LoRA_alpha':32,'LoRA_modules':model.lora_names,'trainable_parameters':sum(p.numel() for p in params),'frozen_base_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)})
   def batch(ds,rr):
    if audit:return model(rr,ds.batch(rr,c['device']))
    z,m=ds.batch(rr,c['device']);return model(rr,z,m,templates)
   def target(rr):
    if audit:return torch.nn.functional.one_hot(torch.tensor([r['label'] for r in rr],device=c['device']),4 if rr[0]['task']=='A4' else 2).float()
    return targets(rr,c['device'])
   for epoch in range(1,c['epochs']+1):
    model.train();schedule=[]
    for q in qs:
     order=rng.permutation(counts[q]);schedule.extend((q,order[i:i+c['batch']]) for i in range(0,counts[q],c['batch']))
    rng.shuffle(schedule);train_seconds=0.;task_seconds={q:0. for q in qs}
    for bi,(q,ii) in enumerate(schedule):
     rr=[train[q][j] for j in ii];optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize(c['device']);started=time.monotonic()
     with torch.autocast('cuda',dtype=torch.bfloat16):out=batch(tr,rr)
     loss=-(target(rr)*out.float().log_softmax(-1)).sum(-1).mean()*weights[q]
     if not torch.isfinite(loss):raise ValueError('Nonfinite B5c loss')
     loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();exposure[q]+=len(rr);torch.cuda.synchronize(c['device']);elapsed=time.monotonic()-started;train_seconds+=elapsed;task_seconds[q]+=elapsed
     if bi%20==0:budget.check();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(schedule),'exposure':exposure,'training_seconds':train_seconds,'task_seconds':task_seconds})
    model.eval();vl={};val_started=time.monotonic()
    with torch.no_grad():
     for q in qs:
      rs=valid[q];loss=0.
      for start in range(0,len(rs),c['eval_batch']):
       rr=rs[start:start+c['eval_batch']]
       with torch.autocast('cuda',dtype=torch.bfloat16):out=batch(va,rr)
       loss+=float(-(target(rr)*out.float().log_softmax(-1)).sum());
       if start%(20*c['eval_batch'])==0:budget.check()
      vl[q]=loss/len(rs)
    score=float(np.mean(list(vl.values())));row={'epoch':epoch,'val_nll':vl,'selection_nll':score,'exposure':dict(exposure),'training_seconds':train_seconds,'task_seconds':task_seconds,'validation_seconds':time.monotonic()-val_started};history.append(row);print(json.dumps(row),flush=True);write_json(run.path/'history.json',history)
    if score<best:
     best=score;torch.save(model.trainable_state(),run.path/'best.pt');write_json(run.path/'best.json',row)
   checkpoint=run.path/'best.pt';ref={'run':str(run.path),'config':c,'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()};folder=Path(c['output'])/'trained';folder.mkdir(exist_ok=True);write_json(folder/f'{c["case"]}_{c["seed"]}_{c["mode"]}.json',ref)
   budget.check(status='completed',test_opened=False,gate_passed=True,peak_allocated_GiB=torch.cuda.max_memory_allocated(c['device'])/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved(c['device'])/2**30);live(stage='B5c_training_completed',last_run=str(run.path))
  except BaseException:
   budget.check(status='failed',test_opened=False);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
