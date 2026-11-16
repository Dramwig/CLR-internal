"""M10 source controls/B6/per-type CLR with explicit budgets and exposure."""
import sys
sys.dont_write_bytecode=True
import argparse,copy,hashlib,json,time,fcntl
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.m10_runtime import Budget,live
from clr.data.m10 import GSMDataset,ProbeCache,SerializedCache
from clr.models.m10 import ControlProbe,AttentionProbe,OptionFeatures
from clr.models.m6 import NativeLite
from clr.models.m5b_controls import current_only
from clr.models.clr_v0 import Serializer
from clr.data.m4.reader import targets
from clr.data.questions.core import seed as hashseed

def build(c,q=None):
 dim=c['sources'][c['source']]['dim'];kind=c['kind']
 if kind in ['B0','B3-prefix']:return ControlProbe(kind,q,dim)
 if kind=='B6':return AttentionProbe(dim,hidden=c['hidden'])
 return NativeLite(dim)

def train_once(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c);start=time.monotonic()
  try:
   if c['mode']=='full':
    ref=json.loads(Path(c['smoke_ref']).read_text());met=json.loads((Path(ref['run'])/'metrics.json').read_text())
    if met['status']!='completed':raise ValueError('Training smoke required')
   kinds=c['kind'];qtypes=c.get('qtypes',['Q1','Q2','Q3','Q4']);limit=c.get('smoke_problems') if c['mode']=='smoke' else None
   tr=GSMDataset(c,'reader_train','train',limit,qtypes);va=GSMDataset(c,'reader_val','validate',limit,qtypes)
   template=load_config(c['templates']);serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True),template) if kinds.startswith('CLR') else None
   text_caches={id(ds):{q:SerializedCache(serializer,ds.groups[q]) for q in qtypes} for ds in [tr,va]} if serializer else {}
   option=OptionFeatures(c['tokenizer_path']) if kinds=='B6' else None
   caches=[ProbeCache(ds,kinds,template,c['device']) for ds in [tr,va]] if kinds in ['B0','B3-prefix'] else None
   grouped=kinds!='CLR-lite-single';taskgroups=[[q] for q in qtypes] if grouped else [qtypes]
   counts={q:len(tr.groups[q]) for q in qtypes};history=[];checkpoints={};manifest={};exposures={}
   if kinds=='CLR-per-type' and c['mode']=='full':
    target=json.loads(Path(c['reference_exposure']).read_text())[c['source']][str(c['seed'])]
    if counts!=target['counts'] or any(counts[q]*c['reference_epochs']!=target['exposure'][q] for q in qtypes):raise ValueError('Per-type exposure differs from frozen multi-type checkpoint')
    write_json(run.path/'exposure_target.json',target)
   write_json(run.path/'data_manifest.json',{'counts':counts,'validation_counts':{q:len(va.groups[q]) for q in qtypes},'train_problem_ids':sorted({r['problem_id'] for r in tr.rows}),'validation_problem_ids':sorted({r['problem_id'] for r in va.rows}),'test_opened':False})
   budget.check();training_seconds=0.;validation_seconds=0.
   for active in taskgroups:
    key=active[0] if grouped else 'all';best=float('inf');beststate=None;selected=None
    lrs=c['learning_rates'] if kinds in ['B0','B3-prefix','B6'] else [c['learning_rate']]
    epochs=c['epochs']
    if kinds=='CLR-per-type':epochs=c['reference_epochs']
    if c['mode']=='smoke':epochs=1
    for lr in lrs:
     seed_everything(c['seed']);model=build(c,active[0]).to(c['device'])
     if kinds=='B6':model.mean.copy_(tr.mean);model.scale.copy_(tr.scale)
     elif kinds.startswith('CLR'):model.state_mean.copy_(tr.mean);model.state_scale.copy_(tr.scale)
     params=sum(p.numel() for p in model.parameters() if p.requires_grad);manifest[key]={'parameters':params,'counts':{q:counts[q] for q in active}}
     if kinds=='B6':
      baseline_params=sum(p.numel() for p in ControlProbe('B3',active[0],tr.bank.dim).parameters())
      if params<baseline_params:raise ValueError('B6 has fewer parameters than B3')
     opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001);rng=np.random.default_rng(c['seed']);bad=0;exposure={q:0 for q in active}
     def forward(ds,rs,q):
      z,m=ds.batch(rs,c['device'])
      if kinds=='B6':return model(option.batch(rs,c['device']),z,m,q)
      if kinds=='CLR-lite-single':z,m=current_only(z,m)
      return model(text_caches[id(ds)][q].batch(rs,c['device']),z,m,q)
     for epoch in range(1,epochs+1):
      torch.cuda.synchronize(c['device']);epoch_start=time.monotonic()
      model.train();schedule=[]
      for q in active:
       order=(np.random.default_rng(hashseed(c['seed'],epoch-1,'order')).permutation(counts[q]) if kinds in ['B0','B3-prefix'] else rng.permutation(counts[q]))
       schedule.extend((q,order[i:i+c['batch']]) for i in range(0,len(order),c['batch']))
      if len(active)>1:rng.shuffle(schedule)
      for bi,(q,ii) in enumerate(schedule):
       opt.zero_grad(set_to_none=True)
       if caches:
        zz,qq,oo,y=caches[0].batch(q,ii);out=model(zz,qq,oo)
       else:
        rs=[tr.groups[q][i] for i in ii];y=targets(rs,c['device'])
        with torch.autocast('cuda',dtype=torch.bfloat16):out=forward(tr,rs,q)
       loss=-(y*out.float().log_softmax(-1)).sum(-1).mean()
       if len(active)>1:loss=loss*sum(counts.values())/(len(active)*counts[q])
       if not torch.isfinite(loss):raise RuntimeError('Nonfinite M10 loss')
       loss.backward()
       if not caches:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
       opt.step();exposure[q]+=len(ii)
       if bi%100==0:budget.check();write_json(run.path/'progress.json',{'task':key,'epoch':epoch,'batch':bi,'batches':len(schedule),'exposure':exposure})
      torch.cuda.synchronize(c['device']);train_elapsed=time.monotonic()-epoch_start;training_seconds+=train_elapsed
      model.eval();vl={};validation_start=time.monotonic()
      with torch.no_grad():
       for q in active:
        loss_sum=0.;rs=va.groups[q]
        for i in range(0,len(rs),c['eval_batch']):
         rr=rs[i:i+c['eval_batch']]
         if caches:
          zz,qq,oo,y=caches[1].batch(q,np.arange(i,i+len(rr)));out=model(zz,qq,oo)
         else:
          with torch.autocast('cuda',dtype=torch.bfloat16):out=forward(va,rr,q)
          y=targets(rr,c['device'])
         loss_sum+=float(-(y*out.float().log_softmax(-1)).sum())
        vl[q]=loss_sum/len(rs);budget.check()
      torch.cuda.synchronize(c['device']);valid_elapsed=time.monotonic()-validation_start;validation_seconds+=valid_elapsed
      score=float(np.mean(list(vl.values())));row={'task':key,'epoch':epoch,'lr':lr,'val_nll':vl,'selection_nll':score,'exposure':dict(exposure),'training_seconds':train_elapsed,'validation_seconds':valid_elapsed,'gpu_hours':(time.monotonic()-start)/3600};history.append(row);print(json.dumps(row),flush=True)
      eligible=(c['mode']=='smoke' or epoch>=c['min_epochs'])
      if kinds=='CLR-per-type':eligible=epoch==epochs
      if eligible:
       if score<best:best=score;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=row;bad=0
       else:bad+=1
      write_json(run.path/'history.json',history)
      if kinds=='CLR-lite-single' and eligible and bad>=1:break
     exposures[key]={'total_by_type':dict(exposure),'selected':selected}
    if beststate is None:raise ValueError('No eligible checkpoint')
    path=run.path/(key+'_best.pt');torch.save(beststate,path);checkpoints[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected':selected};write_json(run.path/(key+'_best.json'),selected)
    del model
   write_json(run.path/'training_manifest.json',{'kind':kinds,'tasks':manifest,'exposure':exposures,'test_opened':False,'source_text_access':False})
   folder=Path(c['output'])/'trained';folder.mkdir(parents=True,exist_ok=True)
   ref={'run':str(run.path),'config':c,'checkpoints':checkpoints,'training_manifest':str(run.path/'training_manifest.json')};write_json(folder/f'{c["case"]}_{c["seed"]}_{c["mode"]}.json',ref)
   budget.check(status='completed',test_opened=False,training_seconds=training_seconds,validation_seconds=validation_seconds,gate_passed=True);live(stage='training_completed',last_run=str(run.path),test_opened=False)
  except BaseException:
   budget.check(status='failed',test_opened=False);raise
def main(c):
 # Additional queues may use a GPU after its original job finishes. Serialize
 # identical scientific tasks before acquiring a GPU or charging a new run.
 if c['mode']!='full':return train_once(c)
 folder=Path(c['output'])/'task_locks';folder.mkdir(parents=True,exist_ok=True)
 with (folder/f'{c["case"]}_{c["seed"]}_full.lock').open('a') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX)
  ref=Path(c['output'])/'trained'/f'{c["case"]}_{c["seed"]}_full.json'
  if ref.exists():
   r=json.loads(ref.read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
   scientific=lambda x:{k:v for k,v in x.items() if k!='device'}
   if m['status']!='completed' or scientific(r['config'])!=scientific(c):raise ValueError('Existing task reference is incomplete or uses a different protocol')
   print('Already completed; no new GPU run:',ref,flush=True);return
  return train_once(c)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
