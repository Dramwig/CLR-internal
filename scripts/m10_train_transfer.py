"""Paired M9-body initialization versus scratch on nested audit QA labels."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m10_transfer import panel,initialize
from clr.audit.m71_reader import Bank,Serializer
from clr.data.m10 import SerializedCache
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from clr.utils.m10_runtime import Budget,live

def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device'])
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   smoke=c['mode']=='smoke'
   if not smoke:
    ref=json.loads(Path(c['smoke_ref']).read_text());m=json.loads((Path(ref['run'])/'metrics.json').read_text())
    if m['status']!='completed':raise ValueError('Transfer smoke required')
   train=Bank(c['data_root'],'reader_train',limit=c['smoke_groups'] if smoke else None);val=Bank(c['data_root'],'reader_val',limit=c['smoke_groups'] if smoke else c['validation_groups']);norm=train.sequence_normalization()
   serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));valid={t:[r for r in val.rows if r['task']==t] for t in ['A1','A4']};valcache={t:SerializedCache(serializer,rs) for t,rs in valid.items()};references={};history=[]
   for n in c['label_counts']:
    label='full' if n is None else str(n);rows=panel(train.rows,n,c['panel_seed']);groups={t:[r for r in rows if r['task']==t] for t in valid};cache={t:SerializedCache(serializer,rs) for t,rs in groups.items()};counts={t:len(rs) for t,rs in groups.items()};total=sum(counts.values())
    write_json(run.path/f'panel_{label}.json',[{k:r[k] for k in ['group_id','message_id','task','template_id','question_text','options','label']} for r in rows])
    for seed in c['seeds']:
     for init in ['scratch','M9']:
      seed_everything(seed);model=initialize(seed,init,norm).to(c['device']);optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=.01);rng=np.random.default_rng(seed);best=float('inf');beststate=None;bestrow=None;exposure={t:0 for t in valid};key=f'{init}_{label}_{seed}'
      epochs=c['smoke_epochs'] if smoke else c['full_epochs'] if n is None else c['small_epochs']
      for epoch in range(1,epochs+1):
       model.train();schedule=[];started=time.monotonic()
       for t,rs in groups.items():
        order=rng.permutation(len(rs));schedule.extend((t,order[i:i+c['batch']]) for i in range(0,len(rs),c['batch']))
       rng.shuffle(schedule)
       for bi,(task,ii) in enumerate(schedule):
        rr=[groups[task][i] for i in ii];optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):x=model(cache[task].batch(rr,c['device']),train.batch(rr,c['device']),task)
        y=torch.tensor([r['label'] for r in rr],device=c['device']);loss=torch.nn.functional.cross_entropy(x.float(),y)*total/(2*counts[task])
        if not torch.isfinite(loss):raise ValueError('Nonfinite transfer loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();exposure[task]+=len(rr)
        if bi%100==0:budget.check();write_json(run.path/'progress.json',{'model':key,'epoch':epoch,'batch':bi,'batches':len(schedule),'exposure':exposure})
       torch.cuda.synchronize(c['device']);train_seconds=time.monotonic()-started;started=time.monotonic();model.eval();vl={};acc={}
       with torch.no_grad():
        for task,rs in valid.items():
         total_loss=0.;correct=0
         for start in range(0,len(rs),c['eval_batch']):
          rr=rs[start:start+c['eval_batch']]
          with torch.autocast('cuda',dtype=torch.bfloat16):x=model(valcache[task].batch(rr,c['device']),val.batch(rr,c['device']),task)
          y=torch.tensor([r['label'] for r in rr],device=c['device']);total_loss+=float(torch.nn.functional.cross_entropy(x.float(),y,reduction='sum'));correct+=int((x.argmax(1)==y).sum());budget.check()
         vl[task]=total_loss/len(rs);acc[task]=correct/len(rs)
       row={'model':key,'seed':seed,'initialization':init,'N':n,'epoch':epoch,'val_nll':vl,'val_accuracy':acc,'selection_nll':float(np.mean(list(vl.values()))),'exposure':dict(exposure),'training_seconds':train_seconds,'validation_seconds':time.monotonic()-started};history.append(row)
       if row['selection_nll']<best:best=row['selection_nll'];bestrow=row;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
       write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
      path=run.path/f'{key}_best.pt';torch.save(beststate,path);references[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected':bestrow};write_json(run.path/'checkpoints.json',references)
      del model,optimizer,beststate;torch.cuda.empty_cache()
   folder=Path(c['output'])/'trained';folder.mkdir(exist_ok=True);write_json(folder/f'transfer_{c["mode"]}.json',{'run':str(run.path),'config':c,'checkpoints':references,'test_opened':False})
   budget.check(status='completed',test_opened=False,gate_passed=True);live(stage='transfer_training_completed',last_run=str(run.path))
  except BaseException:
   budget.check(status='failed',test_opened=False);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
