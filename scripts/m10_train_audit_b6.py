"""Independent per-task, single-cross-attention probes on frozen M7.1 features."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from clr.models.m10 import AttentionProbe,OptionFeatures
from clr.models.m71 import Reader
from clr.audit.m71_reader import Bank
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m10_runtime import Budget,live

@torch.no_grad()
def infer(model,option,bank,rows,c):
 out=[];model.eval()
 for i in range(0,len(rows),c['eval_batch']):
  rr=rows[i:i+c['eval_batch']]
  with torch.autocast('cuda',dtype=torch.bfloat16):x=model(option.batch(rr,c['device'],audit=True),bank.batch(rr,c['device']),task=rr[0]['task'])
  out.append(x.float().cpu())
 return torch.cat(out)

def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   smoke=c['mode']=='smoke'
   if not smoke:
    r=json.loads(Path(c['smoke_ref']).read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
    if m['status']!='completed':raise ValueError('Audit B6 smoke required')
   train=Bank(c['data_root'],'reader_train',limit=c['train_groups'] if smoke else None);val=Bank(c['data_root'],'reader_val',limit=c['val_groups'] if smoke else None)
   mean,scale=train.sequence_normalization();option=OptionFeatures(c['tokenizer_path']);history=[];checkpoints={};manifest={}
   old=json.loads(Path(c['data_root'],'pool','training',f'B3_{c["seed"]}.json').read_text());minimum=math.ceil(old['exposure']/(8000*48))
   write_json(run.path/'data_manifest.json',{'training_groups':len(train.entries),'validation_groups':len(val.entries),'test_opened':False,'minimum_epochs_from_B3_total_exposure':minimum,'reference':old['run']})
   for task in ['A1','A2','A3','A4']:
    tr=[r for r in train.rows if r['task']==task];va=[r for r in val.rows if r['task']==task];best=float('inf');beststate=None;selected=None
    baseline=Reader('B3',len(train.layer_ids));bparams=sum(p.numel() for p in baseline.heads[task].parameters());del baseline
    for lr in c['learning_rates']:
     seed_everything(c['seed']);model=AttentionProbe(512,hidden=c['hidden'],audit=True).to(c['device']);model.mean.copy_(mean);model.scale.copy_(scale)
     params=sum(p.numel() for p in model.parameters())
     if params<bparams:raise ValueError('B6 parameters below B3')
     optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01);rng=np.random.default_rng(c['seed']);exposure=0
     for epoch in range(1,(1 if smoke else c['epochs'])+1):
      started=time.monotonic();model.train();order=rng.permutation(len(tr))
      for i in range(0,len(tr),c['batch']):
       rr=[tr[j] for j in order[i:i+c['batch']]];optimizer.zero_grad(set_to_none=True)
       with torch.autocast('cuda',dtype=torch.bfloat16):x=model(option.batch(rr,c['device'],audit=True),train.batch(rr,c['device']),task=task)
       y=torch.tensor([r['label'] for r in rr],device=c['device']);loss=torch.nn.functional.cross_entropy(x.float(),y)
       if not torch.isfinite(loss):raise ValueError('Nonfinite B6 audit loss')
       loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();exposure+=len(rr)
       if i%(100*c['batch'])==0:budget.check();write_json(run.path/'progress.json',{'task':task,'epoch':epoch,'lr':lr,'batch':i//c['batch'],'batches':math.ceil(len(tr)/c['batch']),'exposure':exposure})
      torch.cuda.synchronize(c['device']);seconds=time.monotonic()-started;x=infer(model,option,val,va,c);y=torch.tensor([r['label'] for r in va]);nll=float(torch.nn.functional.cross_entropy(x,y));accuracy=float((x.argmax(1)==y).float().mean())
      row={'task':task,'epoch':epoch,'lr':lr,'exposure':exposure,'val_nll':nll,'val_accuracy':accuracy,'training_seconds':seconds};history.append(row);print(json.dumps(row),flush=True)
      if (smoke or epoch>=minimum) and nll<best:best=nll;selected=row;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
      write_json(run.path/'history.json',history);budget.check()
     del model
    if beststate is None:raise ValueError('No B6 checkpoint satisfies B3 exposure')
    path=run.path/f'{task}_best.pt';torch.save(beststate,path);checkpoints[task]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected':selected};manifest[task]={'parameters':params,'B3_parameters':bparams,'count':len(tr)}
   ref={'run':str(run.path),'config':c,'checkpoints':checkpoints,'parameters':manifest};folder=Path(c['output'])/'trained';folder.mkdir(exist_ok=True);write_json(folder/f'{c.get("case","audit_B6")}_{c["seed"]}_{c["mode"]}.json',ref)
   budget.check(status='completed',test_opened=False,gate_passed=True);live(stage='audit_B6_training_completed',last_run=str(run.path))
  except BaseException:
   budget.check(status='failed',test_opened=False);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
