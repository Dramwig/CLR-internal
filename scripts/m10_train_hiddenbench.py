"""Independent-group HiddenBench readers; hyperparameters selected by val NLL."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m10_hiddenbench import Bank,FactSerializer,OptionFeatures,forward,build
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from clr.utils.m10_runtime import Budget

@torch.no_grad()
def infer(model,kind,option,serializer,bank,rows,c):
 model.eval();out=[]
 for i in range(0,len(rows),c['eval_batch']):
  rr=rows[i:i+c['eval_batch']]
  with torch.autocast('cuda',dtype=torch.bfloat16):x=forward(model,kind,option,serializer,bank,rr,c['device'])
  out.append(x.float().cpu())
 return torch.cat(out)

def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);smoke=c['mode']=='smoke'
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   if not smoke:
    ref=json.loads(Path(c['reader_smoke_ref']).read_text())
    if not ref['passed']:raise ValueError('Reader smoke required')
   train=Bank(c,'reader_train');val=Bank(c,'reader_val',limit=1 if smoke else None);norm=train.sequence_normalization();pnorm=train.pool_normalization();option=OptionFeatures(c['tokenizer_path']);serializer=FactSerializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));refs={};history=[]
   groups={t:[r for r in train.rows if r['task']==t] for t in ['A1','A4']};valid={t:[r for r in val.rows if r['task']==t] for t in groups}
   if smoke:groups={t:rs[:max(1,int(len(rs)*.05))] for t,rs in groups.items()}
   for kind in c['kinds']:
    for seed in c['seeds']:
     active_tasks=[['A1','A4']] if kind=='CLR' else [['A1'],['A4']]
     for tasks in active_tasks:
      key=f'{kind}_{seed}_'+('_'.join(tasks));best=float('inf');state=None;selected=None
      for lr in c['learning_rates']:
       seed_everything(seed);model=build(kind).to(c['device']);mean,scale=pnorm if kind=='B3' else norm;model.mean.copy_(mean);model.scale.copy_(scale);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01);rng=np.random.default_rng(seed);exposure={t:0 for t in tasks}
       if kind=='B6' and sum(p.numel() for p in model.parameters())<sum(p.numel() for p in build('B3').parameters()):raise ValueError('B6 parameter budget below B3')
       for epoch in range(1,(1 if smoke else c['epochs'])+1):
        started=time.monotonic();model.train();schedule=[]
        for t in tasks:
         ii=rng.permutation(len(groups[t]));schedule.extend((t,ii[i:i+c['batch']]) for i in range(0,len(ii),c['batch']))
        rng.shuffle(schedule)
        for task,ii in schedule:
         rr=[groups[task][i] for i in ii];opt.zero_grad(set_to_none=True)
         with torch.autocast('cuda',dtype=torch.bfloat16):x=forward(model,kind,option,serializer,train,rr,c['device'])
         loss=torch.nn.functional.cross_entropy(x.float(),torch.tensor([r['label'] for r in rr],device=c['device']))
         if not torch.isfinite(loss):raise ValueError('Nonfinite HiddenBench loss')
         loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();exposure[task]+=len(rr);budget.check()
        torch.cuda.synchronize();seconds=time.monotonic()-started;nll={};acc={}
        for t in tasks:
         x=infer(model,kind,option,serializer,val,valid[t],c);y=torch.tensor([r['label'] for r in valid[t]]);nll[t]=float(torch.nn.functional.cross_entropy(x,y));acc[t]=float((x.argmax(1)==y).float().mean())
        row={'model':key,'epoch':epoch,'lr':lr,'val_nll':nll,'val_accuracy':acc,'exposure':dict(exposure),'training_seconds':seconds};history.append(row);score=np.mean(list(nll.values()))
        # Both probes return the full-exposure checkpoint; LR is selected on val.
        eligible=smoke or epoch==c['epochs'] or kind=='CLR'
        if eligible and score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=row
        write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
       del model,opt
      p=run.path/(key+'.pt');torch.save(state,p);refs[key]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'kind':kind,'seed':seed,'tasks':tasks,'selected':selected};write_json(run.path/'checkpoints.json',refs)
   result={'run':str(run.path),'config':c,'checkpoints':refs,'passed':True,'test_opened':False,'smoke_training_fraction_by_labels':.05 if smoke else None,'validation_groups':len({r['group_id'] for rs in valid.values() for r in rs})};write_json(c['reader_smoke_ref'] if smoke else Path(c['hidden_root'])/'training.json',result);budget.check(status='completed',gate_passed=True,test_opened=False)
  except BaseException:budget.check(status='failed');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
