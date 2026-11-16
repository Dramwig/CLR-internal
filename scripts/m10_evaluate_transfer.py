"""Locked final A1/A4 learning curves for paired M9/scratch initialization."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.special import softmax
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.models.m71 import Reader
from clr.audit.m7_evaluation import fit_temperature,qa_summary
from clr.data.m10 import SerializedCache
from clr.data.m7 import threshold
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from clr.utils.m10_runtime import Budget

def select(bank,start,end):
 b=copy.copy(bank);b.entries=bank.entries[start:end];ids={e['id'] for e in b.entries};b.rows=[r for r in bank.rows if r['episode_id'] in ids];return b
@torch.no_grad()
def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);ref=json.loads(Path(c['training_ref']).read_text());smoke=c['mode']=='smoke';tc=ref['config']
 if not smoke:
  lock=json.loads(Path(c['eval_lock']).read_text())
  if not lock['all_training_locked']:raise ValueError('Training lock required')
  for cp in ref['checkpoints'].values():
   if hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()!=lock['checkpoint_sha256'].get(cp['path']):raise ValueError('Transfer checkpoint not locked')
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   if not smoke:budget.check(test_opened=True)
   proxy=Bank(tc['data_root'],'reader_val',limit=30) if smoke else None;banks={s:select(proxy,*bounds) if smoke else Bank(tc['data_root'],s) for s,bounds in [('cal_A',(0,10)),('cal_B',(10,20)),('reader_val',(20,30))]+([] if smoke else [('test',(0,0))])};serializer=Serializer(AutoTokenizer.from_pretrained(tc['tokenizer_path'],local_files_only=True));rows={s:{t:[r for r in b.rows if r['task']==t] for t in ['A1','A4']} for s,b in banks.items()};caches={s:{t:SerializedCache(serializer,rr) for t,rr in groups.items()} for s,groups in rows.items()};results=[]
   for key,cp in ref['checkpoints'].items():
    model=Reader('CLR').to(c['device']);model.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=True));model.eval();temps={};ths={}
    def predict(split,t):
     rr=rows[split][t];out=[]
     for i in range(0,len(rr),c['eval_batch']):
      batch=rr[i:i+c['eval_batch']]
      with torch.autocast('cuda',dtype=torch.bfloat16):x=model(caches[split][t].batch(batch,c['device']),banks[split].batch(batch,c['device']),t)
      out.append(x.float().cpu().numpy())
      if i%(100*c['eval_batch'])==0:budget.check()
     return np.concatenate(out)
    for t in ['A1','A4']:
     x=predict('cal_A',t);temps[t]=fit_temperature(x,[r['label'] for r in rows['cal_A'][t]]);x=predict('cal_B',t);p=softmax(x/temps[t],axis=1);scores=1-p[np.arange(len(p)),[r['label'] for r in rows['cal_B'][t]]];q=threshold(scores,.1);ths[t]=float(q) if np.isfinite(q) else None
    write_json(run.path/f'{key}_calibration.json',{'temperatures':temps,'Mondrian_qtype_alpha_0.1':ths,'null_means':'positive infinity'})
    for split in ['reader_val']+([] if smoke else ['test']):
     if split=='test':budget.check(test_opened=True)
     for t in ['A1','A4']:
      rr=rows[split][t];x=predict(split,t);p=softmax(x/temps[t],axis=1);stats=qa_summary(rr,p);stats.update(initialization=cp['selected']['initialization'],N=cp['selected']['N'],seed=cp['selected']['seed'],split=split,task=t);results.append(stats)
      records=[];q=float('inf') if ths[t] is None else ths[t]
      for r,prob in zip(rr,p):
       y=r['label'];records.append({k:r[k] for k in ['group_id','episode_id','message_id','category','template_id','label']}|{'accuracy':float(prob.argmax()==y),'nll':float(-np.log(max(prob[y],1e-12))),'coverage':bool(1-prob[y]<=q),'set_size':int(((1-prob)<=q).sum()),'probabilities':prob.tolist()})
      pq.write_table(pa.Table.from_pylist(records),run.path/f'{key}_{split}_{t}.parquet',compression='zstd');write_json(run.path/'results.json',results)
    del model;torch.cuda.empty_cache();print(key,'complete',flush=True)
   write_json(Path(c['output'])/'evaluated'/f'transfer_{c["mode"]}.json',{'run':str(run.path),'training':c['training_ref']});budget.check(status='completed',gate_passed=True,test_opened=not smoke)
  except BaseException:budget.check(status='failed');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
