"""Locked B6 auditing with the frozen M7.1 public-only consistency protocol."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from clr.audit.m71_reader import Bank
from clr.audit.m71_evaluation import calibration_rows,views,consistency_rows,calibration_thresholds,detection
from clr.audit.m7_evaluation import fit_temperature,qa_summary
from clr.models.m10 import AttentionProbe,OptionFeatures
from clr.data.m7 import threshold
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import Budget
from scripts.m10_train_audit_b6 import infer

def subset(bank,lo,hi):
 result=copy.copy(bank);result.entries=bank.entries[lo:hi];ids={e['id'] for e in result.entries};result.rows=[r for r in bank.rows if r['episode_id'] in ids];return result

def dump(path,rows):
 with path.open('x') as f:
  for r in rows:f.write(json.dumps(r)+'\n')

@torch.no_grad()
def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);ref=json.loads(Path(c['training_ref']).read_text());tc=ref['config'];smoke=c['mode']=='smoke'
 if not smoke:
  lock=json.loads(Path(c['eval_lock']).read_text())
  if not lock['all_training_locked']:raise ValueError('Final training lock required')
  for cp in ref['checkpoints'].values():
   if hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()!=lock['checkpoint_sha256'].get(cp['path']):raise ValueError('Checkpoint not locked')
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   models={}
   for task,cp in ref['checkpoints'].items():
    m=AttentionProbe(512,hidden=tc['hidden'],audit=True);m.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=True));models[task]=m.to(c['device']).eval()
   option=OptionFeatures(tc['tokenizer_path'])
   def predict(bank,rr):
    out={}
    for task in models:
     ii=[i for i,r in enumerate(rr) if r['task']==task]
     if not ii:continue
     x=infer(models[task],option,bank,[rr[i] for i in ii],c).numpy()
     for i,v in zip(ii,x):out[i]=v
     budget.check()
    return [out[i] for i in range(len(rr))]
   proxy=Bank(tc['data_root'],'reader_val',limit=30) if smoke else None
   ca=subset(proxy,0,10) if smoke else Bank(tc['data_root'],'cal_A')
   rr=calibration_rows(ca);raw=predict(ca,rr)
   temp={t:fit_temperature([x for r,x in zip(rr,raw) if r['task']==t],[r['label'] for r in rr if r['task']==t]) for t in models}
   del ca
   def scores(bank):
    vv=views(bank);result=[]
    for start in range(0,len(vv),64):
     chunk=vv[start:start+64];rr=[r for v in chunk for r in consistency_rows(v,c['seed'])];raw=predict(bank,rr)
     for i,v in enumerate(chunk):
      logp=[-np.log(max(float(softmax(x/temp[r['task']])[r['label']]),1e-12)) for r,x in zip(rr[i*16:(i+1)*16],raw[i*16:(i+1)*16])]
      result.append({'id':v['entry']['id'],'group_id':v['entry']['group_id'],'category':v['category'],**{str(k):float(np.mean(logp[:k])) for k in [1,4,16]}})
    return result
   cb=subset(proxy,10,20) if smoke else Bank(tc['data_root'],'cal_B');cs=scores(cb);th=calibration_thresholds(cs,['1','4','16']);dump(run.path/'cal_B_scores.jsonl',cs)
   th={k:{a:float(v) if np.isfinite(v) else None for a,v in aa.items()} for k,aa in th.items()}
   rr=calibration_rows(cb);raw=predict(cb,rr);qa_th={t:{str(a):threshold([1-float(softmax(x/temp[t])[r['label']]) for r,x in zip(rr,raw) if r['task']==t],a) for a in [.1]} for t in models}
   qa_th={k:{a:float(v) if np.isfinite(v) else None for a,v in aa.items()} for k,aa in qa_th.items()}
   write_json(run.path/'calibration.json',{'temperatures':temp,'detection_thresholds':th,'QA_thresholds':qa_th,'null_threshold_means':'positive infinity; detection rejects nothing, QA includes all options','expectations':'frozen M8 P0-verified public T,z_pub and programmatic commitment only'})
   del cb
   for split in (['reader_val_proxy'] if smoke else ['reader_val','test']):
    if split=='test':budget.check(test_opened=True)
    bank=subset(proxy,20,30) if smoke else Bank(tc['data_root'],split);rr=bank.rows;raw=predict(bank,rr);qa={};records=[]
    for task in models:
     rows=[r for r in rr if r['task']==task];p=np.stack([softmax(x/temp[task]) for r,x in zip(rr,raw) if r['task']==task]);qa[task]=qa_summary(rows,p)
    for r,x in zip(rr,raw):records.append({**r,'logits':x.tolist(),'probabilities':softmax(x/temp[r['task']]).tolist()})
    dump(run.path/f'{split}_qa.jsonl',records);write_json(run.path/f'{split}_QA.json',qa)
    ss=scores(bank);dump(run.path/f'{split}_scores.jsonl',ss);write_json(run.path/f'{split}_detection.json',detection(ss,th,['1','4','16']));print(split,'completed',flush=True)
   folder=Path(c['output'])/'evaluated';folder.mkdir(exist_ok=True);write_json(folder/f'{tc.get("case","audit_B6")}_{c["seed"]}_{c["mode"]}.json',{'run':str(run.path),'training':c['training_ref']})
   budget.check(status='completed',test_opened=not smoke,gate_passed=True)
  except BaseException:
   budget.check(status='failed');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
