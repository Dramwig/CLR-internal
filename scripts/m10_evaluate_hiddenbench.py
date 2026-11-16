"""Final-locked HiddenBench QA and group-calibrated mismatch detection."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer
from clr.audit.m10_hiddenbench import Bank,FactSerializer,OptionFeatures,build,read_raw,questions,visible_conclusion
from clr.audit.m7_evaluation import fit_temperature
from clr.data.m7 import threshold
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from clr.utils.m10_runtime import Budget
from scripts.m10_train_hiddenbench import infer

def dump(path,rows):
 with path.open('x') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
def grouped(rows):
 d=defaultdict(list)
 for r in rows:d[(r['group_id'],r['category'])].append(r['score'])
 return [{'group_id':g,'category':cat,'score':float(np.mean(v))} for (g,cat),v in d.items()]
def detectors(scores,thresholds):
 clean=[r['score'] for r in scores if r['category']=='clean'];bad=[r['score'] for r in scores if r['category']=='mismatch'];out={'AUROC':float(roc_auc_score([0]*len(clean)+[1]*len(bad),clean+bad)),'clean_groups':len(clean),'mismatch_groups':len(bad)}
 for a,t in thresholds.items():
  t=float('inf') if t is None else t;out['FPR_'+a]=float(np.mean(np.array(clean)>t));out['TPR_'+a]=float(np.mean(np.array(bad)>t))
 return out

def main(c):
 torch.set_num_threads(2);torch.cuda.set_device(c['device']);smoke=c['mode']=='smoke';ref=json.loads(Path(c.get('training_ref',str(Path(c['hidden_root'])/'training.json'))).read_text());tc=ref['config']
 if not smoke:
  lock=json.loads(Path(c['eval_lock']).read_text())
  if not lock['all_training_locked']:raise ValueError('Final training lock required')
  for cp in ref['checkpoints'].values():
   if hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()!=lock['checkpoint_sha256'].get(cp['path']):raise ValueError('Checkpoint lock mismatch')
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c)
  try:
   if not smoke:budget.check(test_opened=True)
   option=OptionFeatures(tc['tokenizer_path']);serializer=FactSerializer(AutoTokenizer.from_pretrained(tc['tokenizer_path'],local_files_only=True));banks={s:Bank(c,'reader_val' if smoke else s,templates=range(6),limit=1 if smoke else None) for s in ['cal_A','cal_B']+(['reader_val'] if smoke else ['reader_val','test'])};results=[];references={}
   # Match M71 norm/text baselines: unlabeled training norm moments and
   # cal_A logistic regression on visible committed/receiver choice fields.
   raw_items={str(e['id']):e for e in read_raw(c)};norms={};bundles={}
   for split,bank in banks.items():
    for mid in bank.z:
     p=Path(c['hidden_root'])/'source'/('reader_val' if smoke else split)/(mid+'.pt');b=torch.load(p,weights_only=True,map_location='cpu');bundles[split,mid]=b;norms[split,mid]=np.log(np.array(b['raw_norms']).flatten().clip(1e-12))
   fit=np.stack([norms['cal_A',mid] for mid in banks['cal_A'].z]) if smoke else np.stack([np.log(np.array(torch.load(p,weights_only=True,map_location='cpu')['raw_norms']).flatten().clip(1e-12)) for p in sorted((Path(c['hidden_root'])/'source/reader_train').glob('*.pt'))]);mu=fit.mean(0);sd=fit.std(0).clip(1e-4)
   def text_features(panel,cat,split):
    mid=panel['episode_id'] if cat=='clean' else panel['donor_id'];b=bundles[split,mid];output=b['receiver']['clean' if cat=='clean' else panel['episode_id']];answer=visible_conclusion(panel['commitment']);a=raw_items[panel['episode_id']]['possible_answers'].index(answer);pred=int(np.argmax(output));return np.r_[np.eye(4)[a],np.eye(4)[pred],np.eye(16)[a*4+pred],float(a==pred)]
   fit_panels=list({p['episode_id']:p for p in banks['cal_A'].entries}.values());tx=np.stack([text_features(panel,cat,'cal_A') for panel in fit_panels for cat in ['clean','mismatch']]);ty=np.array([int(cat!='clean') for panel in fit_panels for cat in ['clean','mismatch']]);text_model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=11).fit(tx,ty)
   for kind in tc['kinds']:
    for seed in tc['seeds']:
     models={}
     for cp in ref['checkpoints'].values():
      if cp['kind']!=kind or cp['seed']!=seed:continue
      model=build(kind).to(c['device']);model.load_state_dict(torch.load(cp['path'],weights_only=True,map_location=c['device']));model.eval()
      for task in cp['tasks']:models[task]=model
     def predict(bank):
      rr=bank.rows;raw={}
      for task,m in models.items():
       ii=[i for i,r in enumerate(rr) if r['task']==task];x=infer(m,kind,option,serializer,bank,[rr[i] for i in ii],c).numpy();raw.update(zip(ii,x));budget.check()
      return [raw[i] for i in range(len(rr))]
     ca=banks['cal_A'];raw=predict(ca);temps={t:fit_temperature([x for r,x in zip(ca.rows,raw) if r['task']==t],[r['label'] for r in ca.rows if r['task']==t]) for t in models};th={}
     for split,bank in banks.items():
      raw=predict(bank);qa=[];scores=[]
      # Fixed question distribution, public commitment expected labels only.
      chosen=defaultdict(list)
      for r,x in zip(bank.rows,raw):
       p=softmax(x/temps[r['task']]);qa.append({k:r[k] for k in ['group_id','episode_id','panel_id','message_id','category','task','template_id','label','expected_label']}|{'probabilities':p.tolist(),'accuracy':float(p.argmax()==r['label']),'nll':float(-np.log(max(p[r['label']],1e-12)))})
       chosen[(r['panel_id'],r['category'])].append((hashlib.sha256(f'{seed}:{r["template_id"]}:{r["task"]}:{r.get("query_fact","")}'.encode()).hexdigest(),r,float(-np.log(max(p[r['expected_label']],1e-12)))))
      for vs in chosen.values():
       sample=sorted(vs,key=lambda a:a[0])[:16];scores.append({'group_id':sample[0][1]['group_id'],'category':sample[0][1]['category'],'score':float(np.mean([v[2] for v in sample]))})
      scores=grouped(scores)
      if split=='cal_B':
       th={str(a):threshold([r['score'] for r in scores if r['category']=='clean'],a) for a in [.05,.01]};th={a:float(t) if np.isfinite(t) else None for a,t in th.items()}
      if split in ['reader_val','test']:
       dump(run.path/f'{kind}_{seed}_{split}_qa.jsonl',qa);dump(run.path/f'{kind}_{seed}_{split}_scores.jsonl',scores)
       results.append({'kind':kind,'seed':seed,'split':split,'QA':{t:{'accuracy':float(np.mean([r['accuracy'] for r in qa if r['task']==t])),'NLL':float(np.mean([r['nll'] for r in qa if r['task']==t]))} for t in models},'detection':detectors(scores,th)})
       budget.check(test_opened=split=='test')
     references[f'{kind}_{seed}']={'temperatures':temps,'thresholds':th,'expectations':'visible commitment audited fact + one-of-four panel constraint','calibration_unit':'independent public-description group'};del models;torch.cuda.empty_cache()
   for kind in ['norm','text']:
    th={}
    for split,bank in banks.items():
     ss=[]
     for p in bank.entries:
      for cat in ['clean','mismatch']:
       mid=p['episode_id'] if cat=='clean' else p['donor_id'];b=bundles[split,mid]
       if kind=='norm':score=float(np.mean(((norms[split,mid]-mu)/sd)**2))
       else:
        # The original answer is explicitly visible in the commitment. The
        # output vector is indexed by the visible receiver choice names.
        score=float(text_model.predict_proba(text_features(p,cat,split)[None])[0,1])
       ss.append({'group_id':p['group_id'],'category':cat,'score':score})
     ss=grouped(ss)
     if split=='cal_B':
      th={str(a):threshold([r['score'] for r in ss if r['category']=='clean'],a) for a in [.05,.01]};th={a:float(t) if np.isfinite(t) else None for a,t in th.items()}
     if split in ['reader_val','test']:
      dump(run.path/f'{kind}_{split}_scores.jsonl',ss);results.append({'kind':kind,'seed':None,'split':split,'detection':detectors(ss,th)})
    references[kind]={'thresholds':th,'parameters':{'weight':text_model.coef_[0].tolist(),'bias':float(text_model.intercept_[0])} if kind=='text' else {'mean':mu.tolist(),'scale':sd.tolist()},'fit_split':'cal_A' if kind=='text' else 'reader_train' if not smoke else 'val_proxy'}
   write_json(run.path/'calibration.json',references);write_json(run.path/'results.json',results);write_json(Path(c['hidden_root'])/('evaluation_smoke.json' if smoke else 'evaluation.json'),{'run':str(run.path),'test_opened':not smoke});budget.check(status='completed',test_opened=not smoke,gate_passed=True)
  except BaseException:budget.check(status='failed');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
