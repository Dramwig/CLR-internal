"""Group bootstrap for locked M10 audit predictions, conditional on fixed seeds."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from clr.audit.m71_statistics import auc_group_draws
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def jsrows(p):return [json.loads(l) for l in Path(p).open()]
def ci(point,draws):
 lo,hi=np.quantile(draws,[.025,.975]);return dict(mean=float(point),lo=float(lo),hi=float(hi))
def weights(n):return np.random.default_rng(11).multinomial(n,np.full(n,1/n),size=10000).astype(np.float32)
def group_summary(frame,column):
 v=frame.groupby(['group_id','seed'])[column].mean().groupby('group_id').mean().sort_index().to_numpy();w=weights(len(v));return dict(groups=len(v),**ci(v.mean(),w@v/len(v)))
def detection_summary(frames,thresholds,key,category,alpha):
 groups=sorted(set(frames[0].group_id));idx={g:i for i,g in enumerate(groups)};w=weights(len(groups));dd=[];ff=[];aa=[]
 for f,threshold in zip(frames,thresholds):
  t=threshold[str(alpha)];t=float('inf') if t is None else t
  clean=f[f.category=='clean'].set_index('group_id').loc[groups];bad=f[f.category==category].set_index('group_id').loc[groups];d=(bad[key].to_numpy()>t).astype(float);false=(clean[key].to_numpy()>t).astype(float);dd.append(d);ff.append(false)
  yy=np.r_[np.zeros(len(groups),int),np.ones(len(groups),int)];scores=np.r_[clean[key],bad[key]];gi=np.tile(np.arange(len(groups)),2);point=auc_group_draws(yy,scores,gi,np.ones((1,len(groups))))[0];draw=auc_group_draws(yy,scores,gi,w);aa.append((point,draw))
 d=np.mean(dd,axis=0);f=np.mean(ff,axis=0);a=np.mean([v[0] for v in aa]);ad=np.mean([v[1] for v in aa],axis=0);out=ci(d.mean(),w@d/len(groups));fci=ci(f.mean(),w@f/len(groups));aci=ci(a,ad)
 return dict(groups=len(groups),**out,FPR=fci['mean'],FPR_lo=fci['lo'],FPR_hi=fci['hi'],AUROC=aci['mean'],AUROC_lo=aci['lo'],AUROC_hi=aci['hi'])

def main(c):
 root=Path(c['output']);dest=root/'paper_inputs';dest.mkdir(exist_ok=True);sources={}
 def remember(p):
  p=Path(p);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
 detection=pd.read_csv(remember('data/M9_v1/paper_inputs/communication_detection.csv'));qa=pd.read_csv(remember('data/M9_v1/paper_inputs/communication_qa.csv'));ds=[];qs=[];auditruns=[];paired=[]
 for seed in [11,22,33]:
  p=root/'evaluated'/f'audit_B6_h896_{seed}_full.json';ref=json.loads(remember(p).read_text());auditruns.append(Path(ref['run']))
 for split in ['reader_val','test']:
  ff=[pd.DataFrame(jsrows(remember(run/f'{split}_scores.jsonl'))) for run in auditruns];tt=[json.loads(remember(run/'calibration.json').read_text())['detection_thresholds'] for run in auditruns]
  for key in ['1','4','16']:
   for category in sorted(set(ff[0].category)-{'clean'}):
    for a in [.01,.05]:ds.append(dict(panel='full',split=split,kind='B6',k=int(key),alpha=a,category=category,**detection_summary(ff,[t[key] for t in tt],key,category,a)))
  rr=[]
  for seed,run in zip([11,22,33],auditruns):
   for r in jsrows(remember(run/f'{split}_qa.jsonl')):
    p=np.array(r['probabilities']);y=r['label'];rr.append({'group_id':r['group_id'],'seed':seed,'task':r['task'],'templates':'seen' if r['template_id']<4 else 'heldout','correct':float(p.argmax()==y),'nll':float(-np.log(max(p[y],1e-12)))})
  frame=pd.DataFrame(rr)
  for (task,templates),f in frame.groupby(['task','templates']):
   for metric in ['correct','nll']:qs.append(dict(split=split,kind='B6',task=task,templates=templates,metric=metric,**group_summary(f,metric)))
 # Paired comparison with frozen CLR predictions; no new CLR inference.
  from clr.audit.m71_storage import load_bundle
  oldruns=[];oldths=[]
  for seed in [11,22,33]:
   r=json.loads(remember(f'data/M71_v1/pool/evaluated/CLR_{seed}.json').read_text());oldruns.append(Path(r['run']));oldths.append(json.loads(remember(f'data/M71_v1/pool/calibration/CLR_{seed}.json').read_text())['thresholds']['16'])
  for task in ['A1','A4']:
   old=[]
   for seed,run in zip([11,22,33],oldruns):
    b=load_bundle(remember(run/f'{split}_{task}_seen.pt.gz'))
    for r,p in zip(b['rows'],b['probs'].numpy()):old.append({'group_id':r['group_id'],'seed':seed,'correct':float(p.argmax()==r['label'])})
   a=pd.DataFrame(old).groupby(['group_id','seed']).correct.mean();b=frame[(frame.task==task)&(frame.templates=='seen')].groupby(['group_id','seed']).correct.mean();d=(a-b).rename('delta').reset_index();paired.append(dict(split=split,model='CLR',baseline='B6',metric=task+'_accuracy',**group_summary(d,'delta')))
  for alpha in [.01,.05]:
   diffs=[]
   for seed,oldrun,newframe,oldt,newt in zip([11,22,33],oldruns,ff,oldths,tt):
    a=pd.DataFrame(jsrows(remember(oldrun/f'{split}_scores.jsonl')));a=a[a.category=='counterfactual'].set_index('group_id')['16'];b=newframe[newframe.category=='counterfactual'].set_index('group_id')['16'];ta=oldt[str(alpha)];tb=newt['16'][str(alpha)];ta=float('inf') if ta is None else ta;tb=float('inf') if tb is None else tb;d=((a>ta).astype(float)-(b>tb).astype(float)).rename('delta').reset_index();d['seed']=seed;diffs.append(d)
   paired.append(dict(split=split,model='CLR',baseline='B6',metric=f'CF_detection_{alpha}',**group_summary(pd.concat(diffs),'delta')))
 pd.DataFrame(paired).to_csv(dest/'M10_audit_paired.csv',index=False,float_format='%.15g')
 pd.concat([detection,pd.DataFrame(ds)],ignore_index=True).to_csv(dest/'communication_detection.csv',index=False,float_format='%.15g');pd.concat([qa,pd.DataFrame(qs)],ignore_index=True).to_csv(dest/'communication_qa.csv',index=False,float_format='%.15g')
 href=json.loads(remember(root/'hiddenbench/evaluation.json').read_text());hrun=Path(href['run']);cal=json.loads(remember(hrun/'calibration.json').read_text());hidden=[]
 for split in ['reader_val','test']:
  for kind in ['CLR','B3','B6','norm','text']:
   seeds=[11,22,33] if kind in ['CLR','B3','B6'] else [None];frames=[];ths=[];qq=[]
   for seed in seeds:
    prefix=f'{kind}_{seed}' if seed is not None else kind;frames.append(pd.DataFrame(jsrows(remember(hrun/f'{prefix}_{split}_scores.jsonl'))));ths.append(cal[prefix]['thresholds'])
    if seed is not None:
     for r in jsrows(remember(hrun/f'{prefix}_{split}_qa.jsonl')):qq.append({**r,'seed':seed})
   for a in [.01,.05]:
    d=detection_summary(frames,ths,'score','mismatch',a);hidden.append(dict(split=split,kind=kind,metric=f'detect_{a}',groups=d['groups'],**{k:d[k] for k in ['mean','lo','hi']}))
    if a==.05:
     for metric,prefix in [('AUROC','AUROC'),('FPR_0.05','FPR')]:hidden.append(dict(split=split,kind=kind,metric=metric,groups=d['groups'],mean=d[prefix],lo=d[prefix+'_lo'],hi=d[prefix+'_hi']))
   if qq:
    for task,g in pd.DataFrame(qq).groupby('task'):hidden.append(dict(split=split,kind=kind,metric=task+'_accuracy',**group_summary(g,'accuracy')))
 pd.DataFrame(hidden).to_csv(dest/'hiddenbench.csv',index=False,float_format='%.15g');write_json(root/'audit_export_sources.json',{'sha256':sources})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
