"""CPU-only paired B5c summaries from three final-locked prediction runs."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from clr.utils.idea_a import write_json
from clr.paper.statistics import metric_summary
from clr.calib.m4_evaluation import unique_indices
from scripts.m9_export import Export

def main(c):
 root=Path(c['output']);ex=Export(c);points=[];cp=[]
 for seed in ([11] if c.get('mode')=='smoke' else [11,22,33]):
  ref=ex.js(('data/M10b_v1/evaluated/B5c_gpt2_11_smoke.json' if c.get('mode')=='smoke' else root/'evaluated'/f'B5c_gpt2_{seed}_full.json'));offset=len(cp);r={'run':ref['run'],'source':'gpt2','model':'B5c','seed':seed,'native':True}
  for split in (['reader_val'] if c.get('mode')=='smoke' else ['reader_val','test']):
   for q in ['Q3','Q4']:
    raw,p,y,f=ex.predict_rows(r,q,split);common=dict(source='gpt2',model='B5c',seed=seed,split=split,qtype=q)
    g=f.groupby(['problem_id','t'])[['accuracy','nll']].mean();g['instances']=f.groupby(['problem_id','t']).size();points.append(g.reset_index().assign(**common))
    conformal=ex.js(Path(r['run'])/'conformal.json');ths={a['t']:a['threshold'] if a['threshold'] is not None else np.inf for a in conformal if a['split']==split and a['qtype']==q and not a['heldout'] and a['alpha']==.1 and a['calibration']=='temperature'}
    for t,thr in ths.items():
     jj=unique_indices(raw,[i for i,a in enumerate(raw) if a['state_ref']['t']==t],q,seed);sets=1-p[jj]<=thr
     for i,ss in zip(jj,sets):cp.append({**common,'problem_id':raw[i]['problem_id'],'t':t,'instances':1,'coverage':float((ss*y[i]).sum()),'size':float(ss.sum())})
  ex.validate_cp(r,cp[offset:])
 frame=pd.concat(points,ignore_index=True);ex.save('B5c_problem_metrics',frame);summary=[];paired=[]
 old=ex.csv('data/M10_v1/paper_inputs/main_problem_metrics.csv');old=old[(old.source=='gpt2')&old.model.isin(['CLR-lite','B0'])&old.qtype.isin(['Q3','Q4'])]
 for (split,q),g in frame.groupby(['split','qtype']):
  co=dict(source='gpt2',model='B5c',split=split,qtype=q);summary.extend({**co,**a} for a in metric_summary(g,['nll','accuracy']))
  for baseline in ['CLR-lite','B0']:
   b=old[(old.split==split)&(old.qtype==q)&(old.model==baseline)]
   if c.get('mode')=='smoke':b=b[b.problem_id.isin(g.problem_id)&b.seed.isin(g.seed)]
   d=b.merge(g,on=['problem_id','t','seed'],suffixes=('_a','_b'),validate='one_to_one')
   if len(d)!=len(g) or len(d)!=len(b):raise ValueError('Paired QA populations differ')
   d['instances']=d.instances_b;d['delta']=d.accuracy_a-d.accuracy_b
   e=metric_summary(d,['delta'])[0];e['metric']='accuracy_difference';paired.append(dict(source='gpt2',model=baseline,baseline='B5c',split=split,qtype=q,**e))
   if baseline=='B0' and q=='Q3':summary.append(dict(**co,metric='Q3_minus_B0',mean=-e['mean'],lo=-e['hi'],hi=-e['lo']))
 ex.save('B5c_summary',summary);ex.save('CLR_minus_B5c',paired)
 main=ex.csv('data/M10_v1/paper_inputs/main.csv');ex.save('main',pd.concat([main,pd.DataFrame(summary)],ignore_index=True))
 cpf=pd.DataFrame(cp);ex.save('B5c_conformal_problem',cpf);cs=[]
 for key,g in cpf.groupby(['source','model','split','qtype']):cs.extend({**dict(zip(['source','model','split','qtype'],key)),**e} for e in metric_summary(g,['coverage','size']))
 ex.save('B5c_conformal',cs);write_json(root/'export_sources.json',{'sha256':ex.used})
