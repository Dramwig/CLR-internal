"""CPU-only exports of frozen evidence; predictions are never recomputed."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib,pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from scipy.special import softmax
from clr.calib.m4 import apply
from clr.calib.m4_evaluation import unique_indices,target_array
from clr.paper.statistics import metric_summary,auc_summary
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m9_runtime import snapshot,hours

class Export:
 def __init__(self,c):self.c=c;self.root=Path(c['output']);self.dest=self.root/'paper_inputs';self.dest.mkdir(exist_ok=True);self.used={};self.qa_metadata={};self.cp_checks=[]
 def remember(self,p):
  p=Path(p)
  if str(p) in self.used:return p
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  self.used[str(p)]=h.hexdigest();return p
 def js(self,p):return json.loads(self.remember(p).read_text())
 def csv(self,p):return pd.read_csv(self.remember(p))
 def pq(self,p,**kw):return pq.read_table(self.remember(p),**kw)
 def save(self,name,rows):
  df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);p=self.dest/(name+'.csv')
  for manifest in self.dest.glob('*freeze*.json'):
   locked=json.loads(manifest.read_text()).get('sha256',{})
   if str(p) in locked:
    payload=df.to_csv(index=False,float_format='%.15g').encode()
    if hashlib.sha256(payload).hexdigest()!=locked[str(p)]:raise ValueError('Frozen derived CSV differs; use a new version: '+str(p))
    return
  df.to_csv(p,index=False,float_format='%.15g')
 def registry(self,include_new):
  idx=self.js('data/M6_v1/evaluation_index.json');regs=[]
  for source in ['gpt2','llama']:
   for kind,label in [('b3','B3'),('lite','CLR-lite')]:
    for seed in [11,22,33]:regs.append(dict(source=source,model=label,seed=seed,run=idx[f'single_{source}_{kind}_{seed}'][source],native=True))
  m41=self.js('data/M41_v1/evaluation_complete.json')['runs'];m5=self.js('data/M5b_v1/completion.json')['A']['runs']
  for seed in [11,22,33]:
   regs.append(dict(source='gpt2',model='B0',seed=seed,run=m41[f'B0_{seed}'],native=False))
   for model in ['B3-prefix','CLR-lite-single']:regs.append(dict(source='gpt2',model=model,seed=seed,run=m5[f'{model}_{seed}'],native=False))
  if include_new:
   idx=self.js(self.root/'evaluation_index.json')
   for seed in [11,22,33]:
    for s in ['gpt2','llama']:regs.append(dict(source=s,model='CLR-lite-multi',seed=seed,run=idx[f'multi_lite_{seed}'][s],native=True))
  return regs
 def predict_rows(self,r,q,split,kind='clean'):
  path=Path(r['run'])/(f'{r["source"]}_{q}_{split}.parquet' if r['native'] else f'{q}_{split}_False.parquet')
  raw=self.pq(path).to_pylist();cal=self.js(Path(r['run'])/'calibration.json')[q]['temperature']
  raw=[x for x in raw if (x['state_ref']['kind'] if r['native'] else x['kind'])==kind]
  if not raw:return [],None,None,None
  x=np.array([x.get('calibrated_logits',x['logits']) for x in raw]);x=x if r['native'] else apply(x,cal)
  p=softmax(x,axis=1);y=np.eye(p.shape[1])[[a['label'] for a in raw]] if q=='Q3' else np.array([[1-a['label_soft'],a['label_soft']] for a in raw])
  frame=pd.DataFrame({'problem_id':[x['problem_id'] for x in raw],'t':[x['state_ref']['t'] if r['native'] else x['t'] for x in raw],
    'accuracy':y[np.arange(len(p)),p.argmax(1)],'nll':-(y*np.log(p.clip(1e-12))).sum(1),'label':[x['label'] for x in raw], 'score':p[:,1]})
  return raw,p,y,frame
 def metadata_rows(self,raw,q,split):
  key=split,q
  def identity(a):
   ref=a.get('state_ref',a)
   return a['problem_id'],ref['t'],ref['traj_id'],a['template_id'],a['replicate']
  if key not in self.qa_metadata:
   rs=self.pq(f'data/M23_v1a/full/{split}.parquet',columns=['problem_id','state_ref','qtype','template_id','options','replicate'],filters=[('qtype','=',q),('is_heldout_template','=',False)]).to_pylist()
   lookup={}
   for a in rs:
    k=identity(a)
    if k in lookup and lookup[k]!=a['options']:raise ValueError('Ambiguous saved QA metadata key')
    lookup[k]=a['options']
   self.qa_metadata[key]=lookup
  return [{**a,'state_ref':{'t':a['t'],'kind':a['kind'],'traj_id':a['traj_id']},'options':self.qa_metadata[key][identity(a)]} for a in raw]
 def validate_cp(self,r,values):
  f=pd.DataFrame(values)
  if f.empty:raise ValueError('No reconstructed conformal observations')
  stored=self.js(Path(r['run'])/'conformal.json')
  for a in stored:
   if a['heldout'] or a['alpha']!=.1 or a.get('calibration','temperature')!='temperature' or a.get('subset','all')!='all':continue
   g=f[(f.split==a['split'])&(f.qtype==a['qtype'])&(f.t==a['t'])]
   if g.empty:continue
   errors={m:abs(float(g[m].mean())-a[m]) for m in ['coverage','size']}
   if max(errors.values())>1e-7:raise ValueError('Reconstructed conformal differs from frozen result: '+str((r,a,errors)))
   self.cp_checks.append(dict(source=r['source'],model=r['model'],seed=r['seed'],split=a['split'],qtype=a['qtype'],t=a['t'],**{m+'_error':v for m,v in errors.items()}))
 def main_tables(self,include_new):
  registry=self.registry(include_new);points=[];aucs=[];cp=[];q1=[]
  for r in registry:
   cache=self.root/'paper_cache'/f'{r["source"]}_{r["model"]}_{r["seed"]}.pkl'
   if cache.exists():
    with cache.open('rb') as stream:cached=pickle.load(stream)
    for path,h in cached['sources'].items():
     self.remember(path)
     if self.used[path]!=h:raise ValueError('Cached source hash differs: '+path)
    points.extend(cached['points']);aucs.extend(cached['aucs']);cp.extend(cached['cp']);q1.extend(cached['q1']);self.validate_cp(r,cached['cp'])
    print('cached model',r['source'],r['model'],r['seed'],flush=True);continue
   lengths=[len(x) for x in [points,aucs,cp,q1]]
   for split in ['reader_val','test']:
    for q in ['Q1','Q2','Q3','Q4']:
     raw,p,y,f=self.predict_rows(r,q,split);common={k:r[k] for k in ['source','model','seed']};common.update(split=split,qtype=q)
     g=f.groupby(['problem_id','t'])[['accuracy','nll']].mean();g['instances']=f.groupby(['problem_id','t']).size();g=g.reset_index()
     points.append(g.assign(**common))
     if q=='Q1' and split=='reader_val':
      q1.append(g.assign(**common,kind='clean'))
      rr,pp,yy,ff=self.predict_rows(r,q,split,'traj')
      if rr:
       gg=ff.groupby(['problem_id','t'])[['accuracy','nll']].mean();gg['instances']=ff.groupby(['problem_id','t']).size()
       q1.append(gg.reset_index().assign(**common,kind='traj'))
     if q=='Q2':aucs.append(f.assign(**common))
     conformal=self.js(Path(r['run'])/'conformal.json');ths={a['t']:a['threshold'] if a['threshold'] is not None else np.inf for a in conformal if a['split']==split and a['qtype']==q and not a['heldout'] and a['alpha']==.1 and a.get('calibration','temperature')=='temperature' and a.get('subset','all')=='all'}
     normalized=raw if r['native'] else self.metadata_rows(raw,q,split)
     for t,thr in ths.items():
      jj=unique_indices(normalized,[i for i,a in enumerate(normalized) if a['state_ref']['t']==t],q,r['seed']);sets=1-p[jj]<=thr
      for i,ss in zip(jj,sets):cp.append({**common,'problem_id':raw[i]['problem_id'],'t':t,'instances':1,'coverage':float((ss*y[i]).sum()),'size':float(ss.sum())})
   print('export model',r['source'],r['model'],r['seed'],flush=True)
   self.validate_cp(r,cp[lengths[2]:])
   cache.parent.mkdir(exist_ok=True)
   sources={p:h for p,h in self.used.items() if Path(r['run']).resolve() in Path(p).resolve().parents or p.startswith('data/M23_v1a/full/')}
   with cache.open('wb') as stream:pickle.dump(dict(points=points[lengths[0]:],aucs=aucs[lengths[1]:],cp=cp[lengths[2]:],q1=q1[lengths[3]:],sources=sources),stream)
  frame=pd.concat(points,ignore_index=True);self.save('main_problem_metrics',frame);result=[]
  for key,g in frame.groupby(['source','model','split','qtype']):
   co=dict(zip(['source','model','split','qtype'],key));result.extend({**co,**a} for a in metric_summary(g,['nll','accuracy']))
  af=pd.concat(aucs,ignore_index=True)
  def auc_job(item):
   key,g=item
   value=dict(zip(['source','model','split'],key),qtype='Q2',**auc_summary(g))
   print('AUROC bootstrap',key,flush=True)
   return value
  with ThreadPoolExecutor(max_workers=self.c.get('auc_workers',1)) as pool:
   result.extend(pool.map(auc_job,af.groupby(['source','model','split'])))
  baseline=frame[(frame.model=='B0')&(frame.qtype=='Q3')]
  for key,g in frame[frame.qtype=='Q3'].groupby(['source','model','split']):
   b=baseline[(baseline.source==key[0])&(baseline.split==key[2])]
   if b.empty:continue
   d=g.merge(b,on=['problem_id','t','seed','source','split','qtype'],suffixes=('_a','_b'),validate='one_to_one');d['delta']=d.accuracy_a-d.accuracy_b;d['instances']=d.instances_a
   e=metric_summary(d,['delta'])[0];e['metric']='Q3_minus_B0';result.append(dict(zip(['source','model','split'],key),qtype='Q3',**e))
  self.save('main',result);cpf=pd.DataFrame(cp);self.save('conformal_problem',cpf);cs=[];paired=[]
  q1f=pd.concat(q1,ignore_index=True);q1s=[]
  for key,g in q1f.groupby(['source','model','split','qtype','kind']):q1s.extend({**dict(zip(['source','model','split','qtype','kind'],key)),**e} for e in metric_summary(g,['nll','accuracy']))
  self.save('reader_val_q1_clean_traj',q1s)
  for key,g in cpf.groupby(['source','model','split','qtype']):cs.extend({**dict(zip(['source','model','split','qtype'],key)),**e} for e in metric_summary(g,['coverage','size']))
  for (s,split,q),g in cpf.groupby(['source','split','qtype']):
   for model,baseline in [('CLR-lite','B3'),('CLR-lite-multi','B3'),('CLR-lite-multi','CLR-lite')]:
    base=g[g.model==baseline]
    a=g[g.model==model]
    if a.empty:continue
    d=a.merge(base,on=['problem_id','t','seed'],suffixes=('_a','_b'),validate='one_to_one');d['delta']=d.size_a-d.size_b;d['instances']=1
    e=metric_summary(d,['delta'])[0];paired.append(dict(source=s,split=split,qtype=q,model=model,baseline=baseline,**e))
  self.save('conformal',cs);self.save('conformal_paired',paired);self.save('conformal_replay_checks',self.cp_checks)
  self.save('equal_coverage_paired',self.csv('data/M5b_v1/analysis/A_matched_coverage_paired.csv'))
 def static(self):
  self.save('type_transfer',self.csv('reports/M41_aggregate.csv'));self.save('source_transfer',self.csv('data/M6_v1/analysis/metrics.csv'))
  self.save('source_transfer_paired',self.csv('data/M6_v1/analysis/transfer_paired.csv'));self.save('shift',self.csv('data/M6_v1/shift/summary.csv'))
  early=[]
  for s,root in [('gpt2','data/dumps/codi_gpt2_v3'),('llama','data/dumps/codi_llama_M6_v1')]:
   for split in ['reader_val','test']:
    f=self.pq(Path(root)/split/'early_exit.parquet').to_pandas();f=f.rename(columns={'correct':'accuracy'});f['instances']=1
    for t,g in f.groupby('t'):
     for e in metric_summary(g,['accuracy']):early.append(dict(source=s,split=split,t=t,**e))
  self.save('early_exit',early);self.save('ranking',self.csv('data/M5b_v1/analysis/B_summary.csv'));self.save('search',self.csv('data/M5b_v1/analysis/C_budget_curves.csv'));self.save('search_settings',self.csv('data/M5b_v1/analysis/C_summary.csv'));self.save('small_sigma',self.csv('data/M6_v1/rq1/summary.csv'))
  self.save('communication_detection',self.csv('data/M71_v1/analysis_v3/detection_group_bootstrap.csv'));self.save('communication_qa',self.csv('data/M71_v1/analysis_v3/qa_group_bootstrap.csv'));self.save('localization',self.csv('data/M8a_v1/analysis_v1/P0_QA.csv'))
  self.save('weak_D0',self.csv('data/M8a_v1/analysis_v1/robustness_summary.csv'));self.save('weak_designs',self.csv('data/M8b_v1/analysis_v1/robustness_summary.csv'));self.save('dependent',self.csv('data/M8c_v1/analysis_final21/robustness_summary.csv'))
  self.save('noise_llama',self.csv('data/M6_v1/noise/eight_point_summary_ci.csv'))
  # Cost rows come from the closed per-milestone ledgers, retaining missing-count flags.
  specifications=[('M4.1','reports/M41_costs.csv','gpu_hours'),('M5a','data/M5a_v13/analysis/cost_ledger.csv','gpu_hours_conservative_charge'),('M5b','data/M5b_v1/analysis/costs.csv','gpu_hours'),('M6','data/M6_v1/analysis/costs.csv','gpu_hours'),('M7','data/M7_v1/analysis/costs.csv','gpu_hours'),('M7.1','data/M71_v1/analysis_v3/costs.csv','gpu_hours'),('M8a','data/M8a_v1/analysis_v1/costs.csv','GPUh'),('M8b','data/M8b_v1/analysis_v1/costs.csv','gpu_hours'),('M8c','data/M8c_v1/analysis_final21/costs.csv','gpu_hours')]
  costs=[]
  for milestone,path,field in specifications:
   f=self.csv(path);calls=pd.to_numeric(f.get('source_forward_calls',pd.Series(np.nan,index=f.index)),errors='coerce');costs.append(dict(milestone=milestone,gpu_hours=f[field].sum(),source_forward_calls=calls.sum(min_count=1),missing_source_counts=int(calls.isna().sum()),cost_scope='sum of run wall time; see original missing/upper-bound notes',evidence=path))
  # Exact closed totals can include conservative additions absent from row ledgers.
  for row in costs:
   name={'M4.1':'M41','M5a':'M5a','M5b':'M5b','M6':'M6','M7':'M7','M7.1':'M71','M8a':'M8a','M8b':'M8b','M8c':'M8c'}[row['milestone']]
   roots={'M5a':'M5a_v13','M41':'M41_v1'};p=Path('data')/roots.get(name,name+'_v1')/'freeze.json';d=self.js(p)
   total=d.get('gpu_hours_conservative',d.get('cost_gpu_hours_conservative',d.get('gpu_hours',d.get('report',{}).get('total_gpu_hours'))))
   if total is not None:row['gpu_hours']=total
  costs.append(dict(milestone='M9-A',gpu_hours=sum(x['gpu_hours'] for x in hours()),source_forward_calls=0,missing_source_counts=0,cost_scope='includes M9-A smoke/failures/calibration; CPU paper work=0',evidence='data/M9_v1/experiment_complete.json'))
  from scripts.m9_appendix import export_appendices,export_legacy_costs
  export_appendices(self);costs=export_legacy_costs(self,costs)
  self.save('costs',costs)
 def finish(self):
  write_json(self.dest/'provenance.json',{'source_sha256':self.used,'protocol':'CPU-only derived statistics from final frozen evidence; no model inference'})
  if self.c.get('freeze_inputs',True):
   paths=sorted(self.dest.glob('*.csv'))+[self.dest/'provenance.json'];write_json(self.dest/'freeze.json',{'version':'M9_paper_inputs_v1','sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})

def main(c):
 with recorded(c) as run:
  snapshot(run,c);e=Export(c);e.static()
  if c.get('include_main',True):e.main_tables(c.get('include_new',True))
  e.finish();run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))
