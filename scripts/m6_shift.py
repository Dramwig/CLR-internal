"""RQ4 score shift diagnostics. Domain fitting uses cal_A covariates only."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import softmax
from clr.calib.m23 import conformal_threshold
from clr.calib.m6 import fit_ratio,domain_features,weighted_threshold,coverage
from clr.calib.m4_evaluation import target_array
from clr.data.questions.core import seed
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.m6_runtime import snapshot
from clr.utils.idea_a import write_json

def select(rs,q,part=None,steps=None,random_seed=11):
    selected={}
    for r in rs:
        if r['state_ref']['traj_id']!=-1 or (part and r['cal_part']!=part) or (steps is not None and r['state_ref']['t'] not in steps):continue
        pid=r['problem_id'];rank=seed(random_seed,'M6-CP-instance',pid,q,r['state_ref']['t'],tuple(r['options']),r['template_id'])
        if pid not in selected or rank<selected[pid][0]:selected[pid]=(rank,r)
    return [v[1] for _,v in sorted(selected.items())]

def arrays(rs,q,random_seed):
    logits=np.asarray([r['calibrated_logits'] for r in rs]);p=softmax(logits,axis=1)
    yy=[r['bernoulli_outcomes'][seed(random_seed,'M6-CP-Bernoulli',r['problem_id'],q)%len(r['bernoulli_outcomes'])] if q=='Q1' else r['label'] for r in rs]
    return p,1-p[np.arange(len(rs)),yy],domain_features(logits,[r['state_ref']['t'] for r in rs]),target_array(rs,q)

def bootstrap_matrix(values,random_seed=11):
    values=np.asarray(values);n=len(values);rng=np.random.default_rng(random_seed);out=[]
    for start in range(0,10000,500):
        ix=rng.integers(n,size=(min(500,10000-start),n));out.append(values[ix].mean(1))
    out=np.concatenate(out);return np.quantile(out,[.025,.975],axis=0)

def main(c):
 with recorded(c) as run:
    snapshot(run,{**c,'section':c.get('section','CD')})
    print(run.path,flush=True)
    start=time.monotonic();index=json.loads(Path(c['evaluation_index']).read_text());out=Path(c['output'])/'shift';out.mkdir(parents=True,exist_ok=True);summary=[];allpoints=[]
    if not json.loads(Path(c['eval_lock']).read_text()).get('all_training_locked'):raise ValueError('RQ4 requires final evaluation lock')
    proxy=c.get('mode')=='smoke';eval_split='val' if proxy else 'test'
    if c.get('proxy_sources') and not proxy:raise ValueError('Source aliases are val-only smoke fixtures')
    for kind in ['lite','b3']:
      for sd in c['seeds']:
       for q in ['Q1','Q2','Q3','Q4']:
        def get(source,split):
            key=f'single_{source}_{kind}_{sd}';r=Path(index[key][source]);prefix=c.get('proxy_sources',{}).get(source,source)
            rs=pq.read_table(r/f'{prefix}_{q}_{split}.parquet').to_pylist()
            if proxy and split=='cal':
                ids=sorted({row['problem_id'] for row in rs});aid=set(ids[:len(ids)//2])
                for row in rs:row['cal_part']='cal_A' if row['problem_id'] in aid else 'cal_B'
            return rs
        ca={s:get(s,'cal') for s in ['gpt2','llama']};evals={s:get(s,eval_split) for s in ['gpt2','llama']}
        cases=[]
        for a,b in [('gpt2','llama'),('llama','gpt2')]:cases.append((f'source_{a}_to_{b}',select(ca[a],q,'cal_A',random_seed=sd),select(ca[b],q,'cal_A',random_seed=sd),select(ca[a],q,'cal_B',random_seed=sd),select(ca[b],q,'cal_B',random_seed=sd),select(evals[b],q,random_seed=sd),'source'))
        for s in ['gpt2','llama']:cases.append((f'step_{s}_early_to_late',select(ca[s],q,'cal_A',[0,1,2],sd),select(ca[s],q,'cal_A',[4,5,6],sd),select(ca[s],q,'cal_B',[0,1,2],sd),None,select(evals[s],q,steps=[4,5,6],random_seed=sd),'step'))
        for source in ['gpt2','llama']:
         for pool in ['svamp','asdiv']:
            try:
                unseen_cal=get(source,'unseen_train');unseen_eval=get(source,'unseen_eval')
            except FileNotFoundError:
                continue
            tr=[r for r in unseen_cal if r['problem_id'].lower().startswith(pool+':')];ev=[r for r in unseen_eval if r['problem_id'].lower().startswith(pool+':')]
            # Existing unseen_train IDs are a disjoint target calibration pool;
            # no CLR/baseline training in M6 uses this pool.
            ids=sorted({r['problem_id'] for r in tr},key=lambda p:seed(11,'M6-unseen-cal',p));aid=set(ids[::2]);bid=set(ids[1::2])
            for r in tr:r['cal_part']='cal_A' if r['problem_id'] in aid else 'cal_B'
            cases.append((f'dataset_{source}_gsm_to_{pool}',select(ca[source],q,'cal_A',random_seed=sd),select(tr,q,'cal_A',random_seed=sd),select(ca[source],q,'cal_B',random_seed=sd),select(tr,q,'cal_B',random_seed=sd),select(ev,q,random_seed=sd),'dataset'))
        for name,a_train,b_train,a_cal,b_cal,ev,shift in cases:
            if min(map(len,[a_train,b_train,a_cal,ev]))<2:raise ValueError('Missing RQ4 domain calibration data: '+name)
            _,_,ax,_=arrays(a_train,q,sd);_,_,bx,_=arrays(b_train,q,sd);ratio,clf=fit_ratio(ax,bx)
            _,ascore,acx,_=arrays(a_cal,q,sd);ep,_,ex,ey=arrays(ev,q,sd);aw=ratio(acx);ew=ratio(ex)
            for alpha in [.05,.1,.2]:
                methods={'split':np.repeat(conformal_threshold(ascore,alpha),len(ev)),
                    'weighted':weighted_threshold(ascore,aw,ew,alpha),
                    'mondrian_source_only':np.repeat(conformal_threshold(ascore,alpha) if shift=='step' else np.inf,len(ev))}
                if b_cal:
                    _,bscore,_,_=arrays(b_cal,q,sd);methods['mondrian_target_cal']=np.repeat(conformal_threshold(bscore,alpha),len(ev))
                for method,threshold in methods.items():
                    cov,size=coverage(ep,ey,threshold)
                    common={'model':kind,'seed':sd,'qtype':q,'shift':name,'method':method,'alpha':alpha,'cal_n':len(a_cal),'target_cal_n':len(b_cal) if method=='mondrian_target_cal' and b_cal else 0}
                    summary.append({**common,'coverage':float(cov.mean()),'coverage_minus_target':float(cov.mean()-(1-alpha)),'size':float(size.mean()),'cal_weight_ess':float(aw.sum()**2/(aw@aw)),'cal_weight_min':float(aw.min()),'cal_weight_max':float(aw.max())})
                    allpoints.extend([{**common,'problem_id':r['problem_id'],'coverage':float(v),'size':int(n)} for r,v,n in zip(ev,cov,size)])
        print(kind,sd,q,flush=True)
        pd.DataFrame(summary).to_csv(out/'seed_summary.csv',index=False)
    frame=pd.DataFrame(allpoints);pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),out/'problem_results.parquet',compression='zstd')
    aggregate=[]
    for key,group in frame.groupby(['model','qtype','shift','method','alpha']):
        means=group.groupby('problem_id')[['coverage','size']].mean();ci=bootstrap_matrix(means.values)
        aggregate.append(dict(zip(['model','qtype','shift','method','alpha'],key))|{'n':len(means),'coverage':float(means.coverage.mean()),'coverage_minus_target':float(means.coverage.mean()-(1-key[4])),'coverage_ci_low':float(ci[0,0]),'coverage_ci_high':float(ci[1,0]),'size':float(means['size'].mean()),'size_ci_low':float(ci[0,1]),'size_ci_high':float(ci[1,1])})
    pd.DataFrame(aggregate).to_csv(out/'summary.csv',index=False)
    write_json(out/'protocol.json',{'mode':c.get('mode','full'),'test_opened':not proxy,'seeds':c['seeds'],'domain_classifier':'L2 logistic C=1; cal_A only; sorted predicted probability + t/6 + entropy; equal-domain prior correction; ratio clipped [.1,10]','threshold_labels':'cal_B only, one fixed question per problem/qtype; Q1 one fixed Bernoulli calibration draw; eval coverage averaged over K16 outcomes','weighted_quantile':'includes test weight as infinity atom','mondrian_source_only':'unseen source/dataset -> infinity/full set; same source early/late -> original stratum','mondrian_target_cal':'additional target cal_B labels; distinct availability diagnostic, not source-only recovery','scope':'source-specific C1 score functions; no same-representation or exact density-ratio guarantee','CI':'10000 problem cluster bootstrap, average fixed training seeds; seed variation separate in seed_summary.csv'})
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,rows=len(aggregate))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))
