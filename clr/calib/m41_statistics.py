"""Problem-cluster paired uncertainty for M4.1; all steps stay together."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from clr.calib.m4 import apply
from clr.calib.m4_statistics import WeightedAUC
from clr.calib.m4_evaluation import unique_indices,target_array
from clr.data.m4.reader import rows_for

def prediction(suite,case,seed,split,q):
    path=Path(suite[f'{case}_{seed}']);frame=pd.read_parquet(path/f'{q}_{split}_False.parquet');cal=json.loads((path/'calibration.json').read_text())[q]['temperature'];probs=softmax(apply(np.stack(frame.logits),cal),axis=1)
    return path,frame,probs

def paired(suite,a,b,seeds,split,q,metric,rng,draws=10000):
    differences=[];designs=[];counts=None;pids=None
    for seed in seeds:
        outputs=[];auc=[]
        for case in [a,b]:
            _,r,p=prediction(suite,case,seed,split,q);clean=r.kind=='clean';p=p[clean];r=r[clean].copy()
            if metric=='auroc':
                r['prob']=p[:,1];r=r.set_index(['problem_id','t']).sort_index();ids=sorted(r.index.get_level_values(0).unique())
                if pids is None:pids=ids
                if pids!=ids:raise ValueError('AUC problem alignment')
                y=r.label.unstack('t').to_numpy();pr=r.prob.unstack('t').to_numpy();auc.append([WeightedAUC(y[:,t],pr[:,t]) for t in range(y.shape[1])]);continue
            if metric=='nll':r['value']=-(r.label_soft*np.log(np.maximum(p[:,1],1e-12))+(1-r.label_soft)*np.log(np.maximum(p[:,0],1e-12)))
            elif metric=='accuracy':r['value']=(p.argmax(1)==r.label).astype(float) if q=='Q3' else np.where(p.argmax(1)==1,r.label_soft,1-r.label_soft)
            else:raise ValueError(metric)
            g=r.groupby(['problem_id','t']).value;outputs.append((g.mean(),g.size()))
        if metric=='auroc':designs.append(auc)
        else:
            if not outputs[0][0].index.equals(outputs[1][0].index) or not outputs[0][1].equals(outputs[1][1]):raise ValueError('Paired keys/counts mismatch')
            differences.append(outputs[0][0]-outputs[1][0]);counts=outputs[0][1]
    if metric=='auroc':
        n=len(pids)
        def calc(w):return np.mean([np.mean([d(w) for d in pair[0]],axis=0)-np.mean([d(w) for d in pair[1]],axis=0) for pair in designs],axis=0)
        delta=float(calc(np.ones((1,n)))[0]);boot=[]
        for j in range(0,draws,100):
            size=min(100,draws-j);ii=rng.integers(n,size=(size,n));w=np.zeros((size,n));np.add.at(w,(np.arange(size)[:,None],ii),1);boot.extend(calc(w))
    else:
        d=pd.concat(differences,axis=1).mean(axis=1).unstack('t');x=d.fillna(0).to_numpy();w=counts.unstack('t').reindex_like(d).fillna(0).to_numpy();n=len(x);delta=float(np.nanmean((x*w).sum(0)/w.sum(0)));boot=[]
        for j in range(0,draws,100):
            ii=rng.integers(n,size=(min(100,draws-j),n));boot.extend(np.nanmean((x[ii]*w[ii]).sum(1)/w[ii].sum(1),axis=1))
    lo,hi=np.quantile(boot,[.025,.975]);return {'a':a,'b':b,'split':split,'qtype':q,'metric':metric,'delta':delta,'ci_low':float(lo),'ci_high':float(hi),'bootstrap':draws}

def cp_paired(c,suite,a,b,split,q,rng):
    qa=rows_for(c,split,'final',qtypes=[q]);diffs=[]
    for seed in c['seeds']:
        outputs=[]
        for case in [a,b]:
            path,frame,probs=prediction(suite,case,seed,split,q)
            if len(qa)!=len(frame) or [r['problem_id'] for r in qa]!=frame.problem_id.tolist():raise ValueError('CP QA ordering mismatch')
            target=target_array(qa,q);th=json.loads((path/'thresholds.json').read_text())[q]['temperature'];records=[]
            for t in range(7):
                ii=unique_indices(qa,[i for i,r in enumerate(qa) if r['state_ref']['kind']=='clean' and r['state_ref']['t']==t],q,seed);threshold=th[f'{t}:0.1'];sets=1-probs[ii]<=(np.inf if threshold is None else threshold)
                records.extend({'problem_id':qa[index]['problem_id'],'t':t,'coverage':float((sets[j]*target[index]).sum()),'size':float(sets[j].sum())} for j,index in enumerate(ii))
            outputs.append(pd.DataFrame(records).set_index(['problem_id','t']).sort_index())
        diffs.append(outputs[0]-outputs[1])
    results=[]
    for metric in ['coverage','size']:
        x=pd.concat([d[metric] for d in diffs],axis=1).mean(axis=1).unstack('t').to_numpy();n=len(x);boot=[]
        for _ in range(100):
            ii=rng.integers(n,size=(100,n));boot.extend(np.nanmean(np.nanmean(x[ii],axis=1),axis=1))
        lo,hi=np.quantile(boot,[.025,.975]);results.append({'a':a,'b':b,'split':split,'qtype':q,'metric':metric,'delta':float(np.nanmean(np.nanmean(x,axis=0))),'ci_low':float(lo),'ci_high':float(hi),'alpha':.1,'bootstrap':10000})
    return results
