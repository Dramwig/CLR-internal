"""Frozen M10 estimands, exposing the same draws for 90/95 percentiles."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from clr.calib.m12_statistics import cluster_counts
from clr.calib.m4_statistics import WeightedAUC

def intervals(point,draws):
    q=np.quantile(draws,[.025,.05,.95,.975],axis=0)
    return [{'mean':float(point[i]),'lo':float(q[0,i]),'hi':float(q[3,i]),'lo90':float(q[1,i]),'hi90':float(q[2,i])} for i in range(len(point))]

def linear(frame,columns):
    g=frame.groupby(['problem_id','t'])[[*columns,'instances']].mean()
    ids=sorted(g.index.get_level_values(0).unique());steps=sorted(g.index.get_level_values(1).unique())
    g=g.reindex(pd.MultiIndex.from_product([ids,steps],names=['problem_id','t']))
    x=g[columns].to_numpy().reshape(len(ids),len(steps),len(columns));w=g.instances.fillna(0).to_numpy().reshape(len(ids),len(steps),1)*np.isfinite(x)
    num=np.nan_to_num(x)*w
    def finish(n,d):
        v=np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)
        return np.divide(np.nansum(v,axis=-2),np.isfinite(v).sum(axis=-2),out=np.full(v.shape[:-2]+(v.shape[-1],),np.nan),where=np.isfinite(v).sum(axis=-2)>0)
    point=finish(num.sum(0),w.sum(0));draws=[]
    with threadpool_limits(limits=2):
        for start in range(0,10000,500):
            cw=cluster_counts(len(ids))[start:start+500].astype(float)
            shape=(len(cw),len(steps),len(columns))
            draws.append(finish((cw@num.reshape(len(ids),-1)).reshape(shape),(cw@w.reshape(len(ids),-1)).reshape(shape)))
    return intervals(point,np.concatenate(draws))

def align(a,b,metric):
    if set(b.seed)=={0}:
        b=pd.concat([b.assign(seed=s) for s in sorted(a.seed.unique())],ignore_index=True)
    if set(a.seed)=={0}:
        a=pd.concat([a.assign(seed=s) for s in sorted(b.seed.unique())],ignore_index=True)
    d=a.merge(b,on=['seed','problem_id','t'],suffixes=('_a','_b'),how='outer',indicator=True,validate='one_to_one')
    if not (d['_merge']=='both').all() or not np.array_equal(d.instances_a,d.instances_b):raise ValueError('Paired outcome/count support mismatch')
    d['instances']=d.instances_a;d['value']=d[metric+'_a'];d['baseline']=d[metric+'_b'];d['delta']=d.value-d.baseline
    return d

def paired(a,b,metric):
    d=align(a,b,metric);return dict(zip(['value','baseline','delta'],linear(d,['value','baseline','delta'])))

def auc(a,b):
    ids=sorted(a.problem_id.unique());assert ids==sorted(b.problem_id.unique());lookup={p:i for i,p in enumerate(ids)}
    # Validate per-seed labels. Deterministic CLUE is evaluated once, no pseudo-seeds.
    def make(f):
        panels=[]
        for _,g in f.groupby(['seed','t']):
            ix=np.array([lookup[p] for p in g.problem_id]);panels.append((WeightedAUC(g.label.to_numpy(),g.score.to_numpy()),ix))
        return panels
    for f in [a,b]:
        check=f.groupby(['problem_id','t']).label.nunique();assert (check==1).all()
    la=a.groupby(['problem_id','t']).label.first();lb=b.groupby(['problem_id','t']).label.first();assert la.equals(lb)
    pa,pb=make(a),make(b)
    def calculate(w):
        av=np.mean([fun(w[:,ix]) for fun,ix in pa],axis=0);bv=np.mean([fun(w[:,ix]) for fun,ix in pb],axis=0)
        return np.stack([av,bv,av-bv],1)
    point=calculate(np.ones((1,len(ids))))[0];draws=[];counts=cluster_counts(len(ids))
    for first in range(0,10000,100):draws.append(calculate(counts[first:first+100]))
    return dict(zip(['value','baseline','delta'],intervals(point,np.concatenate(draws))))

def direction(r,metric):
    if r['lo']<=0<=r['hi']:return '不显著'
    good=(r['hi']<0) if metric in ['nll','size'] else (r['lo']>0)
    return '显著更好' if good else '显著更差'
