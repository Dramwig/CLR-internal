"""Problem-cluster intervals, preserving step and seed estimands."""
import numpy as np
import pandas as pd
from scripts.m6_report import cluster_estimate

def metric_summary(frame,columns):
 point,ci=cluster_estimate(frame,columns)
 return [{'metric':m,'mean':point[i],'lo':ci[0,i],'hi':ci[1,i]} for i,m in enumerate(columns)]

def auc_summary(frame,resamples=10000):
 """Exact weighted AUROC with shared problem bootstrap across steps/seeds."""
 ids=sorted(frame.problem_id.unique());lookup={p:i for i,p in enumerate(ids)};panels=[]
 for _,g in frame.groupby(['seed','t']):
  y=g.label.to_numpy();score=g.score.to_numpy();order=np.argsort(score,kind='stable');score=score[order];y=y[order];ix=np.array([lookup[p] for p in g.problem_id])[order]
  first=np.r_[0,np.flatnonzero(np.diff(score))+1];panels.append((ix,y,first))
 def calculate(counts):
  values=[]
  for ix,y,first in panels:
   w=counts[:,ix];pos=np.add.reduceat(w*y,first,axis=1);neg=np.add.reduceat(w*(1-y),first,axis=1)
   denom=pos.sum(1)*neg.sum(1)
   values.append(np.divide((pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1),denom,out=np.full(len(counts),np.nan),where=denom>0))
  values=np.asarray(values);valid=np.isfinite(values)
  return np.divide(np.nansum(values,axis=0),valid.sum(0),out=np.full(values.shape[1],np.nan),where=valid.sum(0)>0)
 point=calculate(np.ones((1,len(ids))))[0];rng=np.random.default_rng(11);draws=[]
 for start in range(0,resamples,100):
  ii=rng.integers(len(ids),size=(min(100,resamples-start),len(ids)));cnt=np.zeros(ii.shape,dtype=np.int32)
  for j in range(len(ii)):cnt[j]=np.bincount(ii[j],minlength=len(ids))
  draws.extend(calculate(cnt))
 lo,hi=np.nanquantile(draws,[.025,.975]);return {'metric':'auroc','mean':point,'lo':lo,'hi':hi}

def monotone_envelope(frame,value='mean',radius='epsilon'):
 result=frame.sort_values(radius).copy();result['envelope']=np.maximum.accumulate(result[value].to_numpy());return result
