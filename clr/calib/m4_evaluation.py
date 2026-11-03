"""Fixed M23 metrics and Mondrian strata; one question per problem per stratum."""
from collections import defaultdict
import numpy as np
from scipy.special import softmax
from clr.calib.m23 import metrics,conformal_threshold
from clr.calib.m4 import apply
from clr.data.questions.core import seed

def target_array(rows,q):
    if q=='Q3':return np.eye(4)[[r['label'] for r in rows]]
    y=np.array([r['label_soft'] for r in rows]);return np.stack([1-y,y],1)

def unique_indices(rows,indices,q,random_seed):
    unique={}
    def rank(i):
        r=rows[i];return seed(random_seed,r['problem_id'],q,r['state_ref']['t'],tuple(r['options']),r['template_id'])
    for i in indices:
        p=rows[i]['problem_id']
        if p not in unique or rank(i)<rank(unique[p]):unique[p]=i
    return list(unique.values())

def thresholds(rows,logits,calibration,q,random_seed):
    p=softmax(apply(logits,calibration),axis=1);out={}
    for t in range(7):
        ii=unique_indices(rows,[i for i,r in enumerate(rows) if r['state_ref']['t']==t],q,random_seed)
        if not ii:continue
        labels=[rows[i]['bernoulli_outcomes'][seed(random_seed,rows[i]['problem_id'],t,'cp')%len(rows[i]['bernoulli_outcomes'])] if q=='Q1' else rows[i]['label'] for i in ii]
        scores=1-p[ii,labels]
        for alpha in [.05,.1,.2]:out[f'{t}:{alpha}']=conformal_threshold(scores,alpha)
    return out

def summarize(rows,logits,q,calibrations,threshold_map,random_seed):
    y=target_array(rows,q);groups=defaultdict(list);stats=[];cp=[]
    for i,r in enumerate(rows):groups[r['state_ref']['t'],r['state_ref']['kind'],r['is_heldout_template']].append(i)
    for method in ['raw',*calibrations]:
        x=logits if method=='raw' else apply(logits,calibrations[method]);p=softmax(x,axis=1)
        for (t,kind,held),ii in groups.items():
            stats.append({'qtype':q,'t':t,'kind':kind,'heldout':held,'calibration':method,'n':len(ii),**metrics(x[ii],y[ii],q=='Q3')})
            if method=='raw' or kind!='clean':continue
            jj=unique_indices(rows,ii,q,random_seed)
            for alpha in [.05,.1,.2]:
                key=f'{t}:{alpha}'
                if key not in threshold_map[method]:continue
                threshold=threshold_map[method][key];sets=1-p[jj]<=threshold
                cp.append({'qtype':q,'t':t,'heldout':held,'calibration':method,'alpha':alpha,'n':len(jj),'coverage':float((sets*y[jj]).sum(1).mean()),'size':float(sets.sum(1).mean()),'threshold':float(threshold) if np.isfinite(threshold) else None,'threshold_is_infinite':not np.isfinite(threshold)})
    return stats,cp
