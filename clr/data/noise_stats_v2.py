"""K-rollout metrics: one observation per problem, never per latent step."""
from collections import Counter
import numpy as np
from clr.utils.stats_v2 import bootstrap_mean

def summarize(rows,k):
    if not rows or any(len(r['correct'])!=k or len(r['answers'])!=k for r in rows):
        raise ValueError('Incomplete problem rollout panel')
    values=np.array([sum(r['correct'])/k for r in rows])
    return {'accuracy':bootstrap_mean(values),
            'nondegenerate':bootstrap_mean((values>0)&(values<1)),
            'invalid_rate':bootstrap_mean([sum(a is None for a in r['answers'])/k for r in rows]),
            'distinct_answers':bootstrap_mean([len(set(r['answers'])-{None}) for r in rows]),
            'distinct_answers_including_invalid':bootstrap_mean([len(set(r['answers'])) for r in rows]),
            'V_histogram':{'correct_counts':list(range(k+1)),'counts':np.bincount([sum(r['correct']) for r in rows],minlength=k+1).tolist(),'K':k}}

def recommend(settings):
    eligible=[s for s in settings if s['temperature']==0 and s['sigma']>0 and s['accuracy_drop']<.05 and s['nondegenerate']>=.30]
    if not eligible:return {'selected':None,'reason':'No Gaussian setting meets drop <5pp and nondegenerate >=30%; review required.'}
    best=max(eligible,key=lambda s:(s['nondegenerate'],-s['accuracy_drop'],-s['sigma'],-s['from_step']))
    return {'selected':best,'tie_break':'nondegenerate descending, drop ascending, sigma ascending, from_step ascending'}
