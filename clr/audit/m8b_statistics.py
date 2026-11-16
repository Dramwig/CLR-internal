"""Outcome-blind attack candidate selection and group-level uncertainty."""
import numpy as np

def select_candidate(rows):
    # Only objective-side information enters; no report-panel score/decision.
    return min(rows,key=lambda r:(not r['target_achieved'],float('inf') if r['selection_nll'] is None else r['selection_nll'],-r['target_logp']))

def group_ci(values,seed=20261108):
    x=np.asarray(values,float);rng=np.random.default_rng(seed);boot=[]
    for start in range(0,10000,200):boot.extend(x[rng.integers(0,len(x),(min(200,10000-start),len(x)))].mean(1).tolist())
    return float(x.mean()),*map(float,np.quantile(boot,[.025,.975]))
