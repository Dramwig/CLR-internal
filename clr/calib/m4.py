"""Permutation-safe affine calibration and paired problem-level comparisons."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from clr.calib.m23 import temperature

def fit(logits,targets,method='temperature'):
    if method not in ['temperature','affine']:raise ValueError(method)
    logits=np.asarray(logits,dtype=np.float64);targets=np.asarray(targets,dtype=np.float64)
    t=temperature(logits,targets)
    # Positional class biases are forbidden for unordered choice options.
    if method=='temperature' or logits.shape[1]!=2:return {'temperature':t,'bias':0.,'method':method,'shared_choice_bias':logits.shape[1]!=2}
    delta=logits[:,1]-logits[:,0]
    def objective(x):
        v=delta/np.exp(x[0])+x[1];return float((np.logaddexp(0,v)-targets[:,1]*v).mean())
    result=minimize(objective,[np.log(t),0.],method='L-BFGS-B',bounds=[(-4,4),(-10,10)])
    if not result.success:raise RuntimeError('Affine calibration failed: '+str(result.message))
    return {'temperature':float(np.exp(result.x[0])),'bias':float(result.x[1]),'method':method,'shared_choice_bias':False}

def apply(logits,calibration):
    out=np.array(logits,dtype=np.float64)/calibration['temperature']
    if out.shape[1]==2:out[:,1]+=calibration['bias']
    return out
