"""Temperature scaling and finite-sample Mondrian thresholds."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,softmax

def temperature(logits,targets):
    def loss(x):
        a=logits/np.exp(x);return float(-(targets*(a-logsumexp(a,axis=1,keepdims=True))).sum(1).mean())
    fit=minimize_scalar(loss,bounds=(-4,4),method='bounded')
    return float(np.exp(fit.x))

def conformal_threshold(scores,alpha):
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.sort(scores)[k-1]) if k<=len(scores) else float('inf')

def metrics(logits,targets,choice=False):
    from sklearn.metrics import roc_auc_score
    p=softmax(logits,axis=1);pred=p.argmax(1);confidence=p.max(1)
    correctness=targets[np.arange(len(p)),pred]
    # Fractional Q1 targets exactly average the Bernoulli outcomes; Brier
    # includes irreducible sampling variance, not squared error to V_hat.
    nll=float(-(targets*np.log(np.maximum(p,1e-12))).sum(1).mean())
    brier=float((1-2*(targets*p).sum(1)+(p*p).sum(1)).mean())/(1 if choice else 2)
    ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
        if mask.any():ece+=mask.mean()*abs(confidence[mask].mean()-correctness[mask].mean())
    out={'nll':nll,'brier':brier,'accuracy':float(correctness.mean()),'ece15':float(ece)}
    if not choice:
        out['mae']=float(abs(p[:,1]-targets[:,1]).mean())
        hard=np.all((targets[:,1]==0)|(targets[:,1]==1))
        out['auroc']=float(roc_auc_score(targets[:,1],p[:,1])) if hard and len(np.unique(targets[:,1]))==2 else None
    return out
