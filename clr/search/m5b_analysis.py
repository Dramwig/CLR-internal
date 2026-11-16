"""Problem-cluster statistics for M5b diagnostics and compute-matched searches."""
import numpy as np
from scipy.stats import rankdata


def soft_auc(scores,truth):
    """AUROC over the K Bernoulli outcomes, retaining fractional class mass."""
    x=np.asarray(scores);y=np.asarray(truth);order=np.argsort(x,kind='stable');x=x[order];y=y[order]
    pos=y.sum();neg=(1-y).sum()
    if pos<=0 or neg<=0:return np.nan
    starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1];pp=np.add.reduceat(y,starts);nn=np.add.reduceat(1-y,starts)
    return float(np.sum(pp*(np.cumsum(nn)-nn+.5*nn))/(pos*neg))


def bootstrap_mean(x,rng,draws=10000):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return {'mean':None,'ci_low':None,'ci_high':None,'n':0}
    boot=[]
    for j in range(0,draws,100):boot.extend(x[rng.integers(len(x),size=(min(100,draws-j),len(x)))].mean(1))
    lo,hi=np.quantile(boot,[.025,.975]);return {'mean':float(x.mean()),'ci_low':float(lo),'ci_high':float(hi),'n':len(x)}


class MatchedCoverage:
    """Empirical randomized prediction-set size at exact requested coverage.

    One item per problem/stratum. The returned diagnostic recomputes the
    coverage threshold in each paired bootstrap sample; this is an evaluation
    curve, not a deployable cal_B conformal threshold or a coverage guarantee.
    """
    def __init__(self,probs,targets):
        probs=np.asarray(probs);targets=np.asarray(targets);n,k=probs.shape
        self.n=n;flat=(1-probs).ravel();order=np.argsort(flat,kind='stable');self.pid=np.repeat(np.arange(n),k)[order];self.mass=targets.ravel()[order]
        sorted_score=flat[order];starts=np.r_[0,np.flatnonzero(np.diff(sorted_score)!=0)+1];ends=np.r_[starts[1:]-1,len(order)-1]
        self.start=np.repeat(starts,ends-starts+1);self.end=np.repeat(ends,ends-starts+1)
    def __call__(self,weights,coverage=.9):
        w=np.asarray(weights,dtype=float);cell=w[:,self.pid];cum_mass=np.cumsum(cell*self.mass,axis=1);cum_count=np.cumsum(cell,axis=1)
        target=coverage*w.sum(1);at=(cum_mass>=target[:,None]-1e-12).argmax(1);begin=self.start[at];end=self.end[at];rows=np.arange(len(w))
        mass_before=np.where(begin>0,cum_mass[rows,np.maximum(begin-1,0)],0);count_before=np.where(begin>0,cum_count[rows,np.maximum(begin-1,0)],0)
        mass_end=cum_mass[rows,end];count_end=cum_count[rows,end];fraction=(target-mass_before)/np.maximum(mass_end-mass_before,1e-12)
        return (count_before+fraction*(count_end-count_before))/w.sum(1)
