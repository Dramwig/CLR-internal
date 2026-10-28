"""Question-level resampling; rollout repeats never increase the question count."""
import numpy as np

def bootstrap_mean(values, resamples=10000, rng_seed=20261028):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Expected nonempty finite per-question values')
    rng=np.random.default_rng(rng_seed);means=[]
    for offset in range(0,resamples,250):
        idx=rng.integers(len(x),size=(min(250,resamples-offset),len(x)))
        means.extend(x[idx].mean(axis=1))
    lo,hi=np.quantile(means,[.025,.975])
    return {'mean':float(x.mean()),'ci95':[float(lo),float(hi)],'n_questions':len(x),'bootstrap_resamples':resamples,'bootstrap_rng_seed':rng_seed}
