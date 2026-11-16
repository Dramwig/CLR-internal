"""Frozen-protocol robustness summaries; paired episode-group uncertainty."""
import numpy as np


def select_candidate(candidates,seed):
    """Finite-search adversary selected without the held-out audit questions.

    Candidate selection is part of white-box attack evaluation, with all
    lambda runs charged. It never changes trained models or clean thresholds.
    The same candidate is used at both nominal alpha values.
    """
    hits=[r for r in candidates if r['target_achieved']]
    return max(hits or candidates,key=lambda r:(r['target_logp'],-r['lambda']))


def metric_vector(row,seed,alpha):
    alarm=bool(row['audit'][str(seed)][str(alpha)])
    target=bool(row['target_achieved']);changed=bool(row['changed_to_target'])
    return {'robust_failure':float(changed and not alarm),
            'target_and_accepted':float(target and not alarm),
            'target_rate':float(target),'changed_target_rate':float(changed),
            'detection_rate':float(alarm),'clean_FPR':float(row['clean_audit'][str(seed)][str(alpha)]),
            'clean_correct':float(row['clean_correct']),
            'clean_correct_failure':float(row['clean_correct'] and changed and not alarm)}


def bootstrap_mean(values,repeats=10000,seed=20261108):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x):raise ValueError('Need nonempty group means')
    rng=np.random.default_rng(seed);draws=[]
    for _ in range((repeats+99)//100):
        ii=rng.integers(len(x),size=(min(100,repeats-len(draws)),len(x)))
        draws.extend(x[ii].mean(1).tolist())
    return float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))
