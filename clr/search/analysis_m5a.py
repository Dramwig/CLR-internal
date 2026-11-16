"""Question-paired, compute-accounted M5a summaries (no source/data loading)."""
from bisect import bisect_right
import numpy as np
from .m5a import majority


def setting(row):
    return (row['method'], row['value'], row['N'], row['b'], row['horizon'])


def capacity(row):
    # Declared before full results; accuracy and cost ties never select winners.
    return (row['N'] * row['b'], row['N'], row['b'])


def vote_at(pool, budget):
    costs = pool['cumulative_source_forwards']
    n = bisect_right(costs, budget)
    if n == 0:
        return None
    return {'answer': majority(pool['answers'][:n]), 'compute': costs[n-1],
            'n': n, 'residual': budget-costs[n-1],
            'coverage': costs[-1] >= budget}


def bootstrap_weights(n, repetitions=10000, seed=20261105):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1/n), repetitions).astype(float) / n


def estimate(values, weights):
    values = np.asarray(values, dtype=float)
    samples = weights @ values
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {'mean': float(values.mean()), 'low': float(lo), 'high': float(hi)}


def fraction(numerator, denominator, weights):
    """Never turn an uncertain oracle gain into a stable-looking percentage."""
    den = estimate(denominator, weights)
    if den['low'] <= 0:
        return {'identifiable': False, 'reason': 'oracle gain 95% CI includes zero'}
    num_samples = weights @ np.asarray(numerator)
    den_samples = weights @ np.asarray(denominator)
    if np.any(den_samples <= 0):
        return {'identifiable': False, 'reason': 'nonpositive oracle bootstrap denominators'}
    lo, hi = np.quantile(num_samples / den_samples, [.025, .975])
    return {'identifiable': True, 'mean': float(np.mean(numerator)/den['mean']),
            'low': float(lo), 'high': float(hi)}
