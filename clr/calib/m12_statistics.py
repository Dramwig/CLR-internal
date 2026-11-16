"""M4.1 problem bootstrap estimand with counted draws and bounded CPU threads."""
from functools import lru_cache
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=8)
def cluster_counts(n):
    """Exactly the original RNG(11) integer draws, compressed to multiplicities."""
    rng = np.random.default_rng(11)
    counts = np.zeros((10000, n), dtype=np.uint16 if n <= 65535 else np.uint32)
    for start in range(0, 10000, 500):
        indices = rng.integers(n, size=(500, n))
        for j, row in enumerate(indices):
            counts[start + j] = np.bincount(row, minlength=n)
    return counts


def cluster_estimate(frame, columns):
    g = frame.groupby(['problem_id', 't'])[[*columns, 'instances']].mean()
    ids = sorted(g.index.get_level_values(0).unique())
    steps = sorted(g.index.get_level_values(1).unique())
    g = g.reindex(pd.MultiIndex.from_product([ids, steps], names=['problem_id', 't']))
    x = g[columns].to_numpy().reshape(len(ids), len(steps), len(columns))
    weight = g.instances.fillna(0).to_numpy().reshape(len(ids), len(steps), 1) * np.isfinite(x)
    numerator = np.nan_to_num(x) * weight

    def finish(n, d):
        v = np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)
        return np.divide(np.nansum(v, axis=-2), np.isfinite(v).sum(axis=-2),
                         out=np.full(v.shape[:-2] + (v.shape[-1],), np.nan),
                         where=np.isfinite(v).sum(axis=-2) > 0)

    point = finish(numerator.sum(0), weight.sum(0))
    shape = (500, len(steps), len(columns))
    num = numerator.reshape(len(ids), -1)
    den = weight.reshape(len(ids), -1)
    draws = []
    with threadpool_limits(limits=2):
        counts = cluster_counts(len(ids))
        for start in range(0, 10000, 500):
            w = counts[start:start + 500].astype(float)
            draws.append(finish((w @ num).reshape(shape), (w @ den).reshape(shape)))
    return point, np.quantile(np.concatenate(draws), [.025, .975], axis=0)


def metric_summary(frame, columns):
    point, ci = cluster_estimate(frame, columns)
    return [dict(metric=m, mean=point[i], lo=ci[0, i], hi=ci[1, i])
            for i, m in enumerate(columns)]
