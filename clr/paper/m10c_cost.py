"""Read-only measurement helpers. No optimizer, calibration or test inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def verify(manifest):
    d = json.loads(Path(manifest).read_text())
    bad = [p for p, h in d['sha256'].items() if sha(p) != h]
    if bad:
        raise ValueError('Frozen input mismatch: ' + str(bad))
    return len(d['sha256'])


def paired_conformal(clr, b5c):
    """Join exactly the same frozen calibration-selected problem/step/seed rows."""
    keys = ['source', 'seed', 'split', 'qtype', 'problem_id', 't']
    a = clr[(clr.source == 'gpt2') & (clr.model == 'CLR-lite') & clr.qtype.isin(['Q3', 'Q4'])]
    b = b5c[(b5c.source == 'gpt2') & (b5c.model == 'B5c')]
    d = b.merge(a, on=keys, suffixes=('_b5c', '_clr'), validate='one_to_one', how='outer', indicator=True)
    if not (d['_merge'] == 'both').all():
        raise ValueError('Unpaired conformal rows')
    if not ((d.instances_b5c == 1) & (d.instances_clr == 1)).all():
        raise ValueError('Expected one frozen selected question per problem/step/seed')
    for m in ['coverage', 'size']:
        d[m + '_difference'] = d[m + '_b5c'] - d[m + '_clr']
    d['instances'] = 1
    return d


def summarize_conformal(clr_path, b5c_path):
    from clr.paper.statistics import metric_summary
    a = pd.concat([g[(g.source == 'gpt2') & (g.model == 'CLR-lite') & g.qtype.isin(['Q3', 'Q4'])]
                   for g in pd.read_csv(clr_path, chunksize=200000)], ignore_index=True)
    b = pd.read_csv(b5c_path)
    d = paired_conformal(a, b)
    result = []
    for (split, q), g in d.groupby(['split', 'qtype']):
        cols = [m + '_' + k for k in ['b5c', 'clr', 'difference'] for m in ['coverage', 'size']]
        for row in metric_summary(g, cols):
            result.append(dict(source='gpt2', split=split, qtype=q, alpha=.1,
                               n_problems=g.problem_id.nunique(), **row))
    return pd.DataFrame(result), d


def measured_median(seconds, nqueries):
    if len(seconds) != 3 or min(seconds) <= 0:
        raise ValueError('Exactly three positive timed rounds required')
    median = float(np.median(seconds))
    return median / nqueries, nqueries / median
