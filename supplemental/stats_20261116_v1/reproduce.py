"""Re-run frozen M10/M10b statistics before any new analysis."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
from clr.utils.config import load_config
from clr.paper.statistics import metric_summary as original_summary
from clr.calib.m12_statistics import metric_summary as counted_summary


def main(c):
    root = Path(c['output'])
    f = pd.read_csv('data/M10_v1/paper_inputs/main_problem_metrics.csv')
    v = pd.read_csv('data/M10b_v1/paper_inputs/B5c_problem_metrics.csv')
    checks = []
    for base, q, expected in [
        ('CLR-per-type', 'Q4', [2.51, 1.45, 3.60]),
        ('CLR-per-type', 'Q3', [None, -1.31, .261]),
        ('B3', 'Q3', [17.5, 16.1, 19.0]),
        ('B3', 'Q4', [7.93, 6.31, 9.56]),
        ('B5c', 'Q3', [-.659, -1.51, .190]),
        ('B5c', 'Q4', [-.0864, -.984, .830]),
    ]:
        a = f[(f.source == 'gpt2') & (f.model == 'CLR-lite') & (f.split == 'test') & (f.qtype == q)]
        b = v[(v.split == 'test') & (v.qtype == q)] if base == 'B5c' else f[(f.source == 'gpt2') & (f.model == base) & (f.split == 'test') & (f.qtype == q)]
        d = a.merge(b, on=['seed','problem_id','t'], suffixes=('_a','_b'), validate='one_to_one')
        assert len(d) == len(a) == len(b)
        assert np.array_equal(d.instances_a, d.instances_b)
        d['instances'] = d.instances_a
        d['delta'] = d.accuracy_a - d.accuracy_b
        old = original_summary(d, ['delta'])[0]
        fast = counted_summary(d, ['delta'])[0]
        values = [100*old[k] for k in ['mean','lo','hi']]
        error = max(abs(x-y) for x,y in zip(values,expected) if y is not None)
        counted_error = max(abs(old[k]-fast[k]) for k in ['mean','lo','hi'])
        row = dict(baseline=base, qtype=q, estimate_pp=values[0], lo_pp=values[1], hi_pp=values[2], expected=expected, max_error_pp=error, counted_error=counted_error, passed=bool(error <= .1 and counted_error < 1e-10))
        checks.append(row)
        print(json.dumps(row), flush=True)
        (root/'reproduction.json').write_text(json.dumps(checks, indent=2))
        if not row['passed']:
            raise RuntimeError('Required reproduction gate failed; stop new analysis')
    (root/'reproduction_passed.json').write_text(json.dumps({'passed': True, 'checks': len(checks)}))


if __name__ == '__main__':
    main(load_config(sys.argv[1]))
