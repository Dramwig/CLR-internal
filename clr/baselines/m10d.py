"""M3 scalar reward protocol in native LLaMA dimension; frozen code stays intact."""
import numpy as np
import torch
from torch import nn
from clr.baselines.probes import Probe


def scalar_model(dim):
    # Retain M3's construction, unused parameters, output and optimizer behavior.
    model = Probe('B4', 'Q1')
    model.head = nn.Linear(dim, 1)
    return model


def equal_step_statistics(rows, logits, temperature, qtype):
    from clr.calib.m23 import metrics
    out = []
    for kind in ['clean', 'traj']:
        for t in sorted({r['state_ref']['t'] for r in rows}):
            ii = [i for i, r in enumerate(rows)
                  if r['state_ref']['kind'] == kind and r['state_ref']['t'] == t]
            if not ii:
                continue
            y = np.array([[1-rows[i]['label_soft'], rows[i]['label_soft']] for i in ii])
            out.append(dict(qtype=qtype, kind=kind, t=t, n=len(ii),
                            **metrics(logits[ii] / temperature, y)))
    return out
