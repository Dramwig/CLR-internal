"""Val/synthetic-only verification of bootstrap pairing and perturbation gates."""
import numpy as np
import torch  # Load the project C++ runtime before pyarrow/Pillow.
import pandas as pd
from clr.calib import m41_statistics as mod

def test_paired_bootstrap_keeps_problem_steps_and_sign(monkeypatch):
    rows=[]
    for p in range(10):
        for t in range(7):rows.append({'problem_id':str(p),'t':t,'kind':'clean','label':p%2,'label_soft':float(p%2)})
    frame=pd.DataFrame(rows)
    def fake(suite,case,seed,split,q):
        correct=.9 if case=='good' else .1;p=np.where(frame.label.to_numpy()==1,correct,1-correct);return None,frame.copy(),np.stack([1-p,p],axis=1)
    monkeypatch.setattr(mod,'prediction',fake)
    out=mod.paired({},'good','bad',[11,22,33],'reader_val','Q4','accuracy',np.random.default_rng(11),draws=200)
    assert out['delta']==1 and out['ci_low']==1 and out['ci_high']==1
