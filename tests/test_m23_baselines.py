import numpy as np
import torch
from clr.baselines.probes import Probe
from clr.calib.m23 import conformal_threshold,metrics

def test_choice_equivariance_and_zero_state():
    torch.manual_seed(11);z=torch.randn(8,768);q=torch.randn(8,64);o=torch.randn(8,4,8);perm=[2,0,3,1]
    for name in ['B0','B1','B2','B3']:
        m=Probe(name,'Q3')
        torch.testing.assert_close(m(z,q,o[:,:, :])[:,perm],m(z,q,o[:,perm]))
        if name=='B0':torch.testing.assert_close(m(z,q,o),m(z*10,q,o))

def test_conformal_rank_and_bernoulli_brier():
    assert conformal_threshold([.1,.2,.3,.4],.1)==float('inf')
    assert conformal_threshold([.1,.2,.3,.4],.4)==.3
    m=metrics(np.array([[0.,0.]]),np.array([[.5,.5]]))
    assert abs(m['brier']-.25)<1e-8 # must not be zero versus MC mean

def test_exchangeable_conformal_empirical_coverage():
    rng=np.random.default_rng(11);coverage=[]
    for _ in range(1000):
        scores=rng.random(101);q=conformal_threshold(scores[:100],.1)
        coverage.append(scores[-1]<=q)
    assert .87<np.mean(coverage)<.94
