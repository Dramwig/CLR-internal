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

def test_mlp_value_effect_depends_on_state():
    torch.manual_seed(11);m=Probe('B3','Q4');q=torch.zeros(1,64)
    options=torch.randn(1,1,8,requires_grad=True)
    a=torch.autograd.grad(m(torch.zeros(1,768),q,options)[0,1],options)[0]
    b=torch.autograd.grad(m(torch.randn(1,768)*3,q,options)[0,1],options)[0]
    assert not torch.allclose(a,b)

def test_mc_metrics_equal_individual_bernoulli_samples():
    logits=np.array([[.2,1.1],[-.5,.3],[.7,-.8]])
    outcomes=np.array([[0]*4+[1]*12,[0]*9+[1]*7,[0]*16])
    means=outcomes.mean(1)
    aggregated=metrics(logits,np.stack([1-means,means],1))
    expanded=metrics(np.repeat(logits,16,axis=0),np.eye(2)[outcomes.reshape(-1)])
    for name in ['nll','brier','ece15','accuracy']:
        np.testing.assert_allclose(aggregated[name],expanded[name],atol=1e-12)
