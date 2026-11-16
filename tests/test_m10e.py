import numpy as np
import pytest
import torch
from clr.baselines.m10e import HSRM, linear_probe, centroid_score, fit_logistic, calibrated_logits
from clr.baselines.probes import Probe
from clr.data.m4.reader import require_split


def test_hsrm_mask_and_gradient():
    torch.manual_seed(11);m=HSRM(768).eval();x=torch.randn(3,7,768,requires_grad=True)
    mask=torch.arange(7)[None,:]<torch.tensor([1,4,7])[:,None]
    a=m(x,mask);v=x.detach().clone();v[~mask]=1000
    torch.testing.assert_close(a,m(v,mask),atol=1e-6,rtol=1e-5)
    a[:,1].sum().backward();assert x.grad[mask].abs().sum()>0;assert x.grad[~mask].abs().sum()==0
    assert 1.7e6<sum(p.numel() for p in m.parameters())<2.2e6


def test_b2_matches_m3_and_option_order():
    for q in ['Q1','Q2','Q3','Q4']:
        old=Probe('B2',q);new=linear_probe(768,q);new.load_state_dict(old.state_dict())
        z=torch.randn(4,768);t=torch.randn(4,64);o=torch.randn(4,4 if q=='Q3' else 1,8)
        torch.testing.assert_close(old(z,t,o),new(z,t,o),atol=0,rtol=0)
    m=linear_probe(2048,'Q3');z=torch.randn(4,2048);q=torch.randn(4,64);o=torch.randn(4,4,8);p=[2,0,3,1]
    torch.testing.assert_close(m(z,q,o)[:,p],m(z,q,o[:,p]))


def test_clue_distance_and_logistic():
    x=torch.tensor([[0.,0.],[3.,4.]]);centers=torch.tensor([[0.,0.],[3.,4.]])
    np.testing.assert_allclose(centroid_score(x,centers),[-5,5])
    s=np.array([-2.,-1.,0.,0.,1.,2.]);y=np.array([0,0,0,1,1,1])
    fit=fit_logistic(s,y);assert fit['slope']>0
    assert np.isfinite(calibrated_logits(s,fit)).all()
    assert fit==fit_logistic(s,y)
    # Q1 independent Bernoulli NLL equals fractional-label NLL exactly.
    p=.37;assert np.mean([-np.log(p),-np.log(1-p)])==pytest.approx(-.5*np.log(p)-.5*np.log(1-p))


def test_split_guard():
    with pytest.raises(ValueError):require_split('test','validate')
    with pytest.raises(ValueError):require_split('test','train')


def test_serialized_near_tie_matches_table1_precision():
    from scipy.special import softmax
    x=np.array([[0.,1e-8],[0.,-1e-8]],dtype=np.float32)
    # Actual regression: float32 probability rounding creates an artificial
    # tie even though serialized logits have a strictly positive margin.
    assert softmax(x,axis=1).argmax(1).tolist()==[0,0]
    canonical=np.asarray(x.tolist(),dtype=np.float64)
    assert softmax(canonical,axis=1).argmax(1).tolist()==[1,0]
    np.testing.assert_array_equal(softmax(canonical,axis=1).argmax(1),canonical.argmax(1))
