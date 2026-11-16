import numpy as np
import torch
from clr.baselines.m10d import scalar_model, equal_step_statistics
from clr.data.m4.reader import require_split
import pytest


def test_scalar_uses_current_state_only_and_q2_reuses_score():
    model=scalar_model(2048)
    z=torch.randn(3,2048)
    a=model(z,torch.zeros(3,64),torch.zeros(3,1,8))
    b=model(z,torch.randn(3,64),torch.randn(3,1,8))
    assert torch.equal(a,b)
    assert a.shape==(3,2)
    assert torch.equal(a[:,0],torch.zeros(3))
    a[:,1].sum().backward()
    assert model.head.weight.grad.abs().sum()>0


def test_bernoulli_replication_keeps_weights_and_step_strata():
    def row(t,y):
        return dict(state_ref=dict(kind='clean',t=t),label_soft=y)
    rs=[row(0,0.),row(0,1.),row(0,1.),row(1,0.)]
    x=np.array([[0.,1.]]*4)
    ss=equal_step_statistics(rs,x,1.,'Q1')
    assert [r['n'] for r in ss]==[3,1]
    assert ss[0]['accuracy']==pytest.approx(2/3)
    assert ss[1]['accuracy']==0
    assert np.mean([r['accuracy'] for r in ss])==pytest.approx(1/3)


def test_test_is_final_only():
    for phase in ['train','validate','calibrate']:
        with pytest.raises(ValueError):require_split('test',phase)
    require_split('test','final')
