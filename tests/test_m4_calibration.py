import numpy as np
from clr.calib.m4 import fit,apply
from clr.calib.m23 import metrics

def test_binary_affine_can_correct_prior_and_choice_is_equivariant():
    logits=np.tile([0.,1.],(100,1));targets=np.eye(2)[[1]*20+[0]*80]
    a=fit(logits,targets,'affine');t=fit(logits,targets,'temperature')
    assert metrics(apply(logits,a),targets)['nll']<metrics(apply(logits,t),targets)['nll']-.1
    rng=np.random.default_rng(11);x=rng.normal(size=(100,4));y=np.eye(4)[rng.integers(0,4,100)];p=[2,0,3,1]
    a=fit(x,y,'affine');b=fit(x[:,p],y[:,p],'affine')
    np.testing.assert_allclose(apply(x,a)[:,p],apply(x[:,p],b),rtol=1e-6)
    assert a['bias']==b['bias']==0
