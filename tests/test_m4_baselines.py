import torch
from clr.baselines.b3_multi_v0 import B3Multi

def test_shared_mlp_option_permutation_and_binary_heads():
    torch.manual_seed(5);model=B3Multi().eval();z=torch.randn(3,768);q=torch.randn(3,64);options=torch.randn(3,4,8);order=[3,1,0,2]
    with torch.no_grad():
        scores=model(z,q,options,'Q3');permuted=model(z,q,options[:,order],'Q3')
        torch.testing.assert_close(scores[:,order],permuted,atol=1e-7,rtol=1e-6)
        for kind in ['Q1','Q2','Q4']:
            logits=model(z,q,options[:,:1],kind)
            assert logits.shape==(3,2) and torch.isfinite(logits).all()
