import pytest
import torch
from clr.sources.common import RelativeNoise,max_latent_difference,question

def test_relative_noise_formula_boundary_and_rng():
    z=torch.arange(1,17,dtype=torch.float32).reshape(1,1,16)
    spec={'from_step':2,'sigma':.1,'seed':7}
    before=torch.get_rng_state().clone();noise=RelativeNoise(spec,'cpu')
    assert noise(z,1) is z
    eps=torch.randn(z.shape,generator=torch.Generator().manual_seed(7))
    expected=z+.1*z.norm(dim=-1,keepdim=True)*eps/4
    assert torch.equal(noise(z,2),expected)
    assert torch.equal(torch.get_rng_state(),before)
    assert torch.equal(RelativeNoise(spec,'cpu')(z,2),expected)
    assert not torch.equal(noise(z,3),expected)

def test_zero_noise_exact_identity_and_zero_norm():
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        z=torch.ones(10,dtype=dtype)
        assert RelativeNoise({'from_step':0,'sigma':0.,'seed':1},'cpu')(z,0) is z
        zero=torch.zeros(10,dtype=dtype)
        assert torch.equal(RelativeNoise({'from_step':0,'sigma':.3,'seed':1},'cpu')(zero,0),zero)

@pytest.mark.parametrize('sigma',[-.1,float('inf'),float('nan')])
def test_invalid_noise(sigma):
    with pytest.raises(ValueError): RelativeNoise({'from_step':0,'sigma':sigma,'seed':1},'cpu')

def test_latent_comparison_and_no_gold_input():
    assert max_latent_difference([torch.tensor([1.])],[torch.tensor([1.125])])==.125
    with pytest.raises(ValueError): max_latent_difference([],[])
    with pytest.raises(ValueError): max_latent_difference([torch.ones(2)],[torch.ones(3)])
    assert question({'question':'Q','answer':'SECRET','steps':['SECRET']})=='Q'
