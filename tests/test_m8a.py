import torch
from clr.audit.m8a_perturb import pairing,to_vectors,from_vectors,representable_pair_delta

def test_roundtrip_head_concatenation():
    k=torch.randn(1,2,9,3);v=torch.randn_like(k)
    kk,vv=from_vectors(to_vectors((k,v)),2,3)
    assert torch.equal(k,kk) and torch.equal(v,vv)

def test_kernel_pairs_respect_boundaries_and_tail():
    a,b=pairing(torch.ones(1,24,dtype=torch.bool),{'record_spans':[[0,5],[5,24]]})
    assert len(a)==len(b) and (b<16).all()
    assert (a<5).eq(b<5).all()

def test_quantized_pair_preserves_actual_transmission_sum():
    torch.manual_seed(11)
    a=torch.randn(300,20).bfloat16().float();b=torch.randn_like(a).bfloat16().float()
    d=torch.randn_like(a)*.02
    snap=representable_pair_delta(a,b,d)
    left=(a+snap).bfloat16().float()-a;right=(b-snap).bfloat16().float()-b
    assert torch.equal(left,-right)
    assert (snap.abs()<=d.abs()+1e-7).all()
    assert snap.abs().sum()>0

def test_pair_projection_is_in_pooling_kernel_and_differentiable():
    x=torch.randn(12,7,requires_grad=True)
    a=torch.tensor([0,2,4,6]);b=a+1
    coordinates=(x[a]-x[b])/2
    projected=torch.zeros_like(x).index_copy(0,a,coordinates).index_copy(0,b,-coordinates)
    assert torch.equal(projected.sum(0),torch.zeros(7))
    projected.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
