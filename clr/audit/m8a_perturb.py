"""Constrained raw-KV perturbations; E2 uses an exact pooling-kernel subspace.

The kernel projector pairs non-tail tokens within each record and removes
their common component. Quantization uses a common BF16 lattice per pair;
nonrepresentable updates are zeroed, preserving opposing deltas after cast.
"""
import math
import torch


def to_vectors(pair):
    k,v=pair
    return torch.cat([k.permute(0,2,1,3).flatten(2),v.permute(0,2,1,3).flatten(2)],-1)


def from_vectors(x,heads,dim):
    k,v=x.split(heads*dim,-1)
    return tuple(t.reshape(x.shape[0],x.shape[1],heads,dim).permute(0,2,1,3).contiguous() for t in [k,v])


def pairing(mask,metadata,tail_tokens=8):
    valid=mask[0].bool();tail=set(valid.nonzero().flatten()[-tail_tokens:].tolist())
    offset=metadata.get('physical_offset',0);a=[];b=[]
    for start,end in metadata['record_spans']:
        positions=[i for i in range(start+offset,end+offset) if bool(valid[i]) and i not in tail]
        a.extend(positions[0:len(positions)//2*2:2]);b.extend(positions[1:len(positions)//2*2:2])
    if not a:raise ValueError('No non-tail record token pairs')
    return torch.tensor(a,device=mask.device),torch.tensor(b,device=mask.device)


def representable_pair_delta(a,b,delta):
    # Values are BF16 exactly represented in FP32. A common power-of-two
    # lattice makes updates representable at both endpoints unless exponent
    # crossings occur; detect those rather than assume them away.
    scale=torch.maximum(a.abs(),b.abs()).clamp_min(2.**-120)
    unit=torch.pow(2.,torch.floor(torch.log2(scale))-7)
    snapped=delta.sign()*torch.floor(delta.abs()/unit)*unit
    left=(a+snapped).to(torch.bfloat16).float()-a
    right=(b-snapped).to(torch.bfloat16).float()-b
    valid=(left==-right)&(left.abs()<=snapped.abs())
    return torch.where(valid,left,torch.zeros_like(left))


class Perturbation:
    def __init__(self,prefix,metadata,kind,read_layers):
        self.prefix,self.kind=prefix,kind
        self.layers=list(read_layers) if kind=='E2' else [i for i in range(len(prefix.layers)) if i not in read_layers] if kind=='E1' else list(range(len(prefix.layers)))
        self.base={l:to_vectors(prefix.layers[l]).float() for l in self.layers}
        self.norm={l:x.norm() for l,x in self.base.items()}
        self.valid=prefix.mask[...,None]
        self.pairs=pairing(prefix.mask,metadata) if kind=='E2' else None
        self.delta={l:torch.zeros_like(x[:,self.pairs[0]]) if kind=='E2' else torch.zeros_like(x) for l,x in self.base.items()}

    def reset(self):
        for d in self.delta.values():d.zero_()

    def leaf(self):
        self.delta={l:d.detach().requires_grad_(True) for l,d in self.delta.items()}

    def build(self,straight_through=True):
        from latdef.inject.cache import PrefixKV
        pairs=list(self.prefix.layers)
        for l in self.layers:
            base=self.base[l];d=self.delta[l]
            if self.kind=='E2':
                a,b=self.pairs
                snap=representable_pair_delta(base[:,a],base[:,b],d)
                actual=d+(snap-d).detach() if straight_through else snap
                change=torch.zeros_like(base).index_copy(1,a,actual).index_copy(1,b,-actual)
            else:
                raw=d*self.valid
                rounded=(base+raw).to(torch.bfloat16).float()-base
                # Rounding must not silently violate the raw-KV norm ball.
                good=(rounded.abs()<=raw.abs())&(rounded*raw>=0)
                snap=torch.where(good,rounded,torch.zeros_like(rounded))
                change=raw+(snap-raw).detach() if straight_through else snap
            x=(base+change).to(self.prefix.layers[l][0].dtype)
            heads,dim=self.prefix.layers[l][0].shape[1],self.prefix.layers[l][0].shape[3]
            pairs[l]=from_vectors(x,heads,dim)
        return PrefixKV(tuple(pairs),self.prefix.mask,self.prefix.active_layers,self.prefix.gates)

    @torch.no_grad()
    def step(self,grads,epsilon,steps,step_multiplier=2.5):
        for (l,d),g in zip(self.delta.items(),grads):
            if g is None or not torch.isfinite(g).all():raise ValueError('Invalid raw-KV PGD gradient')
            # E2 coordinates occur twice in the full KV norm.
            factor=math.sqrt(2) if self.kind=='E2' else 1.
            radius=(.3 if epsilon is None else epsilon)*self.norm[l]/factor
            updated=d+step_multiplier*radius/max(steps,1)*g/g.norm().clamp_min(1e-20)
            if epsilon is not None:updated*=torch.clamp(radius/updated.norm().clamp_min(1e-20),max=1.)
            if self.kind!='E2':updated*=self.valid
            self.delta[l]=updated.detach()

    @torch.no_grad()
    def ratios(self,prefix):
        return {str(l):float((to_vectors(prefix.layers[l]).float()-self.base[l]).norm()/self.norm[l]) for l in self.layers}

    @torch.no_grad()
    def enforce_fp16_identity(self,transform,metadata,clean):
        """Discard record updates where GEMM rounding changes stored reader input.

        Opposing BF16 deltas are in the real-arithmetic kernel. This additional
        restriction makes the actual FP16 reader features bitwise equal, while
        still reporting the FP32 accumulation residual separately.
        """
        if self.kind!='E2':return
        prefix=self.build(False);feature=transform(prefix,prefix.mask,metadata)
        bad=(feature.values.half()!=clean.values.half()).any(-1)
        a,_=self.pairs
        for layer in self.layers:
            ranges=feature.token_ranges[bad&(feature.layers==layer)]
            if not len(ranges):continue
            affected=((a[:,None]>=ranges[None,:,0])&(a[:,None]<ranges[None,:,1])).any(-1)
            self.delta[layer][:,affected]=0


def feature_difference(clean,changed):
    diff=(clean.values-changed.values).abs()
    return {'maximum_absolute':float(diff.max()),'fp16_exact_fraction':float((clean.values.half()==changed.values.half()).float().mean()),
            'relative_L2':float(diff.norm()/clean.values.norm().clamp_min(1e-20))}
