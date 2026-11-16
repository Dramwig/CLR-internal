"""Nonadaptive controls on actual KV, upstream of differentiable compression."""
import torch


def raw_control(prefix,kind,layers,seed,relative_noise=.1):
    from latdef.inject.cache import PrefixKV
    chosen=set(layers)
    if kind=='truncate':
        keep=max(1,prefix.length//2)
        return PrefixKV(tuple((k[:,:,:keep],v[:,:,:keep]) for k,v in prefix.layers),
                        prefix.mask[:,:keep],prefix.active_layers,prefix.gates)
    if kind=='zero_layer':
        # Partial layer zeroing: half of the selected audit layers.
        zero=set(sorted(chosen)[::2])
        pairs=tuple((torch.zeros_like(k),torch.zeros_like(v)) if i in zero else (k,v)
                    for i,(k,v) in enumerate(prefix.layers))
    elif kind=='noise':
        # One norm constraint per message, jointly across selected K/V layers.
        generator=torch.Generator(device=prefix.mask.device).manual_seed(seed)
        noise={};energy=torch.zeros(len(prefix.mask),device=prefix.mask.device)
        norm=energy.clone()
        for i in sorted(chosen):
            for j,tensor in enumerate(prefix.layers[i]):
                eps=torch.randn(tensor.shape,device=tensor.device,dtype=torch.float32,generator=generator)
                valid=prefix.mask[:,None,:,None].to(eps)
                eps*=valid;noise[i,j]=eps
                energy+=eps.square().sum((1,2,3))
                norm+=(tensor.float()*valid).square().sum((1,2,3))
        scale=relative_noise*(norm/energy.clamp_min(1e-12)).sqrt()
        for attempt in range(12):
            pairs=[];actual=torch.zeros_like(norm)
            for i,(k,v) in enumerate(prefix.layers):
                pair=tuple((t.float()+noise[i,j]*scale[:,None,None,None]).to(t)
                           for j,t in enumerate((k,v))) if i in chosen else (k,v)
                pairs.append(pair)
                if i in chosen:
                    for before,after in zip((k,v),pair):
                        actual+=(after.float()-before.float()).square().sum((1,2,3))
            ratio=(actual/norm.clamp_min(1e-12)).sqrt()
            excess=ratio>relative_noise*(1+1e-6)
            if not excess.any():break
            scale*=torch.where(excess,relative_noise/ratio.clamp_min(1e-12)*.999,1.)
        else:raise ValueError('Quantized raw KV noise cannot satisfy the norm bound')
        pairs=tuple(pairs)
    else:raise ValueError('Unknown raw KV control '+kind)
    return PrefixKV(pairs,prefix.mask,prefix.active_layers,prefix.gates)


def raw_norms(prefix,layers,batch_index=0):
    valid=prefix.mask[batch_index].bool()
    if not valid.any():raise ValueError('Control removed every valid message token')
    return [[float(t[batch_index,:,valid].float().square().mean()) for t in prefix.layers[i]]
            for i in layers]
